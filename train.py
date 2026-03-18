from typing import Any
import jax.numpy as jnp
from absl import app, flags
from functools import partial
import numpy as np
import tqdm
import jax
import jax.numpy as jnp
import flax
import optax
import wandb

from ml_collections import config_flags
import ml_collections

from utils.wandb import setup_wandb, default_wandb_config
from utils.train_state import TrainStateEma
from utils.checkpoint import Checkpoint
from utils.stable_vae import StableVAE
from utils.sharding import create_sharding, all_gather
from utils.datasets import get_dataset
from model import DiT as DiTBase
from helper_eval import eval_model
from helper_inference import do_inference

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_name', 'imagenet256', 'Environment name.')
flags.DEFINE_string(
    'load_dir', None, 'Logging dir (if not None, save params).')
flags.DEFINE_string(
    'save_dir', None, 'Logging dir (if not None, save params).')
flags.DEFINE_string('fid_stats', None, 'FID stats file.')
# Must be the same across all processes.
flags.DEFINE_integer('seed', 10, 'Random seed.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 20000, 'Eval interval.')
flags.DEFINE_integer('save_interval', 100000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 32, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1_000_000), 'Number of training steps.')
flags.DEFINE_integer('debug_overfit', 0, 'Debug overfitting.')
flags.DEFINE_string('mode', 'train', 'train or inference.')

model_config = ml_collections.ConfigDict({
    'lr': 0.0001,
    'beta1': 0.9,
    'beta2': 0.999,
    'weight_decay': 0.1,
    'use_cosine': 0,
    'warmup': 0,
    'dropout': 0.0,
    'hidden_size': 768,  # change this!
    'patch_size': 8,  # change this!
    'depth': 2,  # change this!
    'num_heads': 2,  # change this!
    'mlp_ratio': 1,  # change this!
    'class_dropout_prob': 0.1,
    'num_classes': 1000,
    'denoise_timesteps': 128,
    'cfg_scale': 4.0,
    'target_update_rate': 0.999,
    'use_ema': 1,
    'use_stable_vae': 1,
    'sharding': 'dp',  # dp or fsdp.
    't_sampling': 'discrete-dt',
    'dt_sampling': 'uniform',
    'bootstrap_cfg': 0,
    'bootstrap_every': 4,  # Make sure its a divisor of batch size.
    'bootstrap_ema': 1,
    'bootstrap_dt_bias': 0,
    'train_type': 'shortcut',  # or naive, projected_diag_gmm.
    # GMM config (only used when train_type='projected_diag_gmm').
    'gmm_num_modes': 8,
    'gmm_top_m': 1,
    'gmm_use_router_cond': 0,
    'gmm_mix_weight': 1.0,
    'gmm_mix_normalize_by_dim': 1,
    'gmm_bal_weight': 0.01,
    'gmm_varreg_weight': 0.01,
    'gmm_proj_eps': 1e-6,
    'gmm_cov_eps': 1e-6,
    'gmm_use_warmup': 0,
    'gmm_warmup_iters': 0,
    'gmm_warmup_mode': 'mix_only',
    'gmm_radius_low': 0.9,
    'gmm_radius_high': 1.1,
    'gmm_radius_mu': 0.0,
    'gmm_radius_sigma': 0.25,
    'gmm_use_prior_ema': 1,
    'gmm_stop_gradient_q_top': 0,
})


wandb_config = default_wandb_config()
wandb_config.update({
    'project': 'shortcut',
    'name': 'shortcut_{dataset_name}',
})

config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('model', model_config, lock_config=False)

##############################################
# Training Code.
##############################################


def main(_):

    np.random.seed(FLAGS.seed)
    print("Using devices", jax.local_devices())
    device_count = len(jax.local_devices())
    global_device_count = jax.device_count()
    print("Device count", device_count)
    print("Global device count", global_device_count)
    local_batch_size = FLAGS.batch_size // (
        global_device_count // device_count)
    print("Global Batch: ", FLAGS.batch_size)
    print("Node Batch: ", local_batch_size)
    print("Device Batch:", local_batch_size // device_count)

    # Create wandb logger
    if jax.process_index() == 0 and FLAGS.mode == 'train':
        setup_wandb(FLAGS.model.to_dict(), **FLAGS.wandb)

    dataset = get_dataset(FLAGS.dataset_name,
                          local_batch_size, True, FLAGS.debug_overfit)
    dataset_valid = get_dataset(
        FLAGS.dataset_name, local_batch_size, False, FLAGS.debug_overfit)
    example_obs, example_labels = next(dataset)
    example_obs = example_obs[:1]
    example_obs_shape = example_obs.shape

    if FLAGS.model.use_stable_vae:
        vae = StableVAE.create()
        if 'latent' in FLAGS.dataset_name:
            example_obs = example_obs[:, :, :, example_obs.shape[-1] // 2:]
            example_obs_shape = example_obs.shape
        else:
            example_obs = vae.encode(jax.random.PRNGKey(0), example_obs)
        example_obs_shape = example_obs.shape
        vae_rng = jax.random.PRNGKey(42)
        vae_encode = jax.jit(vae.encode)
        vae_decode = jax.jit(vae.decode)

    if FLAGS.fid_stats is not None:
        from utils.fid import get_fid_network, fid_from_stats
        get_fid_activations = get_fid_network()
        truth_fid_stats = np.load(FLAGS.fid_stats)
    else:
        get_fid_activations = None
        truth_fid_stats = None

    ###################################
    # Creating Model and put on devices.
    ###################################
    FLAGS.model.image_channels = example_obs_shape[-1]
    FLAGS.model.image_size = example_obs_shape[1]
    dit_args = {
        'patch_size': FLAGS.model['patch_size'],
        'hidden_size': FLAGS.model['hidden_size'],
        'depth': FLAGS.model['depth'],
        'num_heads': FLAGS.model['num_heads'],
        'mlp_ratio': FLAGS.model['mlp_ratio'],
        'out_channels': example_obs_shape[-1],
        'class_dropout_prob': FLAGS.model['class_dropout_prob'],
        'num_classes': FLAGS.model['num_classes'],
        'dropout': FLAGS.model['dropout'],
        'ignore_dt': False if (FLAGS.model['train_type'] in ('shortcut', 'livereflow', 'projected_diag_gmm')) else True,
    }
    if FLAGS.model['train_type'] == 'projected_diag_gmm' and FLAGS.model['gmm_use_router_cond']:
        from model_router_cond import DiT as DiTRouterCond
        dit_args['num_modes'] = FLAGS.model['gmm_num_modes']
        model_def = DiTRouterCond(**dit_args)
        tabulate_fn = flax.linen.tabulate(model_def, jax.random.PRNGKey(0))
        example_rc = jnp.zeros((1, FLAGS.model['gmm_num_modes']))
        print(tabulate_fn(example_obs, jnp.zeros((1,)),
              jnp.zeros((1,)), jnp.zeros((1,), dtype=jnp.int32), example_rc))
    else:
        model_def = DiTBase(**dit_args)
        tabulate_fn = flax.linen.tabulate(model_def, jax.random.PRNGKey(0))
        print(tabulate_fn(example_obs, jnp.zeros((1,)),
              jnp.zeros((1,)), jnp.zeros((1,), dtype=jnp.int32)))

    if FLAGS.model.use_cosine:
        lr_schedule = optax.warmup_cosine_decay_schedule(
            0.0, FLAGS.model['lr'], FLAGS.model['warmup'], FLAGS.max_steps)
    elif FLAGS.model.warmup > 0:
        lr_schedule = optax.linear_schedule(
            0.0, FLAGS.model['lr'], FLAGS.model['warmup'])
    else:
        def lr_schedule(x): return FLAGS.model['lr']
    adam = optax.adamw(learning_rate=lr_schedule,
                       b1=FLAGS.model['beta1'], b2=FLAGS.model['beta2'], weight_decay=FLAGS.model['weight_decay'])
    tx = optax.chain(adam)

    def init(rng):
        param_key, dropout_key, dropout2_key, prior_key = jax.random.split(rng, 4)
        example_t = jnp.zeros((1,))
        example_dt = jnp.zeros((1,))
        example_label = jnp.zeros((1,), dtype=jnp.int32)
        example_obs = jnp.zeros(example_obs_shape)
        model_rngs = {'params': param_key,
                      'label_dropout': dropout_key, 'dropout': dropout2_key}

        if FLAGS.model['train_type'] == 'projected_diag_gmm':
            from utils.train_state_projected_gmm import TrainStateProjectedGMMEma

            if FLAGS.model['gmm_use_router_cond']:
                example_rc = jnp.zeros((1, FLAGS.model['gmm_num_modes']))
                model_params = model_def.init(model_rngs, example_obs,
                                              example_t, example_dt, example_label, example_rc)['params']
            else:
                model_params = model_def.init(model_rngs, example_obs,
                                              example_t, example_dt, example_label)['params']

            D = int(np.prod(example_obs_shape[1:]))
            K = FLAGS.model['gmm_num_modes']
            assert FLAGS.model['gmm_top_m'] <= K, f"gmm_top_m={FLAGS.model['gmm_top_m']} > gmm_num_modes={K}"
            assert FLAGS.model['gmm_warmup_mode'] in ('mix_only', 'mix_bal'), \
                f"Unknown gmm_warmup_mode: {FLAGS.model['gmm_warmup_mode']}"
            prior_params = {
                'pi_logits': jnp.zeros((K,), dtype=jnp.float32),
                'mu': 0.02 * jax.random.normal(prior_key, (K, D), dtype=jnp.float32),
                'r_raw': jnp.zeros((K, D), dtype=jnp.float32),
            }
            print(f"GMM prior: K={K}, D={D}, prior param count={K + 2*K*D}")

            params = {"model": model_params, "prior": prior_params}
            opt_state = tx.init(params)
            return TrainStateProjectedGMMEma.create(model_def, params, rng=rng, tx=tx, opt_state=opt_state)
        else:
            params = model_def.init(model_rngs, example_obs,
                                    example_t, example_dt, example_label)['params']
            opt_state = tx.init(params)
            return TrainStateEma.create(model_def, params, rng=rng, tx=tx, opt_state=opt_state)

    rng = jax.random.PRNGKey(FLAGS.seed)
    train_state_shape = jax.eval_shape(init, rng)

    data_sharding, train_state_sharding, no_shard, shard_data, global_to_local = create_sharding(
        FLAGS.model.sharding, train_state_shape)
    train_state = jax.jit(init, out_shardings=train_state_sharding)(rng)
    if FLAGS.model['train_type'] == 'projected_diag_gmm':
        _p = train_state.params['model']
    else:
        _p = train_state.params
    jax.debug.visualize_array_sharding(
        _p['FinalLayer_0']['Dense_0']['kernel'])
    jax.debug.visualize_array_sharding(
        _p['TimestepEmbedder_1']['Dense_0']['kernel'])
    jax.experimental.multihost_utils.assert_equal(
        _p['TimestepEmbedder_1']['Dense_0']['kernel'])
    start_step = 1

    if FLAGS.load_dir is not None:
        cp = Checkpoint(FLAGS.load_dir)
        replace_dict = cp.load_as_dict()['train_state']
        del replace_dict['opt_state']  # Debug
        # Handle loading non-GMM checkpoint into GMM mode.
        if FLAGS.model['train_type'] == 'projected_diag_gmm':
            if 'model' not in replace_dict.get('params', {}):
                print("Converting non-GMM checkpoint to GMM format (keeping current prior).")
                replace_dict['params'] = {"model": replace_dict['params'], "prior": train_state.params["prior"]}
                replace_dict['params_ema'] = {"model": replace_dict['params_ema'], "prior": train_state.params_ema["prior"]}
        train_state = train_state.replace(**replace_dict)
        if FLAGS.wandb.run_id != "None":  # If we are continuing a run.
            start_step = train_state.step
        train_state = jax.jit(
            lambda x: x, out_shardings=train_state_sharding)(train_state)
        print("Loaded model with step", train_state.step)
        train_state = train_state.replace(step=0)
        if FLAGS.model['train_type'] == 'projected_diag_gmm':
            _p_load = train_state.params['model']
        else:
            _p_load = train_state.params
        jax.debug.visualize_array_sharding(
            _p_load['FinalLayer_0']['Dense_0']['kernel'])
        del cp

    if FLAGS.model.train_type == 'progressive' or FLAGS.model.train_type == 'consistency-distillation':
        train_state_teacher = jax.jit(
            lambda x: x, out_shardings=train_state_sharding)(train_state)
    else:
        train_state_teacher = None

    visualize_labels = example_labels
    visualize_labels = shard_data(visualize_labels)
    visualize_labels = jax.experimental.multihost_utils.process_allgather(
        visualize_labels)
    imagenet_labels = open('data/imagenet_labels.txt').read().splitlines()

    ###################################
    # Update Function
    ###################################

    @partial(jax.jit, out_shardings=(train_state_sharding, no_shard))
    def update(train_state, train_state_teacher, images, labels, force_t=-1, force_dt=-1):
        new_rng, targets_key, dropout_key, perm_key = jax.random.split(
            train_state.rng, 4)
        info = {}

        id_perm = jax.random.permutation(perm_key, images.shape[0])
        images = images[id_perm]
        labels = labels[id_perm]
        images = jax.lax.with_sharding_constraint(images, data_sharding)
        labels = jax.lax.with_sharding_constraint(labels, data_sharding)

        if FLAGS.model['cfg_scale'] == 0:  # For unconditional generation.
            labels = jnp.ones(
                labels.shape[0], dtype=jnp.int32) * FLAGS.model['num_classes']

        if FLAGS.model['train_type'] == 'projected_diag_gmm':
            from baselines.targets_projected_diag_gmm import sample_time_and_labels, apply_label_dropout, sample_batch_radius
            from utils.projected_diag_gmm import (
                flatten_latent, unflatten_latent, compute_router_posterior,
                select_top_m, sample_projected_sources, apply_sample_radius,
                build_sparse_router_cond, compute_mix_loss, compute_bal_loss,
                compute_var_reg_loss
            )

            time_key, radius_key, source_key, label_key = jax.random.split(targets_key, 4)
            t, dt_base, pre_info = sample_time_and_labels(time_key, images, FLAGS)
            labels_dropped, dropped_ratio = apply_label_dropout(label_key, labels, FLAGS)
            info['dropped_ratio'] = dropped_ratio

            # Handle force_t.
            force_t_vec = jnp.ones(images.shape[0], dtype=jnp.float32) * force_t
            t = jnp.where(force_t_vec != -1, force_t_vec, t)

            r_scalar = sample_batch_radius(
                radius_key, images.shape[0],
                FLAGS.model['gmm_radius_low'], FLAGS.model['gmm_radius_high'],
                FLAGS.model['gmm_radius_mu'], FLAGS.model['gmm_radius_sigma'])

            B = images.shape[0]
            H, W, C = images.shape[1], images.shape[2], images.shape[3]
            M = FLAGS.model['gmm_top_m']

            def loss_fn(grad_params):
                prior_params = grad_params["prior"]
                x1_flat = flatten_latent(images)  # [B, D]

                # Router posterior (differentiable w.r.t. prior_params).
                q_full, log_mixprob, router_stats = compute_router_posterior(
                    x1_flat, prior_params, FLAGS.model['gmm_cov_eps'])

                # Top-M selection.
                top_idx, q_top = select_top_m(q_full, M)

                # Sample projected sources (differentiable via reparameterization).
                x0_dir_bmd, source_stats = sample_projected_sources(
                    source_key, prior_params, top_idx, FLAGS.model['gmm_proj_eps'])

                # Apply per-sample radius (stop_gradient on r).
                x0_flat_bmd = apply_sample_radius(x0_dir_bmd, r_scalar)  # [B, M, D]

                # Unflatten to spatial.
                x0_bmhwc = unflatten_latent(x0_flat_bmd, (H, W, C))  # [B, M, H, W, C]
                x1_bmhwc = images[:, None, :, :, :]  # [B, 1, H, W, C] -> broadcast

                # Build flow pair.
                t_full = t[:, None, None, None, None]  # [B, 1, 1, 1, 1]
                x_t_bmhwc = (1 - (1 - 1e-5) * t_full) * x0_bmhwc + t_full * x1_bmhwc
                v_t_bmhwc = x1_bmhwc - (1 - 1e-5) * x0_bmhwc

                # Reshape [B, M, H, W, C] -> [B*M, H, W, C] for single model forward.
                x_t_flat = x_t_bmhwc.reshape(B * M, H, W, C)
                v_t_flat = v_t_bmhwc.reshape(B * M, H, W, C)
                t_rep = jnp.repeat(t, M)                    # [B*M]
                dt_base_rep = jnp.repeat(dt_base, M)        # [B*M]
                labels_rep = jnp.repeat(labels_dropped, M)  # [B*M]

                # Model forward.
                if FLAGS.model['gmm_use_router_cond']:
                    router_cond_sparse = build_sparse_router_cond(
                        top_idx, q_top, FLAGS.model['gmm_num_modes'])
                    router_cond_rep = jnp.repeat(router_cond_sparse, M, axis=0)
                    v_prime, logvars, activations = train_state.call_model(
                        x_t_flat, t_rep, dt_base_rep, labels_rep,
                        router_cond_rep,
                        train=True, rngs={'dropout': dropout_key},
                        params=grad_params, return_activations=True)
                else:
                    v_prime, logvars, activations = train_state.call_model(
                        x_t_flat, t_rep, dt_base_rep, labels_rep,
                        train=True, rngs={'dropout': dropout_key},
                        params=grad_params, return_activations=True)

                # MSE per sample*mode -> reshape [B, M].
                mse_flat = jnp.mean((v_prime - v_t_flat) ** 2, axis=(1, 2, 3))  # [B*M]
                mse_bm = mse_flat.reshape(B, M)

                # Weight by q_top.
                if FLAGS.model['gmm_stop_gradient_q_top']:
                    q_top_w = jax.lax.stop_gradient(q_top)
                else:
                    q_top_w = q_top
                weighted_mse = jnp.sum(q_top_w * mse_bm, axis=-1)  # [B]
                loss_fm = jnp.mean(weighted_mse)

                # Auxiliary losses (use dense full-K posterior).
                loss_mix_raw = compute_mix_loss(log_mixprob)
                if FLAGS.model['gmm_mix_normalize_by_dim']:
                    mix_norm_factor = jnp.asarray(
                        x1_flat.shape[-1], dtype=loss_mix_raw.dtype)
                    loss_mix = loss_mix_raw / mix_norm_factor
                else:
                    mix_norm_factor = jnp.asarray(1.0, dtype=loss_mix_raw.dtype)
                    loss_mix = loss_mix_raw
                loss_bal = compute_bal_loss(q_full)
                loss_varreg = compute_var_reg_loss(
                    prior_params['r_raw'], FLAGS.model['gmm_cov_eps'])

                # Warmup logic.
                lambda_mix = FLAGS.model['gmm_mix_weight']
                lambda_bal = FLAGS.model['gmm_bal_weight']
                lambda_varreg = FLAGS.model['gmm_varreg_weight']
                if FLAGS.model['gmm_use_warmup']:
                    is_warmup = train_state.step < FLAGS.model['gmm_warmup_iters']
                    if FLAGS.model['gmm_warmup_mode'] == 'mix_only':
                        warmup_loss = lambda_mix * loss_mix + lambda_varreg * loss_varreg
                    else:  # mix_bal
                        warmup_loss = lambda_mix * loss_mix + \
                            lambda_bal * loss_bal + lambda_varreg * loss_varreg
                    full_loss = loss_fm + lambda_mix * \
                        loss_mix + lambda_bal * loss_bal + lambda_varreg * loss_varreg
                    loss = jnp.where(is_warmup, warmup_loss, full_loss)
                else:
                    loss = loss_fm + lambda_mix * loss_mix + \
                        lambda_bal * loss_bal + lambda_varreg * loss_varreg

                # Logging.
                x0_norms = jnp.sqrt(jnp.sum(x0_flat_bmd ** 2, axis=-1))  # [B, M]
                mean_q = jnp.mean(q_full, axis=0)  # [K]
                var_all = jax.nn.softplus(prior_params['r_raw']) ** 2 + \
                    FLAGS.model['gmm_cov_eps']
                info = {
                    'loss': loss,
                    'loss_flow': loss_fm,
                    'loss_mix': loss_mix,
                    'loss_mix_raw': loss_mix_raw,
                    'loss_mix_norm_factor': mix_norm_factor,
                    'loss_bal': loss_bal,
                    'loss_varreg': loss_varreg,
                    'in_gmm_warmup': jnp.where(
                        FLAGS.model['gmm_use_warmup'],
                        (train_state.step < FLAGS.model['gmm_warmup_iters']).astype(jnp.float32),
                        0.0),
                    'v_magnitude_prime': jnp.sqrt(jnp.mean(jnp.square(v_prime))),
                    'source_x0_norm_mean': jnp.mean(x0_norms),
                    'source_x0_norm_std': jnp.std(x0_norms),
                    'router_usage_min': jnp.min(mean_q),
                    'router_usage_max': jnp.max(mean_q),
                    'prior_var_mean': jnp.mean(var_all),
                    'prior_var_min': jnp.min(var_all),
                    'prior_var_max': jnp.max(var_all),
                    'prior_var_dev_abs_mean': jnp.mean(jnp.abs(var_all - 1.0)),
                    'has_nan_loss': jnp.any(jnp.isnan(loss)).astype(jnp.float32),
                    **router_stats,
                    **source_stats,
                    **{'activations/' + k: jnp.sqrt(jnp.mean(jnp.square(v))) for k, v in activations.items()},
                }
                return loss, info

        else:
            # All other train types: get_targets outside loss_fn.
            if FLAGS.model['train_type'] == 'naive':
                from baselines.targets_naive import get_targets
                x_t, v_t, t, dt_base, labels, info = get_targets(
                    FLAGS, targets_key, train_state, images, labels, force_t, force_dt)
            elif FLAGS.model['train_type'] == 'shortcut':
                from targets_shortcut import get_targets
                x_t, v_t, t, dt_base, labels, info = get_targets(
                    FLAGS, targets_key, train_state, images, labels, force_t, force_dt)
            elif FLAGS.model['train_type'] == 'progressive':
                from baselines.targets_progressive import get_targets
                x_t, v_t, t, dt_base, labels, info = get_targets(
                    FLAGS, targets_key, train_state, train_state_teacher, images, labels, force_t, force_dt)
            elif FLAGS.model['train_type'] == 'consistency-distillation':
                from baselines.targets_consistency_distillation import get_targets
                x_t, v_t, t, dt_base, labels, info = get_targets(
                    FLAGS, targets_key, train_state, train_state_teacher, images, labels, force_t, force_dt)
            elif FLAGS.model['train_type'] == 'consistency':
                from baselines.targets_consistency_training import get_targets
                x_t, v_t, t, dt_base, labels, info = get_targets(
                    FLAGS, targets_key, train_state, images, labels, force_t, force_dt)
            elif FLAGS.model['train_type'] == 'livereflow':
                from baselines.targets_livereflow import get_targets
                x_t, v_t, t, dt_base, labels, info = get_targets(
                    FLAGS, targets_key, train_state, images, labels, force_t, force_dt)

            def loss_fn(grad_params):
                v_prime, logvars, activations = train_state.call_model(x_t, t, dt_base, labels, train=True, rngs={
                                                                       'dropout': dropout_key}, params=grad_params, return_activations=True)
                mse_v = jnp.mean((v_prime - v_t) ** 2, axis=(1, 2, 3))
                loss = jnp.mean(mse_v)

                info = {
                    'loss': loss,
                    'v_magnitude_prime': jnp.sqrt(jnp.mean(jnp.square(v_prime))),
                    **{'activations/' + k: jnp.sqrt(jnp.mean(jnp.square(v))) for k, v in activations.items()},
                }

                if FLAGS.model['train_type'] == 'shortcut' or FLAGS.model['train_type'] == 'livereflow':
                    bootstrap_size = FLAGS.batch_size // FLAGS.model['bootstrap_every']
                    info['loss_flow'] = jnp.mean(mse_v[bootstrap_size:])
                    info['loss_bootstrap'] = jnp.mean(mse_v[:bootstrap_size])

                return loss, info

        grads, new_info = jax.grad(loss_fn, has_aux=True)(train_state.params)
        info = {**info, **new_info}
        updates, new_opt_state = train_state.tx.update(
            grads, train_state.opt_state, train_state.params)
        new_params = optax.apply_updates(train_state.params, updates)

        info['grad_norm'] = optax.global_norm(grads)
        info['update_norm'] = optax.global_norm(updates)
        info['param_norm'] = optax.global_norm(new_params)
        info['lr'] = lr_schedule(train_state.step)

        train_state = train_state.replace(
            rng=new_rng, step=train_state.step + 1, params=new_params, opt_state=new_opt_state)
        train_state = train_state.update_ema(FLAGS.model['target_update_rate'])
        return train_state, info

    if FLAGS.mode != 'train':
        if FLAGS.model['train_type'] == 'projected_diag_gmm':
            from helper_inference_projected_gmm import do_inference as do_inference_gmm
            do_inference_gmm(FLAGS, train_state, None, dataset, dataset_valid, shard_data, vae_encode, vae_decode, update,
                         get_fid_activations, imagenet_labels, visualize_labels,
                         fid_from_stats, truth_fid_stats)
        else:
            do_inference(FLAGS, train_state, None, dataset, dataset_valid, shard_data, vae_encode, vae_decode, update,
                         get_fid_activations, imagenet_labels, visualize_labels,
                         fid_from_stats, truth_fid_stats)
        return

    ###################################
    # Train Loop
    ###################################

    for i in tqdm.tqdm(range(1 + start_step, FLAGS.max_steps + 1 + start_step),
                       smoothing=0.1,
                       dynamic_ncols=True):

        # Sample data.
        if not FLAGS.debug_overfit or i == 1:
            batch_images, batch_labels = shard_data(*next(dataset))
            if FLAGS.model.use_stable_vae and 'latent' not in FLAGS.dataset_name:
                vae_rng, vae_key = jax.random.split(vae_rng)
                batch_images = vae_encode(vae_key, batch_images)

        # Train update.
        train_state, update_info = update(
            train_state, train_state_teacher, batch_images, batch_labels)

        if i % FLAGS.log_interval == 0 or i == 1:
            update_info = jax.device_get(update_info)
            update_info = jax.tree_map(lambda x: np.array(x), update_info)
            update_info = jax.tree_map(lambda x: x.mean(), update_info)
            train_metrics = {f'training/{k}': v for k,
                             v in update_info.items()}

            valid_images, valid_labels = shard_data(*next(dataset_valid))
            if FLAGS.model.use_stable_vae and 'latent' not in FLAGS.dataset_name:
                valid_images = vae_encode(vae_rng, valid_images)
            _, valid_update_info = update(
                train_state, train_state_teacher, valid_images, valid_labels)
            valid_update_info = jax.device_get(valid_update_info)
            valid_update_info = jax.tree_map(
                lambda x: x.mean(), valid_update_info)
            train_metrics['training/loss_valid'] = valid_update_info['loss']

            if jax.process_index() == 0:
                wandb.log(train_metrics, step=i)

        if FLAGS.model['train_type'] == 'progressive':
            num_sections = np.log2(
                FLAGS.model['denoise_timesteps']).astype(jnp.int32)
            if i % (FLAGS.max_steps // num_sections) == 0:
                train_state_teacher = jax.jit(
                    lambda x: x, out_shardings=train_state_sharding)(train_state)

        if i % FLAGS.eval_interval == 0:
            if FLAGS.model['train_type'] == 'projected_diag_gmm':
                from helper_eval_projected_gmm import eval_model as eval_model_gmm
                eval_model_gmm(FLAGS, train_state, train_state_teacher, i, dataset, dataset_valid, shard_data, vae_encode, vae_decode, update,
                           get_fid_activations, imagenet_labels, visualize_labels,
                           fid_from_stats, truth_fid_stats)
            else:
                eval_model(FLAGS, train_state, train_state_teacher, i, dataset, dataset_valid, shard_data, vae_encode, vae_decode, update,
                           get_fid_activations, imagenet_labels, visualize_labels,
                           fid_from_stats, truth_fid_stats)

        if i % FLAGS.save_interval == 0 and FLAGS.save_dir is not None:
            train_state_gather = jax.experimental.multihost_utils.process_allgather(
                train_state)
            if jax.process_index() == 0:
                cp = Checkpoint(
                    FLAGS.save_dir+str(train_state_gather.step+1), parallel=False)
                cp.train_state = train_state_gather
                cp.save()
                del cp
            del train_state_gather


if __name__ == '__main__':
    app.run(main)
