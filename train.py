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
    # Stage scheduler: A (GMM warmup), B (FM Gaussian warmup), C (joint).
    'gmm_stage_a_iters': 0,
    'gmm_stage_b_iters': 0,
    # Prior update accumulation during stage C.
    'gmm_prior_accum_steps': 10,
    # Stage A loss weights.
    'gmm_stage_a_mix_weight': 1.0,
    'gmm_stage_a_bal_weight': 0.01,
    'gmm_stage_a_var_weight': 0.01,
    # Stage C joint loss weights.
    'gmm_joint_bal_weight': 0.01,
    'gmm_joint_var_weight': 0.01,
    # Stage B FM source controls.
    'gmm_stage_b_noise_std': 1.0,
    'gmm_stage_b_random_pair': 1,
    # Stage C within-cluster data coupling controls (gmm_use_cluster_data=1).
    'gmm_use_cluster_data': 1,
    'gmm_cluster_sample_uniform': 1,
    'gmm_cluster_reassign_interval': 5000,
    'gmm_cluster_save_path': '/tmp/gmm_clusters/',
    # Stage C best-of-n fallback controls (gmm_use_cluster_data=0, kept for ablation).
    'gmm_best_of_n': 1,
    'gmm_best_of_n_threshold': 16,
    'gmm_best_of_n_chunk': 4,
    'gmm_proj_eps': 1e-6,
    'gmm_cov_eps': 1e-6,
    # Legacy flags (kept only for compatibility; ignored by the new stage-based implementation).
    'gmm_mix_weight': 1.0,
    'gmm_bal_weight': 0.01,
    'gmm_varreg_weight': 0.01,
    'gmm_mix_normalize_by_dim': 1,
    'gmm_fm_pretrain_iters': 0,
    'gmm_fm_pretrain_noise_std': 1.0,
    'gmm_fm_pretrain_random_pair': 1,
    'gmm_use_warmup': 0,
    'gmm_warmup_iters': 0,
    'gmm_warmup_mode': 'mix_only',
    'gmm_radius_low': 63.75,
    'gmm_radius_high': 64.25,
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
    if FLAGS.model['train_type'] == 'projected_diag_gmm':
        if FLAGS.model['gmm_top_m'] != 1:
            print(
                f"[projected_diag_gmm] hard top-1 is enforced; ignoring gmm_top_m={FLAGS.model['gmm_top_m']}."
            )
        if FLAGS.model.get('gmm_use_warmup', 0) or FLAGS.model.get('gmm_warmup_iters', 0) > 0:
            print("[projected_diag_gmm] legacy gmm_use_warmup/gmm_warmup_* is ignored in stage-based training.")
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
            assert K >= 1, f"gmm_num_modes must be >=1, got {K}"
            assert FLAGS.model['gmm_prior_accum_steps'] >= 1, \
                f"gmm_prior_accum_steps must be >=1, got {FLAGS.model['gmm_prior_accum_steps']}"
            assert FLAGS.model['gmm_stage_a_iters'] >= 0 and FLAGS.model['gmm_stage_b_iters'] >= 0, \
                "gmm_stage_a_iters and gmm_stage_b_iters must be non-negative."
            assert FLAGS.model.get('gmm_cluster_reassign_interval', 5000) >= 1, \
                f"gmm_cluster_reassign_interval must be >=1"
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
            from baselines.targets_projected_diag_gmm import sample_time_and_labels, apply_label_dropout
            from utils.projected_diag_gmm import (
                flatten_latent, unflatten_latent, project_to_shell,
                compute_router_posterior, select_hard_top1,
                sample_projected_sources_hard_top1,
                sample_projected_sources_hard_top1_best_of_n, sample_chi_radius,
                compute_mix_loss, compute_bal_loss, compute_var_reg_loss
            )

            time_key, source_key, radius_key, label_key, pair_key, noise_key = jax.random.split(
                targets_key, 6
            )
            t, dt_base, pre_info = sample_time_and_labels(time_key, images, FLAGS)
            labels_dropped, dropped_ratio = apply_label_dropout(label_key, labels, FLAGS)
            info['dropped_ratio'] = dropped_ratio

            # Handle force_t.
            force_t_vec = jnp.ones(images.shape[0], dtype=jnp.float32) * force_t
            t = jnp.where(force_t_vec != -1, force_t_vec, t)

            B = images.shape[0]
            H, W, C = images.shape[1], images.shape[2], images.shape[3]
            D = H * W * C
            K = FLAGS.model['gmm_num_modes']
            target_shell_radius = jnp.sqrt(jnp.asarray(D, dtype=jnp.float32))

            stage_a_iters = jnp.asarray(
                FLAGS.model['gmm_stage_a_iters'], dtype=jnp.int32)
            stage_b_iters = jnp.asarray(
                FLAGS.model['gmm_stage_b_iters'], dtype=jnp.int32)
            stage_b_end = stage_a_iters + stage_b_iters
            is_stage_a = train_state.step < stage_a_iters
            is_stage_b = jnp.logical_and(
                train_state.step >= stage_a_iters, train_state.step < stage_b_end)
            is_stage_c = jnp.logical_not(jnp.logical_or(is_stage_a, is_stage_b))

            def loss_fn(grad_params):
                prior_params = grad_params["prior"]
                x1_flat = flatten_latent(images)  # [B, D]

                # Route on shell-targets y = R0 * x1 / ||x1||.
                x1_shell_flat, shell_denom_min = project_to_shell(
                    x1_flat, target_shell_radius, FLAGS.model['gmm_proj_eps'])
                q_full, log_mixprob, router_stats = compute_router_posterior(
                    x1_shell_flat, prior_params, FLAGS.model['gmm_cov_eps'])

                # Hard top-1 routing index (no gradient through argmax).
                top_idx = select_hard_top1(q_full)  # [B]

                # Stage C source: top-1 component + best-of-n directional pairing.
                # Stage A/B fallback: top-1 single sample (faster, ignored by stage losses).
                def _sample_stage_c(_):
                    return sample_projected_sources_hard_top1_best_of_n(
                        source_key,
                        prior_params,
                        top_idx,
                        x1_flat,
                        best_of_n=FLAGS.model['gmm_best_of_n'],
                        best_of_n_threshold=FLAGS.model['gmm_best_of_n_threshold'],
                        best_of_n_chunk=FLAGS.model['gmm_best_of_n_chunk'],
                        eps_proj=FLAGS.model['gmm_proj_eps'])

                def _sample_not_stage_c(_):
                    x0_dir_fallback, stats_fallback = sample_projected_sources_hard_top1(
                        source_key, prior_params, top_idx, FLAGS.model['gmm_proj_eps'])
                    stats_fallback = {
                        **stats_fallback,
                        'best_of_n': jnp.asarray(0.0, dtype=jnp.float32),
                        'best_of_n_cos_selected_mean': jnp.asarray(0.0, dtype=jnp.float32),
                        'best_of_n_cost_min_mean': jnp.asarray(0.0, dtype=jnp.float32),
                    }
                    return x0_dir_fallback, stats_fallback

                x0_dir_stage_c, source_stats_gmm = jax.lax.cond(
                    is_stage_c, _sample_stage_c, _sample_not_stage_c, operand=None
                )  # [B, D]
                r_chi = sample_chi_radius(radius_key, B, D)  # [B]
                x0_flat_stage_c = x0_dir_stage_c * r_chi[:, None]  # [B, D]

                # Stage B source: Gaussian FM warmup (optionally random pairing).
                if FLAGS.model['gmm_stage_b_random_pair']:
                    pair_perm = jax.random.permutation(pair_key, B)
                    x1_flow_stage_b = images[pair_perm]
                else:
                    x1_flow_stage_b = images
                noise_std = jnp.asarray(
                    FLAGS.model['gmm_stage_b_noise_std'], dtype=jnp.float32)
                x0_flat_stage_b = noise_std * jax.random.normal(
                    noise_key, (B, D), dtype=jnp.float32)

                # Select stage-dependent flow pairs.
                x1_flow = jnp.where(is_stage_b, x1_flow_stage_b, images)
                x0_flat = jnp.where(is_stage_b, x0_flat_stage_b, x0_flat_stage_c)

                # Unflatten to spatial.
                x0_bhwc = unflatten_latent(x0_flat, (H, W, C))  # [B, H, W, C]

                # Build flow pair.
                t_full = t[:, None, None, None]  # [B, 1, 1, 1]
                x_t = (1 - (1 - 1e-5) * t_full) * x0_bhwc + t_full * x1_flow
                v_t = x1_flow - (1 - 1e-5) * x0_bhwc

                # Model forward.
                if FLAGS.model['gmm_use_router_cond']:
                    router_cond_hard = jax.nn.one_hot(
                        top_idx, K, dtype=jnp.float32)
                    router_cond = jnp.where(
                        is_stage_c, router_cond_hard, jnp.zeros_like(router_cond_hard))
                    v_prime, logvars, activations = train_state.call_model(
                        x_t, t, dt_base, labels_dropped,
                        router_cond,
                        train=True, rngs={'dropout': dropout_key},
                        params=grad_params, return_activations=True)
                else:
                    v_prime, logvars, activations = train_state.call_model(
                        x_t, t, dt_base, labels_dropped,
                        train=True, rngs={'dropout': dropout_key},
                        params=grad_params, return_activations=True)

                # FM loss.
                mse_v = jnp.mean((v_prime - v_t) ** 2, axis=(1, 2, 3))
                loss_fm = jnp.mean(mse_v)

                # GMM losses.
                mix_norm_factor = jnp.asarray(D, dtype=jnp.float32)
                loss_mix_pre_raw = compute_mix_loss(log_mixprob)
                loss_mix_pre = loss_mix_pre_raw / mix_norm_factor
                loss_bal = compute_bal_loss(q_full)
                loss_varreg = compute_var_reg_loss(
                    prior_params['r_raw'], FLAGS.model['gmm_cov_eps'])

                # Stage losses.
                lambda_mix_pre = jnp.asarray(
                    FLAGS.model['gmm_stage_a_mix_weight'], dtype=jnp.float32)
                lambda_bal_pre = jnp.asarray(
                    FLAGS.model['gmm_stage_a_bal_weight'], dtype=jnp.float32)
                lambda_var_pre = jnp.asarray(
                    FLAGS.model['gmm_stage_a_var_weight'], dtype=jnp.float32)
                lambda_bal_joint = jnp.asarray(
                    FLAGS.model['gmm_joint_bal_weight'], dtype=jnp.float32)
                lambda_var_joint = jnp.asarray(
                    FLAGS.model['gmm_joint_var_weight'], dtype=jnp.float32)

                loss_stage_a = lambda_mix_pre * loss_mix_pre + \
                    lambda_bal_pre * loss_bal + lambda_var_pre * loss_varreg
                loss_stage_b = loss_fm
                loss_stage_c = loss_fm + lambda_bal_joint * \
                    loss_bal + lambda_var_joint * loss_varreg
                loss = jnp.where(is_stage_a, loss_stage_a,
                                 jnp.where(is_stage_b, loss_stage_b, loss_stage_c))

                # Stage-aware logging (no loss_mix branch in stage C).
                loss_mix_pre_logged = jnp.where(is_stage_a, loss_mix_pre, 0.0)
                loss_mix_pre_raw_logged = jnp.where(
                    is_stage_a, loss_mix_pre_raw, 0.0)
                loss_bal_logged = jnp.where(is_stage_b, 0.0, loss_bal)
                loss_varreg_logged = jnp.where(is_stage_b, 0.0, loss_varreg)

                pre_norm = jnp.sqrt(jnp.sum(x0_flat_stage_b ** 2, axis=-1))
                source_proj_denom_min = jnp.where(
                    is_stage_b,
                    jnp.min(pre_norm),
                    source_stats_gmm['source_proj_denom_min'])
                has_nan_source = jnp.where(
                    is_stage_b,
                    jnp.any(jnp.isnan(x0_flat_stage_b)).astype(jnp.float32),
                    source_stats_gmm['has_nan_source'])
                source_stats = {
                    'source_proj_denom_min': source_proj_denom_min,
                    'has_nan_source': has_nan_source,
                    'target_shell_denom_min': shell_denom_min,
                    'best_of_n': jnp.where(is_stage_c, source_stats_gmm['best_of_n'], 0.0),
                    'best_of_n_cos_selected_mean': jnp.where(
                        is_stage_c, source_stats_gmm['best_of_n_cos_selected_mean'], 0.0),
                    'best_of_n_cost_min_mean': jnp.where(
                        is_stage_c, source_stats_gmm['best_of_n_cost_min_mean'], 0.0),
                }

                # Logging.
                x0_norms = jnp.sqrt(jnp.sum(x0_flat ** 2, axis=-1))  # [B]
                mean_q = jnp.mean(q_full, axis=0)  # [K]
                var_all = jax.nn.softplus(prior_params['r_raw']) ** 2 + \
                    FLAGS.model['gmm_cov_eps']
                stage_id = jnp.where(is_stage_a, 0.0,
                                     jnp.where(is_stage_b, 1.0, 2.0))
                info = {
                    'loss': loss,
                    'loss_flow': loss_fm,
                    'loss_mix_pre': loss_mix_pre_logged,
                    'loss_mix_pre_raw': loss_mix_pre_raw_logged,
                    'loss_mix_pre_norm_factor': mix_norm_factor,
                    'loss_bal': loss_bal_logged,
                    'loss_varreg': loss_varreg_logged,
                    'gmm_stage_id': stage_id,
                    'in_gmm_stage_a': is_stage_a.astype(jnp.float32),
                    'in_gmm_stage_b': is_stage_b.astype(jnp.float32),
                    'in_gmm_stage_c': is_stage_c.astype(jnp.float32),
                    'v_magnitude_prime': jnp.sqrt(jnp.mean(jnp.square(v_prime))),
                    'source_x0_norm_mean': jnp.mean(x0_norms),
                    'source_x0_norm_std': jnp.std(x0_norms),
                    'source_radius_mean': jnp.mean(r_chi),
                    'source_radius_std': jnp.std(r_chi),
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
        if FLAGS.model['train_type'] == 'projected_diag_gmm':
            stage_a_iters = jnp.asarray(
                FLAGS.model['gmm_stage_a_iters'], dtype=jnp.int32)
            stage_b_iters = jnp.asarray(
                FLAGS.model['gmm_stage_b_iters'], dtype=jnp.int32)
            stage_b_end = stage_a_iters + stage_b_iters
            is_stage_a_global = train_state.step < stage_a_iters
            is_stage_b_global = jnp.logical_and(
                train_state.step >= stage_a_iters, train_state.step < stage_b_end)
            is_stage_c_global = jnp.logical_not(
                jnp.logical_or(is_stage_a_global, is_stage_b_global))

            zero_model_grads = jax.tree_map(jnp.zeros_like, grads['model'])
            zero_prior_grads = jax.tree_map(jnp.zeros_like, grads['prior'])
            model_grads = jax.tree_map(
                lambda g, z: jnp.where(is_stage_a_global, z, g),
                grads['model'],
                zero_model_grads,
            )
            prior_grads_active = jax.tree_map(
                lambda g, z: jnp.where(is_stage_b_global, z, g),
                grads['prior'],
                zero_prior_grads,
            )

            accum_steps = jnp.asarray(
                FLAGS.model['gmm_prior_accum_steps'], dtype=jnp.int32)
            accum_steps_f = jnp.asarray(
                FLAGS.model['gmm_prior_accum_steps'], dtype=jnp.float32)
            accum_candidate = jax.tree_map(
                lambda acc, g: jnp.where(is_stage_c_global, acc + g, jnp.zeros_like(acc)),
                train_state.prior_grad_accum,
                prior_grads_active,
            )
            count_candidate = jnp.where(
                is_stage_c_global, train_state.prior_accum_count + 1, 0)
            prior_update_stage_c = jnp.logical_and(
                is_stage_c_global, count_candidate >= accum_steps)
            prior_grads_avg_stage_c = jax.tree_map(
                lambda acc: acc / accum_steps_f,
                accum_candidate,
            )

            prior_grads_for_update = jax.tree_map(
                lambda g_a, g_c, z: jnp.where(
                    is_stage_a_global, g_a, jnp.where(prior_update_stage_c, g_c, z)
                ),
                prior_grads_active,
                prior_grads_avg_stage_c,
                zero_prior_grads,
            )

            grads = {
                **grads,
                'model': model_grads,
                'prior': prior_grads_for_update,
            }

            # Accumulator update: active only in stage C; reset otherwise.
            zero_prior_accum = jax.tree_map(jnp.zeros_like, accum_candidate)
            prior_accum_next = jax.tree_map(
                lambda acc, z: jnp.where(
                    jnp.logical_and(is_stage_c_global, jnp.logical_not(prior_update_stage_c)),
                    acc,
                    z,
                ),
                accum_candidate,
                zero_prior_accum,
            )
            prior_count_next = jnp.where(
                jnp.logical_and(is_stage_c_global, jnp.logical_not(prior_update_stage_c)),
                count_candidate,
                0,
            )
            prior_update_applied = jnp.logical_or(
                is_stage_a_global, prior_update_stage_c)

            info['prior_update_applied'] = prior_update_applied.astype(jnp.float32)
            info['prior_accum_count'] = prior_count_next.astype(jnp.float32)
            info['prior_accum_progress'] = prior_count_next.astype(
                jnp.float32) / accum_steps_f
            info['prior_frozen'] = jnp.logical_not(prior_update_applied).astype(
                jnp.float32)
            info['model_frozen'] = is_stage_a_global.astype(jnp.float32)
        updates, new_opt_state = train_state.tx.update(
            grads, train_state.opt_state, train_state.params)
        if FLAGS.model['train_type'] == 'projected_diag_gmm':
            zero_model_updates = jax.tree_map(jnp.zeros_like, updates['model'])
            zero_prior_updates = jax.tree_map(jnp.zeros_like, updates['prior'])
            updates = {
                **updates,
                'model': jax.tree_map(
                    lambda u, z: jnp.where(is_stage_a_global, z, u),
                    updates['model'],
                    zero_model_updates),
                'prior': jax.tree_map(
                    lambda u, z: jnp.where(prior_update_applied, u, z),
                    updates['prior'],
                    zero_prior_updates),
            }
        new_params = optax.apply_updates(train_state.params, updates)

        info['grad_norm'] = optax.global_norm(grads)
        info['update_norm'] = optax.global_norm(updates)
        info['param_norm'] = optax.global_norm(new_params)
        info['lr'] = lr_schedule(train_state.step)

        if FLAGS.model['train_type'] == 'projected_diag_gmm':
            train_state = train_state.replace(
                rng=new_rng,
                step=train_state.step + 1,
                params=new_params,
                opt_state=new_opt_state,
                prior_grad_accum=prior_accum_next,
                prior_accum_count=prior_count_next,
            )
        else:
            train_state = train_state.replace(
                rng=new_rng, step=train_state.step + 1, params=new_params, opt_state=new_opt_state)
        train_state = train_state.update_ema(FLAGS.model['target_update_rate'])
        return train_state, info

    @partial(jax.jit, out_shardings=(train_state_sharding, no_shard))
    def update_cluster(train_state, train_state_teacher, images, labels, k_batch):
        """Stage C update with within-cluster data coupling.

        images:  [B, lH, lW, 4] x1 latents sampled from cluster k (host-side)
        labels:  [B] int32 class labels from the fetched cluster images
        k_batch: [B] int32 cluster index for each sample (host-sampled)
        """
        from baselines.targets_projected_diag_gmm import sample_time_and_labels, apply_label_dropout
        from utils.projected_diag_gmm import (
            flatten_latent, unflatten_latent, project_to_shell,
            compute_router_posterior,
            sample_projected_sources_hard_top1,
            sample_chi_radius, compute_bal_loss, compute_var_reg_loss
        )

        new_rng, time_key, source_key, radius_key, label_key, dropout_key = jax.random.split(
            train_state.rng, 6)
        info = {}

        images = jax.lax.with_sharding_constraint(images, data_sharding)
        labels = jax.lax.with_sharding_constraint(labels, data_sharding)

        if FLAGS.model['cfg_scale'] == 0:
            labels = jnp.ones(labels.shape[0], dtype=jnp.int32) * FLAGS.model['num_classes']

        t, dt_base, _ = sample_time_and_labels(time_key, images, FLAGS)
        labels_dropped, dropped_ratio = apply_label_dropout(label_key, labels, FLAGS)

        B = images.shape[0]
        H, W, C = images.shape[1], images.shape[2], images.shape[3]
        D = H * W * C
        target_shell_radius = jnp.sqrt(jnp.asarray(D, dtype=jnp.float32))

        def loss_fn(grad_params):
            prior_params = grad_params["prior"]

            x1_flat = flatten_latent(images)  # [B, D]

            # x0: sample from GMM component k (k_batch is host-sampled)
            top_idx = jax.lax.stop_gradient(k_batch.astype(jnp.int32))  # [B]
            x0_dir, source_stats = sample_projected_sources_hard_top1(
                source_key, prior_params, top_idx, FLAGS.model['gmm_proj_eps'])  # [B, D]
            r_chi = sample_chi_radius(radius_key, B, D)  # [B]
            x0_flat = x0_dir * r_chi[:, None]  # [B, D]

            x0_bhwc = unflatten_latent(x0_flat, (H, W, C))  # [B, H, W, C]

            # Flow pair
            t_full = t[:, None, None, None]  # [B, 1, 1, 1]
            x_t = (1 - (1 - 1e-5) * t_full) * x0_bhwc + t_full * images
            v_t = images - (1 - 1e-5) * x0_bhwc

            # Model forward (no router conditioning in cluster-data mode)
            v_prime, logvars, activations = train_state.call_model(
                x_t, t, dt_base, labels_dropped,
                train=True, rngs={'dropout': dropout_key},
                params=grad_params, return_activations=True)

            # FM loss
            mse_v = jnp.mean((v_prime - v_t) ** 2, axis=(1, 2, 3))  # [B]
            loss_fm = jnp.mean(mse_v)

            # Balance + VarReg losses (computed on cluster-fetched x1)
            x1_shell_flat, shell_denom_min = project_to_shell(
                x1_flat, target_shell_radius, FLAGS.model['gmm_proj_eps'])
            q_full, _, router_stats = compute_router_posterior(
                x1_shell_flat, prior_params, FLAGS.model['gmm_cov_eps'])
            loss_bal = compute_bal_loss(q_full)
            loss_varreg = compute_var_reg_loss(
                prior_params['r_raw'], FLAGS.model['gmm_cov_eps'])

            lambda_bal_joint = jnp.asarray(
                FLAGS.model['gmm_joint_bal_weight'], dtype=jnp.float32)
            lambda_var_joint = jnp.asarray(
                FLAGS.model['gmm_joint_var_weight'], dtype=jnp.float32)
            loss = loss_fm + lambda_bal_joint * loss_bal + lambda_var_joint * loss_varreg

            x0_norms = jnp.sqrt(jnp.sum(x0_flat ** 2, axis=-1))  # [B]
            mean_q = jnp.mean(q_full, axis=0)  # [K]
            var_all = jax.nn.softplus(prior_params['r_raw']) ** 2 + FLAGS.model['gmm_cov_eps']

            inner_info = {
                'loss': loss,
                'loss_flow': loss_fm,
                'loss_mix_pre': jnp.asarray(0.0),
                'loss_mix_pre_raw': jnp.asarray(0.0),
                'loss_mix_pre_norm_factor': jnp.asarray(float(D)),
                'loss_bal': loss_bal,
                'loss_varreg': loss_varreg,
                'gmm_stage_id': jnp.asarray(2.0),
                'in_gmm_stage_a': jnp.asarray(0.0),
                'in_gmm_stage_b': jnp.asarray(0.0),
                'in_gmm_stage_c': jnp.asarray(1.0),
                'v_magnitude_prime': jnp.sqrt(jnp.mean(jnp.square(v_prime))),
                'source_x0_norm_mean': jnp.mean(x0_norms),
                'source_x0_norm_std': jnp.std(x0_norms),
                'source_radius_mean': jnp.mean(r_chi),
                'source_radius_std': jnp.std(r_chi),
                'source_proj_denom_min': source_stats['source_proj_denom_min'],
                'target_shell_denom_min': shell_denom_min,
                'router_usage_min': jnp.min(mean_q),
                'router_usage_max': jnp.max(mean_q),
                'prior_var_mean': jnp.mean(var_all),
                'prior_var_min': jnp.min(var_all),
                'prior_var_max': jnp.max(var_all),
                'prior_var_dev_abs_mean': jnp.mean(jnp.abs(var_all - 1.0)),
                'has_nan_loss': jnp.any(jnp.isnan(loss)).astype(jnp.float32),
                'has_nan_source': source_stats['has_nan_source'],
                'dropped_ratio': dropped_ratio,
                **router_stats,
                **{'activations/' + k_name: jnp.sqrt(jnp.mean(jnp.square(v)))
                   for k_name, v in activations.items()},
            }
            return loss, inner_info

        grads, new_info = jax.grad(loss_fn, has_aux=True)(train_state.params)
        info.update(new_info)

        # --- Gradient routing: Stage C only ---
        # Model: always updated every step.
        # Prior: accumulated over gmm_prior_accum_steps, then averaged and applied.
        zero_prior_grads = jax.tree_map(jnp.zeros_like, grads['prior'])
        accum_steps = jnp.asarray(FLAGS.model['gmm_prior_accum_steps'], dtype=jnp.int32)
        accum_steps_f = jnp.asarray(FLAGS.model['gmm_prior_accum_steps'], dtype=jnp.float32)

        accum_candidate = jax.tree_map(
            lambda acc, g: acc + g,
            train_state.prior_grad_accum,
            grads['prior'],
        )
        count_candidate = train_state.prior_accum_count + 1
        prior_update_now = count_candidate >= accum_steps

        prior_grads_avg = jax.tree_map(
            lambda acc: acc / accum_steps_f, accum_candidate)
        prior_grads_for_update = jax.tree_map(
            lambda g, z: jnp.where(prior_update_now, g, z),
            prior_grads_avg, zero_prior_grads)

        grads = {'model': grads['model'], 'prior': prior_grads_for_update}

        updates, new_opt_state = train_state.tx.update(
            grads, train_state.opt_state, train_state.params)

        zero_prior_updates = jax.tree_map(jnp.zeros_like, updates['prior'])
        updates = {
            **updates,
            'prior': jax.tree_map(
                lambda u, z: jnp.where(prior_update_now, u, z),
                updates['prior'], zero_prior_updates),
        }

        new_params = optax.apply_updates(train_state.params, updates)

        # Accumulator reset
        zero_prior_accum = jax.tree_map(jnp.zeros_like, accum_candidate)
        prior_accum_next = jax.tree_map(
            lambda acc, z: jnp.where(prior_update_now, z, acc),
            accum_candidate, zero_prior_accum)
        prior_count_next = jnp.where(prior_update_now, 0, count_candidate)

        info['prior_update_applied'] = prior_update_now.astype(jnp.float32)
        info['prior_accum_count'] = prior_count_next.astype(jnp.float32)
        info['prior_accum_progress'] = prior_count_next.astype(jnp.float32) / accum_steps_f
        info['prior_frozen'] = jnp.logical_not(prior_update_now).astype(jnp.float32)
        info['model_frozen'] = jnp.asarray(0.0, dtype=jnp.float32)
        info['grad_norm'] = optax.global_norm(grads)
        info['update_norm'] = optax.global_norm(updates)
        info['param_norm'] = optax.global_norm(new_params)
        info['lr'] = lr_schedule(train_state.step)

        train_state = train_state.replace(
            rng=new_rng,
            step=train_state.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            prior_grad_accum=prior_accum_next,
            prior_accum_count=prior_count_next,
        )
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

    # -- Host-side cluster state for Stage C within-cluster data coupling --
    cluster_indices = None
    cluster_sizes = None
    random_access_source = None
    cluster_step_counter = 0
    use_cluster_data = (
        FLAGS.model['train_type'] == 'projected_diag_gmm'
        and FLAGS.model.get('gmm_use_cluster_data', 0)
        and not FLAGS.debug_overfit
    )
    if use_cluster_data:
        from utils.datasets import get_random_access_dataset, get_ordered_dataset, sample_cluster_batch
        from utils.projected_diag_gmm import precompute_cluster_assignments, load_cluster_assignments
        import os as _os
        random_access_source = get_random_access_dataset(FLAGS.dataset_name, is_train=True)
        _cluster_save_path = FLAGS.model['gmm_cluster_save_path']
        _sizes_path = _os.path.join(_cluster_save_path, 'cluster_sizes.npy')
        if _os.path.exists(_sizes_path):
            _K = FLAGS.model['gmm_num_modes']
            cluster_indices, cluster_sizes = load_cluster_assignments(_cluster_save_path, _K)
            print(f"[GMM] Loaded cluster assignments from {_cluster_save_path}: sizes={cluster_sizes}")

    def _run_cluster_precomputation():
        """Run one-pass cluster assignment with current GMM params."""
        _D = int(np.prod(example_obs_shape[1:]))
        _K = FLAGS.model['gmm_num_modes']
        _prior_np = jax.device_get(train_state.get_prior_params(use_ema=False))
        _ordered_iter, _ = get_ordered_dataset(FLAGS.dataset_name, batch_size=64)
        _vae_enc = vae_encode if FLAGS.model.use_stable_vae else None
        _vae_rng_local = jax.random.PRNGKey(int(jax.device_get(train_state.step)))
        _ci, _cs = precompute_cluster_assignments(
            _ordered_iter, _prior_np, _vae_enc, _vae_rng_local,
            D=_D, K=_K,
            gmm_proj_eps=FLAGS.model['gmm_proj_eps'],
            gmm_cov_eps=FLAGS.model['gmm_cov_eps'],
            save_path=FLAGS.model['gmm_cluster_save_path'],
        )
        return _ci, _cs

    # Python step counter: avoids jax.device_get sync every iteration.
    _python_step = int(jax.device_get(train_state.step)) if use_cluster_data else 0

    for i in tqdm.tqdm(range(1 + start_step, FLAGS.max_steps + 1 + start_step),
                       smoothing=0.1,
                       dynamic_ncols=True):

        # -- Determine Stage C entry for cluster path --
        in_stage_c_py = False
        if use_cluster_data:
            _current_step = _python_step
            _stage_c_start = FLAGS.model['gmm_stage_a_iters'] + FLAGS.model['gmm_stage_b_iters']
            in_stage_c_py = (_current_step >= _stage_c_start)

            if in_stage_c_py:
                if cluster_indices is None:
                    print(f"[GMM] Entering Stage C at step {_current_step}. Running cluster precomputation...")
                    cluster_indices, cluster_sizes = _run_cluster_precomputation()
                    cluster_step_counter = 0
                else:
                    _reassign_interval = FLAGS.model.get('gmm_cluster_reassign_interval', 5000)
                    if cluster_step_counter > 0 and cluster_step_counter % _reassign_interval == 0:
                        print(f"[GMM] Reassigning clusters (Stage-C step {cluster_step_counter}, global {_current_step})...")
                        cluster_indices, cluster_sizes = _run_cluster_precomputation()

        # -- Sample data --
        if use_cluster_data and in_stage_c_py and cluster_indices is not None:
            # Cluster-aware path: k is sampled first, x1 comes from cluster k.
            _prior_np = jax.device_get(train_state.get_prior_params(use_ema=False))
            if FLAGS.model.get('gmm_cluster_sample_uniform', 1):
                _pi_np = None
            else:
                import jax.numpy as _jnp
                _pi_np = np.array(
                    jax.nn.softmax(_jnp.array(_prior_np['pi_logits'], dtype=_jnp.float32)))

            x1_raw, x1_labels_np, k_batch_np = sample_cluster_batch(
                cluster_indices, cluster_sizes, local_batch_size,
                random_access_source, FLAGS.dataset_name, pi_np=_pi_np)

            if FLAGS.model.use_stable_vae and 'latent' not in FLAGS.dataset_name:
                vae_rng, vae_key = jax.random.split(vae_rng)
                batch_images = vae_encode(vae_key, x1_raw)
            else:
                batch_images = x1_raw
            batch_images = shard_data(batch_images)
            batch_labels = shard_data(x1_labels_np)
            k_batch_sharded = shard_data(k_batch_np)

            train_state, update_info = update_cluster(
                train_state, train_state_teacher, batch_images, batch_labels, k_batch_sharded)
            cluster_step_counter += 1
        else:
            # Existing streaming path (Stage A, B, or cluster_data=0).
            if not FLAGS.debug_overfit or i == 1:
                batch_images, batch_labels = shard_data(*next(dataset))
                if FLAGS.model.use_stable_vae and 'latent' not in FLAGS.dataset_name:
                    vae_rng, vae_key = jax.random.split(vae_rng)
                    batch_images = vae_encode(vae_key, batch_images)

            train_state, update_info = update(
                train_state, train_state_teacher, batch_images, batch_labels)

        _python_step += 1

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
            if 'loss_flow' in valid_update_info:
                train_metrics['training/loss_flow_valid'] = valid_update_info['loss_flow']
            # Log key valid components to compare apples-to-apples with training metrics.
            valid_keys = (
                'loss_mix_pre',
                'loss_mix_pre_raw',
                'loss_bal',
                'loss_varreg',
                'prior_var_dev_abs_mean',
                'gmm_stage_id',
                'prior_update_applied',
                'prior_accum_count',
                'prior_accum_progress',
                'router_usage_min',
                'router_usage_max',
                'cluster_step_counter',
            )
            for k in valid_keys:
                if k in valid_update_info:
                    train_metrics[f'training/{k}_valid'] = valid_update_info[k]

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
