from typing import Any
import csv
import os
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
from model import DiT
from helper_eval import eval_model
from helper_inference import do_inference
from gmm_utils import (
    assign_angular_codes,
    build_source_base,
    categorical_entropy,
    categorical_kl,
    directions_from_local,
    flatten_and_standardize,
    flatten_latents,
    load_gmm_stats,
    local_coordinates_from_standardized,
    posterior_from_standardized,
    sample_lognormal_radius,
    standardize_latents,
    temperature_smooth_probs,
)
from moe_source import (
    SourceBaseNet,
    floor_log_sigma,
    sample_source_gaussian,
    source_sigma_floor_loss,
)


def _parse_csv_steps(step_text):
    if not step_text:
        return set()
    steps = set()
    for item in step_text.split(','):
        item = item.strip()
        if item:
            steps.add(int(item))
    return steps


def _csv_scalar(value):
    arr = np.asarray(value)
    if arr.size != 1:
        return ''
    item = arr.reshape(()).item()
    if isinstance(item, (np.integer, int)):
        return int(item)
    if isinstance(item, (np.floating, float)):
        return float(item)
    return item


def _write_summary_csv(path, step, metrics):
    if path is None:
        return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    row = {'step': int(step)}
    row.update({key: _csv_scalar(value) for key, value in metrics.items()})
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    if file_exists:
        with open(path, 'r', newline='') as f:
            fieldnames = next(csv.reader(f))
    else:
        fieldnames = ['step'] + sorted(metrics.keys())
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _upload_summary_csv_to_wandb(path):
    if path is None or wandb.run is None:
        return
    base_path = os.path.dirname(path) or '.'
    wandb.save(path, base_path=base_path, policy='now')


FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_name', 'imagenet256', 'Environment name.')
flags.DEFINE_string('tfds_data_dir', None, 'Optional TFDS data directory.')
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
flags.DEFINE_string(
    'summary_csv_path', None,
    'Optional CSV path for scalar training summaries. Defaults to '
    '<save_dir>/training_summary.csv when save_dir is set.')
flags.DEFINE_string(
    'summary_csv_steps', '',
    'Comma-separated step list for CSV summaries. Empty means every log step.')
flags.DEFINE_integer(
    'summary_csv_wandb_upload', 1,
    'Upload the summary CSV to W&B run files after each CSV write.')
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
    'train_type': 'shortcut',  # or naive, naive-moe-source, khoat-fm.
    'gmm_stats_path': '',
    'gmm_num_modes': 16,
    'angular_num_submodes': 4,
    'local_eta': 0.5,
    'source_condition_dim': 64,
    'source_channels': 128,
    'source_num_blocks': 6,
    'source_kernel_size': 3,
    'source_sigma_init': 0.7,
    'source_sigma_min': 0.7,
    'source_sigma_hard_floor': 0,
    'source_sensitivity_metrics': 1,
    'source_eps': 1e-8,
    'loss_post_weight': 0.1,
    'posterior_temperature': 2.0,
    'loss_var_weight': 0.05,
    'loss_align_weight': 0.01,

    # ===== Khoat Flow Matching defaults =====
    'kfm_p_min': 0.20,          # P_min = 75%
    'kfm_alpha': 0.9,           # alpha = 0.9
    # 12.5% of batch forced to t=0 (stratified sampling)
    'kfm_t0_ratio': 0.125,
    'kfm_dt_min_exp': 0,        # default
    'kfm_dt_max_exp': -1,       # default: -1 => auto = log2(denoise_timesteps)
    'kfm_schedule_type': 'linear',  # default
    'kfm_schedule': '',         # default unused for now
    'kfm_eps': 1e-5,            # keep same epsilon style as current codebase
})


wandb_config = default_wandb_config()
wandb_config.update({
    'project': 'shortcut',
    'name': 'shortcut_{dataset_name}',
})

config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('model', model_config, lock_config=False)


def get_backbone_params(params):
    if hasattr(params, 'keys') and 'backbone' in params:
        return params['backbone']
    return params

##############################################
# Training Code.
##############################################


def main(_):

    np.random.seed(FLAGS.seed)
    if FLAGS.model.train_type == 'naive-moe-source':
        if not FLAGS.model.gmm_stats_path:
            raise ValueError("--model.gmm_stats_path is required for naive-moe-source.")
        if not FLAGS.model.use_stable_vae:
            raise ValueError("naive-moe-source requires --model.use_stable_vae=1.")
        if 'latent' in FLAGS.dataset_name:
            raise ValueError("naive-moe-source does not support pre-paired latent datasets.")

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

    dataset = get_dataset(
        FLAGS.dataset_name,
        local_batch_size,
        True,
        FLAGS.debug_overfit,
        data_dir=FLAGS.tfds_data_dir,
    )
    dataset_valid = get_dataset(
        FLAGS.dataset_name,
        local_batch_size,
        False,
        FLAGS.debug_overfit,
        data_dir=FLAGS.tfds_data_dir,
    )
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

    if FLAGS.model.train_type == 'naive-moe-source':
        gmm_state = load_gmm_stats(FLAGS.model.gmm_stats_path)
        required_source_keys = [
            'angular_centers',
            'angular_pi',
            'radius_log_mean',
            'radius_log_std',
            'latent_shape',
        ]
        missing_keys = [key for key in required_source_keys if key not in gmm_state]
        if missing_keys:
            raise ValueError(
                "naive-moe-source requires expanded source stats. "
                f"Missing keys: {missing_keys}")
        gmm_standardize_eps = float(np.asarray(
            gmm_state.get('standardize_eps', np.array(1e-6, dtype=np.float32))))
        gmm_local_eta = float(np.asarray(
            gmm_state.get('local_eta', np.array(FLAGS.model.local_eta, dtype=np.float32))))
        if int(gmm_state['pi'].shape[0]) != int(FLAGS.model.gmm_num_modes):
            raise ValueError("model.gmm_num_modes must match stats pi size.")
        if int(gmm_state['angular_centers'].shape[1]) != int(FLAGS.model.angular_num_submodes):
            raise ValueError("model.angular_num_submodes must match stats angular centers.")
        stats_latent_shape = tuple(np.asarray(gmm_state['latent_shape']).astype(np.int32).tolist())
        if stats_latent_shape != tuple(example_obs_shape[1:]):
            raise ValueError(
                f"Stats latent_shape {stats_latent_shape} does not match "
                f"current latent shape {tuple(example_obs_shape[1:])}.")
    else:
        gmm_state = None
        gmm_standardize_eps = 1e-6
        gmm_local_eta = 0.5

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
        'ignore_dt': False if (FLAGS.model['train_type'] in ('shortcut', 'livereflow', 'khoat-fm')) else True,
    }
    model_def = DiT(**dit_args)
    if FLAGS.model.train_type == 'naive-moe-source':
        source_model_def = SourceBaseNet(
            num_modes=FLAGS.model['gmm_num_modes'],
            angular_num_submodes=FLAGS.model['angular_num_submodes'],
            condition_dim=FLAGS.model['source_condition_dim'],
            channels=FLAGS.model['source_channels'],
            num_blocks=FLAGS.model['source_num_blocks'],
            out_channels=example_obs_shape[-1],
            sigma_init=FLAGS.model['source_sigma_init'],
            kernel_size=FLAGS.model['source_kernel_size'],
        )
    else:
        source_model_def = None
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
        if source_model_def is None:
            param_key, dropout_key, dropout2_key = jax.random.split(rng, 3)
            source_key = None
        else:
            param_key, source_key, dropout_key, dropout2_key = jax.random.split(
                rng, 4)
        example_t = jnp.zeros((1,))
        example_dt = jnp.zeros((1,))
        example_label = jnp.zeros((1,), dtype=jnp.int32)
        example_obs = jnp.zeros(example_obs_shape)
        model_rngs = {'params': param_key,
                      'label_dropout': dropout_key, 'dropout': dropout2_key}
        params = model_def.init(model_rngs, example_obs,
                                example_t, example_dt, example_label)['params']
        if source_model_def is None:
            opt_state = tx.init(params)
            return TrainStateEma.create(
                model_def, params, rng=rng, tx=tx, opt_state=opt_state)

        source_params = source_model_def.init(
            {'params': source_key},
            jnp.zeros(example_obs_shape),
            jnp.zeros(example_obs_shape),
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.zeros((1,), dtype=jnp.float32),
        )['params']
        opt_state = tx.init({'backbone': params, 'source': source_params})
        return TrainStateEma.create(
            model_def,
            params,
            rng=rng,
            tx=tx,
            opt_state=opt_state,
            source_model_def=source_model_def,
            source_params=source_params,
        )

    rng = jax.random.PRNGKey(FLAGS.seed)
    train_state_shape = jax.eval_shape(init, rng)

    data_sharding, train_state_sharding, no_shard, shard_data, global_to_local = create_sharding(
        FLAGS.model.sharding, train_state_shape)
    train_state = jax.jit(init, out_shardings=train_state_sharding)(rng)
    backbone_params = get_backbone_params(train_state.params)
    jax.debug.visualize_array_sharding(
        backbone_params['FinalLayer_0']['Dense_0']['kernel'])
    jax.debug.visualize_array_sharding(
        backbone_params['TimestepEmbedder_1']['Dense_0']['kernel'])
    jax.experimental.multihost_utils.assert_equal(
        backbone_params['TimestepEmbedder_1']['Dense_0']['kernel'])
    start_step = 1

    if FLAGS.load_dir is not None:
        cp = Checkpoint(FLAGS.load_dir)
        replace_dict = cp.load_as_dict()['train_state']
        del replace_dict['opt_state']  # Debug
        if source_model_def is not None and 'source' not in replace_dict['params']:
            replace_dict['params'] = {
                'backbone': replace_dict['params'],
                'source': train_state.params['source'],
            }
            replace_dict['params_ema'] = {
                'backbone': replace_dict['params_ema'],
                'source': train_state.params_ema['source'],
            }
        elif source_model_def is None and hasattr(replace_dict['params'], 'keys') and 'backbone' in replace_dict['params']:
            replace_dict['params'] = replace_dict['params']['backbone']
            replace_dict['params_ema'] = replace_dict['params_ema']['backbone']
        train_state = train_state.replace(**replace_dict)
        if FLAGS.wandb.run_id != "None":  # If we are continuing a run.
            start_step = train_state.step
        train_state = jax.jit(
            lambda x: x, out_shardings=train_state_sharding)(train_state)
        print("Loaded model with step", train_state.step)
        train_state = train_state.replace(step=0)
        backbone_params = get_backbone_params(train_state.params)
        jax.debug.visualize_array_sharding(
            backbone_params['FinalLayer_0']['Dense_0']['kernel'])
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

        if FLAGS.model['train_type'] == 'naive-moe-source':

            def loss_fn(grad_params):
                label_key, time_key, z_key, radius_key, x0_key, sens_key = (
                    jax.random.split(targets_key, 6))
                z_sens_key, _ = jax.random.split(sens_key)

                labels_dropout = jax.random.bernoulli(
                    label_key,
                    FLAGS.model['class_dropout_prob'],
                    (labels.shape[0],),
                )
                labels_dropped = jnp.where(
                    labels_dropout,
                    FLAGS.model['num_classes'],
                    labels,
                )

                t = jax.random.randint(
                    time_key,
                    (images.shape[0],),
                    minval=0,
                    maxval=FLAGS.model['denoise_timesteps'],
                ).astype(jnp.float32)
                t /= FLAGS.model['denoise_timesteps']
                force_t_vec = jnp.ones(images.shape[0], dtype=jnp.float32) * force_t
                t = jnp.where(force_t_vec != -1, force_t_vec, t)
                t_full = t[:, None, None, None]

                latents_std = flatten_and_standardize(
                    images,
                    gmm_state['mean'],
                    gmm_state['std'],
                    gmm_standardize_eps,
                )
                q1 = posterior_from_standardized(
                    latents_std,
                    gmm_state['log_pi'],
                    gmm_state['mu'],
                    gmm_state['var'],
                )
                q1 = jax.lax.stop_gradient(q1)
                mode_indices = jnp.argmax(q1, axis=-1).astype(jnp.int32)
                angular_indices, _ = assign_angular_codes(
                    latents_std,
                    mode_indices,
                    gmm_state['mu'],
                    gmm_state['var'],
                    gmm_state['angular_centers'],
                    gmm_local_eta,
                    eps=FLAGS.model['source_eps'],
                )
                angular_indices = angular_indices.astype(jnp.int32)
                radius, log_radius = sample_lognormal_radius(
                    radius_key,
                    mode_indices,
                    angular_indices,
                    gmm_state['radius_log_mean'],
                    gmm_state['radius_log_std'],
                )
                x_base, _ = build_source_base(
                    mode_indices,
                    angular_indices,
                    radius,
                    images.shape[1:],
                    gmm_state['mean'],
                    gmm_state['std'],
                    gmm_standardize_eps,
                    gmm_state['mu'],
                    gmm_state['var'],
                    gmm_state['angular_centers'],
                    gmm_local_eta,
                    eps=FLAGS.model['source_eps'],
                )
                z = jax.random.normal(z_key, images.shape)
                mu_x0, raw_log_sigma, delta_mu = train_state.call_source(
                    z,
                    x_base,
                    mode_indices,
                    angular_indices,
                    log_radius,
                    params=grad_params,
                )
                if FLAGS.model['source_sigma_hard_floor']:
                    log_sigma = floor_log_sigma(
                        raw_log_sigma, FLAGS.model['source_sigma_min'])
                else:
                    log_sigma = raw_log_sigma
                x_0 = sample_source_gaussian(x0_key, mu_x0, log_sigma)
                x_t = (1 - t_full) * x_0 + t_full * images
                v_t = images - x_0
                dt_flow = np.log2(FLAGS.model['denoise_timesteps']).astype(jnp.int32)
                dt_base = jnp.ones(images.shape[0], dtype=jnp.int32) * dt_flow

                v_prime, logvars, activations = train_state.call_model(
                    x_t,
                    t,
                    dt_base,
                    labels_dropped,
                    train=True,
                    rngs={'dropout': dropout_key},
                    params=grad_params,
                    return_activations=True,
                )
                mse_v = jnp.mean((v_prime - v_t) ** 2, axis=(1, 2, 3))
                loss_fm = jnp.mean(mse_v)
                mu_x0_std = standardize_latents(
                    flatten_latents(mu_x0),
                    gmm_state['mean'],
                    gmm_state['std'],
                    gmm_standardize_eps,
                )
                q0 = posterior_from_standardized(
                    mu_x0_std,
                    gmm_state['log_pi'],
                    gmm_state['mu'],
                    gmm_state['var'],
                )
                q1_target = jax.lax.stop_gradient(temperature_smooth_probs(
                    q1,
                    FLAGS.model['posterior_temperature'],
                    eps=FLAGS.model['source_eps'],
                ))
                loss_post = jnp.mean(categorical_kl(
                    q1_target, q0, eps=FLAGS.model['source_eps']))
                loss_var = source_sigma_floor_loss(
                    raw_log_sigma, FLAGS.model['source_sigma_min'])
                u_mu = local_coordinates_from_standardized(
                    mu_x0_std,
                    mode_indices,
                    gmm_state['mu'],
                    gmm_state['var'],
                    gmm_local_eta,
                    eps=FLAGS.model['source_eps'],
                )
                s_mu = directions_from_local(u_mu, eps=FLAGS.model['source_eps'])
                target_dirs = gmm_state['angular_centers'][mode_indices, angular_indices]
                align_cos = jnp.sum(s_mu * target_dirs, axis=-1)
                loss_align = jnp.mean(1.0 - align_cos)
                sigma = jnp.exp(log_sigma)
                raw_sigma = jnp.exp(raw_log_sigma)
                log_sigma_min = jnp.log(jnp.asarray(
                    FLAGS.model['source_sigma_min'], dtype=raw_log_sigma.dtype))
                sigma_floor_frac = jnp.mean(
                    (raw_log_sigma < log_sigma_min).astype(jnp.float32))
                x0_minus_base_norm = jnp.sqrt(jnp.mean(jnp.square(x_0 - x_base)))
                x0_minus_x1_norm = jnp.sqrt(jnp.mean(jnp.square(x_0 - images)))
                mu_x0_minus_x1_norm = jnp.sqrt(
                    jnp.mean(jnp.square(mu_x0 - images)))
                base_minus_x1_norm = jnp.sqrt(
                    jnp.mean(jnp.square(x_base - images)))
                delta_mu_norm = jnp.sqrt(jnp.mean(jnp.square(delta_mu)))
                if FLAGS.model['source_sensitivity_metrics']:
                    z_alt = jax.random.normal(z_sens_key, images.shape)
                    mu_x0_z_alt, _, _ = train_state.call_source(
                        z_alt,
                        x_base,
                        mode_indices,
                        angular_indices,
                        log_radius,
                        params=grad_params,
                    )
                    angular_alt = (
                        angular_indices + 1
                    ) % FLAGS.model['angular_num_submodes']
                    mu_x0_a_alt, _, _ = train_state.call_source(
                        z,
                        x_base,
                        mode_indices,
                        angular_alt,
                        log_radius,
                        params=grad_params,
                    )
                    z_sensitivity_mu = jnp.sqrt(
                        jnp.mean(jnp.square(mu_x0_z_alt - mu_x0)))
                    angular_sensitivity_mu = jnp.sqrt(
                        jnp.mean(jnp.square(mu_x0_a_alt - mu_x0)))
                else:
                    z_sensitivity_mu = jnp.asarray(0.0, dtype=images.dtype)
                    angular_sensitivity_mu = jnp.asarray(0.0, dtype=images.dtype)
                q0_entropy = jnp.mean(categorical_entropy(
                    q0, eps=FLAGS.model['source_eps']))
                q1_entropy = jnp.mean(categorical_entropy(
                    q1, eps=FLAGS.model['source_eps']))
                posterior_agreement = jnp.mean(
                    (jnp.argmax(q0, axis=-1) == mode_indices).astype(jnp.float32))
                angular_alignment = jnp.mean(align_cos)
                loss = (
                    loss_fm
                    + FLAGS.model['loss_post_weight'] * loss_post
                    + FLAGS.model['loss_var_weight'] * loss_var
                    + FLAGS.model['loss_align_weight'] * loss_align
                )

                info = {
                    'loss': loss,
                    'loss/fm': loss_fm,
                    'loss/post': loss_post,
                    'loss/var': loss_var,
                    'loss/align': loss_align,
                    'source/x0_sample_norm': jnp.sqrt(jnp.mean(jnp.square(x_0))),
                    'source/mu_x0_norm': jnp.sqrt(jnp.mean(jnp.square(mu_x0))),
                    'source/sigma_mean': jnp.mean(sigma),
                    'source/sigma_min': jnp.min(sigma),
                    'source/sigma_max': jnp.max(sigma),
                    'source/sigma_raw_mean': jnp.mean(raw_sigma),
                    'source/sigma_raw_min': jnp.min(raw_sigma),
                    'source/sigma_raw_max': jnp.max(raw_sigma),
                    'source/sigma_floor_frac': sigma_floor_frac,
                    'source/x0_minus_base_norm': x0_minus_base_norm,
                    'source/x0_minus_x1_norm': x0_minus_x1_norm,
                    'source/mu_x0_minus_x1_norm': mu_x0_minus_x1_norm,
                    'source/base_minus_x1_norm': base_minus_x1_norm,
                    'source/delta_mu_norm': delta_mu_norm,
                    'source/base_norm': jnp.sqrt(jnp.mean(jnp.square(x_base))),
                    'condition/z_sensitivity_mu': z_sensitivity_mu,
                    'condition/angular_sensitivity_mu': angular_sensitivity_mu,
                    'posterior/agreement_argmax_mu_x0_x1': posterior_agreement,
                    'posterior/q0_entropy': q0_entropy,
                    'posterior/q1_entropy': q1_entropy,
                    'angular/source_alignment_cosine': angular_alignment,
                    'v_magnitude_prime': jnp.sqrt(jnp.mean(jnp.square(v_prime))),
                    'dropped_ratio': jnp.mean(
                        labels_dropped == FLAGS.model['num_classes']),
                    **{'activations/' + k: jnp.sqrt(jnp.mean(jnp.square(v))) for k, v in activations.items()},
                }
                for idx in range(FLAGS.model['gmm_num_modes']):
                    info[f'condition/mode_usage_{idx}'] = jnp.mean(
                        (mode_indices == idx).astype(jnp.float32))
                    info[f'posterior/q1_mean_{idx}'] = jnp.mean(q1[:, idx])
                    info[f'posterior/q0_mean_{idx}'] = jnp.mean(q0[:, idx])
                for idx in range(FLAGS.model['angular_num_submodes']):
                    info[f'condition/angular_usage_{idx}'] = jnp.mean(
                        (angular_indices == idx).astype(jnp.float32))

                return loss, info

            grads, info = jax.grad(loss_fn, has_aux=True)(train_state.params)
        else:
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
            elif FLAGS.model['train_type'] == 'khoat-fm':
                from targets_khoat_fm import get_targets
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
        if FLAGS.model['train_type'] == 'naive-moe-source':
            info['grad/source_total_norm'] = optax.global_norm(grads['source'])

        train_state = train_state.replace(
            rng=new_rng, step=train_state.step + 1, params=new_params, opt_state=new_opt_state)
        train_state = train_state.update_ema(FLAGS.model['target_update_rate'])
        return train_state, info

    if FLAGS.mode != 'train':
        do_inference(FLAGS, train_state, None, dataset, dataset_valid, shard_data, vae_encode, vae_decode, update,
                     get_fid_activations, imagenet_labels, visualize_labels,
                     fid_from_stats, truth_fid_stats, gmm_state)
        return

    ###################################
    # Train Loop
    ###################################

    summary_csv_steps = _parse_csv_steps(FLAGS.summary_csv_steps)
    summary_csv_path = FLAGS.summary_csv_path
    if summary_csv_path is None and FLAGS.save_dir is not None:
        summary_csv_path = os.path.join(FLAGS.save_dir, 'training_summary.csv')

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

        should_log = i % FLAGS.log_interval == 0 or i == 1
        should_write_summary = (
            should_log if not summary_csv_steps else i in summary_csv_steps)
        if should_log or should_write_summary:
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
                if should_log:
                    wandb.log(train_metrics, step=i)
                if should_write_summary:
                    _write_summary_csv(summary_csv_path, i, train_metrics)
                    if FLAGS.summary_csv_wandb_upload:
                        _upload_summary_csv_to_wandb(summary_csv_path)

        if FLAGS.model['train_type'] == 'progressive':
            num_sections = np.log2(
                FLAGS.model['denoise_timesteps']).astype(jnp.int32)
            if i % (FLAGS.max_steps // num_sections) == 0:
                train_state_teacher = jax.jit(
                    lambda x: x, out_shardings=train_state_sharding)(train_state)

        if i % FLAGS.eval_interval == 0:
            eval_model(FLAGS, train_state, train_state_teacher, i, dataset, dataset_valid, shard_data, vae_encode, vae_decode, update,
                       get_fid_activations, imagenet_labels, visualize_labels,
                       fid_from_stats, truth_fid_stats, gmm_state)

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
