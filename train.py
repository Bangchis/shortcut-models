from typing import Any
import json
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
from gmm_utils import flatten_latents, load_gmm_stats, posterior_from_stats
from moe_source import SourceMoE, sample_diag_gaussian, var_only_kld_loss

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
flags.DEFINE_integer('batch_size', 32, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1_000_000), 'Number of training steps.')
flags.DEFINE_integer('debug_overfit', 0, 'Debug overfitting.')
flags.DEFINE_string('mode', 'train', 'train or inference.')
flags.DEFINE_string('metrics_output_path', None, 'Optional JSON path for final metrics.')
flags.DEFINE_string('final_save_dir', None, 'Optional final checkpoint path.')
flags.DEFINE_integer('run_final_inference', 0, 'Whether to run final inference after training.')
flags.DEFINE_integer('dump_source_stats', 0, 'Whether inference should dump source/data stats.')
flags.DEFINE_integer('source_stats_samples', 4096, 'How many samples to collect for source stats dumps.')
flags.DEFINE_string(
    'train_metrics_level',
    'full',
    'Training diagnostics level: "summary" keeps ranking-critical metrics; "full" also logs heavy activation/expert diagnostics.',
)
flags.DEFINE_string(
    'eval_fid_timesteps',
    '1,4,32,128',
    'Comma-separated denoise step counts to score in helper_eval.',
)
flags.DEFINE_string(
    'moe1_k_list',
    '8,16,24,32',
    'Comma-separated GMM mode counts for mode=moe1-naive-k-ablation.',
)
flags.DEFINE_string(
    'moe1_root_dir',
    '/kaggle/working/moe1_naive_k_ablation',
    'Root output directory for mode=moe1-naive-k-ablation.',
)
flags.DEFINE_integer(
    'moe1_train_naive_ref',
    1,
    'Whether mode=moe1-naive-k-ablation should train the naive reference.',
)
flags.DEFINE_integer(
    'moe1_keep_checkpoints',
    0,
    'Whether mode=moe1-naive-k-ablation may save model checkpoints/final.pkl.',
)
flags.DEFINE_integer(
    'moe1_gmm_fit_samples',
    -1,
    'Train samples for each GMM fit in mode=moe1-naive-k-ablation. -1 means full split.',
)
flags.DEFINE_integer(
    'moe1_gmm_valid_samples',
    -1,
    'Validation samples for each GMM fit in mode=moe1-naive-k-ablation. -1 means full split.',
)
flags.DEFINE_string(
    'moe1_tau_values',
    '0.75,1.0,1.5,2.0,3.0',
    'Comma-separated source_tau values for greedy mode=moe1-naive-k-ablation.',
)
flags.DEFINE_string(
    'moe1_balance_values',
    '0.0,0.05,0.1,0.2,0.4,0.8,1.2,1.6',
    'Comma-separated loss_balance_weight values for greedy mode=moe1-naive-k-ablation.',
)
flags.DEFINE_string(
    'moe1_entropy_values',
    '0.0,0.005,0.01,0.03,0.1,0.2,0.5,1.0',
    'Comma-separated loss_entropy_weight values for greedy mode=moe1-naive-k-ablation.',
)
flags.DEFINE_string(
    'moe1_weight_decay_values',
    '0.001,0.003,0.01,0.03,0.1',
    'Comma-separated weight_decay values for greedy mode=moe1-naive-k-ablation.',
)
flags.DEFINE_string(
    'moe1_var_target_values',
    '0.5,0.7,1.0,1.3',
    'Comma-separated source_var_target_std values for greedy mode=moe1-naive-k-ablation.',
)
flags.DEFINE_integer('moe1_base_k', 8, 'Base K for greedy mode=moe1-naive-k-ablation.')
flags.DEFINE_float('moe1_base_tau', 1.5, 'Base source_tau for greedy mode=moe1-naive-k-ablation.')
flags.DEFINE_float('moe1_base_balance', 0.1, 'Base loss_balance_weight for greedy mode=moe1-naive-k-ablation.')
flags.DEFINE_float('moe1_base_entropy', 0.01, 'Base loss_entropy_weight for greedy mode=moe1-naive-k-ablation.')
flags.DEFINE_float('moe1_base_weight_decay', 0.01, 'Base weight_decay for greedy mode=moe1-naive-k-ablation.')
flags.DEFINE_float('moe1_base_var_target', 0.7, 'Base source_var_target_std for greedy mode=moe1-naive-k-ablation.')
flags.DEFINE_string(
    'moe1_greedy_metric',
    'fid',
    'Primary greedy selection metric for mode=moe1-naive-k-ablation. Currently supports "fid".',
)

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
    'gmm_num_modes': 4,
    'source_soft_moe': 1,
    'source_condition_dim': 16,
    'source_hidden_channels': 64,
    'source_tau': 2.0,
    'loss_balance_weight': 0.1,
    'loss_entropy_weight': 0.01,
    'source_var_weight': 1.0,
    'source_var_target_std': 1.0,
    'source_var_eps': 1e-6,
    'source_logvar_min': -8.0,
    'source_logvar_max': 4.0,
    'source_zero_init': 1,
    'source_dtype': 'float32',

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


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_metrics_json(path, payload):
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_json_ready(payload), f, indent=2, sort_keys=True)


def _resolve_jax_dtype(dtype_name):
    normalized = str(dtype_name).lower()
    if normalized in ('bf16', 'bfloat16'):
        return jnp.bfloat16
    if normalized in ('fp32', 'float32'):
        return jnp.float32
    if normalized in ('fp16', 'float16'):
        return jnp.float16
    raise ValueError(f'Unsupported dtype: {dtype_name}')

##############################################
# Training Code.
##############################################


def main(_):
    if FLAGS.mode == 'moe1-naive-k-ablation':
        from moe1_naive_k_ablation import run as run_moe1_naive_k_ablation
        run_moe1_naive_k_ablation(FLAGS)
        return

    if FLAGS.mode == 'moe1-ablation':
        from moe1_ablation import run as run_moe1_ablation
        run_moe1_ablation(FLAGS)
        return

    if FLAGS.train_metrics_level not in ('summary', 'full'):
        raise ValueError('--train_metrics_level must be "summary" or "full".')

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
    if FLAGS.batch_size % global_device_count != 0:
        raise ValueError(
            f'--batch_size={FLAGS.batch_size} must be divisible by global device count '
            f'{global_device_count} for multihost sharding.'
        )
    if local_batch_size % device_count != 0:
        raise ValueError(
            f'Local batch size {local_batch_size} must be divisible by local device count '
            f'{device_count} for multihost sharding.'
        )
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
        gmm_standardize_eps = float(np.asarray(
            gmm_state.get('standardize_eps', np.array(1e-6, dtype=np.float32))))
    else:
        gmm_state = None
        gmm_standardize_eps = 1e-6

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
        source_model_def = SourceMoE(
            num_modes=FLAGS.model['gmm_num_modes'],
            condition_dim=FLAGS.model['source_condition_dim'],
            hidden_channels=FLAGS.model['source_hidden_channels'],
            out_channels=example_obs_shape[-1],
            tau=FLAGS.model['source_tau'],
            soft_moe=bool(FLAGS.model['source_soft_moe']),
            var_eps=FLAGS.model['source_var_eps'],
            logvar_min=FLAGS.model['source_logvar_min'],
            logvar_max=FLAGS.model['source_logvar_max'],
            zero_init=bool(FLAGS.model['source_zero_init']),
            dtype=_resolve_jax_dtype(FLAGS.model['source_dtype']),
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
            jnp.zeros((1, FLAGS.model['gmm_num_modes']), dtype=jnp.float32),
            return_experts=True,
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

    collect_full_train_diagnostics = FLAGS.train_metrics_level == 'full'

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
                label_key, time_key, z_key, mode_key, x0_key = jax.random.split(
                    targets_key, 5)

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

                latents_flat = flatten_latents(images)
                q = posterior_from_stats(
                    latents_flat,
                    gmm_state['mean'],
                    gmm_state['std'],
                    gmm_standardize_eps,
                    gmm_state['log_pi'],
                    gmm_state['mu'],
                    gmm_state['var'],
                )
                q = jax.lax.stop_gradient(q)
                sampled_modes = jax.random.categorical(
                    mode_key,
                    jnp.log(jnp.maximum(q, 1e-8)),
                    axis=-1,
                )
                condition_weights = jax.nn.one_hot(
                    sampled_modes,
                    FLAGS.model['gmm_num_modes'],
                    dtype=jnp.float32,
                )
                z = jax.random.normal(z_key, images.shape)
                source_out = train_state.call_source(
                    z,
                    condition_weights,
                    params=grad_params,
                    return_experts=collect_full_train_diagnostics,
                )
                if collect_full_train_diagnostics:
                    mu_x0, logvar_x0, var_x0, alpha, expert_mu, expert_logvar, router_logits = source_out
                else:
                    mu_x0, logvar_x0, var_x0, alpha = source_out
                x_0 = sample_diag_gaussian(x0_key, mu_x0, logvar_x0)
                x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * images
                v_t = images - (1 - 1e-5) * x_0
                dt_flow = np.log2(FLAGS.model['denoise_timesteps']).astype(jnp.int32)
                dt_base = jnp.ones(images.shape[0], dtype=jnp.int32) * dt_flow

                model_out = train_state.call_model(
                    x_t,
                    t,
                    dt_base,
                    labels_dropped,
                    train=True,
                    rngs={'dropout': dropout_key},
                    params=grad_params,
                    return_activations=collect_full_train_diagnostics,
                )
                if collect_full_train_diagnostics:
                    v_prime, logvars, activations = model_out
                else:
                    v_prime = model_out
                mse_v = jnp.mean((v_prime - v_t) ** 2, axis=(1, 2, 3))
                loss_fm = jnp.mean(mse_v)
                alpha_mean = jnp.mean(alpha, axis=0)
                hard_usage = jnp.mean(
                    jax.nn.one_hot(
                        jnp.argmax(alpha, axis=-1),
                        FLAGS.model['gmm_num_modes'],
                        dtype=jnp.float32,
                    ),
                    axis=0,
                )
                loss_balance = jnp.sum(
                    (alpha_mean - (1.0 / FLAGS.model['gmm_num_modes'])) ** 2)
                entropy = -jnp.sum(
                    alpha * jnp.log(jnp.maximum(alpha, 1e-8)), axis=-1)
                loss_entropy = jnp.mean(entropy)
                loss_var = var_only_kld_loss(
                    var_x0,
                    logvar_x0,
                    target_std=FLAGS.model['source_var_target_std'],
                    eps=FLAGS.model['source_var_eps'],
                )
                loss = (
                    loss_fm
                    + FLAGS.model['loss_balance_weight'] * loss_balance
                    - FLAGS.model['loss_entropy_weight'] * loss_entropy
                    + FLAGS.model['source_var_weight'] * loss_var
                )

                info = {
                    'loss': loss,
                    'loss/fm': loss_fm,
                    'loss/balance': loss_balance,
                    'loss/entropy': loss_entropy,
                    'loss/var': loss_var,
                    'source/shift_norm': jnp.sqrt(jnp.mean(jnp.square(x_0 - z))),
                    'source/x0_sample_norm': jnp.sqrt(jnp.mean(jnp.square(x_0))),
                    'source/mu_x0_norm': jnp.sqrt(jnp.mean(jnp.square(mu_x0))),
                    'source/logvar_mean': jnp.mean(logvar_x0),
                    'source/var_mean': jnp.mean(var_x0),
                    'source/var_min': jnp.min(var_x0),
                    'source/var_max': jnp.max(var_x0),
                    'max_usage': jnp.max(hard_usage),
                    'max_soft_usage': jnp.max(alpha_mean),
                    'router/entropy_mean': loss_entropy,
                    'posterior/q_entropy_mean': jnp.mean(
                        -jnp.sum(q * jnp.log(jnp.maximum(q, 1e-8)), axis=-1)),
                    'q_alpha_agreement': jnp.mean(
                        jnp.argmax(q, axis=-1) == jnp.argmax(alpha, axis=-1)),
                    'v_magnitude_prime': jnp.sqrt(jnp.mean(jnp.square(v_prime))),
                    'dropped_ratio': jnp.mean(
                        labels_dropped == FLAGS.model['num_classes']),
                }

                if collect_full_train_diagnostics:
                    info['router/logit_norm'] = jnp.sqrt(jnp.mean(jnp.square(router_logits)))
                    info.update({
                        'activations/' + k: jnp.sqrt(jnp.mean(jnp.square(v)))
                        for k, v in activations.items()
                    })
                    expert_means = jnp.mean(expert_mu, axis=0).reshape(
                        (FLAGS.model['gmm_num_modes'], -1))
                    expert_norms = jnp.linalg.norm(expert_means, axis=-1) + 1e-8
                    for idx in range(FLAGS.model['gmm_num_modes']):
                        mu_i = expert_mu[:, idx]
                        logvar_i = expert_logvar[:, idx]
                        var_i = jnp.exp(logvar_i)
                        weighted_mu_i = alpha[:, idx, None, None, None] * mu_i
                        info[f'router/usage_mean_{idx}'] = alpha_mean[idx]
                        info[f'router/argmax_freq_{idx}'] = hard_usage[idx]
                        info[f'posterior/q_mean_{idx}'] = jnp.mean(q[:, idx])
                        info[f'expert/mu_norm_{idx}'] = jnp.sqrt(
                            jnp.mean(jnp.square(mu_i)))
                        info[f'expert/logvar_mean_{idx}'] = jnp.mean(logvar_i)
                        info[f'expert/var_mean_{idx}'] = jnp.mean(var_i)
                        info[f'expert/weighted_mu_norm_{idx}'] = jnp.sqrt(
                            jnp.mean(jnp.square(weighted_mu_i)))
                    for i in range(FLAGS.model['gmm_num_modes']):
                        for j in range(i + 1, FLAGS.model['gmm_num_modes']):
                            cosine = jnp.sum(expert_means[i] * expert_means[j])
                            cosine /= expert_norms[i] * expert_norms[j]
                            info[f'expert/cosine_{i}_{j}'] = cosine

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
                model_out = train_state.call_model(
                    x_t,
                    t,
                    dt_base,
                    labels,
                    train=True,
                    rngs={'dropout': dropout_key},
                    params=grad_params,
                    return_activations=collect_full_train_diagnostics,
                )
                if collect_full_train_diagnostics:
                    v_prime, logvars, activations = model_out
                else:
                    v_prime = model_out
                mse_v = jnp.mean((v_prime - v_t) ** 2, axis=(1, 2, 3))
                loss = jnp.mean(mse_v)

                info = {
                    'loss': loss,
                    'v_magnitude_prime': jnp.sqrt(jnp.mean(jnp.square(v_prime))),
                }
                if collect_full_train_diagnostics:
                    info.update({
                        'activations/' + k: jnp.sqrt(jnp.mean(jnp.square(v)))
                        for k, v in activations.items()
                    })

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

        info['lr'] = lr_schedule(train_state.step)
        if collect_full_train_diagnostics:
            info['grad_norm'] = optax.global_norm(grads)
            info['update_norm'] = optax.global_norm(updates)
            info['param_norm'] = optax.global_norm(new_params)
            if FLAGS.model['train_type'] == 'naive-moe-source':
                info['grad/source_total_norm'] = optax.global_norm(grads['source'])
                info['grad/router_norm'] = optax.global_norm(grads['source']['router'])
                for idx in range(FLAGS.model['gmm_num_modes']):
                    info[f'grad/expert_{idx}_norm'] = optax.global_norm(
                        grads['source'][f'expert_{idx}'])

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

    def summarize_moe_metrics(metric_dict):
        summary = {}
        if FLAGS.model['train_type'] != 'naive-moe-source':
            return summary
        usage_keys = [k for k in metric_dict if k.startswith('router/argmax_freq_')]
        soft_usage_keys = [k for k in metric_dict if k.startswith('router/usage_mean_')]
        summary['max_usage'] = float(
            metric_dict.get(
                'max_usage',
                max((float(metric_dict[k]) for k in usage_keys), default=0.0),
            )
        )
        summary['max_soft_usage'] = float(
            metric_dict.get(
                'max_soft_usage',
                max((float(metric_dict[k]) for k in soft_usage_keys), default=0.0),
            )
        )
        for key in (
            'router/entropy_mean',
            'q_alpha_agreement',
            'source/var_mean',
            'source/var_min',
            'source/mu_x0_norm',
        ):
            if key in metric_dict:
                summary[key] = float(metric_dict[key])
        return summary

    last_summary_metrics = None

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

            valid_moe_summary = summarize_moe_metrics(valid_update_info)
            last_summary_metrics = {
                'step': int(i),
                'train_loss': float(update_info['loss']),
                'valid_loss': float(valid_update_info['loss']),
                'lr': float(update_info['lr']),
                **valid_moe_summary,
            }

            if jax.process_index() == 0:
                summary_metrics = {
                    'summary/train_loss': float(update_info['loss']),
                    'summary/valid_loss': float(valid_update_info['loss']),
                    'summary/lr': float(update_info['lr']),
                }
                for key, value in valid_moe_summary.items():
                    summary_metrics[f'summary/{key.replace("/", "_")}'] = float(value)
                wandb.log({**train_metrics, **summary_metrics}, step=i)

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

    final_inference_metrics = None
    if FLAGS.run_final_inference:
        if FLAGS.fid_stats is None:
            raise ValueError("--fid_stats is required when --run_final_inference=1.")
        final_inference_metrics = do_inference(
            FLAGS,
            train_state,
            int(train_state.step),
            dataset,
            dataset_valid,
            shard_data,
            vae_encode,
            vae_decode,
            update,
            get_fid_activations,
            imagenet_labels,
            visualize_labels,
            fid_from_stats,
            truth_fid_stats,
            gmm_state,
        )

    if FLAGS.final_save_dir is not None:
        train_state_gather = jax.experimental.multihost_utils.process_allgather(train_state)
        if jax.process_index() == 0:
            cp = Checkpoint(FLAGS.final_save_dir, parallel=False)
            cp.train_state = train_state_gather
            cp.save()
            del cp
        del train_state_gather

    if jax.process_index() == 0 and FLAGS.metrics_output_path:
        _write_metrics_json(
            FLAGS.metrics_output_path,
            {
                'train': last_summary_metrics or {},
                'inference': final_inference_metrics or {},
                'mode': FLAGS.mode,
                'train_type': FLAGS.model['train_type'],
                'max_steps': int(FLAGS.max_steps),
            },
        )


if __name__ == '__main__':
    app.run(main)
