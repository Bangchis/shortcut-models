from typing import Any
import jax.numpy as jnp
from absl import app, flags
from functools import partial
import numpy as np
import tensorflow_datasets as tfds
import tqdm
import jax
import jax.numpy as jnp
import flax
import optax
import wandb

import os
import matplotlib.pyplot as plt
from ml_collections import config_flags
import ml_collections

from utils.wandb import setup_wandb, default_wandb_config
from utils.train_state import TrainStateEma
from utils.checkpoint import Checkpoint
from utils.stable_vae import StableVAE
from utils.sharding import create_sharding, all_gather
from utils.gmm_em import EMConfig, fit_gmm_em_streaming
from utils.gmm_prior import load_prior_npz, save_prior_npz
from utils.datasets import get_dataset
from model import DiT
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
    'train_type': 'shortcut',  # or naive, khoat-fm.

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

    # ===== GMM-FM (new mode) defaults =====
    # Fit a diagonal GMM prior in StableVAE latent space, then use it to sample x0.
    'gmm_K': 32,
    # EM_subset in [0,1]: fraction of training split used to fit the GMM.
    'gmm_em_subset': 0.10,
    # EM iterations
    'gmm_em_max_iters': 50,
    # Early stop: stop if loglik improvement < tol for `patience` consecutive iters.
    'gmm_em_tol': 1e-4,
    'gmm_em_patience': 3,
    # Numerical stability
    'gmm_var_floor': 1e-5,
    # Where to save/load the prior (npz). If empty, auto-path under save_dir or cwd.
    'gmm_cache_path': '',
    # Force refit even if cache exists.
    'gmm_force_refit': False,
    # Assignment mode used in training: "soft_sample" (default), "moment", "hard"
    'gmm_assign_mode': 'soft_sample',
    # Temperature for responsibilities softmax(log p(k|x1)/T). (T=1 default)
    'gmm_resp_temperature': 1.0,
    # PCA debug plot
    'gmm_pca_max_points': 2000,
    'gmm_log_artifact': True,

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
    def _get_num_examples(dataset_name: str, is_train: bool = True) -> int:
        """Best-effort dataset size lookup for EM_subset."""
        # Map common aliases used in this repo to TFDS builder names.
        if 'imagenet' in dataset_name:
            tfds_name = 'imagenet2012'
            split = 'train' if is_train else 'validation'
        elif 'celebahq256' in dataset_name:
            tfds_name = 'celebahq256'
            split = 'train'  # TFDS celebA-HQ typically uses only train split
        else:
            tfds_name = dataset_name
            split = 'train' if is_train else 'validation'
        try:
            builder = tfds.builder(tfds_name)
            splits = builder.info.splits
            if split in splits:
                return int(splits[split].num_examples)
            # fallback: first available split
            return int(next(iter(splits.values())).num_examples)
        except Exception as e:
            if jax.process_index() == 0:
                print(
                    f"[GMM-FM] WARNING: could not read TFDS num_examples for {tfds_name} ({e}). Using 100000 as fallback.")
            return 100000

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

    # ------------------------------------------------------------
    # (GMM-FM) Preprocess: fit/load a diagonal GMM prior in latent space.
    # ------------------------------------------------------------
    gmm_prior = None
    if FLAGS.model['train_type'] == 'gmm-fm':
        if not FLAGS.model.use_stable_vae:
            raise ValueError(
                "gmm-fm requires model.use_stable_vae=True (we fit GMM in StableVAE latent space).")

        cache_path = str(FLAGS.model.get('gmm_cache_path', '') or '')
        if cache_path == '':
            base_dir = FLAGS.save_dir if FLAGS.save_dir is not None else os.getcwd()
            cache_path = os.path.join(
                base_dir, f"gmm_prior_{FLAGS.dataset_name}_K{int(FLAGS.model['gmm_K'])}.npz")
        # store back to config for reproducibility
        FLAGS.model.gmm_cache_path = cache_path

        if (not bool(FLAGS.model.get('gmm_force_refit', False))) and os.path.exists(cache_path):
            if jax.process_index() == 0:
                print("[GMM-FM] Loading GMM prior from:", cache_path)
            gmm_prior = load_prior_npz(cache_path)
        else:
            if jax.process_index() == 0:
                print("[GMM-FM] Fitting GMM prior with EM...")
                print("[GMM-FM] cache_path:", cache_path)

            # Determine how many examples to use for EM.
            em_subset = float(FLAGS.model.get('gmm_em_subset', 0.1))
            em_subset = max(0.0, min(1.0, em_subset))
            num_examples = int(_get_num_examples(
                FLAGS.dataset_name, is_train=True))
            target_examples = max(
                int(num_examples * em_subset), int(FLAGS.model['gmm_K']) * 4)
            num_batches = int(np.ceil(target_examples / local_batch_size))
            if jax.process_index() == 0:
                print(f"[GMM-FM] Train split examples: {num_examples}")
                print(
                    f"[GMM-FM] EM_subset={em_subset} => target_examples={target_examples} (~{num_batches} batches)")

            # Fresh iterator for EM (so we don't disturb training RNG/iterator)
            dataset_em = get_dataset(
                FLAGS.dataset_name, local_batch_size, True, FLAGS.debug_overfit)

            gmm_rng = jax.random.PRNGKey(int(FLAGS.seed) + 12345)

            def _encode_to_latent(batch_images, key):
                # Match training behavior: if dataset isn't already latent, encode with StableVAE.
                if FLAGS.model.use_stable_vae and 'latent' not in FLAGS.dataset_name:
                    batch_images = vae_encode(key, batch_images)
                # If latent dataset contains concatenated tensors, keep the "x1 half" convention.
                if 'latent' in FLAGS.dataset_name and batch_images.shape[-1] > 4:
                    batch_images = batch_images[...,
                                                batch_images.shape[-1] // 2:]
                return batch_images

            # Build init points for mu initialization
            init_buf = []
            need = int(FLAGS.model['gmm_K'])
            while need > 0:
                batch_images, _ = next(dataset_em)
                gmm_rng, k = jax.random.split(gmm_rng)
                lat = _encode_to_latent(batch_images, k)
                x = jnp.asarray(lat).reshape(
                    (lat.shape[0], -1)).astype(jnp.float32)
                take = min(int(x.shape[0]), need)
                init_buf.append(x[:take])
                need -= take
            init_points = jnp.concatenate(init_buf, axis=0)

            D = int(init_points.shape[1])

            def batch_iterator():
                nonlocal gmm_rng
                while True:
                    batch_images, _ = next(dataset_em)
                    gmm_rng, k = jax.random.split(gmm_rng)
                    lat = _encode_to_latent(batch_images, k)
                    x = jnp.asarray(lat).reshape(
                        (lat.shape[0], -1)).astype(jnp.float32)
                    yield x

            it = batch_iterator()

            em_cfg = EMConfig(
                K=int(FLAGS.model['gmm_K']),
                max_iters=int(FLAGS.model.get('gmm_em_max_iters', 50)),
                var_floor=float(FLAGS.model.get('gmm_var_floor', 1e-5)),
                tol=float(FLAGS.model.get('gmm_em_tol', 1e-4)),
                patience=int(FLAGS.model.get('gmm_em_patience', 3)),
            )

            def on_iter_end(iter_idx, logs):
                if jax.process_index() == 0 and wandb.run is not None:
                    wandb.log({
                        "gmm_em/iter": int(iter_idx),
                        "gmm_em/loglik": float(logs["loglik"]),
                        "gmm_em/min_Nk": float(logs["min_Nk"]),
                        "gmm_em/max_Nk": float(logs["max_Nk"]),
                    })

            gmm_prior, em_logs = fit_gmm_em_streaming(
                it,
                num_batches=num_batches,
                D=D,
                cfg=em_cfg,
                rng=gmm_rng,
                init_points=init_points,
                on_iter_end=on_iter_end,
            )

            # Save + log artifacts only on process 0
            if jax.process_index() == 0:
                cache_dir = os.path.dirname(cache_path)
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                save_prior_npz(cache_path, gmm_prior)
                print("[GMM-FM] Saved GMM prior to:", cache_path)

                if wandb.run is not None and bool(FLAGS.model.get("gmm_log_artifact", True)):
                    try:
                        art = wandb.Artifact(
                            name=f"gmm_prior_{FLAGS.dataset_name}_K{int(FLAGS.model['gmm_K'])}", type="gmm_prior")
                        art.add_file(cache_path)
                        wandb.log_artifact(art)
                    except Exception as e:
                        print("[GMM-FM] W&B artifact logging failed:", e)

                # PCA 2D visualization (debug/insight)
                if wandb.run is not None:
                    try:
                        max_pts = int(FLAGS.model.get(
                            "gmm_pca_max_points", 2000))
                        max_pts = max(200, max_pts)
                        # New iterator for PCA sampling
                        dataset_pca = get_dataset(
                            FLAGS.dataset_name, local_batch_size, True, FLAGS.debug_overfit)
                        pts = []
                        while sum([p.shape[0] for p in pts]) < max_pts:
                            batch_images, _ = next(dataset_pca)
                            gmm_rng, k = jax.random.split(gmm_rng)
                            lat = _encode_to_latent(batch_images, k)
                            x = np.array(jax.device_get(jnp.asarray(lat).reshape(
                                (lat.shape[0], -1)).astype(jnp.float32)))
                            pts.append(x)
                        X = np.concatenate(pts, axis=0)[:max_pts]  # (N,D)
                        mean = X.mean(axis=0, keepdims=True)
                        Xc = X - mean

                        # PCA via Gram matrix (N <= D is typical here)
                        G = (Xc @ Xc.T) / max(Xc.shape[0] - 1, 1)  # (N,N)
                        evals, evecs = np.linalg.eigh(G)
                        idx2 = np.argsort(evals)[-2:]
                        evals2 = evals[idx2]
                        U2 = evecs[:, idx2]
                        Z = U2 * np.sqrt(np.maximum(evals2, 1e-12))  # (N,2)

                        # Component centers projected
                        mu = np.array(jax.device_get(gmm_prior.mu))
                        mu_c = mu - mean
                        # components in feature space: V = Xc^T U / sqrt(evals)
                        V2 = (Xc.T @ U2) / \
                            np.sqrt(np.maximum(evals2, 1e-12))[None, :]
                        centers_2d = mu_c @ V2  # (K,2)

                        # Color points by hard assignment under the fitted GMM
                        from utils.gmm_prior import posterior_logp
                        log_r = np.array(jax.device_get(
                            posterior_logp(gmm_prior, jnp.asarray(X))))
                        mode = log_r.argmax(axis=-1)

                        fig = plt.figure()
                        ax = fig.add_subplot(111)
                        sc = ax.scatter(Z[:, 0], Z[:, 1],
                                        c=mode, s=6, alpha=0.6)
                        ax.scatter(
                            centers_2d[:, 0], centers_2d[:, 1], marker='x', s=80)
                        ax.set_title("GMM-FM: PCA 2D of latents + centers")
                        ax.set_xlabel("PC1")
                        ax.set_ylabel("PC2")
                        fig.tight_layout()

                        tmp_path = os.path.join(os.path.dirname(cache_path) if os.path.dirname(
                            cache_path) else ".", "gmm_pca2d.png")
                        fig.savefig(tmp_path, dpi=150)
                        plt.close(fig)
                        wandb.log({"gmm_em/pca2d": wandb.Image(tmp_path)})
                    except Exception as e:
                        print("[GMM-FM] PCA viz failed:", e)

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
        'ignore_dt': False if (FLAGS.model['train_type'] in ('shortcut', 'livereflow', 'khoat-fm')) else True,
    }
    model_def = DiT(**dit_args)
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
        param_key, dropout_key, dropout2_key = jax.random.split(rng, 3)
        example_t = jnp.zeros((1,))
        example_dt = jnp.zeros((1,))
        example_label = jnp.zeros((1,), dtype=jnp.int32)
        example_obs = jnp.zeros(example_obs_shape)
        model_rngs = {'params': param_key,
                      'label_dropout': dropout_key, 'dropout': dropout2_key}
        params = model_def.init(model_rngs, example_obs,
                                example_t, example_dt, example_label)['params']
        opt_state = tx.init(params)
        return TrainStateEma.create(model_def, params, rng=rng, tx=tx, opt_state=opt_state)

    rng = jax.random.PRNGKey(FLAGS.seed)
    train_state_shape = jax.eval_shape(init, rng)

    data_sharding, train_state_sharding, no_shard, shard_data, global_to_local = create_sharding(
        FLAGS.model.sharding, train_state_shape)
    train_state = jax.jit(init, out_shardings=train_state_sharding)(rng)
    jax.debug.visualize_array_sharding(
        train_state.params['FinalLayer_0']['Dense_0']['kernel'])
    jax.debug.visualize_array_sharding(
        train_state.params['TimestepEmbedder_1']['Dense_0']['kernel'])
    jax.experimental.multihost_utils.assert_equal(
        train_state.params['TimestepEmbedder_1']['Dense_0']['kernel'])
    start_step = 1

    if FLAGS.load_dir is not None:
        cp = Checkpoint(FLAGS.load_dir)
        replace_dict = cp.load_as_dict()['train_state']
        del replace_dict['opt_state']  # Debug
        train_state = train_state.replace(**replace_dict)
        if FLAGS.wandb.run_id != "None":  # If we are continuing a run.
            start_step = train_state.step
        train_state = jax.jit(
            lambda x: x, out_shardings=train_state_sharding)(train_state)
        print("Loaded model with step", train_state.step)
        train_state = train_state.replace(step=0)
        jax.debug.visualize_array_sharding(
            train_state.params['FinalLayer_0']['Dense_0']['kernel'])
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

        elif FLAGS.model['train_type'] == 'gmm-fm':
            from targets_gmm_fm import get_targets
            x_t, v_t, t, dt_base, labels, info = get_targets(
                FLAGS, targets_key, train_state, images, labels, force_t, force_dt, gmm_prior=gmm_prior)

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
        do_inference(FLAGS, train_state, None, dataset, dataset_valid, shard_data, vae_encode, vae_decode, update,
                     get_fid_activations, imagenet_labels, visualize_labels,
                     fid_from_stats, truth_fid_stats, gmm_prior=gmm_prior)
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
            eval_model(FLAGS, train_state, train_state_teacher, i, dataset, dataset_valid, shard_data, vae_encode, vae_decode, update,
                       get_fid_activations, imagenet_labels, visualize_labels,
                       fid_from_stats, truth_fid_stats, gmm_prior=gmm_prior)

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
