import json
import os
import time
from functools import partial

import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import tqdm
import wandb
from absl import flags

from gmm_utils import flatten_latents, posterior_from_stats

flags.DEFINE_integer('inference_timesteps', 128, 'Number of timesteps for inference.')
flags.DEFINE_integer('inference_generations', 4096, 'Number of generations for inference.')
flags.DEFINE_float('inference_cfg_scale', 1.0, 'CFG scale for inference.')
flags.DEFINE_integer('save_x_render', 1, 'Whether to save rendered inference grid to save_dir/x_render.npy.')
flags.DEFINE_integer('dump_flow_viz', 0, 'Whether final inference should dump compact latent endpoints and ODE paths.')
flags.DEFINE_integer('flow_viz_samples', 512, 'Number of samples to cache for latent endpoint/path visualization.')


def _compress_latents(arr):
    arr = np.asarray(arr)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float16)
    return arr.astype(np.float16, copy=False)


def _flow_viz_completed_steps(num_steps):
    candidates = [0, 1, 2, 4, 8, 16, 32, 64, max(1, num_steps - 1), num_steps]
    ordered = []
    seen = set()
    for value in candidates:
        value = int(max(0, min(num_steps, value)))
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def do_inference(
    FLAGS,
    train_state,
    step,
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
    gmm_state=None,
):
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

    def _write_metrics(path, payload):
        if not path:
            return
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(_json_ready(payload), f, indent=2, sort_keys=True)

    with jax.spmd_mode('allow_all'):
        key = jax.random.PRNGKey(42 + jax.process_index())
        batch_images, batch_labels = next(dataset)
        valid_images, valid_labels = next(dataset_valid)
        if FLAGS.model.use_stable_vae and 'latent' not in FLAGS.dataset_name:
            batch_images = vae_encode(key, batch_images)
            valid_images = vae_encode(key, valid_images)
        batch_labels_sharded, valid_labels_sharded = shard_data(batch_labels, valid_labels)
        labels_uncond = shard_data(jnp.ones(batch_labels.shape, dtype=jnp.int32) * FLAGS.model['num_classes'])

        @partial(jax.jit, static_argnums=(5,))
        def call_model(train_state, images, t, dt, labels, use_ema=True):
            if use_ema and FLAGS.model.use_ema:
                call_fn = train_state.call_model_ema
            else:
                call_fn = train_state.call_model
            return call_fn(images, t, dt, labels, train=False)

        @partial(jax.jit, static_argnums=(3, 4))
        def call_source(train_state, latents, condition, return_experts=False, use_ema=True):
            if use_ema and FLAGS.model.use_ema:
                call_fn = train_state.call_source_ema
            else:
                call_fn = train_state.call_source
            return call_fn(latents, condition, return_experts=return_experts)

        images_shape = batch_images.shape

        def sample_source_prior(sample_key):
            if FLAGS.model.train_type != 'naive-moe-source':
                latents = jax.random.normal(sample_key, images_shape)
                return shard_data(latents)
            z_key, cond_key, x0_key = jax.random.split(sample_key, 3)
            z = jax.random.normal(z_key, images_shape)
            sampled_modes = jax.random.categorical(
                cond_key,
                jnp.log(jnp.maximum(gmm_state['pi'], 1e-8)),
                shape=(images_shape[0],),
            )
            condition = jax.nn.one_hot(
                sampled_modes,
                FLAGS.model['gmm_num_modes'],
                dtype=jnp.float32,
            )
            z, condition = shard_data(z, condition)
            mu_x0, logvar_x0, _, _ = call_source(train_state, z, condition)
            x0_key = shard_data(jax.random.normal(x0_key, images_shape))
            return mu_x0 + x0_key * jnp.exp(0.5 * logvar_x0)

        def sample_source_posterior(sample_key, latents):
            if FLAGS.model.train_type != 'naive-moe-source':
                return latents, None
            z_key, mode_key, x0_key = jax.random.split(sample_key, 3)
            flat_latents = flatten_latents(latents)
            q = posterior_from_stats(
                flat_latents,
                gmm_state['mean'],
                gmm_state['std'],
                float(np.asarray(gmm_state.get('standardize_eps', np.array(1e-6, dtype=np.float32)))),
                gmm_state['log_pi'],
                gmm_state['mu'],
                gmm_state['var'],
            )
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
            z = jax.random.normal(z_key, latents.shape)
            z, condition_weights = shard_data(z, condition_weights)
            mu_x0, logvar_x0, _, alpha = call_source(train_state, z, condition_weights)
            x0_key = shard_data(jax.random.normal(x0_key, latents.shape))
            x0 = mu_x0 + x0_key * jnp.exp(0.5 * logvar_x0)
            stats = {
                'q_posterior': np.array(q),
                'alpha_posterior': np.array(jax.experimental.multihost_utils.process_allgather(alpha)[0]),
                'conditioned_mode': np.array(sampled_modes),
            }
            return x0, stats

        analysis_needed = 0
        if FLAGS.dump_source_stats or FLAGS.dump_flow_viz:
            analysis_needed = max(
                int(getattr(FLAGS, 'source_stats_samples', 0)),
                int(getattr(FLAGS, 'flow_viz_samples', 0) if FLAGS.dump_flow_viz else 0),
                images_shape[0],
            )
        flow_viz_needed = int(getattr(FLAGS, 'flow_viz_samples', 0) if FLAGS.dump_flow_viz else 0)
        flow_completed_steps = _flow_viz_completed_steps(int(FLAGS.inference_timesteps)) if flow_viz_needed > 0 else []
        flow_step_set = {step_value for step_value in flow_completed_steps if step_value > 0}

        prior_chunks = []
        generated_chunks = []
        label_chunks = []
        flow_states_by_step = {step_value: [] for step_value in flow_completed_steps}
        collected_analysis = 0
        collected_flow = 0

        def dump_source_stats_if_needed():
            if FLAGS.save_dir is None or not (FLAGS.dump_source_stats or FLAGS.dump_flow_viz):
                return None
            os.makedirs(FLAGS.save_dir, exist_ok=True)
            num_needed = max(analysis_needed, images_shape[0])
            x1_chunks = []
            x0_posterior_chunks = []
            q_chunks = []
            alpha_chunks = []
            mode_chunks = []
            gathered = 0
            stats_key = jax.random.PRNGKey(1701 + jax.process_index())
            while gathered < num_needed:
                batch_images_local, _ = next(dataset_valid)
                if FLAGS.model.use_stable_vae and 'latent' not in FLAGS.dataset_name:
                    local_key = jax.random.fold_in(stats_key, gathered)
                    batch_images_local = vae_encode(local_key, batch_images_local)
                batch_take = min(num_needed - gathered, batch_images_local.shape[0])
                batch_latents = batch_images_local[:batch_take]
                x1_chunks.append(np.array(batch_latents))
                x0_posterior, posterior_stats = sample_source_posterior(
                    jax.random.fold_in(stats_key, gathered + 1),
                    batch_latents,
                )
                if posterior_stats is not None:
                    x0_posterior_chunks.append(
                        np.array(jax.experimental.multihost_utils.process_allgather(x0_posterior)[0][:batch_take])
                    )
                    q_chunks.append(posterior_stats['q_posterior'][:batch_take])
                    alpha_chunks.append(posterior_stats['alpha_posterior'][:batch_take])
                    mode_chunks.append(posterior_stats['conditioned_mode'][:batch_take])
                gathered += batch_take

            payload = {
                'x1_data': _compress_latents(np.concatenate(x1_chunks, axis=0)[:num_needed]),
                'x0_prior': _compress_latents(np.concatenate(prior_chunks, axis=0)[:num_needed]) if prior_chunks else np.zeros((0,), dtype=np.float16),
                'x1_final': _compress_latents(np.concatenate(generated_chunks, axis=0)[:num_needed]) if generated_chunks else np.zeros((0,), dtype=np.float16),
                'labels': np.concatenate(label_chunks, axis=0)[:num_needed] if label_chunks else np.zeros((0,), dtype=np.int32),
            }
            if x0_posterior_chunks:
                payload['x0_posterior'] = _compress_latents(np.concatenate(x0_posterior_chunks, axis=0)[:num_needed])
                payload['q_posterior'] = np.concatenate(q_chunks, axis=0)[:num_needed].astype(np.float32, copy=False)
                payload['alpha_posterior'] = np.concatenate(alpha_chunks, axis=0)[:num_needed].astype(np.float32, copy=False)
                payload['conditioned_mode'] = np.concatenate(mode_chunks, axis=0)[:num_needed].astype(np.int32, copy=False)
            if flow_viz_needed > 0 and flow_completed_steps:
                path_chunks = []
                for step_value in flow_completed_steps:
                    if flow_states_by_step[step_value]:
                        path_chunks.append(_compress_latents(np.concatenate(flow_states_by_step[step_value], axis=0)[:flow_viz_needed]))
                if len(path_chunks) == len(flow_completed_steps) and path_chunks:
                    payload['flow_path_states'] = np.stack(path_chunks, axis=1)
                    payload['flow_path_times'] = (np.asarray(flow_completed_steps, dtype=np.float32) / float(FLAGS.inference_timesteps))
                    payload['flow_path_completed_steps'] = np.asarray(flow_completed_steps, dtype=np.int32)
            output_path = os.path.join(FLAGS.save_dir, 'source_stats.npz')
            np.savez_compressed(output_path, **payload)
            return output_path

        denoise_timesteps = FLAGS.inference_timesteps
        num_generations = FLAGS.inference_generations
        cfg_scale = FLAGS.inference_cfg_scale
        should_save_x_render = bool(FLAGS.save_x_render)
        alpha = float(FLAGS.model['kfm_alpha']) if FLAGS.model['train_type'] == 'khoat-fm' else 1.0
        x_render = []
        activations = []
        generation_time = 0.0
        print(f"Calc FID for CFG {cfg_scale} and denoise_timesteps {denoise_timesteps}")
        for fid_it in tqdm.tqdm(range(num_generations // FLAGS.batch_size)):
            key = jax.random.PRNGKey(42)
            key = jax.random.fold_in(key, fid_it)
            key = jax.random.fold_in(key, jax.process_index())
            eps_key, label_key = jax.random.split(key)
            batch_start = time.perf_counter()
            x = sample_source_prior(eps_key)
            labels = jax.random.randint(label_key, (images_shape[0],), 0, FLAGS.model.num_classes)
            labels = shard_data(labels)
            x0_initial = x

            batch_analysis_take = 0
            batch_flow_take = 0
            prior_batch = None
            if analysis_needed > collected_analysis or flow_viz_needed > collected_flow:
                prior_batch = np.array(jax.experimental.multihost_utils.process_allgather(x)[0])
                if analysis_needed > collected_analysis:
                    batch_analysis_take = min(analysis_needed - collected_analysis, prior_batch.shape[0])
                    prior_chunks.append(prior_batch[:batch_analysis_take])
                if flow_viz_needed > collected_flow:
                    batch_flow_take = min(flow_viz_needed - collected_flow, prior_batch.shape[0])
                    if batch_flow_take > 0:
                        flow_states_by_step[0].append(prior_batch[:batch_flow_take])

            delta_t = 1.0 / denoise_timesteps
            for ti in range(denoise_timesteps):
                t = ti / denoise_timesteps
                t_vector = jnp.full((images_shape[0],), t)
                if FLAGS.model.train_type in ('naive', 'naive-moe-source'):
                    dt_flow = np.log2(FLAGS.model['denoise_timesteps']).astype(jnp.int32)
                    dt_base = jnp.ones(images_shape[0], dtype=jnp.int32) * dt_flow
                else:
                    dt_flow = np.log2(denoise_timesteps).astype(jnp.int32)
                    dt_base = jnp.ones(images_shape[0], dtype=jnp.int32) * dt_flow
                t_vector, dt_base = shard_data(t_vector, dt_base)
                if cfg_scale == 1:
                    v = call_model(train_state, x, t_vector, dt_base, labels)
                elif cfg_scale == 0:
                    v = call_model(train_state, x, t_vector, dt_base, labels_uncond)
                else:
                    v_pred_uncond = call_model(train_state, x, t_vector, dt_base, labels_uncond)
                    v_pred_label = call_model(train_state, x, t_vector, dt_base, labels)
                    v = v_pred_uncond + cfg_scale * (v_pred_label - v_pred_uncond)

                if FLAGS.model.train_type == 'khoat-fm':
                    if ti == 0:
                        x = (1.0 - alpha) * x0_initial + alpha * (delta_t * v)
                    else:
                        x = x + alpha * (delta_t * v)
                elif FLAGS.model.train_type == 'consistency':
                    eps = shard_data(jax.random.normal(jax.random.fold_in(eps_key, ti), images_shape))
                    x1pred = x + v * (1 - t)
                    x = x1pred * (t + delta_t) + eps * (1 - t - delta_t)
                else:
                    x = x + v * delta_t

                completed_step = ti + 1
                if batch_flow_take > 0 and completed_step in flow_step_set:
                    flow_states_by_step[completed_step].append(
                        np.array(jax.experimental.multihost_utils.process_allgather(x)[0][:batch_flow_take])
                    )

            x = jax.block_until_ready(x)
            generation_time += time.perf_counter() - batch_start

            if batch_analysis_take > 0:
                generated_batch = np.array(jax.experimental.multihost_utils.process_allgather(x)[0])
                label_batch = np.array(jax.experimental.multihost_utils.process_allgather(labels)[0])
                generated_chunks.append(generated_batch[:batch_analysis_take])
                label_chunks.append(label_batch[:batch_analysis_take])
                collected_analysis += batch_analysis_take
            if batch_flow_take > 0:
                collected_flow += batch_flow_take

            if FLAGS.model.use_stable_vae:
                x = vae_decode(x)
                if should_save_x_render and num_generations < 10000:
                    x_render.append(np.array(jax.experimental.multihost_utils.process_allgather(x)))
            x = jax.image.resize(x, (x.shape[0], 299, 299, 3), method='bilinear', antialias=False)
            x = jnp.clip(x, -1, 1)
            acts = get_fid_activations(x)[..., 0, 0, :]
            acts = jax.experimental.multihost_utils.process_allgather(acts)
            activations.append(np.array(acts))

        if jax.process_index() == 0:
            activations = np.concatenate(activations, axis=0)
            activations = activations.reshape((-1, activations.shape[-1]))
            mu1 = np.mean(activations, axis=0)
            sigma1 = np.cov(activations, rowvar=False)
            fid = fid_from_stats(mu1, sigma1, truth_fid_stats['mu'], truth_fid_stats['sigma'])
            print(f"FID is {fid}")
            print(f"FID is {fid}")
            print(f"FID is {fid}")
            latency = generation_time / max(num_generations, 1)
            throughput = num_generations / max(generation_time, 1e-8)
            source_stats_path = dump_source_stats_if_needed()
            fid_key = f'fid{int(denoise_timesteps)}_{int(num_generations)}'
            latency_key = f'latency_{int(denoise_timesteps)}'
            throughput_key = f'throughput_{int(denoise_timesteps)}'
            summary = {
                fid_key: float(fid),
                'denoise_timesteps': int(denoise_timesteps),
                'num_generations': int(num_generations),
                latency_key: float(latency),
                throughput_key: float(throughput),
                'source_stats_path': source_stats_path,
            }
            if wandb.run is not None and step is not None:
                wandb.log(
                    {
                        f'final/{fid_key}': float(fid),
                        f'final/{latency_key}': float(latency),
                        f'final/{throughput_key}': float(throughput),
                    },
                    step=int(step),
                )
            _write_metrics(FLAGS.metrics_output_path, summary)

            if FLAGS.save_dir is not None:
                os.makedirs(FLAGS.save_dir, exist_ok=True)
                if should_save_x_render and x_render:
                    np.save(os.path.join(FLAGS.save_dir, 'x_render.npy'), np.concatenate(x_render, axis=0))
            return summary
        return {}
