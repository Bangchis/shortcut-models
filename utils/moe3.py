import concurrent.futures
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from utils.datasets import load_tfds_split


META_VERSION = 2


def _wandb_log(metrics, step=0):
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is not None:
        wandb.log(metrics, step=step)


def _wandb_table(name, columns, rows, step=0):
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is not None:
        wandb.log({name: wandb.Table(columns=columns, data=rows)}, step=step)


@jax.jit
def _score_chunk(x, centroids):
    x = x.reshape((x.shape[0], -1)).astype(jnp.float32)
    x = x / jnp.maximum(jnp.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x @ centroids.T


def _as_channel_last(x):
    if x.ndim == 4 and x.shape[1] == 4 and x.shape[-1] != 4:
        return np.transpose(x, (0, 2, 3, 1))
    return x


def _normalize_np(x):
    x = x.reshape((x.shape[0], -1)).astype(np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def _prior_quotas(priors, total):
    priors = np.asarray(priors, dtype=np.float64)
    priors = priors / np.maximum(np.sum(priors), 1e-12)
    raw = priors * int(total)
    quotas = np.floor(raw).astype(np.int32)
    remainder = int(total) - int(np.sum(quotas))
    if remainder > 0:
        order = np.argsort(raw - quotas)[::-1]
        quotas[order[:remainder]] += 1

    nonzero = priors > 0
    if total >= int(np.sum(nonzero)) and np.any(nonzero & (quotas == 0)):
        quotas[nonzero & (quotas == 0)] = 1
        overflow = int(np.sum(quotas)) - int(total)
        for idx in np.argsort(quotas)[::-1]:
            if overflow <= 0:
                break
            reducible = max(0, int(quotas[idx]) - 1)
            take = min(reducible, overflow)
            quotas[idx] -= take
            overflow -= take

    diff = int(total) - int(np.sum(quotas))
    if diff > 0:
        order = np.argsort(raw - quotas)[::-1]
        quotas[order[:diff]] += 1
    elif diff < 0:
        for idx in np.argsort(quotas)[::-1]:
            if diff == 0:
                break
            take = min(int(quotas[idx]), -diff)
            quotas[idx] -= take
            diff += take
    return quotas.astype(np.int32)


def _paths(cache_dir):
    return {
        'meta': os.path.join(cache_dir, 'meta.json'),
        'train_latents': os.path.join(cache_dir, 'train_latents.npy'),
        'val_latents': os.path.join(cache_dir, 'val_latents.npy'),
        'centroids': os.path.join(cache_dir, 'centroids.npy'),
        'train_assignments': os.path.join(cache_dir, 'train_assignments.npy'),
        'val_assignments': os.path.join(cache_dir, 'val_assignments.npy'),
        'train_priors': os.path.join(cache_dir, 'train_priors.npy'),
        'inference_bias': os.path.join(cache_dir, 'inference_bias.npy'),
    }


def _meta_matches(path, FLAGS):
    if not os.path.exists(path):
        return False
    with open(path, 'r') as f:
        meta = json.load(f)
    return (
        meta.get('version') == META_VERSION
        and meta.get('dataset_name') == FLAGS.dataset_name
        and meta.get('num_clusters') == int(FLAGS.model.moe3_num_clusters)
        and meta.get('balance_lambda') == float(FLAGS.model.moe3_balance_lambda)
        and meta.get('balance_min_factor') == float(FLAGS.model.moe3_balance_min_factor)
        and meta.get('balance_max_factor') == float(FLAGS.model.moe3_balance_max_factor)
    )


def _image_dataset(dataset_name, split, batch_size, data_dir):
    if dataset_name != 'celebahq256':
        raise ValueError('moe3 cache currently supports dataset_name=celebahq256')

    def deserialization_fn(data):
        image = data['image']
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - 0.5) / 0.5
        return image

    dataset = load_tfds_split(
        dataset_name, split, data_dir=data_dir, aliases=('celeb_a_hq',))
    dataset = dataset.map(deserialization_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return tfds.as_numpy(dataset)


def _encode_split(path, dataset_name, split, data_dir, batch_size, vae_encode, seed):
    if os.path.exists(path):
        return np.load(path, mmap_mode='r')

    print(f'Building moe3 latent cache for {split}')
    latents = []
    key = jax.random.PRNGKey(seed)
    for i, images in enumerate(_image_dataset(dataset_name, split, batch_size, data_dir)):
        batch_key = jax.random.fold_in(key, i)
        z = np.asarray(vae_encode(batch_key, images))
        z = _as_channel_last(z).astype(np.float32)
        latents.append(z)
    latents = np.concatenate(latents, axis=0)
    np.save(path, latents)
    return np.load(path, mmap_mode='r')


def _score_latents(latents, centroids, chunk_size):
    centroids_jax = jnp.asarray(centroids, dtype=jnp.float32)
    scores = np.empty((latents.shape[0], centroids.shape[0]), dtype=np.float32)
    for start in range(0, latents.shape[0], chunk_size):
        end = min(start + chunk_size, latents.shape[0])
        scores[start:end] = np.asarray(_score_chunk(jnp.asarray(latents[start:end]), centroids_jax))
    return scores


def _centroid_stats(centroids):
    sim = centroids @ centroids.T
    mask = ~np.eye(centroids.shape[0], dtype=bool)
    off_diag = sim[mask]
    return {
        'centroid_cos_mean': float(np.mean(off_diag)),
        'centroid_cos_max': float(np.max(off_diag)),
        'centroid_cos_min': float(np.min(off_diag)),
    }


def _assignment_stats(scores, assignments, num_clusters):
    row_ids = np.arange(scores.shape[0])
    assigned_scores = scores[row_ids, assignments]
    other_scores = scores.copy()
    other_scores[row_ids, assignments] = -np.inf
    best_other = np.max(other_scores, axis=1)
    margin = assigned_scores - best_other
    counts = np.bincount(assignments, minlength=num_clusters)
    priors = counts.astype(np.float64) / max(1, int(np.sum(counts)))
    target = 1.0 / num_clusters
    return {
        'score_mean': float(np.mean(assigned_scores)),
        'score_std': float(np.std(assigned_scores)),
        'score_min': float(np.min(assigned_scores)),
        'score_max': float(np.max(assigned_scores)),
        'margin_mean': float(np.mean(margin)),
        'margin_min': float(np.min(margin)),
        'top1_match_ratio': float(np.mean(assignments == np.argmax(scores, axis=1))),
        'count_min': float(np.min(counts)),
        'count_max': float(np.max(counts)),
        'count_std': float(np.std(counts)),
        'count_max_over_min': float(np.max(counts) / max(1, np.min(counts))),
        'prior_min': float(np.min(priors)),
        'prior_max': float(np.max(priors)),
        'prior_std': float(np.std(priors)),
        'prior_l1_uniform': float(np.sum(np.abs(priors - target))),
        'prior_l2_uniform': float(np.sqrt(np.mean((priors - target) ** 2))),
        'prior_min_over_target': float(np.min(priors) / target),
        'prior_max_over_target': float(np.max(priors) / target),
        'empty_clusters': float(np.sum(counts == 0)),
    }


def balanced_assign_greedy(scores, quotas):
    scores = np.asarray(scores, dtype=np.float32)
    quotas = np.asarray(quotas, dtype=np.int32).copy()
    n, k = scores.shape
    if int(np.sum(quotas)) != n:
        raise ValueError(f'Quotas must sum to {n}, got {np.sum(quotas)}')

    assignments = np.full(n, -1, dtype=np.int32)
    order = np.argsort(scores.reshape(-1))[::-1]
    assigned = 0
    for flat_idx in order:
        i = flat_idx // k
        bucket = flat_idx - i * k
        if assignments[i] == -1 and quotas[bucket] > 0:
            assignments[i] = bucket
            quotas[bucket] -= 1
            assigned += 1
            if assigned == n:
                break
    if assigned != n:
        raise RuntimeError(f'Balanced assignment failed: assigned {assigned}/{n}')
    return assignments


def _soft_balance_assign(scores, balance_lambda, min_factor, max_factor, rng):
    scores = np.asarray(scores, dtype=np.float32)
    n, k = scores.shape
    assignments = np.full(n, -1, dtype=np.int32)
    counts = np.zeros(k, dtype=np.int32)
    target = 1.0 / k
    min_counts = int(np.floor(float(min_factor) * n / k))
    max_counts = int(np.ceil(float(max_factor) * n / k))
    if min_counts * k > n:
        min_counts = 0
    if max_counts * k < n:
        max_counts = n

    for pos, i in enumerate(rng.permutation(n)):
        remaining = n - pos
        deficits = np.maximum(min_counts - counts, 0)
        valid = counts < max_counts
        if np.sum(deficits) >= remaining:
            valid = deficits > 0
        if not np.any(valid):
            valid = np.ones(k, dtype=bool)
        penalty = float(balance_lambda) * (
            2.0 * (counts.astype(np.float32) / n - target) + 1.0 / n)
        effective = scores[i] - penalty
        effective = np.where(valid, effective, -np.inf)
        bucket = int(np.argmax(effective))
        assignments[i] = bucket
        counts[bucket] += 1
    return assignments


def _update_centroids(latents, assignments, num_clusters, chunk_size, old_centroids=None):
    dim = int(np.prod(latents.shape[1:]))
    sums = np.zeros((num_clusters, dim), dtype=np.float32)
    for start in range(0, latents.shape[0], chunk_size):
        end = min(start + chunk_size, latents.shape[0])
        dirs = _normalize_np(latents[start:end])
        np.add.at(sums, assignments[start:end], dirs)
    norms = np.linalg.norm(sums, axis=1, keepdims=True)
    centroids = sums / np.maximum(norms, 1e-12)
    empty = norms[:, 0] < 1e-12
    if old_centroids is not None and np.any(empty):
        centroids[empty] = old_centroids[empty]
    return centroids


def _soft_balanced_spherical_kmeans(
    latents,
    num_clusters,
    iters,
    chunk_size,
    seed,
    balance_lambda,
    min_factor,
    max_factor,
):
    if latents.shape[0] < num_clusters:
        raise ValueError('Need at least one training sample per moe3 cluster')
    rng = np.random.default_rng(seed)
    init_ids = rng.choice(latents.shape[0], size=num_clusters, replace=False)
    centroids = _normalize_np(np.asarray(latents[init_ids]))
    assignments = None
    metric_rows = []
    kmeans_start = time.time()
    metric_columns = [
        'iter',
        'score_mean',
        'score_std',
        'score_min',
        'score_max',
        'margin_mean',
        'margin_min',
        'top1_match_ratio',
        'centroid_shift_mean',
        'centroid_shift_max',
        'centroid_cos_mean',
        'centroid_cos_max',
        'centroid_cos_min',
        'count_min',
        'count_max',
        'count_std',
        'count_max_over_min',
        'empty_clusters',
        'prior_min',
        'prior_max',
        'prior_std',
        'prior_l1_uniform',
        'prior_l2_uniform',
        'prior_min_over_target',
        'prior_max_over_target',
        'balance_penalty',
        'objective_per_sample',
        'seconds',
    ]

    for i in range(iters):
        iter_start = time.time()
        old_centroids = centroids
        scores = _score_latents(latents, centroids, chunk_size)
        assignments = _soft_balance_assign(
            scores, balance_lambda, min_factor, max_factor, rng)
        assign_stats = _assignment_stats(scores, assignments, num_clusters)
        centroids = _update_centroids(
            latents, assignments, num_clusters, chunk_size, old_centroids)
        centroid_cos = np.sum(old_centroids * centroids, axis=1)
        centroid_shift = 1.0 - centroid_cos
        centroid_stats = _centroid_stats(centroids)
        prior_delta = assign_stats['prior_l2_uniform'] ** 2 * num_clusters
        balance_penalty = float(balance_lambda) * prior_delta
        metrics = {
            **assign_stats,
            **centroid_stats,
            'centroid_shift_mean': float(np.mean(centroid_shift)),
            'centroid_shift_max': float(np.max(centroid_shift)),
            'balance_penalty': balance_penalty,
            'objective_per_sample': assign_stats['score_mean'] - balance_penalty,
            'seconds': time.time() - iter_start,
        }
        _wandb_log({
            'moe3_kmeans/iteration': float(i + 1),
            'moe3_kmeans/score_mean': metrics['score_mean'],
            'moe3_kmeans/score_std': metrics['score_std'],
            'moe3_kmeans/score_min': metrics['score_min'],
            'moe3_kmeans/score_max': metrics['score_max'],
            'moe3_kmeans/margin_mean': metrics['margin_mean'],
            'moe3_kmeans/margin_min': metrics['margin_min'],
            'moe3_kmeans/top1_match_ratio': metrics['top1_match_ratio'],
            'moe3_kmeans/centroid_shift_mean': metrics['centroid_shift_mean'],
            'moe3_kmeans/centroid_shift_max': metrics['centroid_shift_max'],
            'moe3_kmeans/centroid_cos_mean': metrics['centroid_cos_mean'],
            'moe3_kmeans/centroid_cos_max': metrics['centroid_cos_max'],
            'moe3_kmeans/centroid_cos_min': metrics['centroid_cos_min'],
            'moe3_kmeans/count_min': metrics['count_min'],
            'moe3_kmeans/count_max': metrics['count_max'],
            'moe3_kmeans/count_std': metrics['count_std'],
            'moe3_kmeans/count_max_over_min': metrics['count_max_over_min'],
            'moe3_kmeans/empty_clusters': metrics['empty_clusters'],
            'moe3_kmeans/prior_min': metrics['prior_min'],
            'moe3_kmeans/prior_max': metrics['prior_max'],
            'moe3_kmeans/prior_std': metrics['prior_std'],
            'moe3_kmeans/prior_l1_uniform': metrics['prior_l1_uniform'],
            'moe3_kmeans/prior_l2_uniform': metrics['prior_l2_uniform'],
            'moe3_kmeans/prior_min_over_target': metrics['prior_min_over_target'],
            'moe3_kmeans/prior_max_over_target': metrics['prior_max_over_target'],
            'moe3_kmeans/balance_penalty': metrics['balance_penalty'],
            'moe3_kmeans/objective_per_sample': metrics['objective_per_sample'],
            'moe3_kmeans/iter_seconds': metrics['seconds'],
            'moe3_kmeans/elapsed_seconds': time.time() - kmeans_start,
        }, step=i + 1)
        metric_rows.append([i + 1] + [metrics[k] for k in metric_columns[1:]])
        print(
            f"moe3 kmeans iter {i + 1}/{iters}: "
            f"score {metrics['score_mean']:.5f}, "
            f"margin {metrics['margin_mean']:.5f}, "
            f"shift {metrics['centroid_shift_mean']:.5f}, "
            f"centroid cos min/max {metrics['centroid_cos_min']:.5f}/{metrics['centroid_cos_max']:.5f}, "
            f"count min/max {metrics['count_min']:.0f}/{metrics['count_max']:.0f}"
        )

    if metric_rows:
        _wandb_table('moe3_preprocess/kmeans_table', metric_columns, metric_rows)
        last = dict(zip(metric_columns, metric_rows[-1]))
        _wandb_log({
            'moe3_preprocess/kmeans_final_score_mean': last['score_mean'],
            'moe3_preprocess/kmeans_final_margin_mean': last['margin_mean'],
            'moe3_preprocess/kmeans_final_margin_min': last['margin_min'],
            'moe3_preprocess/kmeans_final_top1_match_ratio': last['top1_match_ratio'],
            'moe3_preprocess/kmeans_final_centroid_shift_mean': last['centroid_shift_mean'],
            'moe3_preprocess/kmeans_final_centroid_cos_max': last['centroid_cos_max'],
            'moe3_preprocess/kmeans_final_centroid_cos_min': last['centroid_cos_min'],
            'moe3_preprocess/kmeans_final_prior_min': last['prior_min'],
            'moe3_preprocess/kmeans_final_prior_max': last['prior_max'],
            'moe3_preprocess/kmeans_final_prior_l1_uniform': last['prior_l1_uniform'],
            'moe3_preprocess/kmeans_final_count_max_over_min': last['count_max_over_min'],
            'moe3_preprocess/kmeans_final_empty_clusters': last['empty_clusters'],
            'moe3_preprocess/kmeans_final_objective_per_sample': last['objective_per_sample'],
            'moe3_preprocess/kmeans_iters': float(iters),
            'moe3_preprocess/kmeans_balance_lambda': float(balance_lambda),
        })
    return centroids.astype(np.float32), assignments.astype(np.int32)


def _calibrate_inference_bias(
    centroids,
    priors,
    latent_shape,
    num_samples,
    iters,
    chunk_size,
    seed,
):
    num_samples = int(num_samples)
    iters = int(iters)
    if num_samples <= 0 or iters <= 0:
        return np.zeros((centroids.shape[0],), dtype=np.float32)

    rng = np.random.default_rng(seed)
    centroids_jax = jnp.asarray(centroids, dtype=jnp.float32)
    scores = np.empty((num_samples, centroids.shape[0]), dtype=np.float32)
    for start in range(0, num_samples, chunk_size):
        end = min(start + chunk_size, num_samples)
        x = rng.standard_normal((end - start, *latent_shape)).astype(np.float32)
        scores[start:end] = np.asarray(_score_chunk(jnp.asarray(x), centroids_jax))

    priors = np.asarray(priors, dtype=np.float32)
    priors = priors / np.maximum(np.sum(priors), 1e-12)
    bias = np.zeros((centroids.shape[0],), dtype=np.float32)
    for _ in range(iters):
        labels = np.argmax(scores + bias[None], axis=1)
        probs = np.bincount(labels, minlength=centroids.shape[0]).astype(np.float32)
        probs /= float(num_samples)
        bias += 0.5 * (priors - probs)
        bias -= np.mean(bias)

    labels = np.argmax(scores + bias[None], axis=1)
    probs = np.bincount(labels, minlength=centroids.shape[0]).astype(np.float32)
    probs /= float(num_samples)
    _wandb_log({
        'moe3_preprocess/bias_calibration_samples': float(num_samples),
        'moe3_preprocess/bias_calibration_iters': float(iters),
        'moe3_preprocess/bias_prior_l1_error': float(np.sum(np.abs(probs - priors))),
        'moe3_preprocess/bias_prior_max_abs_error': float(np.max(np.abs(probs - priors))),
        'moe3_preprocess/bias_min': float(np.min(bias)),
        'moe3_preprocess/bias_max': float(np.max(bias)),
        'moe3_preprocess/bias_std': float(np.std(bias)),
    })
    print(
        "moe3 inference bias calibration: "
        f"prior l1 error {np.sum(np.abs(probs - priors)):.5f}, "
        f"max abs error {np.max(np.abs(probs - priors)):.5f}"
    )
    return bias.astype(np.float32)


def prepare_moe3_cache(FLAGS, vae_encode, data_dir=None):
    if jax.process_count() != 1:
        raise ValueError('moe3 currently expects a single host process')

    cache_dir = FLAGS.model.moe3_cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    paths = _paths(cache_dir)

    required = [
        paths['train_latents'],
        paths['val_latents'],
        paths['centroids'],
        paths['train_assignments'],
        paths['val_assignments'],
        paths['train_priors'],
        paths['inference_bias'],
    ]
    cache_start = time.time()
    if _meta_matches(paths['meta'], FLAGS) and all(os.path.exists(p) for p in required):
        print(f'Loading moe3 cache from {cache_dir}')
        _wandb_log({'moe3_preprocess/cache_loaded': 1.0})
    else:
        _wandb_log({'moe3_preprocess/cache_loaded': 0.0})
        encode_start = time.time()
        train_latents = _encode_split(
            paths['train_latents'],
            FLAGS.dataset_name,
            'train',
            data_dir,
            FLAGS.model.moe3_preprocess_batch_size,
            vae_encode,
            FLAGS.seed,
        )
        val_latents = _encode_split(
            paths['val_latents'],
            FLAGS.dataset_name,
            'validation',
            data_dir,
            FLAGS.model.moe3_preprocess_batch_size,
            vae_encode,
            FLAGS.seed + 1,
        )
        encode_metrics = {
            'moe3_preprocess/train_latents': float(train_latents.shape[0]),
            'moe3_preprocess/val_latents': float(val_latents.shape[0]),
            'moe3_preprocess/latent_dim': float(np.prod(train_latents.shape[1:])),
            'moe3_preprocess/encode_seconds': time.time() - encode_start,
        }
        _wandb_log(encode_metrics)
        kmeans_start = time.time()
        centroids, train_assignments = _soft_balanced_spherical_kmeans(
            train_latents,
            int(FLAGS.model.moe3_num_clusters),
            int(FLAGS.model.moe3_kmeans_iters),
            int(FLAGS.model.moe3_assignment_chunk_size),
            int(FLAGS.seed),
            float(FLAGS.model.moe3_balance_lambda),
            float(FLAGS.model.moe3_balance_min_factor),
            float(FLAGS.model.moe3_balance_max_factor),
        )
        kmeans_metrics = {'moe3_preprocess/kmeans_seconds': time.time() - kmeans_start}
        _wandb_log(kmeans_metrics)
        train_counts = np.bincount(
            train_assignments, minlength=int(FLAGS.model.moe3_num_clusters))
        train_priors = train_counts.astype(np.float32) / float(np.sum(train_counts))
        val_scores = _score_latents(
            val_latents,
            centroids,
            int(FLAGS.model.moe3_assignment_chunk_size),
        )
        val_assignments = np.argmax(val_scores, axis=1).astype(np.int32)
        val_stats = _assignment_stats(
            val_scores, val_assignments, int(FLAGS.model.moe3_num_clusters))
        val_metrics = {
            'moe3_preprocess/val_score_mean': val_stats['score_mean'],
            'moe3_preprocess/val_score_std': val_stats['score_std'],
            'moe3_preprocess/val_score_min': val_stats['score_min'],
            'moe3_preprocess/val_margin_mean': val_stats['margin_mean'],
            'moe3_preprocess/val_margin_min': val_stats['margin_min'],
            'moe3_preprocess/val_count_min': val_stats['count_min'],
            'moe3_preprocess/val_count_max': val_stats['count_max'],
            'moe3_preprocess/val_count_std': val_stats['count_std'],
        }
        _wandb_log(val_metrics)
        inference_bias = _calibrate_inference_bias(
            centroids,
            train_priors,
            train_latents.shape[1:],
            int(FLAGS.model.moe3_bias_calibration_samples),
            int(FLAGS.model.moe3_bias_calibration_iters),
            int(FLAGS.model.moe3_assignment_chunk_size),
            int(FLAGS.seed) + 123,
        )

        np.save(paths['centroids'], centroids)
        np.save(paths['train_assignments'], train_assignments)
        np.save(paths['val_assignments'], val_assignments)
        np.save(paths['train_priors'], train_priors)
        np.save(paths['inference_bias'], inference_bias)
        with open(paths['meta'], 'w') as f:
            json.dump({
                'version': META_VERSION,
                'dataset_name': FLAGS.dataset_name,
                'num_clusters': int(FLAGS.model.moe3_num_clusters),
                'balance_lambda': float(FLAGS.model.moe3_balance_lambda),
                'balance_min_factor': float(FLAGS.model.moe3_balance_min_factor),
                'balance_max_factor': float(FLAGS.model.moe3_balance_max_factor),
                'latent_shape': list(train_latents.shape[1:]),
            }, f)

    cache_open_start = time.time()
    cache = {
        'cache_dir': cache_dir,
        'train_latents': np.load(paths['train_latents'], mmap_mode='r'),
        'val_latents': np.load(paths['val_latents'], mmap_mode='r'),
        'centroids': np.load(paths['centroids']).astype(np.float32),
        'train_assignments': np.load(paths['train_assignments']).astype(np.int32),
        'val_assignments': np.load(paths['val_assignments']).astype(np.int32),
        'train_priors': np.load(paths['train_priors']).astype(np.float32),
        'inference_bias': np.load(paths['inference_bias']).astype(np.float32),
    }
    cache_open_seconds = time.time() - cache_open_start
    train_counts = np.bincount(
        cache['train_assignments'], minlength=int(FLAGS.model.moe3_num_clusters))
    val_counts = np.bincount(
        cache['val_assignments'], minlength=int(FLAGS.model.moe3_num_clusters))
    centroid_stats = _centroid_stats(cache['centroids'])
    final_metrics = {
        'moe3_preprocess/cache_total_seconds': time.time() - cache_start,
        'moe3_preprocess/cache_open_seconds': cache_open_seconds,
        'moe3_preprocess/train_count_min': float(np.min(train_counts)),
        'moe3_preprocess/train_count_max': float(np.max(train_counts)),
        'moe3_preprocess/train_count_std': float(np.std(train_counts)),
        'moe3_preprocess/train_count_max_over_min': float(np.max(train_counts) / max(1, np.min(train_counts))),
        'moe3_preprocess/train_prior_min': float(np.min(cache['train_priors'])),
        'moe3_preprocess/train_prior_max': float(np.max(cache['train_priors'])),
        'moe3_preprocess/train_prior_std': float(np.std(cache['train_priors'])),
        'moe3_preprocess/train_prior_min_over_target': float(np.min(cache['train_priors']) * FLAGS.model.moe3_num_clusters),
        'moe3_preprocess/train_prior_max_over_target': float(np.max(cache['train_priors']) * FLAGS.model.moe3_num_clusters),
        'moe3_preprocess/val_count_min_loaded': float(np.min(val_counts)),
        'moe3_preprocess/val_count_max_loaded': float(np.max(val_counts)),
        'moe3_preprocess/val_count_std_loaded': float(np.std(val_counts)),
        'moe3_preprocess/centroid_cos_mean': centroid_stats['centroid_cos_mean'],
        'moe3_preprocess/centroid_cos_max': centroid_stats['centroid_cos_max'],
        'moe3_preprocess/centroid_cos_min': centroid_stats['centroid_cos_min'],
    }
    _wandb_log(final_metrics)
    print(
        "moe3 cluster priors: "
        f"train count min/max {np.min(train_counts)}/{np.max(train_counts)}, "
        f"prior min/max {np.min(cache['train_priors']):.5f}/{np.max(cache['train_priors']):.5f}, "
        f"centroid cos min/max {centroid_stats['centroid_cos_min']:.5f}/{centroid_stats['centroid_cos_max']:.5f}"
    )
    return cache


def load_moe3_centroids(cache_dir):
    return np.load(os.path.join(cache_dir, 'centroids.npy')).astype(np.float32)


def load_moe3_inference_state(cache_dir):
    centroids = load_moe3_centroids(cache_dir)
    bias_path = os.path.join(cache_dir, 'inference_bias.npy')
    bias = np.load(bias_path).astype(np.float32) if os.path.exists(bias_path) else None
    return centroids, bias


def assign_labels_from_noise(x, centroids, bias=None):
    dirs = _normalize_np(np.asarray(x))
    scores = dirs @ centroids.T
    if bias is not None:
        scores = scores + np.asarray(bias, dtype=np.float32)[None]
    return np.argmax(scores, axis=1).astype(np.int32)


class Moe3WindowIterator:
    def __init__(self, cache, batch_size, window_length, seed, prefetch_windows=1):
        self.latents = cache['train_latents']
        self.val_latents = cache['val_latents']
        self.centroids = cache['centroids']
        self.assignments = cache['train_assignments']
        self.val_assignments = cache['val_assignments']
        self.priors = cache['train_priors']
        self.batch_size = int(batch_size)
        self.window_length = int(window_length)
        self.num_clusters = int(self.centroids.shape[0])
        self.prefetch_windows = max(1, int(prefetch_windows))
        self.rng = np.random.default_rng(seed)

        self.window_size = self.batch_size * self.window_length
        self.window_quotas = _prior_quotas(self.priors, self.window_size)
        if int(np.sum(self.window_quotas)) != self.window_size:
            raise ValueError('moe3 window quotas do not sum to window size')

        self.cluster_indices = []
        self.cluster_pos = []
        for k in range(self.num_clusters):
            ids = np.where(self.assignments == k)[0].astype(np.int64)
            if len(ids) == 0:
                raise ValueError(f'moe3 cluster {k} has no training samples')
            self.rng.shuffle(ids)
            self.cluster_indices.append(ids)
            self.cluster_pos.append(0)

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.future = self.executor.submit(self._build_window)
        self.window_images = None
        self.window_labels = None
        self.window_info = None
        self.window_pos = 0

    def _take_cluster(self, k, n):
        if n <= 0:
            return np.zeros((0,), dtype=np.int64)
        ids = self.cluster_indices[k]
        out = []
        remaining = n
        while remaining > 0:
            pos = self.cluster_pos[k]
            if pos >= len(ids):
                self.rng.shuffle(ids)
                pos = 0
            take = min(remaining, len(ids) - pos)
            out.append(ids[pos:pos + take])
            pos += take
            self.cluster_pos[k] = pos
            remaining -= take
        return np.concatenate(out, axis=0)

    def _pair_cos(self, x0, x1):
        u0 = _normalize_np(x0)
        u1 = _normalize_np(x1)
        return np.sum(u0 * u1, axis=1)

    def _build_window(self):
        start_time = time.time()
        x1 = np.empty((self.window_size, *self.latents.shape[1:]), dtype=np.float32)
        labels = np.empty((self.window_size,), dtype=np.int32)

        offset = 0
        data_slices = []
        for k, quota in enumerate(self.window_quotas):
            ids = self._take_cluster(k, int(quota))
            data_slice = slice(offset, offset + int(quota))
            x1[data_slice] = self.latents[ids]
            labels[data_slice] = k
            data_slices.append(data_slice)
            offset += int(quota)

        x0_pool = self.rng.standard_normal(x1.shape).astype(np.float32)
        scores = _normalize_np(x0_pool) @ self.centroids.T
        source_assignments = balanced_assign_greedy(scores, self.window_quotas)

        x0 = np.empty_like(x1)
        paired_x1 = np.empty_like(x1)
        paired_labels = np.empty_like(labels)
        out = 0
        for k, quota in enumerate(self.window_quotas):
            if quota <= 0:
                continue
            data_slice = data_slices[k]
            source_ids = np.where(source_assignments == k)[0]
            source_ids = source_ids[self.rng.permutation(len(source_ids))]
            out_slice = slice(out, out + int(quota))
            x0[out_slice] = x0_pool[source_ids]
            paired_x1[out_slice] = x1[data_slice]
            paired_labels[out_slice] = k
            out += int(quota)

        pair_cos = self._pair_cos(x0, paired_x1)
        random_cos = self._pair_cos(x0[self.rng.permutation(self.window_size)], paired_x1)
        order = self.rng.permutation(self.window_size)
        images = np.concatenate([x0, paired_x1], axis=-1)[order]
        labels = paired_labels[order]

        info = {
            'moe3/window_build_seconds': time.time() - start_time,
            'moe3/pair_cos': float(np.mean(pair_cos)),
            'moe3/random_pair_cos': float(np.mean(random_cos)),
            'moe3/delta_cos': float(np.mean(pair_cos) - np.mean(random_cos)),
            'moe3/source_assign_score': float(np.mean(scores[np.arange(self.window_size), source_assignments])),
            'moe3/cluster_count_min': float(np.min(self.window_quotas)),
            'moe3/cluster_count_max': float(np.max(self.window_quotas)),
            'moe3/cluster_count_std': float(np.std(self.window_quotas)),
            'moe3/cluster_prior_min': float(np.min(self.priors)),
            'moe3/cluster_prior_max': float(np.max(self.priors)),
        }
        return images, labels, info

    def next(self):
        if self.window_images is None or self.window_pos >= self.window_size:
            wait_start = time.time()
            self.window_images, self.window_labels, self.window_info = self.future.result()
            self.window_info['moe3/window_wait_seconds'] = time.time() - wait_start
            self.future = self.executor.submit(self._build_window)
            self.window_pos = 0

        start = self.window_pos
        end = start + self.batch_size
        self.window_pos = end
        info = dict(self.window_info)
        if start != 0:
            info['moe3/window_build_seconds'] = 0.0
            info['moe3/window_wait_seconds'] = 0.0
        return self.window_images[start:end], self.window_labels[start:end], info

    def validation_batch(self, seed, batch_size):
        quotas = _prior_quotas(self.priors, batch_size)
        if int(np.sum(quotas)) != int(batch_size):
            raise ValueError('moe3 validation quotas do not sum to batch size')
        rng = np.random.default_rng(seed)
        x1 = np.empty((batch_size, *self.val_latents.shape[1:]), dtype=np.float32)
        labels = np.empty((batch_size,), dtype=np.int32)
        offset = 0
        data_slices = []
        for k, quota in enumerate(quotas):
            ids = np.where(self.val_assignments == k)[0]
            if len(ids) == 0 and quota > 0:
                raise ValueError(f'moe3 validation cluster {k} is empty')
            chosen = rng.choice(ids, size=int(quota), replace=len(ids) < quota)
            data_slice = slice(offset, offset + int(quota))
            x1[data_slice] = self.val_latents[chosen]
            labels[data_slice] = k
            data_slices.append(data_slice)
            offset += int(quota)

        x0_pool = rng.standard_normal(x1.shape).astype(np.float32)
        scores = _normalize_np(x0_pool) @ self.centroids.T
        source_assignments = balanced_assign_greedy(scores, quotas)
        x0 = np.empty_like(x1)
        paired_x1 = np.empty_like(x1)
        paired_labels = np.empty_like(labels)
        out = 0
        for k, quota in enumerate(quotas):
            if quota <= 0:
                continue
            data_slice = data_slices[k]
            source_ids = np.where(source_assignments == k)[0]
            source_ids = source_ids[rng.permutation(len(source_ids))]
            out_slice = slice(out, out + int(quota))
            x0[out_slice] = x0_pool[source_ids]
            paired_x1[out_slice] = x1[data_slice]
            paired_labels[out_slice] = k
            out += int(quota)
        order = rng.permutation(batch_size)
        return np.concatenate([x0, paired_x1], axis=-1)[order], paired_labels[order]
