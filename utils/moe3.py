import concurrent.futures
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from utils.datasets import load_tfds_split


META_VERSION = 1


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


def _quotas(n, k):
    quotas = np.full(k, n // k, dtype=np.int32)
    quotas[:n % k] += 1
    return quotas


def _paths(cache_dir):
    return {
        'meta': os.path.join(cache_dir, 'meta.json'),
        'train_latents': os.path.join(cache_dir, 'train_latents.npy'),
        'val_latents': os.path.join(cache_dir, 'val_latents.npy'),
        'centroids': os.path.join(cache_dir, 'centroids.npy'),
        'train_assignments': os.path.join(cache_dir, 'train_assignments.npy'),
        'val_assignments': os.path.join(cache_dir, 'val_assignments.npy'),
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


def _update_centroids(latents, assignments, num_clusters, chunk_size):
    dim = int(np.prod(latents.shape[1:]))
    sums = np.zeros((num_clusters, dim), dtype=np.float32)
    for start in range(0, latents.shape[0], chunk_size):
        end = min(start + chunk_size, latents.shape[0])
        dirs = _normalize_np(latents[start:end])
        np.add.at(sums, assignments[start:end], dirs)
    norms = np.linalg.norm(sums, axis=1, keepdims=True)
    return sums / np.maximum(norms, 1e-12)


def _balanced_spherical_kmeans(latents, num_clusters, iters, chunk_size, seed):
    if latents.shape[0] < num_clusters:
        raise ValueError('Need at least one training sample per moe3 cluster')
    rng = np.random.default_rng(seed)
    init_ids = rng.choice(latents.shape[0], size=num_clusters, replace=False)
    centroids = _normalize_np(np.asarray(latents[init_ids]))
    quotas = _quotas(latents.shape[0], num_clusters)
    assignments = None
    metric_rows = []
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
        'seconds',
    ]

    for i in range(iters):
        iter_start = time.time()
        old_centroids = centroids
        scores = _score_latents(latents, centroids, chunk_size)
        assignments = balanced_assign_greedy(scores, quotas)
        assign_stats = _assignment_stats(scores, assignments, num_clusters)
        centroids = _update_centroids(latents, assignments, num_clusters, chunk_size)
        centroid_cos = np.sum(old_centroids * centroids, axis=1)
        centroid_shift = 1.0 - centroid_cos
        centroid_stats = _centroid_stats(centroids)
        metrics = {
            **assign_stats,
            **centroid_stats,
            'centroid_shift_mean': float(np.mean(centroid_shift)),
            'centroid_shift_max': float(np.max(centroid_shift)),
            'seconds': time.time() - iter_start,
        }
        metric_rows.append([i + 1] + [metrics[k] for k in metric_columns[1:]])
        print(
            f"moe3 kmeans iter {i + 1}/{iters}: "
            f"score {metrics['score_mean']:.5f}, "
            f"margin {metrics['margin_mean']:.5f}, "
            f"shift {metrics['centroid_shift_mean']:.5f}, "
            f"max centroid cos {metrics['centroid_cos_max']:.5f}"
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
            'moe3_preprocess/kmeans_iters': float(iters),
        })
    return centroids.astype(np.float32), assignments.astype(np.int32)


def prepare_moe3_cache(FLAGS, vae_encode):
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
            FLAGS.dataset_data_dir,
            FLAGS.model.moe3_preprocess_batch_size,
            vae_encode,
            FLAGS.seed,
        )
        val_latents = _encode_split(
            paths['val_latents'],
            FLAGS.dataset_name,
            'validation',
            FLAGS.dataset_data_dir,
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
        centroids, train_assignments = _balanced_spherical_kmeans(
            train_latents,
            int(FLAGS.model.moe3_num_clusters),
            int(FLAGS.model.moe3_kmeans_iters),
            int(FLAGS.model.moe3_assignment_chunk_size),
            int(FLAGS.seed),
        )
        kmeans_metrics = {'moe3_preprocess/kmeans_seconds': time.time() - kmeans_start}
        _wandb_log(kmeans_metrics)
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

        np.save(paths['centroids'], centroids)
        np.save(paths['train_assignments'], train_assignments)
        np.save(paths['val_assignments'], val_assignments)
        with open(paths['meta'], 'w') as f:
            json.dump({
                'version': META_VERSION,
                'dataset_name': FLAGS.dataset_name,
                'num_clusters': int(FLAGS.model.moe3_num_clusters),
                'latent_shape': list(train_latents.shape[1:]),
            }, f)

    cache = {
        'cache_dir': cache_dir,
        'train_latents': np.load(paths['train_latents'], mmap_mode='r'),
        'val_latents': np.load(paths['val_latents'], mmap_mode='r'),
        'centroids': np.load(paths['centroids']).astype(np.float32),
        'train_assignments': np.load(paths['train_assignments']).astype(np.int32),
        'val_assignments': np.load(paths['val_assignments']).astype(np.int32),
    }
    train_counts = np.bincount(
        cache['train_assignments'], minlength=int(FLAGS.model.moe3_num_clusters))
    val_counts = np.bincount(
        cache['val_assignments'], minlength=int(FLAGS.model.moe3_num_clusters))
    centroid_stats = _centroid_stats(cache['centroids'])
    final_metrics = {
        'moe3_preprocess/cache_total_seconds': time.time() - cache_start,
        'moe3_preprocess/train_count_min': float(np.min(train_counts)),
        'moe3_preprocess/train_count_max': float(np.max(train_counts)),
        'moe3_preprocess/train_count_std': float(np.std(train_counts)),
        'moe3_preprocess/val_count_min_loaded': float(np.min(val_counts)),
        'moe3_preprocess/val_count_max_loaded': float(np.max(val_counts)),
        'moe3_preprocess/val_count_std_loaded': float(np.std(val_counts)),
        'moe3_preprocess/centroid_cos_mean': centroid_stats['centroid_cos_mean'],
        'moe3_preprocess/centroid_cos_max': centroid_stats['centroid_cos_max'],
        'moe3_preprocess/centroid_cos_min': centroid_stats['centroid_cos_min'],
    }
    _wandb_log(final_metrics)
    return cache


def load_moe3_centroids(cache_dir):
    return np.load(os.path.join(cache_dir, 'centroids.npy')).astype(np.float32)


def assign_labels_from_noise(x, centroids):
    dirs = _normalize_np(np.asarray(x))
    return np.argmax(dirs @ centroids.T, axis=1).astype(np.int32)


class Moe3WindowIterator:
    def __init__(self, cache, batch_size, window_length, seed, prefetch_windows=1):
        self.latents = cache['train_latents']
        self.val_latents = cache['val_latents']
        self.centroids = cache['centroids']
        self.assignments = cache['train_assignments']
        self.val_assignments = cache['val_assignments']
        self.batch_size = int(batch_size)
        self.window_length = int(window_length)
        self.num_clusters = int(self.centroids.shape[0])
        self.prefetch_windows = max(1, int(prefetch_windows))
        self.rng = np.random.default_rng(seed)

        self.window_size = self.batch_size * self.window_length
        if self.window_size % self.num_clusters != 0:
            raise ValueError('moe3 requires batch_size * window_length divisible by num_clusters')
        self.quota = self.window_size // self.num_clusters

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
        for k in range(self.num_clusters):
            ids = self._take_cluster(k, self.quota)
            x1[offset:offset + self.quota] = self.latents[ids]
            labels[offset:offset + self.quota] = k
            offset += self.quota

        x0_pool = self.rng.standard_normal(x1.shape).astype(np.float32)
        scores = _normalize_np(x0_pool) @ self.centroids.T
        source_assignments = balanced_assign_greedy(
            scores, np.ones(self.num_clusters, dtype=np.int32) * self.quota)

        x0 = np.empty_like(x1)
        paired_x1 = np.empty_like(x1)
        paired_labels = np.empty_like(labels)
        out = 0
        for k in range(self.num_clusters):
            data_slice = slice(k * self.quota, (k + 1) * self.quota)
            source_ids = np.where(source_assignments == k)[0]
            source_ids = source_ids[self.rng.permutation(len(source_ids))]
            x0[out:out + self.quota] = x0_pool[source_ids]
            paired_x1[out:out + self.quota] = x1[data_slice]
            paired_labels[out:out + self.quota] = k
            out += self.quota

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
            'moe3/cluster_count_min': float(self.quota),
            'moe3/cluster_count_max': float(self.quota),
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
        quota = batch_size // self.num_clusters
        if quota < 1 or quota * self.num_clusters != batch_size:
            raise ValueError('moe3 validation requires batch_size divisible by num_clusters')
        rng = np.random.default_rng(seed)
        x1 = np.empty((batch_size, *self.val_latents.shape[1:]), dtype=np.float32)
        labels = np.empty((batch_size,), dtype=np.int32)
        offset = 0
        for k in range(self.num_clusters):
            ids = np.where(self.val_assignments == k)[0]
            if len(ids) == 0:
                raise ValueError(f'moe3 validation cluster {k} is empty')
            chosen = rng.choice(ids, size=quota, replace=len(ids) < quota)
            x1[offset:offset + quota] = self.val_latents[chosen]
            labels[offset:offset + quota] = k
            offset += quota

        x0_pool = rng.standard_normal(x1.shape).astype(np.float32)
        scores = _normalize_np(x0_pool) @ self.centroids.T
        source_assignments = balanced_assign_greedy(
            scores, np.ones(self.num_clusters, dtype=np.int32) * quota)
        x0 = np.empty_like(x1)
        paired_x1 = np.empty_like(x1)
        paired_labels = np.empty_like(labels)
        out = 0
        for k in range(self.num_clusters):
            data_slice = slice(k * quota, (k + 1) * quota)
            source_ids = np.where(source_assignments == k)[0]
            source_ids = source_ids[rng.permutation(len(source_ids))]
            x0[out:out + quota] = x0_pool[source_ids]
            paired_x1[out:out + quota] = x1[data_slice]
            paired_labels[out:out + quota] = k
            out += quota
        order = rng.permutation(batch_size)
        return np.concatenate([x0, paired_x1], axis=-1)[order], paired_labels[order]
