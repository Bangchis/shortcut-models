import math

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np


def flatten_latents(latents):
    return latents.reshape((latents.shape[0], -1))


def standardize_latents(latents_flat, mean, std, eps):
    return (latents_flat - mean) / (std + eps)


@jax.jit
def diag_gmm_log_prob(x, log_pi, mu, var):
    dim = x.shape[-1]
    diff = x[:, None, :] - mu[None, :, :]
    quad = jnp.sum((diff * diff) / var[None, :, :], axis=-1)
    log_det = jnp.sum(jnp.log(var), axis=-1)
    normalizer = dim * math.log(2.0 * math.pi)
    return log_pi[None, :] - 0.5 * (normalizer + log_det[None, :] + quad)


@jax.jit
def posterior_from_stats(latents_flat, mean, std, eps, log_pi, mu, var):
    latents_std = standardize_latents(latents_flat, mean, std, eps)
    log_prob = diag_gmm_log_prob(latents_std, log_pi, mu, var)
    log_norm = jsp.special.logsumexp(log_prob, axis=-1, keepdims=True)
    return jnp.exp(log_prob - log_norm)


@jax.jit
def chunk_em_stats(latents_std, log_pi, mu, var):
    log_prob = diag_gmm_log_prob(latents_std, log_pi, mu, var)
    log_norm = jsp.special.logsumexp(log_prob, axis=-1, keepdims=True)
    responsibilities = jnp.exp(log_prob - log_norm)
    counts = jnp.sum(responsibilities, axis=0)
    sum_x = jnp.einsum('bk,bd->kd', responsibilities, latents_std)
    sum_x2 = jnp.einsum('bk,bd->kd', responsibilities, latents_std * latents_std)
    nll = -jnp.sum(log_norm)
    return counts, sum_x, sum_x2, nll


@jax.jit
def squared_distance_chunk(latents_std, center):
    diff = latents_std - center[None, :]
    return jnp.sum(diff * diff, axis=-1)


def _chunk_slices(num_examples, chunk_size):
    for start in range(0, num_examples, chunk_size):
        stop = min(start + chunk_size, num_examples)
        yield start, stop


def _choose_random_center(latents_std, rng):
    idx = int(rng.integers(0, latents_std.shape[0]))
    return np.asarray(latents_std[idx], dtype=np.float32)


def kmeanspp_init(latents_std, num_modes, seed, chunk_size):
    rng = np.random.default_rng(seed)
    centers = [_choose_random_center(latents_std, rng)]
    min_distances = np.full(latents_std.shape[0], np.inf, dtype=np.float64)

    for _ in range(1, num_modes):
        center = jnp.asarray(centers[-1], dtype=jnp.float32)
        for start, stop in _chunk_slices(latents_std.shape[0], chunk_size):
            latents_chunk = jnp.asarray(np.asarray(latents_std[start:stop]), dtype=jnp.float32)
            dist_chunk = np.asarray(jax.device_get(squared_distance_chunk(latents_chunk, center)))
            min_distances[start:stop] = np.minimum(min_distances[start:stop], dist_chunk)

        probs = min_distances / np.sum(min_distances)
        idx = int(rng.choice(latents_std.shape[0], p=probs))
        centers.append(np.asarray(latents_std[idx], dtype=np.float32))

    return np.stack(centers, axis=0)


def random_init(latents_std, num_modes, seed):
    rng = np.random.default_rng(seed)
    indices = rng.choice(latents_std.shape[0], size=num_modes, replace=False)
    return np.asarray(latents_std[indices], dtype=np.float32)


def initialize_gmm_params(
    latents_std,
    num_modes,
    seed,
    chunk_size,
    var_floor,
    use_kmeanspp,
):
    if use_kmeanspp:
        mu = kmeanspp_init(latents_std, num_modes, seed, chunk_size)
    else:
        mu = random_init(latents_std, num_modes, seed)

    global_var = np.var(np.asarray(latents_std), axis=0, dtype=np.float64)
    global_var = np.maximum(global_var.astype(np.float32), var_floor)
    var = np.repeat(global_var[None, :], num_modes, axis=0)
    pi = np.ones((num_modes,), dtype=np.float32) / num_modes
    return pi, mu.astype(np.float32), var.astype(np.float32)


def fit_diag_gmm(
    latents_std,
    num_modes,
    em_iters,
    restarts,
    seed,
    chunk_size,
    var_floor,
    weight_prior,
    use_kmeanspp,
):
    num_examples, latent_dim = latents_std.shape
    best_state = None
    best_nll = np.inf

    for restart_idx in range(restarts):
        pi, mu, var = initialize_gmm_params(
            latents_std=latents_std,
            num_modes=num_modes,
            seed=seed + restart_idx,
            chunk_size=chunk_size,
            var_floor=var_floor,
            use_kmeanspp=use_kmeanspp,
        )
        nll_trace = []
        counts_trace = []
        var_min_trace = []
        var_max_trace = []

        for _ in range(em_iters):
            counts = np.zeros((num_modes,), dtype=np.float64)
            sum_x = np.zeros((num_modes, latent_dim), dtype=np.float64)
            sum_x2 = np.zeros((num_modes, latent_dim), dtype=np.float64)
            total_nll = 0.0

            log_pi = jnp.asarray(np.log(np.maximum(pi, 1e-8)), dtype=jnp.float32)
            mu_device = jnp.asarray(mu, dtype=jnp.float32)
            var_device = jnp.asarray(var, dtype=jnp.float32)

            for start, stop in _chunk_slices(num_examples, chunk_size):
                latents_chunk = jnp.asarray(
                    np.asarray(latents_std[start:stop]),
                    dtype=jnp.float32,
                )
                counts_chunk, sum_x_chunk, sum_x2_chunk, nll_chunk = jax.device_get(
                    chunk_em_stats(latents_chunk, log_pi, mu_device, var_device)
                )
                counts += np.asarray(counts_chunk, dtype=np.float64)
                sum_x += np.asarray(sum_x_chunk, dtype=np.float64)
                sum_x2 += np.asarray(sum_x2_chunk, dtype=np.float64)
                total_nll += float(nll_chunk)

            safe_counts = np.maximum(counts, 1e-6)
            mu_new = (sum_x / safe_counts[:, None]).astype(np.float32)
            second_moment = (sum_x2 / safe_counts[:, None]).astype(np.float32)
            var_new = np.maximum(second_moment - mu_new * mu_new, var_floor).astype(np.float32)

            dead_mask = counts < 1.0
            if np.any(dead_mask):
                fallback = random_init(latents_std, int(np.sum(dead_mask)), seed + restart_idx + 123)
                mu_new[dead_mask] = fallback
                global_var = np.var(np.asarray(latents_std), axis=0, dtype=np.float64).astype(np.float32)
                var_new[dead_mask] = np.maximum(global_var, var_floor)

            counts_with_prior = counts + weight_prior
            pi_new = (counts_with_prior / np.sum(counts_with_prior)).astype(np.float32)

            pi = pi_new
            mu = mu_new
            var = var_new

            nll_trace.append(total_nll / num_examples)
            counts_trace.append(counts.astype(np.float32))
            var_min_trace.append(float(np.min(var)))
            var_max_trace.append(float(np.max(var)))

        final_nll = nll_trace[-1]
        if final_nll < best_nll:
            best_nll = final_nll
            best_state = {
                'pi': pi,
                'mu': mu,
                'var': var,
                'nll_trace': np.asarray(nll_trace, dtype=np.float32),
                'counts_trace': np.asarray(counts_trace, dtype=np.float32),
                'var_min_trace': np.asarray(var_min_trace, dtype=np.float32),
                'var_max_trace': np.asarray(var_max_trace, dtype=np.float32),
                'restart_index': restart_idx,
                'final_counts': counts.astype(np.float32),
            }

    return best_state


def save_gmm_stats(path, stats_dict):
    np.savez(path, **stats_dict)


def load_gmm_stats(path):
    raw = np.load(path, allow_pickle=False)
    stats = {key: raw[key] for key in raw.files}
    stats['mean'] = jnp.asarray(stats['mean'], dtype=jnp.float32)
    stats['std'] = jnp.asarray(stats['std'], dtype=jnp.float32)
    stats['pi'] = jnp.asarray(stats['pi'], dtype=jnp.float32)
    stats['mu'] = jnp.asarray(stats['mu'], dtype=jnp.float32)
    stats['var'] = jnp.asarray(stats['var'], dtype=jnp.float32)
    stats['log_pi'] = jnp.log(jnp.maximum(stats['pi'], 1e-8))
    return stats


def sample_categorical_onehot(key, probs, num_modes):
    indices = jax.random.categorical(key, jnp.log(jnp.maximum(probs, 1e-8)), axis=-1)
    one_hot = jax.nn.one_hot(indices, num_modes, dtype=jnp.float32)
    return indices, one_hot
