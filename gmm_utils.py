import math

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np


def flatten_latents(latents):
    return latents.reshape((latents.shape[0], -1))


def standardize_latents(latents_flat, mean, std, eps):
    return (latents_flat - mean) / (std + eps)


def unstandardize_latents(latents_std, mean, std, eps):
    return latents_std * (std + eps) + mean


def flatten_and_standardize(latents, mean, std, eps):
    return standardize_latents(flatten_latents(latents), mean, std, eps)


@jax.jit
def diag_gmm_log_prob(x_std, log_pi, mu, var):
    dim = x_std.shape[-1]
    diff = x_std[:, None, :] - mu[None, :, :]
    quad = jnp.sum((diff * diff) / var[None, :, :], axis=-1)
    log_det = jnp.sum(jnp.log(var), axis=-1)
    normalizer = dim * math.log(2.0 * math.pi)
    return log_pi[None, :] - 0.5 * (normalizer + log_det[None, :] + quad)


@jax.jit
def posterior_from_standardized(x_std, log_pi, mu, var):
    log_prob = diag_gmm_log_prob(x_std, log_pi, mu, var)
    log_norm = jsp.special.logsumexp(log_prob, axis=-1, keepdims=True)
    return jnp.exp(log_prob - log_norm)


@jax.jit
def posterior_from_stats(latents_flat, mean, std, eps, log_pi, mu, var):
    latents_std = standardize_latents(latents_flat, mean, std, eps)
    return posterior_from_standardized(latents_std, log_pi, mu, var)


@jax.jit
def chunk_em_stats(latents_std, log_pi, mu, var):
    log_prob = diag_gmm_log_prob(latents_std, log_pi, mu, var)
    log_norm = jsp.special.logsumexp(log_prob, axis=-1, keepdims=True)
    responsibilities = jnp.exp(log_prob - log_norm)
    counts = jnp.sum(responsibilities, axis=0)
    sum_x = jnp.einsum('bk,bd->kd', responsibilities, latents_std)
    sum_x2 = jnp.einsum('bk,bd->kd', responsibilities, latents_std * latents_std)
    nll = -jnp.sum(log_norm)
    entropy = -jnp.sum(
        responsibilities * jnp.log(jnp.maximum(responsibilities, 1e-8)))
    hard_counts = jnp.bincount(
        jnp.argmax(responsibilities, axis=-1), length=mu.shape[0])
    return counts, sum_x, sum_x2, nll, entropy, hard_counts


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
            latents_chunk = jnp.asarray(
                np.asarray(latents_std[start:stop]), dtype=jnp.float32)
            dist_chunk = np.asarray(jax.device_get(
                squared_distance_chunk(latents_chunk, center)))
            min_distances[start:stop] = np.minimum(
                min_distances[start:stop], dist_chunk)

        probs = min_distances / np.sum(min_distances)
        idx = int(rng.choice(latents_std.shape[0], p=probs))
        centers.append(np.asarray(latents_std[idx], dtype=np.float32))

    return np.stack(centers, axis=0)


def random_init(latents_std, num_modes, seed):
    rng = np.random.default_rng(seed)
    replace = latents_std.shape[0] < num_modes
    indices = rng.choice(latents_std.shape[0], size=num_modes, replace=replace)
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


def _pi_kl_uniform_to_pi(pi):
    num_modes = pi.shape[0]
    uniform = np.ones_like(pi, dtype=np.float32) / num_modes
    return float(np.sum(uniform * (
        np.log(np.maximum(uniform, 1e-8)) - np.log(np.maximum(pi, 1e-8)))))


def fit_diag_gmm(
    latents_std,
    num_modes,
    em_iters,
    restarts,
    seed,
    chunk_size,
    var_floor,
    use_kmeanspp,
    pi_uniform_beta=0.01,
    var_update='scale',
    var_beta=0.05,
    var_target=1.0,
    var_eps=1e-8,
    var_scale_min=0.5,
    var_scale_max=2.0,
    min_component_count=1.0,
):
    num_examples, latent_dim = latents_std.shape
    best_state = None
    best_objective = -np.inf
    uniform_pi = np.ones((num_modes,), dtype=np.float32) / num_modes
    pi_uniform_beta = float(pi_uniform_beta)
    var_beta = float(var_beta)
    var_target = float(var_target)
    var_eps = float(var_eps)
    var_scale_min = float(var_scale_min)
    var_scale_max = float(var_scale_max)
    min_component_count = float(min_component_count)

    if not 0.0 <= pi_uniform_beta <= 1.0:
        raise ValueError("pi_uniform_beta must be in [0, 1].")
    if var_update not in ('none', 'scale'):
        raise ValueError("var_update must be 'none' or 'scale'.")
    if var_beta < 0:
        raise ValueError("var_beta must be non-negative.")
    if var_target <= 0:
        raise ValueError("var_target must be positive.")
    if var_eps <= 0:
        raise ValueError("var_eps must be positive.")
    if var_scale_min <= 0 or var_scale_max <= 0:
        raise ValueError("var scale bounds must be positive.")
    if var_scale_min > var_scale_max:
        raise ValueError("var_scale_min must be <= var_scale_max.")
    if min_component_count < 0:
        raise ValueError("min_component_count must be non-negative.")

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
        objective_trace = []
        counts_trace = []
        hard_counts_trace = []
        var_min_trace = []
        var_max_trace = []
        floor_frac_trace = []
        dead_count_trace = []
        min_count_trace = []
        max_count_trace = []
        pi_kl_u_to_pi_trace = []
        pi_entropy_trace = []
        effective_components_trace = []
        posterior_entropy_trace = []
        var_target_mse_trace = []
        var_scale_min_trace = []
        var_scale_max_trace = []

        for _ in range(em_iters):
            counts = np.zeros((num_modes,), dtype=np.float64)
            hard_counts = np.zeros((num_modes,), dtype=np.float64)
            sum_x = np.zeros((num_modes, latent_dim), dtype=np.float64)
            sum_x2 = np.zeros((num_modes, latent_dim), dtype=np.float64)
            total_nll = 0.0
            total_posterior_entropy = 0.0

            log_pi = jnp.asarray(np.log(np.maximum(pi, 1e-8)), dtype=jnp.float32)
            mu_device = jnp.asarray(mu, dtype=jnp.float32)
            var_device = jnp.asarray(var, dtype=jnp.float32)

            for start, stop in _chunk_slices(num_examples, chunk_size):
                latents_chunk = jnp.asarray(
                    np.asarray(latents_std[start:stop]), dtype=jnp.float32)
                stats = jax.device_get(chunk_em_stats(
                    latents_chunk, log_pi, mu_device, var_device))
                counts_chunk, sum_x_chunk, sum_x2_chunk, nll_chunk, entropy_chunk, hard_chunk = stats
                counts += np.asarray(counts_chunk, dtype=np.float64)
                sum_x += np.asarray(sum_x_chunk, dtype=np.float64)
                sum_x2 += np.asarray(sum_x2_chunk, dtype=np.float64)
                hard_counts += np.asarray(hard_chunk, dtype=np.float64)
                total_nll += float(nll_chunk)
                total_posterior_entropy += float(entropy_chunk)

            safe_counts = np.maximum(counts, 1e-6)
            mu_new = (sum_x / safe_counts[:, None]).astype(np.float32)
            second_moment = (sum_x2 / safe_counts[:, None]).astype(np.float32)
            var_em = np.maximum(second_moment - mu_new * mu_new, 0.0).astype(np.float32)

            if var_update == 'scale' and var_beta > 0:
                mean_var = np.mean(var_em, axis=1)
                var_scale = (var_target / (mean_var + var_eps)) ** var_beta
                var_scale = np.clip(var_scale, var_scale_min, var_scale_max)
                var_new = var_em * var_scale[:, None]
            else:
                var_scale = np.ones((num_modes,), dtype=np.float32)
                var_new = var_em
            var_new = np.maximum(var_new, var_floor).astype(np.float32)

            pi_em = (counts / max(float(num_examples), 1.0)).astype(np.float32)
            pi_new = (
                (1.0 - pi_uniform_beta) * pi_em
                + pi_uniform_beta * uniform_pi
            ).astype(np.float32)
            pi_new = pi_new / np.sum(pi_new)

            pi = pi_new
            mu = mu_new
            var = var_new

            nll = total_nll / num_examples
            pi_kl = _pi_kl_uniform_to_pi(pi)
            pi_entropy = float(-np.sum(pi * np.log(np.maximum(pi, 1e-8))))
            var_target_mse = float(np.mean(
                (np.mean(var, axis=1) - var_target) ** 2))
            objective = -nll - pi_kl - var_target_mse

            nll_trace.append(nll)
            objective_trace.append(objective)
            counts_trace.append(counts.astype(np.float32))
            hard_counts_trace.append(hard_counts.astype(np.float32))
            var_min_trace.append(float(np.min(var)))
            var_max_trace.append(float(np.max(var)))
            floor_frac_trace.append(float(np.mean(var <= var_floor * (1.0 + 1e-6))))
            dead_count_trace.append(float(np.sum(counts < min_component_count)))
            min_count_trace.append(float(np.min(counts)))
            max_count_trace.append(float(np.max(counts)))
            pi_kl_u_to_pi_trace.append(pi_kl)
            pi_entropy_trace.append(pi_entropy)
            effective_components_trace.append(float(np.exp(pi_entropy)))
            posterior_entropy_trace.append(total_posterior_entropy / num_examples)
            var_target_mse_trace.append(var_target_mse)
            var_scale_min_trace.append(float(np.min(var_scale)))
            var_scale_max_trace.append(float(np.max(var_scale)))

        final_objective = objective_trace[-1]
        if final_objective > best_objective:
            best_objective = final_objective
            best_state = {
                'pi': pi,
                'mu': mu,
                'var': var,
                'nll_trace': np.asarray(nll_trace, dtype=np.float32),
                'objective_trace': np.asarray(objective_trace, dtype=np.float32),
                'counts_trace': np.asarray(counts_trace, dtype=np.float32),
                'hard_counts_trace': np.asarray(hard_counts_trace, dtype=np.float32),
                'var_min_trace': np.asarray(var_min_trace, dtype=np.float32),
                'var_max_trace': np.asarray(var_max_trace, dtype=np.float32),
                'floor_frac_trace': np.asarray(floor_frac_trace, dtype=np.float32),
                'dead_count_trace': np.asarray(dead_count_trace, dtype=np.float32),
                'min_count_trace': np.asarray(min_count_trace, dtype=np.float32),
                'max_count_trace': np.asarray(max_count_trace, dtype=np.float32),
                'pi_kl_u_to_pi_trace': np.asarray(pi_kl_u_to_pi_trace, dtype=np.float32),
                'pi_entropy_trace': np.asarray(pi_entropy_trace, dtype=np.float32),
                'effective_components_trace': np.asarray(
                    effective_components_trace, dtype=np.float32),
                'posterior_entropy_trace': np.asarray(
                    posterior_entropy_trace, dtype=np.float32),
                'var_target_mse_trace': np.asarray(var_target_mse_trace, dtype=np.float32),
                'var_scale_min_trace': np.asarray(var_scale_min_trace, dtype=np.float32),
                'var_scale_max_trace': np.asarray(var_scale_max_trace, dtype=np.float32),
                'restart_index': restart_idx,
                'final_counts': counts.astype(np.float32),
                'final_hard_counts': hard_counts.astype(np.float32),
            }

    return best_state


def local_coordinates_from_standardized(x_std, modes, mu, var, eta, eps=1e-8):
    mode_mu = mu[modes]
    mode_var = jnp.maximum(var[modes], eps)
    return (x_std - mode_mu) * (mode_var ** (-0.5 * eta))


def directions_from_local(local_coords, eps=1e-8):
    norm = jnp.linalg.norm(local_coords, axis=-1, keepdims=True)
    return local_coords / (norm + eps)


def assign_angular_codes(x_std, modes, mu, var, angular_centers, eta, eps=1e-8):
    local_coords = local_coordinates_from_standardized(
        x_std, modes, mu, var, eta, eps=eps)
    directions = directions_from_local(local_coords, eps=eps)
    centers = angular_centers[modes]
    scores = jnp.einsum('bd,bad->ba', directions, centers)
    return jnp.argmax(scores, axis=-1), directions


def build_source_base_flat(
    modes,
    angular_codes,
    radius,
    mean,
    std,
    standardize_eps,
    mu,
    var,
    angular_centers,
    eta,
    eps=1e-8,
):
    mode_mu = mu[modes]
    mode_var = jnp.maximum(var[modes], eps)
    centers = angular_centers[modes, angular_codes]
    x_base_std = mode_mu + (mode_var ** (0.5 * eta)) * radius[:, None] * centers
    return unstandardize_latents(x_base_std, mean, std, standardize_eps), x_base_std


def build_source_base(
    modes,
    angular_codes,
    radius,
    latent_shape,
    mean,
    std,
    standardize_eps,
    mu,
    var,
    angular_centers,
    eta,
    eps=1e-8,
):
    x_base_flat, x_base_std = build_source_base_flat(
        modes,
        angular_codes,
        radius,
        mean,
        std,
        standardize_eps,
        mu,
        var,
        angular_centers,
        eta,
        eps=eps,
    )
    return x_base_flat.reshape((modes.shape[0],) + tuple(latent_shape)), x_base_std


def radius_to_rho(radius, modes, angular_codes, radius_log_mean, radius_log_std, eps=1e-8):
    log_radius = jnp.log(jnp.maximum(radius, eps))
    log_mean = radius_log_mean[modes, angular_codes]
    log_std = jnp.maximum(radius_log_std[modes, angular_codes], eps)
    return (log_radius - log_mean) / log_std, log_radius


def rho_to_radius(rho, modes, angular_codes, radius_log_mean, radius_log_std):
    log_mean = radius_log_mean[modes, angular_codes]
    log_std = radius_log_std[modes, angular_codes]
    log_radius = log_mean + log_std * rho
    return jnp.exp(log_radius), log_radius


def make_moe_condition(modes, angular_codes, rho):
    return jnp.stack(
        [
            modes.astype(jnp.float32),
            angular_codes.astype(jnp.float32),
            rho.astype(jnp.float32),
        ],
        axis=-1,
    )


def sample_local_direction(key, centers, direction_noise, eps=1e-8):
    dim = centers.shape[-1]
    noise = jax.random.normal(key, centers.shape)
    tangent = noise - jnp.sum(noise * centers, axis=-1, keepdims=True) * centers
    tangent = tangent / jnp.sqrt(jnp.asarray(dim, dtype=tangent.dtype))
    direction = centers + direction_noise * tangent
    return directions_from_local(direction, eps=eps)


def build_conditional_source(
    key,
    modes,
    angular_codes,
    rho,
    latent_shape,
    mean,
    std,
    standardize_eps,
    mu,
    var,
    angular_centers,
    radius_log_mean,
    radius_log_std,
    eta,
    kappa,
    direction_noise,
    eps=1e-8,
):
    radius, log_radius = rho_to_radius(
        rho, modes, angular_codes, radius_log_mean, radius_log_std)
    mode_mu = mu[modes]
    mode_var = jnp.maximum(var[modes], eps)
    centers = angular_centers[modes, angular_codes]
    source_dirs = sample_local_direction(
        key, centers, direction_noise=direction_noise, eps=eps)
    local_source = kappa * radius[:, None] * source_dirs
    x0_std = mode_mu + (mode_var ** (0.5 * eta)) * local_source
    x0_flat = unstandardize_latents(x0_std, mean, std, standardize_eps)
    x0 = x0_flat.reshape((modes.shape[0],) + tuple(latent_shape))
    return x0, x0_std, source_dirs, radius, log_radius


def sample_lognormal_radius(key, modes, angular_codes, radius_log_mean, radius_log_std):
    log_mean = radius_log_mean[modes, angular_codes]
    log_std = radius_log_std[modes, angular_codes]
    log_radius = log_mean + log_std * jax.random.normal(key, log_mean.shape)
    return jnp.exp(log_radius), log_radius


def temperature_smooth_probs(probs, temperature, eps=1e-8):
    if temperature == 1.0:
        return probs
    smoothed = jnp.maximum(probs, eps) ** (1.0 / temperature)
    return smoothed / jnp.sum(smoothed, axis=-1, keepdims=True)


def categorical_kl(target_probs, pred_probs, eps=1e-8):
    target = jnp.maximum(target_probs, eps)
    pred = jnp.maximum(pred_probs, eps)
    return jnp.sum(target * (jnp.log(target) - jnp.log(pred)), axis=-1)


def categorical_entropy(probs, eps=1e-8):
    probs = jnp.maximum(probs, eps)
    return -jnp.sum(probs * jnp.log(probs), axis=-1)


def save_gmm_stats(path, stats_dict):
    np.savez(path, **stats_dict)


def load_gmm_stats(path):
    raw = np.load(path, allow_pickle=False)
    stats = {key: raw[key] for key in raw.files}
    tensor_keys = [
        'mean',
        'std',
        'pi',
        'mu',
        'var',
        'angular_centers',
        'angular_pi',
        'angular_counts',
        'angular_active',
        'radius_log_mean',
        'radius_log_std',
        'radius_counts',
        'radius_backoff',
        'latent_shape',
        'n_train',
        'standardize_eps',
        'local_eta',
    ]
    for key in tensor_keys:
        if key in stats:
            dtype = jnp.float32
            if key == 'latent_shape':
                dtype = jnp.int32
            stats[key] = jnp.asarray(stats[key], dtype=dtype)
    stats['log_pi'] = jnp.log(jnp.maximum(stats['pi'], 1e-8))
    return stats


def sample_categorical_onehot(key, probs, num_modes):
    indices = jax.random.categorical(
        key, jnp.log(jnp.maximum(probs, 1e-8)), axis=-1)
    one_hot = jax.nn.one_hot(indices, num_modes, dtype=jnp.float32)
    return indices, one_hot
