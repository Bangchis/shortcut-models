import jax
import jax.numpy as jnp
import numpy as np

def get_targets(FLAGS, key, train_state, images, labels, force_t=-1, force_dt=-1):
    """
    Bang-FM: Integer-Grid Shortcut Training.

    Grid resolution M = 128.
    Atomic unit delta = 1/128.
    Step size d = K/128 where K in [1, 128].
    dt_base = log2(128/K) (float value for timestep embedding).

    Training composition:
    - 75% Flow Matching (K=1, standard FM targets)
    - 25% Bootstrap (K>=2, weighted average of random split)
    """
    key, t_key, k_key, j_key, noise_key, dropout_key = jax.random.split(key, 6)

    batch_size = images.shape[0]
    M = FLAGS.model.get('bfm_grid_resolution', 128)  # Grid resolution (default 128)
    K_min = FLAGS.model.get('bfm_k_min', 2)  # Min K for bootstrap (default 2)
    K_max = FLAGS.model.get('bfm_k_max', 128)  # Max K for bootstrap (default 128)

    info = {}

    # ==========================================
    # BATCH SPLITTING
    # ==========================================
    num_bootstrap = batch_size // FLAGS.model['bootstrap_every']
    num_flow = batch_size - num_bootstrap

    # ==========================================
    # PART A: BOOTSTRAP (Recursive Case K >= 2)
    # ==========================================
    if num_bootstrap > 0:
        # 1. Sample K uniformly from [K_min, K_max]
        if force_dt != -1:
            # For evaluation sweeps: interpret force_dt as K value
            K_bst = jnp.ones(num_bootstrap, dtype=jnp.int32) * force_dt
        else:
            K_bst = jax.random.randint(k_key, (num_bootstrap,), minval=K_min, maxval=K_max + 1)

        # dt_base = log2(128/K) - FLOAT value
        dt_base_bst = jnp.log2(M / K_bst.astype(jnp.float32))

        # 2. Sample t such that t + K/128 <= 1
        # t_idx max allowed is M - K
        max_t_idx = M - K_bst
        rand_t = jax.random.uniform(t_key, (num_bootstrap,))
        t_idx_bst = jnp.floor(rand_t * (max_t_idx + 1)).astype(jnp.int32)
        t_bst = t_idx_bst.astype(jnp.float32) / M

        if force_t != -1:
            t_bst = jnp.ones_like(t_bst) * force_t

        t_bst_expanded = t_bst[:, None, None, None]

        # 3. Prepare data
        x1_bst = images[:num_bootstrap]
        x0_bst = jax.random.normal(noise_key, x1_bst.shape)
        bst_labels = labels[:num_bootstrap]

        # Create x_t (Input state)
        x_t_bst = (1 - (1 - 1e-5) * t_bst_expanded) * x0_bst + t_bst_expanded * x1_bst

        # 4. Random Split: Sample j from [1, K-1]
        rand_j = jax.random.uniform(j_key, (num_bootstrap,))
        j_val = jnp.floor(rand_j * (K_bst - 1)).astype(jnp.int32) + 1

        # Step sizes (float for computation)
        d1 = j_val.astype(jnp.float32) / M
        d2 = (K_bst - j_val).astype(jnp.float32) / M
        dt_total = K_bst.astype(jnp.float32) / M

        # dt_base for each sub-step (float log2 values)
        dt_base_step1 = jnp.log2(M / j_val.astype(jnp.float32))
        dt_base_step2 = jnp.log2(M / (K_bst - j_val).astype(jnp.float32))

        # 5. Two-step forward with EMA model
        call_model_fn = train_state.call_model_ema if FLAGS.model['use_ema'] else train_state.call_model

        # Step 1: x_t -> x_mid via step size d1
        v1 = call_model_fn(x_t_bst, t_bst, dt_base_step1, bst_labels, train=False)
        x_mid = x_t_bst + d1[:, None, None, None] * v1
        x_mid = jnp.clip(x_mid, -4, 4)  # Stability clip
        t_mid = t_bst + d1

        # Step 2: x_mid -> x_next via step size d2
        v2 = call_model_fn(x_mid, t_mid, dt_base_step2, bst_labels, train=False)

        # 6. Weighted Average for Target
        # v_target = (v1 * d1 + v2 * d2) / dt_total
        w1 = d1 / dt_total
        w2 = d2 / dt_total
        v_target_bst = v1 * w1[:, None, None, None] + v2 * w2[:, None, None, None]
        v_target_bst = jax.lax.stop_gradient(v_target_bst)
        v_target_bst = jnp.clip(v_target_bst, -4, 4)

        # Store bootstrap info
        info['v_magnitude_bootstrap'] = jnp.sqrt(jnp.mean(jnp.square(v_target_bst)))
        info['v_magnitude_b1'] = jnp.sqrt(jnp.mean(jnp.square(v1)))
        info['v_magnitude_b2'] = jnp.sqrt(jnp.mean(jnp.square(v2)))
        info['avg_K'] = jnp.mean(K_bst.astype(jnp.float32))
    else:
        # Fallback if no bootstrap (shouldn't happen normally)
        x_t_bst, v_target_bst, t_bst, dt_base_bst, bst_labels = (
            jnp.zeros((0,) + images.shape[1:]),
            jnp.zeros((0,) + images.shape[1:]),
            jnp.zeros(0),
            jnp.zeros(0),
            jnp.zeros(0, dtype=jnp.int32)
        )

    # ==========================================
    # PART B: FLOW MATCHING (Base Case K=1)
    # ==========================================
    # Target: Ground Truth vector (x1 - x0)
    # Step size: 1/128
    # dt_base = log2(128/1) = log2(128) = 7.0

    x1_flow = images[num_bootstrap:]
    x0_flow = jax.random.normal(jax.random.fold_in(noise_key, 1), x1_flow.shape)
    labels_flow = labels[num_bootstrap:]

    # Sample t on the grid: 0, 1/128, ..., 127/128
    t_idx_flow = jax.random.randint(jax.random.fold_in(t_key, 1), (num_flow,), minval=0, maxval=M)
    t_flow = t_idx_flow.astype(jnp.float32) / M

    # Handle force_t logic for visualization/debugging
    if force_t != -1:
        t_flow = jnp.ones_like(t_flow) * force_t

    t_flow_expanded = t_flow[:, None, None, None]

    # Flow Matching Interpolation
    x_t_flow = (1 - (1 - 1e-5) * t_flow_expanded) * x0_flow + t_flow_expanded * x1_flow
    v_target_flow = x1_flow - (1 - 1e-5) * x0_flow

    # dt_base for flow is log2(128) = 7.0
    dt_base_flow = jnp.ones(num_flow, dtype=jnp.float32) * jnp.log2(M)

    # ==========================================
    # MERGE [BST + FLOW]
    # ==========================================
    # Order matches targets_shortcut.py: [bootstrap, flow]

    if num_bootstrap > 0:
        x_t = jnp.concatenate([x_t_bst, x_t_flow], axis=0)
        v_t = jnp.concatenate([v_target_bst, v_target_flow], axis=0)
        t = jnp.concatenate([t_bst, t_flow], axis=0)
        dt_base = jnp.concatenate([dt_base_bst, dt_base_flow], axis=0)
        labels_final = jnp.concatenate([bst_labels, labels_flow], axis=0)
    else:
        x_t, v_t, t, dt_base, labels_final = (
            x_t_flow, v_target_flow, t_flow, dt_base_flow, labels_flow
        )

    # ==========================================
    # CLASS DROPOUT FOR CFG
    # ==========================================
    dropout_mask = jax.random.bernoulli(dropout_key, FLAGS.model['class_dropout_prob'], (labels_final.shape[0],))
    labels_dropped = jnp.where(dropout_mask, FLAGS.model['num_classes'], labels_final)

    # Info logging
    info['v_mag_target'] = jnp.mean(jnp.square(v_t))
    info['dropped_ratio'] = jnp.mean(labels_dropped == FLAGS.model['num_classes'])
    if num_bootstrap > 0:
        info['bootstrap_ratio'] = num_bootstrap / batch_size

    return x_t, v_t, t, dt_base, labels_dropped, info
