import os
import time
import jax
import numpy as np
import tqdm
import wandb
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from absl import logging
import jax.numpy as jnp

# Import các module nội bộ
from utils.datasets import get_dataset
from utils.stable_vae import StableVAE

def get_or_fit_gmm(FLAGS):
    """
    Hàm quản lý GMM Stats:
    - Nếu file chưa tồn tại: Process 0 chạy fit, log WandB và save. Các process khác đợi.
    - Nếu file tồn tại: Load và trả về dictionary các JAX Arrays.

    Returns:
        dict: {'means': [K, H, W, C], 'covs': [K, H, W, C],
               'weights': [K], 'empirical_probs': [K]}
    """
    gmm_path = FLAGS.gmm_path

    # --- 1. ĐỒNG BỘ HÓA ĐA PROCESS ---
    if not os.path.exists(gmm_path):
        if jax.process_index() == 0:
            logging.info(f"[GMM Manager] File {gmm_path} not found. Starting Preprocessing...")
            _run_gmm_fitting(FLAGS)
            logging.info(f"[GMM Manager] Preprocessing done. Saved to {gmm_path}.")
        else:
            logging.info(f"[GMM Manager] Process {jax.process_index()} waiting for GMM stats...")
            # Chờ tối đa 60 phút
            start_wait = time.time()
            while not os.path.exists(gmm_path):
                time.sleep(10)
                if time.time() - start_wait > 3600:
                    raise TimeoutError("Timed out waiting for GMM stats.")
            logging.info(f"[GMM Manager] File found! Proceeding...")

    # Barrier đơn giản để đảm bảo Process 0 ghi file xong hoàn toàn
    # Trigger small computation để sync
    _ = jax.random.normal(jax.random.PRNGKey(0), (1,))
    if jax.process_index() != 0:
        time.sleep(5)

    # --- 2. LOAD VÀ TRẢ VỀ DATA ---
    logging.info(f"[GMM Manager] Loading GMM stats from {gmm_path}...")
    loaded = np.load(gmm_path)

    stats = {
        'means': jnp.array(loaded['means']),
        'covs': jnp.array(loaded['covs']),
        'weights': jnp.array(loaded['weights']),
        'empirical_probs': jnp.array(loaded['empirical_probs'])
    }

    logging.info(f"[GMM Manager] Loaded GMM with {len(stats['weights'])} components")
    return stats


def _run_gmm_fitting(FLAGS):
    """Logic Fit GMM (Chỉ chạy trên Process 0)"""
    num_samples = FLAGS.gmm_fit_samples
    num_components = FLAGS.model['gmm_components']
    batch_size = 128

    print(f"--- Fitting GMM: K={num_components}, Samples={num_samples}, Dim=4096 ---")

    # 1. Init Data & VAE
    # Lưu ý: Tạo iterator mới để không ảnh hưởng iterator chính của train
    ds_iter = get_dataset(FLAGS.dataset_name, batch_size, is_train=True)
    vae = StableVAE.create()
    vae_encode = jax.jit(vae.encode)
    rng = jax.random.PRNGKey(0)

    # 2. Collect Latents
    latents_buffer = []
    total_collected = 0
    pbar = tqdm.tqdm(total=num_samples, desc="Collecting Latents")

    while total_collected < num_samples:
        batch_images, _ = next(ds_iter)
        # Handle shape if stable_vae needs scaling/encoding
        rng, key = jax.random.split(rng)

        # Encode -> [B, 32, 32, 4]
        batch_latents = vae_encode(key, batch_images)
        batch_latents_np = np.array(batch_latents)

        # Flatten -> [B, 4096]
        batch_flat = batch_latents_np.reshape(batch_latents_np.shape[0], -1)

        latents_buffer.append(batch_flat)
        total_collected += batch_flat.shape[0]
        pbar.update(batch_flat.shape[0])
    pbar.close()

    X = np.concatenate(latents_buffer, axis=0)[:num_samples]
    print(f"Data Collected Shape: {X.shape}")

    # 3. Fit GMM (CPU, Diagonal Covariance)
    print("Fitting GMM (CPU, diag)... this may take a few minutes...")
    gmm = GaussianMixture(
        n_components=num_components,
        covariance_type='diag',  # Bắt buộc dùng diag với dim=4096
        max_iter=100,
        verbose=1,
        random_state=42
    )
    gmm.fit(X)

    # 4. Calculate Empirical Stats for Importance Sampling
    print("Calculating Empirical Probabilities...")
    labels = gmm.predict(X)
    counts = np.bincount(labels, minlength=num_components)
    empirical_probs = counts / np.sum(counts)
    model_weights = gmm.weights_

    # 5. Log to WandB (Nếu đang active trong train.py)
    if wandb.run is not None:
        # Table
        table_data = []
        for k in range(num_components):
            # IS Weight = Pi / P_emp
            is_weight = model_weights[k] / (empirical_probs[k] + 1e-8)
            table_data.append([
                k,
                float(model_weights[k]),
                float(empirical_probs[k]),
                int(counts[k]),
                float(is_weight)
            ])

        wandb.log({"gmm_preprocessing/stats_table": wandb.Table(
            data=table_data,
            columns=["Cluster", "Model_Pi", "Empirical_Prob", "Count", "Est_IS_Weight"]
        )})

        # Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        idx = np.arange(num_components)
        width = 0.35
        ax.bar(idx - width/2, model_weights, width, label='Model Pi')
        ax.bar(idx + width/2, empirical_probs, width, label='Empirical Prob')
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Probability')
        ax.set_title('GMM Weights vs Data Distribution (for Importance Sampling)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        wandb.log({"gmm_preprocessing/distribution": wandb.Image(fig)})
        plt.close(fig)

    # 6. Save .npz
    H, W, C = 32, 32, 4
    # Reshape means/covs về không gian ảnh để tiện dùng
    np.savez(
        FLAGS.gmm_path,
        means=gmm.means_.reshape(num_components, H, W, C),
        covs=gmm.covariances_.reshape(num_components, H, W, C),
        weights=model_weights,
        empirical_probs=empirical_probs
    )

    print(f"Saved GMM stats to {FLAGS.gmm_path}")

    # Cleanup memory
    del X, latents_buffer, gmm
    import gc
    gc.collect()
