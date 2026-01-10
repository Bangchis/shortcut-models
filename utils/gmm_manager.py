import os
# --- FIX CRASH: Set threading to 1 before importing numpy/sklearn ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import jax
import numpy as np
import tqdm
import wandb
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from absl import logging
import jax.numpy as jnp
import warnings

# Import các module nội bộ
from utils.datasets import get_dataset
from utils.stable_vae import StableVAE

def get_or_fit_gmm(FLAGS):
    """
    Hàm quản lý GMM Stats:
    - Nếu file chưa tồn tại: Process 0 chạy fit, log WandB và save. Các process khác đợi.
    - Nếu file tồn tại: Load và trả về dictionary các JAX Arrays.
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
            start_wait = time.time()
            while not os.path.exists(gmm_path):
                time.sleep(10) 
                if time.time() - start_wait > 3600:
                    raise TimeoutError("Timed out waiting for GMM stats.")
            logging.info(f"[GMM Manager] File found! Proceeding...")
    
    # Barrier đơn giản
    jax.random.normal(jax.random.PRNGKey(0), (1,))
    if jax.process_index() != 0:
        time.sleep(5)

    # --- 2. LOAD VÀ TRẢ VỀ DATA ---
    logging.info(f"[GMM Manager] Loading GMM stats from {gmm_path}...")
    loaded = np.load(gmm_path)

    stats = {
        'means': jnp.array(loaded['means']),
        'covs': jnp.array(loaded['covs']),
        'weights': jnp.array(loaded['weights']),
        'empirical_probs': jnp.array(loaded['empirical_probs']),
        # PCA parameters for inverse projection
        'pca_components': jnp.array(loaded['pca_components']),
        'pca_mean': jnp.array(loaded['pca_mean'])
    }
    return stats

def _run_gmm_fitting(FLAGS):
    """Logic Fit PCA -> Transform -> Fit GMM (Chỉ chạy trên Process 0)"""
    num_samples = FLAGS.gmm_fit_samples
    num_components = FLAGS.model['gmm_components']
    pca_dim = FLAGS.gmm_pca_dim
    batch_size = 128

    print(f"--- Fitting GMM: K={num_components}, Samples={num_samples}, PCA Dim={pca_dim} ---")
    
    # 1. Init Data & VAE
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
        rng, key = jax.random.split(rng)
        
        batch_latents = vae_encode(key, batch_images) 
        batch_latents_np = np.array(batch_latents)
        
        batch_flat = batch_latents_np.reshape(batch_latents_np.shape[0], -1)
        
        latents_buffer.append(batch_flat)
        total_collected += batch_flat.shape[0]
        pbar.update(batch_flat.shape[0])
    pbar.close()
    
    X = np.concatenate(latents_buffer, axis=0)[:num_samples]  # [N, 4096]
    print(f"Data Collected Shape: {X.shape}")

    # 3. Fit PCA
    print(f"Fitting PCA (dim={pca_dim}, whiten=False)...")
    pca = PCA(n_components=pca_dim, whiten=False, random_state=42)
    X_pca = pca.fit_transform(X)  # [N, pca_dim]
    print(f"PCA Variance Explained: {np.sum(pca.explained_variance_ratio_):.4f}")

    # 4. Fit GMM on PCA latent space
    print("Fitting GMM on PCA latent space (CPU, diag)...")

    # Use kmeans for initialization since PCA data is cleaner and more stable
    gmm = GaussianMixture(
        n_components=num_components,
        covariance_type='diag',
        init_params='kmeans',  # PCA data is stable, kmeans works well
        reg_covar=1e-5,        # Smaller reg_covar since PCA removes noise
        max_iter=100,
        verbose=1,
        random_state=42
    )
    gmm.fit(X_pca)
    
    # 5. Calculate Empirical Stats for Importance Sampling
    print("Calculating Empirical Probabilities...")
    labels = gmm.predict(X_pca)
    counts = np.bincount(labels, minlength=num_components)
    empirical_probs = counts / np.sum(counts)
    model_weights = gmm.weights_

    # 6. Log to WandB
    if wandb.run is not None:
        table_data = []
        for k in range(num_components):
            is_weight = model_weights[k] / (empirical_probs[k] + 1e-8)
            table_data.append([k, model_weights[k], empirical_probs[k], counts[k], is_weight])

        wandb.log({"gmm_preprocessing/stats_table": wandb.Table(
            data=table_data, columns=["Cluster", "Model_Pi", "Empirical_Prob", "Count", "Est_IS_Weight"]
        )})

    # 7. Save .npz (GMM params in PCA space + PCA transform matrices)
    np.savez(
        FLAGS.gmm_path,
        means=gmm.means_,               # [K, pca_dim]
        covs=gmm.covariances_,          # [K, pca_dim]
        weights=model_weights,          # [K]
        empirical_probs=empirical_probs,# [K]
        pca_components=pca.components_, # [pca_dim, 4096]
        pca_mean=pca.mean_              # [4096]
    )
    
    print(f"Saved GMM+PCA stats to {FLAGS.gmm_path}")

    # Cleanup memory
    del X, X_pca, latents_buffer, gmm, pca
    import gc
    gc.collect()
