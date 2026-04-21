import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import wandb
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from gmm_utils import flatten_latents, load_gmm_stats, posterior_from_stats
from utils.stable_vae import StableVAE


DEFAULT_PROJECT_BASENAME = 'moe1-ablation-celeba256'
PHASE1_GROUP = 'Phase_1A_GMM'
PHASE1B_GROUP = 'Phase_1B_Screen'
PHASE2_GROUP = 'Phase_2_Sweep'
PHASE2_COMPOSED_GROUP = 'Phase_2_Composed'
PHASE3_GROUP = 'Phase_3_Final'
SUMMARY_GROUP = 'Ablation_Summary'
ROOT_DIR = Path(__file__).resolve().parent
PHASE1_MIN_SURVIVORS = 3
PHASE1_SCREEN_TOPK = 5
PHASE1_GMM_K_LIST = [8, 16, 24, 32]
PHASE2_MAX_STEPS = 5000
ABLATION_FID_TIMESTEPS = 32
ABLATION_FID_GENERATIONS = 1024
FID_METRIC_KEY = f'fid{ABLATION_FID_TIMESTEPS}_{ABLATION_FID_GENERATIONS}'
LATENCY_METRIC_KEY = f'latency_{ABLATION_FID_TIMESTEPS}'
THROUGHPUT_METRIC_KEY = f'throughput_{ABLATION_FID_TIMESTEPS}'
FLOW_VIZ_SAMPLES = 512
PATH_PLOT_MAX_LINES = 256


PHASE2_CONFIGS = [
    ('M00_Default', 'default', {}),
    ('M01_Bal_0.0', 'balance', {'loss_balance_weight': 0.0}),
    ('M02_Bal_0.4', 'balance', {'loss_balance_weight': 0.4}),
    ('M03_Bal_0.6', 'balance', {'loss_balance_weight': 0.6}),
    ('M04_Bal_0.8', 'balance', {'loss_balance_weight': 0.8}),
    ('M05_Ent_0.05', 'entropy', {'loss_entropy_weight': 0.05}),
    ('M06_Ent_0.2', 'entropy', {'loss_entropy_weight': 0.2}),
    ('M07_Ent_0.5', 'entropy', {'loss_entropy_weight': 0.5}),
    ('M08_Ent_0.8', 'entropy', {'loss_entropy_weight': 0.8}),
    ('M09_Ent_1.0', 'entropy', {'loss_entropy_weight': 1.0}),
    ('M10_Var_0.0', 'variance', {'source_var_weight': 0.0}),
    ('M11_Var_0.2', 'variance', {'source_var_weight': 0.2}),
    ('M12_Var_0.5', 'variance', {'source_var_weight': 0.5}),
    ('M13_Var_0.8', 'variance', {'source_var_weight': 0.8}),
    ('M14_Target_0.4', 'variance', {'source_var_target_std': 0.4}),
    ('M15_Target_0.6', 'variance', {'source_var_target_std': 0.6}),
    ('M16_Target_0.8', 'variance', {'source_var_target_std': 0.8}),
]


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


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_json_ready(payload), f, indent=2, sort_keys=True)


def _load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _run_subprocess(cmd, label):
    print(f'[{label}] running:\n  ' + ' '.join(cmd))
    subprocess.run(cmd, cwd=str(ROOT_DIR), check=True)


def _sanitize_flag_value(value):
    if isinstance(value, bool):
        return '1' if value else '0'
    return str(value)


def _metric_value(row, key, default=None):
    value = row.get(key)
    if value is not None:
        return value
    for section_name in ('inference', 'train'):
        section = row.get(section_name, {})
        if isinstance(section, dict) and section.get(key) is not None:
            return section[key]
    return default


def _fid_value(row, default=None):
    return _metric_value(row, FID_METRIC_KEY, default)


def _extend_config_flags(args, prefix, values):
    for key, value in values.items():
        if value is None:
            continue
        args.append(f'--{prefix}.{key}={_sanitize_flag_value(value)}')


def _sanitize_project_name(text):
    text = str(text).strip().replace('/', '-').replace('_', '-').replace(' ', '-')
    allowed = ''.join(ch if (ch.isalnum() or ch == '-') else '-' for ch in text.lower())
    while '--' in allowed:
        allowed = allowed.replace('--', '-')
    return allowed.strip('-') or DEFAULT_PROJECT_BASENAME


def _make_ablation_run_id():
    return datetime.now().strftime('dt-%Y%m%d-%H%M%S')


def _resolve_project_name(flags, run_id):
    base = getattr(flags.wandb, 'project', None)
    if not base or base == 'shortcut':
        base = DEFAULT_PROJECT_BASENAME
    base = _sanitize_project_name(base)
    dataset = _sanitize_project_name(flags.dataset_name)
    return f'{base}-{dataset}-{run_id}'


def _base_model_overrides(flags):
    overrides = flags.model.to_dict()
    overrides.update({
        'train_type': 'naive-moe-source',
        'source_tau': 1.0,
        'source_zero_init': 1,
        'source_soft_moe': 1,
        'loss_balance_weight': 0.2,
        'loss_entropy_weight': 0.1,
        'source_var_weight': 1.0,
        'source_var_target_std': 1.0,
        'source_dtype': 'bfloat16',
    })
    return overrides


def _base_train_args(flags, metrics_output_path, save_dir, max_steps, wandb_group, wandb_name, save_x_render=True):
    log_interval = max(1, min(int(flags.log_interval), int(max_steps)))
    args = [
        sys.executable,
        str(ROOT_DIR / 'train.py'),
        f'--dataset_name={flags.dataset_name}',
        f'--batch_size={flags.batch_size}',
        f'--max_steps={max_steps}',
        '--mode=train',
        '--run_final_inference=1',
        f'--log_interval={log_interval}',
        f'--eval_interval={max_steps + 2}',
        f'--save_interval={max_steps + 2}',
        f'--save_dir={save_dir}',
        f'--metrics_output_path={metrics_output_path}',
        f'--inference_timesteps={ABLATION_FID_TIMESTEPS}',
        f'--inference_generations={ABLATION_FID_GENERATIONS}',
        f'--eval_fid_timesteps={ABLATION_FID_TIMESTEPS}',
        '--train_metrics_level=summary',
        f'--save_x_render={_sanitize_flag_value(save_x_render)}',
        f'--wandb.project={flags.wandb.project}',
        f'--wandb.group={wandb_group}',
        f'--wandb.name={wandb_name}',
    ]
    if flags.tfds_data_dir:
        args.append(f'--tfds_data_dir={flags.tfds_data_dir}')
    if flags.fid_stats:
        args.append(f'--fid_stats={flags.fid_stats}')
    if flags.wandb.entity:
        args.append(f'--wandb.entity={flags.wandb.entity}')
    args.append(f'--wandb.offline={_sanitize_flag_value(flags.wandb.offline)}')
    if hasattr(flags.wandb, 'service_wait'):
        args.append(f'--wandb.service_wait={_sanitize_flag_value(flags.wandb.service_wait)}')
    if hasattr(flags.wandb, 'start_method'):
        args.append(f'--wandb.start_method={_sanitize_flag_value(flags.wandb.start_method)}')
    if hasattr(flags.wandb, 'disable_stats'):
        args.append(f'--wandb.disable_stats={_sanitize_flag_value(flags.wandb.disable_stats)}')
    if hasattr(flags.wandb, 'save_code'):
        args.append(f'--wandb.save_code={_sanitize_flag_value(flags.wandb.save_code)}')
    return args


def _run_train_screen(
    flags,
    run_name,
    group,
    run_dir,
    max_steps,
    model_overrides,
    dump_source_stats=False,
    final_save=False,
    save_x_render=True,
    flow_viz_samples=FLOW_VIZ_SAMPLES,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / 'metrics.json'
    save_dir = run_dir / 'artifacts'
    args = _base_train_args(
        flags,
        metrics_path,
        save_dir,
        max_steps,
        group,
        run_name,
        save_x_render=save_x_render,
    )
    combined_model_overrides = dict(flags.model.to_dict())
    combined_model_overrides.update(model_overrides)
    _extend_config_flags(args, 'model', combined_model_overrides)
    if dump_source_stats:
        args.append('--dump_source_stats=1')
        args.append('--dump_flow_viz=1')
        args.append(f'--source_stats_samples={int(flow_viz_samples)}')
        args.append(f'--flow_viz_samples={int(flow_viz_samples)}')
    if final_save:
        args.append(f'--final_save_dir={run_dir / "final.pkl"}')
    _run_subprocess(args, run_name)
    metrics = _load_json(metrics_path)
    row = {
        'run_name': run_name,
        'group': group,
        'run_dir': str(run_dir),
        'artifact_dir': str(save_dir),
        'metrics_path': str(metrics_path),
        'final_save_path': str(run_dir / 'final.pkl') if final_save else None,
        'train': metrics.get('train', {}),
        'inference': metrics.get('inference', {}),
        'model_overrides': model_overrides,
    }
    row['valid'] = _is_valid_run(row)
    _write_json(run_dir / 'ranking_row.json', row)
    return row


def _run_gmm(flags, phase_dir, num_modes):
    run_name = f'P1A_GMM_K{num_modes:02d}'
    run_dir = phase_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / 'metrics.json'
    figures_dir = run_dir / 'figures'
    gmm_path = run_dir / 'gmm_stats.npz'
    cmd = [
        sys.executable,
        str(ROOT_DIR / 'data_prep.py'),
        f'--dataset_name={flags.dataset_name}',
        f'--batch_size={flags.batch_size}',
        f'--gmm_num_modes={num_modes}',
        f'--gmm_save_path={gmm_path}',
        f'--metrics_output_path={metrics_path}',
        f'--figures_dir={figures_dir}',
        f'--wandb.project={flags.wandb.project}',
        f'--wandb.group={PHASE1_GROUP}',
        f'--wandb.name={run_name}',
        f'--wandb.offline={_sanitize_flag_value(flags.wandb.offline)}',
    ]
    if flags.tfds_data_dir:
        cmd.append(f'--tfds_data_dir={flags.tfds_data_dir}')
    if flags.wandb.entity:
        cmd.append(f'--wandb.entity={flags.wandb.entity}')
    _run_subprocess(cmd, run_name)
    row = _load_json(metrics_path)
    row['run_name'] = run_name
    row['gmm_stats_path'] = str(gmm_path)
    row['metrics_path'] = str(metrics_path)
    row['figures_dir'] = str(figures_dir)
    row['valid'] = not (
        row['dead_component_count'] > (num_modes / 2)
        or row['max_component_fraction'] > 0.75
        or row['var_floor_hit_rate'] > 0.25
    )
    return row


def _is_valid_run(row):
    train = row.get('train', {})
    if not train:
        return False
    max_usage = float(train.get('max_usage', 0.0))
    var_x0_mean = float(train.get('source/var_mean', 1.0))
    var_x0_min = float(train.get('source/var_min', 1.0))
    fid = _fid_value(row)
    if fid is None or not np.isfinite(fid):
        return False
    if not np.isfinite(var_x0_mean) or not np.isfinite(var_x0_min):
        return False
    if max_usage > 0.90:
        return False
    if var_x0_mean < 0.2 or var_x0_mean > 2.5:
        return False
    if var_x0_min <= 1e-6:
        return False
    return True


def _sort_run_key(row):
    return (
        float(_fid_value(row, float('inf'))),
        float(row.get('valid_loss', row.get('train', {}).get('valid_loss', float('inf')))),
        float(row.get('max_usage', row.get('train', {}).get('max_usage', float('inf')))),
    )


def _decorate_run_row(row, **metadata):
    train = row.get('train', {})
    inference = row.get('inference', {})
    row[FID_METRIC_KEY] = inference.get(FID_METRIC_KEY)
    row['valid_loss'] = train.get('valid_loss')
    row['router_entropy_mean'] = train.get('router/entropy_mean')
    row['max_usage'] = train.get('max_usage')
    row['max_soft_usage'] = train.get('max_soft_usage')
    row['q_alpha_agreement'] = train.get('q_alpha_agreement')
    row['var_x0_mean'] = train.get('source/var_mean')
    row['var_x0_min'] = train.get('source/var_min')
    row['mu_x0_norm'] = train.get('source/mu_x0_norm')
    for key, value in metadata.items():
        row[key] = value
    if row.get('run_dir'):
        _write_json(Path(row['run_dir']) / 'ranking_row.json', row)
    return row


def _select_phase1_screen_gmms(rows):
    survivors = [row for row in rows if row['valid']]
    if len(survivors) < PHASE1_MIN_SURVIVORS:
        raise ValueError(
            f'Phase 1A produced only {len(survivors)} valid GMMs; need at least {PHASE1_MIN_SURVIVORS} to continue.'
        )
    survivors.sort(key=lambda row: (
        row['valid_nll'],
        -row['occupancy_entropy'] / max(np.log(max(row['gmm_num_modes'], 2)), 1e-8),
        row['max_component_fraction'],
    ))
    return survivors[:min(PHASE1_SCREEN_TOPK, len(survivors))]


def _select_phase1_winner(rows):
    survivors = [
        row for row in rows
        if row['valid']
        and float(row['train'].get('max_usage', 0.0)) <= 0.85
    ]
    if not survivors:
        raise ValueError('Phase 1B produced no valid downstream screen run.')
    survivors.sort(key=_sort_run_key)
    return survivors[0]


def _select_group_winner(rows, group_name):
    group_rows = [row for row in rows if row['phase2_group'] == group_name and row['valid']]
    if not group_rows:
        raise ValueError(f'Phase 2 group "{group_name}" has no valid surviving run.')
    group_rows.sort(key=_sort_run_key)
    return group_rows[0]


def _compose_phase2_config(default_overrides, best_balance, best_entropy, best_variance):
    composed = dict(default_overrides)
    composed['loss_balance_weight'] = best_balance['model_overrides']['loss_balance_weight']
    composed['loss_entropy_weight'] = best_entropy['model_overrides']['loss_entropy_weight']
    variance_overrides = best_variance['model_overrides']
    if 'source_var_weight' in variance_overrides:
        composed['source_var_weight'] = variance_overrides['source_var_weight']
        composed['source_var_target_std'] = 1.0
    else:
        composed['source_var_weight'] = 1.0
        composed['source_var_target_std'] = variance_overrides['source_var_target_std']
    return composed


def _make_table_png(rows, columns, title, output_path):
    fig, ax = plt.subplots(figsize=(max(8, len(columns) * 1.5), max(3, len(rows) * 0.4 + 1.5)))
    ax.axis('off')
    cell_text = []
    for row in rows:
        cell_text.append([row.get(col, '') for col in columns])
    table = ax.table(cellText=cell_text, colLabels=columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _reshape_last_three(arr):
    arr = np.asarray(arr)
    if arr.size == 0:
        return arr
    return arr.reshape((-1,) + arr.shape[-3:])


def _save_image_grid(images, output_path, title, num_images=16):
    images = _reshape_last_three(images)
    if images.size == 0:
        return None
    num = min(num_images, images.shape[0])
    side = int(np.ceil(np.sqrt(num)))
    fig, axs = plt.subplots(side, side, figsize=(side * 2, side * 2))
    axs = np.array(axs).reshape(side, side)
    for idx in range(side * side):
        ax = axs[idx // side, idx % side]
        ax.axis('off')
        if idx < num:
            img = np.clip(images[idx], -1, 1)
            img = (img * 0.5) + 0.5
            ax.imshow(img)
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


_VAE_DECODE = None


def _get_vae_decode():
    global _VAE_DECODE
    if _VAE_DECODE is None:
        vae = StableVAE.create()
        _VAE_DECODE = jax.jit(vae.decode)
    return _VAE_DECODE


def _decode_latents(latents, num_images=16):
    latents = _reshape_last_three(latents)
    if latents.size == 0:
        return np.zeros((0,))
    decode = _get_vae_decode()
    batch = jnp.asarray(latents[:num_images], dtype=jnp.float32)
    decoded = np.array(decode(batch))
    return decoded


def _load_analysis_payload(npz_path):
    with np.load(npz_path, allow_pickle=False) as raw:
        return {key: np.asarray(raw[key]) for key in raw.files}


def _flatten_embeddings(arr):
    arr = _reshape_last_three(arr).astype(np.float32, copy=False)
    if arr.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return arr.reshape((arr.shape[0], -1))


def _cluster_colors(num_clusters):
    cmap = plt.cm.get_cmap('tab20', max(num_clusters, 1))
    return [cmap(idx % cmap.N) for idx in range(max(num_clusters, 1))]


def _compute_cluster_assignments(arr, gmm_state):
    flat = _flatten_embeddings(arr)
    num_clusters = int(np.asarray(gmm_state['mu']).shape[0])
    if flat.size == 0:
        return np.zeros((0, num_clusters), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    eps = float(np.asarray(gmm_state.get('standardize_eps', np.array(1e-6, dtype=np.float32))))
    q = np.array(
        posterior_from_stats(
            jnp.asarray(flat, dtype=jnp.float32),
            gmm_state['mean'],
            gmm_state['std'],
            eps,
            gmm_state['log_pi'],
            gmm_state['mu'],
            gmm_state['var'],
        )
    )
    return q, np.argmax(q, axis=-1).astype(np.int32)


def _project_chunks(chunks, method):
    valid_chunks = [chunk for chunk in chunks if chunk is not None and chunk.shape[0] > 0]
    if not valid_chunks:
        return []
    stacked = np.concatenate(valid_chunks, axis=0)
    if stacked.shape[0] < 2:
        return [np.zeros((chunk.shape[0], 2), dtype=np.float32) for chunk in valid_chunks]
    if method == 'pca':
        coords = PCA(n_components=2, random_state=0).fit_transform(stacked)
    elif method == 'tsne':
        reduced = stacked
        if reduced.shape[1] > 50 and reduced.shape[0] > 2:
            reduced_dims = min(50, reduced.shape[0] - 1, reduced.shape[1])
            reduced = PCA(n_components=reduced_dims, random_state=0).fit_transform(reduced)
        perplexity = min(30, max(5, reduced.shape[0] // 20))
        perplexity = min(perplexity, max(1, reduced.shape[0] - 1))
        coords = TSNE(
            n_components=2,
            random_state=0,
            init='pca',
            learning_rate='auto',
            perplexity=perplexity,
        ).fit_transform(reduced)
    else:
        raise ValueError(f'Unsupported projection method: {method}')
    outputs = []
    start = 0
    for chunk in valid_chunks:
        stop = start + chunk.shape[0]
        outputs.append(coords[start:stop])
        start = stop
    return outputs


def _project_centroids_from_data(group_specs, projected_groups, num_clusters):
    for spec, coords in zip(group_specs, projected_groups):
        if spec['name'] == 'x_data':
            centroids = {}
            for cluster_id in range(num_clusters):
                mask = spec['clusters'] == cluster_id
                if np.any(mask):
                    centroids[cluster_id] = coords[mask].mean(axis=0)
            if centroids:
                return centroids
    return {}


def _save_cluster_endpoint_projection(group_specs, output_path, title, method='pca'):
    prepared = []
    flat_chunks = []
    for spec in group_specs:
        flat = _flatten_embeddings(spec['points'])
        if flat.size == 0:
            continue
        take = min(int(spec.get('max_points', 256)), flat.shape[0])
        prepared.append({
            **spec,
            'flat': flat[:take],
            'clusters': np.asarray(spec['clusters'][:take], dtype=np.int32),
        })
        flat_chunks.append(flat[:take])
    if len(prepared) < 2:
        return None

    projected = _project_chunks(flat_chunks, method)
    num_clusters = max(int(spec['clusters'].max()) + 1 for spec in prepared if spec['clusters'].size > 0)
    colors = _cluster_colors(num_clusters)
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    legend_handles = []

    for spec, coords in zip(prepared, projected):
        for cluster_id in np.unique(spec['clusters']):
            mask = spec['clusters'] == cluster_id
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=20,
                alpha=float(spec.get('alpha', 0.7)),
                marker=spec.get('marker', 'o'),
                color=colors[int(cluster_id)],
            )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=spec.get('marker', 'o'),
                color='black',
                linestyle='None',
                markersize=7,
                label=spec['name'],
            )
        )

    centroids = _project_centroids_from_data(prepared, projected, num_clusters)
    for cluster_id, center in centroids.items():
        ax.scatter(
            [center[0]],
            [center[1]],
            s=220,
            facecolors='none',
            edgecolors=colors[int(cluster_id)],
            linewidths=1.8,
        )
        ax.text(center[0], center[1], f'C{cluster_id}', fontsize=8, ha='center', va='center')

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(handles=legend_handles, loc='best', fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _save_cluster_path_projection(anchor_groups, path_specs, output_path, title, method='pca'):
    prepared_anchors = []
    anchor_chunks = []
    for spec in anchor_groups:
        flat = _flatten_embeddings(spec['points'])
        if flat.size == 0:
            continue
        take = min(int(spec.get('max_points', 128)), flat.shape[0])
        prepared_anchors.append({
            **spec,
            'flat': flat[:take],
            'clusters': np.asarray(spec['clusters'][:take], dtype=np.int32),
        })
        anchor_chunks.append(flat[:take])

    prepared_paths = []
    path_chunks = []
    for spec in path_specs:
        states = np.asarray(spec['path_states'], dtype=np.float32)
        if states.size == 0:
            continue
        take = min(int(spec.get('max_paths', PATH_PLOT_MAX_LINES)), states.shape[0])
        states = states[:take]
        prepared_paths.append({
            **spec,
            'states': states,
            'start_clusters': np.asarray(spec['start_clusters'][:take], dtype=np.int32),
        })
        path_chunks.append(states.reshape((states.shape[0] * states.shape[1], -1)))

    if not prepared_paths:
        return None

    projected = _project_chunks(anchor_chunks + path_chunks, method)
    projected_anchors = projected[:len(prepared_anchors)]
    projected_paths = projected[len(prepared_anchors):]
    anchor_cluster_count = max(
        (int(spec['clusters'].max()) + 1) for spec in prepared_anchors if spec['clusters'].size > 0
    ) if prepared_anchors else 1
    path_cluster_count = max(
        (int(spec['start_clusters'].max()) + 1) for spec in prepared_paths if spec['start_clusters'].size > 0
    ) if prepared_paths else 1
    num_clusters = max(anchor_cluster_count, path_cluster_count, 1)
    colors = _cluster_colors(num_clusters)
    fig, ax = plt.subplots(figsize=(8.5, 7.0))

    for spec, coords in zip(prepared_paths, projected_paths):
        coords = coords.reshape((spec['states'].shape[0], spec['states'].shape[1], 2))
        for path_idx in range(coords.shape[0]):
            cluster_id = int(spec['start_clusters'][path_idx])
            ax.plot(
                coords[path_idx, :, 0],
                coords[path_idx, :, 1],
                color=colors[cluster_id],
                alpha=float(spec.get('line_alpha', 0.18)),
                linewidth=0.9,
                linestyle=spec.get('linestyle', '-'),
            )

    legend_handles = []
    for spec, coords in zip(prepared_anchors, projected_anchors):
        for cluster_id in np.unique(spec['clusters']):
            mask = spec['clusters'] == cluster_id
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=14,
                alpha=float(spec.get('alpha', 0.55)),
                marker=spec.get('marker', 'o'),
                color=colors[int(cluster_id)],
            )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=spec.get('marker', 'o'),
                color='black',
                linestyle='None',
                markersize=7,
                label=spec['name'],
            )
        )
    for spec in prepared_paths:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color='black',
                linestyle=spec.get('linestyle', '-'),
                linewidth=1.5,
                label=f"{spec['name']} paths",
            )
        )

    centroids = _project_centroids_from_data(prepared_anchors, projected_anchors, num_clusters)
    for cluster_id, center in centroids.items():
        ax.scatter(
            [center[0]],
            [center[1]],
            s=220,
            facecolors='none',
            edgecolors=colors[int(cluster_id)],
            linewidths=1.8,
        )
        ax.text(center[0], center[1], f'C{cluster_id}', fontsize=8, ha='center', va='center')

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(handles=legend_handles, loc='best', fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _mean_same_index_distance(a, b):
    flat_a = _flatten_embeddings(a)
    flat_b = _flatten_embeddings(b)
    if flat_a.size == 0 or flat_b.size == 0:
        return None
    take = min(flat_a.shape[0], flat_b.shape[0])
    if take == 0:
        return None
    return float(np.mean(np.linalg.norm(flat_a[:take] - flat_b[:take], axis=-1)))


def _global_stats_row(run_name, split_name, arr, data_ref=None, paired_ref=None):
    flat = _flatten_embeddings(arr)
    if flat.size == 0:
        return None
    feature_var = np.var(flat, axis=0)
    row = {
        'run_name': run_name,
        'split': split_name,
        'sample_count': int(flat.shape[0]),
        'mean_norm': float(np.mean(np.linalg.norm(flat, axis=-1))),
        'mean_abs': float(np.mean(np.abs(flat))),
        'variance_mean': float(np.mean(feature_var)),
        'variance_min': float(np.min(feature_var)),
        'variance_max': float(np.max(feature_var)),
        'covariance_trace': float(np.sum(feature_var)),
        'avg_distance_to_data': _mean_same_index_distance(arr, data_ref),
        'avg_distance_to_pair': _mean_same_index_distance(arr, paired_ref),
    }
    return row


def _data_centroids(flat_data, data_clusters, num_clusters):
    centroids = {}
    if flat_data.size == 0 or data_clusters.size == 0:
        return centroids
    for cluster_id in range(num_clusters):
        mask = data_clusters == cluster_id
        if np.any(mask):
            centroids[cluster_id] = flat_data[mask].mean(axis=0)
    return centroids


def _per_cluster_stats_rows(run_name, split_name, arr, clusters, num_clusters, reference_centroids=None):
    flat = _flatten_embeddings(arr)
    if flat.size == 0 or clusters.size == 0:
        return []
    rows = []
    for cluster_id in range(num_clusters):
        mask = clusters == cluster_id
        if not np.any(mask):
            rows.append({
                'run_name': run_name,
                'split': split_name,
                'cluster_id': cluster_id,
                'occupancy': 0,
                'mean_norm': None,
                'variance_mean': None,
                'centroid_distance_to_data': None,
            })
            continue
        cluster_flat = flat[mask]
        centroid = cluster_flat.mean(axis=0)
        distance = None
        if reference_centroids and cluster_id in reference_centroids:
            distance = float(np.linalg.norm(centroid - reference_centroids[cluster_id]))
        rows.append({
            'run_name': run_name,
            'split': split_name,
            'cluster_id': cluster_id,
            'occupancy': int(mask.sum()),
            'mean_norm': float(np.mean(np.linalg.norm(cluster_flat, axis=-1))),
            'variance_mean': float(np.mean(np.var(cluster_flat, axis=0))),
            'centroid_distance_to_data': distance,
        })
    return rows


def _compute_path_stats(run_name, path_states, gmm_state, data_clusters):
    states = np.asarray(path_states, dtype=np.float32)
    if states.size == 0:
        return None, np.zeros((0, 0), dtype=np.int32)
    flat = states.reshape((states.shape[0], states.shape[1], -1))
    deltas = np.diff(flat, axis=1)
    segment_lengths = np.linalg.norm(deltas, axis=-1)
    path_length = np.sum(segment_lengths, axis=1)
    endpoint_displacement = np.linalg.norm(flat[:, -1] - flat[:, 0], axis=-1)
    straightness = endpoint_displacement / np.maximum(path_length, 1e-8)
    if deltas.shape[1] >= 2:
        unit = deltas / np.maximum(segment_lengths[..., None], 1e-8)
        curvature = np.linalg.norm(np.diff(unit, axis=1), axis=-1).mean(axis=1)
    else:
        curvature = np.zeros((flat.shape[0],), dtype=np.float32)
    _, path_clusters = _compute_cluster_assignments(states.reshape((-1,) + states.shape[-3:]), gmm_state)
    path_clusters = path_clusters.reshape((states.shape[0], states.shape[1]))
    switch_count = np.sum(path_clusters[:, 1:] != path_clusters[:, :-1], axis=1)
    agreement = None
    if data_clusters is not None and len(data_clusters) > 0:
        take = min(len(data_clusters), path_clusters.shape[0])
        agreement = float(np.mean(path_clusters[:take, -1] == np.asarray(data_clusters[:take], dtype=np.int32)))
    row = {
        'run_name': run_name,
        'split': 'path',
        'sample_count': int(states.shape[0]),
        'path_length_mean': float(np.mean(path_length)),
        'endpoint_displacement_mean': float(np.mean(endpoint_displacement)),
        'straightness_ratio_mean': float(np.mean(straightness)),
        'curvature_proxy_mean': float(np.mean(curvature)),
        'cluster_switch_count_mean': float(np.mean(switch_count)),
        'final_cluster_agreement_with_data': agreement,
    }
    return row, path_clusters


def _save_q_alpha_heatmap(q_posterior, alpha_posterior, output_path):
    q_mean = np.mean(q_posterior, axis=0)
    alpha_mean = np.mean(alpha_posterior, axis=0)
    heatmap = np.stack([q_mean, alpha_mean], axis=0)
    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(heatmap, cmap='viridis', aspect='auto')
    ax.set_yticks([0, 1], labels=['mean q', 'mean alpha'])
    ax.set_title('q vs alpha heatmap')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _save_usage_bar(alpha_posterior, output_path):
    expert_argmax = np.argmax(alpha_posterior, axis=-1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(expert_argmax, bins=np.arange(expert_argmax.max() + 2) - 0.5, rwidth=0.8)
    ax.set_title('Expert Usage Histogram')
    ax.set_xlabel('Expert')
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _save_entropy_hist(alpha_posterior, output_path):
    router_entropy = -np.sum(alpha_posterior * np.log(np.maximum(alpha_posterior, 1e-8)), axis=-1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(router_entropy, bins=30)
    ax.set_title('Router Entropy Histogram')
    ax.set_xlabel('Entropy')
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _save_confusion_heatmap(conditioned_mode, alpha_posterior, output_path):
    expert_argmax = np.argmax(alpha_posterior, axis=-1)
    conditioned = conditioned_mode.astype(np.int32)
    num_modes = int(max(conditioned.max(), expert_argmax.max()) + 1)
    confusion = np.zeros((num_modes, num_modes), dtype=np.float32)
    for cond, routed in zip(conditioned, expert_argmax):
        confusion[cond, routed] += 1.0
    row_sums = np.maximum(confusion.sum(axis=1, keepdims=True), 1.0)
    confusion /= row_sums
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion, cmap='magma', aspect='auto', vmin=0.0, vmax=1.0)
    ax.set_title('Conditioned Mode vs Router Argmax')
    ax.set_xlabel('Router Argmax')
    ax.set_ylabel('Conditioned GMM Mode')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _render_single_analysis(run_name, role_label, payload, gmm_state, output_dir, key_prefix):
    output_dir.mkdir(parents=True, exist_ok=True)
    prior = _reshape_last_three(payload.get('x0_prior', np.zeros((0,), dtype=np.float32))).astype(np.float32, copy=False)
    posterior = _reshape_last_three(payload.get('x0_posterior', np.zeros((0,), dtype=np.float32))).astype(np.float32, copy=False)
    x_data = _reshape_last_three(payload.get('x1_data', np.zeros((0,), dtype=np.float32))).astype(np.float32, copy=False)
    generated = _reshape_last_three(payload.get('x1_final', np.zeros((0,), dtype=np.float32))).astype(np.float32, copy=False)
    path_states = np.asarray(payload.get('flow_path_states', np.zeros((0,), dtype=np.float32)), dtype=np.float32)

    _, prior_clusters = _compute_cluster_assignments(prior, gmm_state)
    _, posterior_clusters = _compute_cluster_assignments(posterior, gmm_state)
    _, data_clusters = _compute_cluster_assignments(x_data, gmm_state)
    _, generated_clusters = _compute_cluster_assignments(generated, gmm_state)
    num_clusters = int(np.asarray(gmm_state['mu']).shape[0])
    reference_centroids = _data_centroids(_flatten_embeddings(x_data), data_clusters, num_clusters)

    global_rows = []
    cluster_rows = []
    for split_name, arr, clusters, pair_ref in (
        ('source_prior', prior, prior_clusters, generated),
        ('source_posterior', posterior, posterior_clusters, x_data),
        ('x_data', x_data, data_clusters, None),
        ('generated', generated, generated_clusters, prior),
    ):
        row = _global_stats_row(run_name, split_name, arr, data_ref=x_data, paired_ref=pair_ref)
        if row is not None:
            global_rows.append(row)
            cluster_rows.extend(
                _per_cluster_stats_rows(run_name, split_name, arr, clusters, num_clusters, reference_centroids)
            )

    path_row = None
    path_clusters = np.zeros((0, 0), dtype=np.int32)
    if path_states.size > 0:
        path_row, path_clusters = _compute_path_stats(run_name, path_states, gmm_state, data_clusters)

    image_paths = {}
    image_paths[f'{key_prefix}_source_grid'] = _save_image_grid(
        _decode_latents(prior),
        output_dir / f'{key_prefix}_source_grid.png',
        f'{role_label} Source Grid',
    )
    if posterior.size > 0:
        image_paths[f'{key_prefix}_source_posterior_grid'] = _save_image_grid(
            _decode_latents(posterior),
            output_dir / f'{key_prefix}_source_posterior_grid.png',
            f'{role_label} Source Posterior Grid',
        )
    if generated.size > 0:
        image_paths[f'{key_prefix}_generated_grid'] = _save_image_grid(
            _decode_latents(generated),
            output_dir / f'{key_prefix}_generated_grid.png',
            f'{role_label} Generated Grid',
        )
    if x_data.size > 0:
        image_paths[f'{key_prefix}_x_data_grid'] = _save_image_grid(
            _decode_latents(x_data),
            output_dir / f'{key_prefix}_x_data_grid.png',
            f'{role_label} x_data Grid',
        )

    endpoint_groups = [
        {'name': 'source', 'points': prior, 'clusters': prior_clusters, 'marker': 'o', 'alpha': 0.72, 'max_points': 256},
        {'name': 'x_data', 'points': x_data, 'clusters': data_clusters, 'marker': 's', 'alpha': 0.62, 'max_points': 256},
        {'name': 'generated', 'points': generated, 'clusters': generated_clusters, 'marker': '^', 'alpha': 0.68, 'max_points': 256},
    ]
    if posterior.size > 0:
        endpoint_groups.insert(
            1,
            {'name': 'source_posterior', 'points': posterior, 'clusters': posterior_clusters, 'marker': 'D', 'alpha': 0.70, 'max_points': 256},
        )
    image_paths[f'{key_prefix}_latent_pca_endpoints'] = _save_cluster_endpoint_projection(
        endpoint_groups,
        output_dir / f'{key_prefix}_latent_pca_endpoints.png',
        f'{role_label} Latent PCA Endpoints',
        method='pca',
    )
    image_paths[f'{key_prefix}_latent_tsne_endpoints'] = _save_cluster_endpoint_projection(
        endpoint_groups,
        output_dir / f'{key_prefix}_latent_tsne_endpoints.png',
        f'{role_label} Latent t-SNE Endpoints',
        method='tsne',
    )
    if path_states.size > 0 and path_clusters.size > 0:
        path_specs = [{
            'name': role_label,
            'path_states': path_states,
            'start_clusters': path_clusters[:, 0],
            'linestyle': '-',
            'max_paths': PATH_PLOT_MAX_LINES,
            'line_alpha': 0.16,
        }]
        image_paths[f'{key_prefix}_latent_pca_paths'] = _save_cluster_path_projection(
            endpoint_groups,
            path_specs,
            output_dir / f'{key_prefix}_latent_pca_paths.png',
            f'{role_label} Latent PCA ODE Paths',
            method='pca',
        )
        image_paths[f'{key_prefix}_latent_tsne_paths'] = _save_cluster_path_projection(
            endpoint_groups,
            path_specs,
            output_dir / f'{key_prefix}_latent_tsne_paths.png',
            f'{role_label} Latent t-SNE ODE Paths',
            method='tsne',
        )

    if 'q_posterior' in payload and 'alpha_posterior' in payload:
        q_posterior = np.asarray(payload['q_posterior'], dtype=np.float32)
        alpha_posterior = np.asarray(payload['alpha_posterior'], dtype=np.float32)
        image_paths[f'{key_prefix}_q_vs_alpha_heatmap'] = _save_q_alpha_heatmap(
            q_posterior,
            alpha_posterior,
            output_dir / f'{key_prefix}_q_vs_alpha_heatmap.png',
        )
        image_paths[f'{key_prefix}_expert_usage_hist'] = _save_usage_bar(
            alpha_posterior,
            output_dir / f'{key_prefix}_expert_usage_hist.png',
        )
        image_paths[f'{key_prefix}_router_entropy_hist'] = _save_entropy_hist(
            alpha_posterior,
            output_dir / f'{key_prefix}_router_entropy_hist.png',
        )
        if 'conditioned_mode' in payload:
            image_paths[f'{key_prefix}_mode_vs_router_confusion'] = _save_confusion_heatmap(
                np.asarray(payload['conditioned_mode'], dtype=np.int32),
                alpha_posterior,
                output_dir / f'{key_prefix}_mode_vs_router_confusion.png',
            )

    stats_payload = {
        'global_rows': global_rows,
        'cluster_rows': cluster_rows,
        'path_row': path_row,
    }
    _write_json(output_dir / f'{key_prefix}_latent_stats.json', stats_payload)

    table_paths = {}
    if global_rows:
        table_paths[f'{key_prefix}_latent_stats_table'] = str(_make_table_png(
            global_rows,
            ['run_name', 'split', 'sample_count', 'mean_norm', 'variance_mean', 'covariance_trace', 'avg_distance_to_data', 'avg_distance_to_pair'],
            f'{role_label} Latent Global Stats',
            output_dir / f'{key_prefix}_latent_stats_table.png',
        ))
    if cluster_rows:
        table_paths[f'{key_prefix}_cluster_stats_table'] = str(_make_table_png(
            cluster_rows,
            ['run_name', 'split', 'cluster_id', 'occupancy', 'mean_norm', 'variance_mean', 'centroid_distance_to_data'],
            f'{role_label} Per-Cluster Stats',
            output_dir / f'{key_prefix}_cluster_stats_table.png',
        ))
    if path_row is not None:
        table_paths[f'{key_prefix}_path_stats_table'] = str(_make_table_png(
            [path_row],
            ['run_name', 'sample_count', 'path_length_mean', 'endpoint_displacement_mean', 'straightness_ratio_mean', 'curvature_proxy_mean', 'cluster_switch_count_mean', 'final_cluster_agreement_with_data'],
            f'{role_label} Path Stats',
            output_dir / f'{key_prefix}_path_stats_table.png',
        ))

    image_paths = {k: str(v) for k, v in image_paths.items() if v is not None}
    image_paths.update(table_paths)
    return {
        'image_paths': image_paths,
        'global_rows': global_rows,
        'cluster_rows': cluster_rows,
        'path_row': path_row,
        'path_clusters': path_clusters,
        'data_clusters': data_clusters,
        'endpoint_groups': endpoint_groups,
    }


def _render_phase3_compare(moe_payload, naive_payload, gmm_state, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    moe_prior = _reshape_last_three(moe_payload.get('x0_prior', np.zeros((0,), dtype=np.float32))).astype(np.float32, copy=False)
    moe_post = _reshape_last_three(moe_payload.get('x0_posterior', np.zeros((0,), dtype=np.float32))).astype(np.float32, copy=False)
    moe_generated = _reshape_last_three(moe_payload.get('x1_final', np.zeros((0,), dtype=np.float32))).astype(np.float32, copy=False)
    naive_prior = _reshape_last_three(naive_payload.get('x0_prior', np.zeros((0,), dtype=np.float32))).astype(np.float32, copy=False)
    naive_generated = _reshape_last_three(naive_payload.get('x1_final', np.zeros((0,), dtype=np.float32))).astype(np.float32, copy=False)
    x_data = _reshape_last_three(
        moe_payload.get('x1_data', naive_payload.get('x1_data', np.zeros((0,), dtype=np.float32)))
    ).astype(np.float32, copy=False)

    _, moe_prior_clusters = _compute_cluster_assignments(moe_prior, gmm_state)
    _, moe_post_clusters = _compute_cluster_assignments(moe_post, gmm_state)
    _, moe_generated_clusters = _compute_cluster_assignments(moe_generated, gmm_state)
    _, naive_prior_clusters = _compute_cluster_assignments(naive_prior, gmm_state)
    _, naive_generated_clusters = _compute_cluster_assignments(naive_generated, gmm_state)
    _, data_clusters = _compute_cluster_assignments(x_data, gmm_state)

    endpoint_groups = [
        {'name': 'naive_source', 'points': naive_prior, 'clusters': naive_prior_clusters, 'marker': 'o', 'alpha': 0.68, 'max_points': 256},
        {'name': 'naive_generated', 'points': naive_generated, 'clusters': naive_generated_clusters, 'marker': '^', 'alpha': 0.66, 'max_points': 256},
        {'name': 'moe_source_prior', 'points': moe_prior, 'clusters': moe_prior_clusters, 'marker': 'P', 'alpha': 0.68, 'max_points': 256},
        {'name': 'moe_source_posterior', 'points': moe_post, 'clusters': moe_post_clusters, 'marker': 'D', 'alpha': 0.68, 'max_points': 256},
        {'name': 'moe_generated', 'points': moe_generated, 'clusters': moe_generated_clusters, 'marker': 'X', 'alpha': 0.68, 'max_points': 256},
        {'name': 'x_data', 'points': x_data, 'clusters': data_clusters, 'marker': 's', 'alpha': 0.60, 'max_points': 256},
    ]
    image_paths = {}
    image_paths['phase3_shared_latent_pca_endpoints'] = str(_save_cluster_endpoint_projection(
        endpoint_groups,
        output_dir / 'phase3_shared_latent_pca_endpoints.png',
        'Phase 3 Shared Latent PCA Endpoints',
        method='pca',
    ))
    image_paths['phase3_shared_latent_tsne_endpoints'] = str(_save_cluster_endpoint_projection(
        endpoint_groups,
        output_dir / 'phase3_shared_latent_tsne_endpoints.png',
        'Phase 3 Shared Latent t-SNE Endpoints',
        method='tsne',
    ))

    moe_path_states = np.asarray(moe_payload.get('flow_path_states', np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    naive_path_states = np.asarray(naive_payload.get('flow_path_states', np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    _, moe_path_clusters = _compute_path_stats('P3_Final_Top1MoE', moe_path_states, gmm_state, data_clusters)
    _, naive_path_clusters = _compute_path_stats('P3_Final_NaiveRef', naive_path_states, gmm_state, data_clusters)
    path_specs = []
    if naive_path_states.size > 0 and naive_path_clusters.size > 0:
        path_specs.append({
            'name': 'naive',
            'path_states': naive_path_states,
            'start_clusters': naive_path_clusters[:, 0],
            'linestyle': '--',
            'max_paths': PATH_PLOT_MAX_LINES,
            'line_alpha': 0.13,
        })
    if moe_path_states.size > 0 and moe_path_clusters.size > 0:
        path_specs.append({
            'name': 'best_moe1',
            'path_states': moe_path_states,
            'start_clusters': moe_path_clusters[:, 0],
            'linestyle': '-',
            'max_paths': PATH_PLOT_MAX_LINES,
            'line_alpha': 0.16,
        })
    if path_specs:
        image_paths['phase3_shared_latent_pca_paths'] = str(_save_cluster_path_projection(
            endpoint_groups,
            path_specs,
            output_dir / 'phase3_shared_latent_pca_paths.png',
            'Phase 3 Shared Latent PCA ODE Paths',
            method='pca',
        ))
        image_paths['phase3_shared_latent_tsne_paths'] = str(_save_cluster_path_projection(
            endpoint_groups,
            path_specs,
            output_dir / 'phase3_shared_latent_tsne_paths.png',
            'Phase 3 Shared Latent t-SNE ODE Paths',
            method='tsne',
        ))
    return {k: v for k, v in image_paths.items() if v and os.path.exists(v)}


def _log_summary_run(flags, group, name, summary_metrics, image_paths=None, config=None):
    os.environ.setdefault(
        'WANDB__SERVICE_WAIT',
        str(getattr(flags.wandb, 'service_wait', 300)),
    )
    run = wandb.init(
        project=flags.wandb.project,
        entity=flags.wandb.entity,
        group=group,
        name=name,
        config=config or {},
        mode='offline' if flags.wandb.offline else 'online',
        save_code=False,
        settings=wandb.Settings(
            start_method=getattr(flags.wandb, 'start_method', 'thread'),
            _disable_stats=getattr(flags.wandb, 'disable_stats', True),
            _service_wait=getattr(flags.wandb, 'service_wait', 300),
        ),
        reinit=True,
    )
    for key, value in summary_metrics.items():
        run.summary[key] = value
    if image_paths:
        payload = {}
        for label, path in image_paths.items():
            if path and os.path.exists(path):
                payload[label] = wandb.Image(path)
        if payload:
            wandb.log(payload)
    run.finish()


def _filter_image_paths(image_paths, keep_keys):
    keep = set(keep_keys)
    return {
        key: value
        for key, value in (image_paths or {}).items()
        if key in keep and value and os.path.exists(value)
    }


def _filter_balanced_master_image_paths(image_paths, keep_keys):
    detail_prefixes = ('phase1b_', 'phase2_')
    detail_suffixes = (
        '_source_grid',
        '_source_posterior_grid',
        '_generated_grid',
        '_x_data_grid',
        '_latent_pca_endpoints',
        '_latent_tsne_endpoints',
        '_latent_pca_paths',
        '_latent_tsne_paths',
    )
    keep = set(keep_keys)
    output = {}
    for key, value in (image_paths or {}).items():
        if not value or not os.path.exists(value):
            continue
        if key in keep or (key.startswith(detail_prefixes) and key.endswith(detail_suffixes)):
            output[key] = value
    return output


def _compact_value(value):
    if value is None:
        return ''
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return ''
        return f'{float(value):.6g}'
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return str(value)


def _markdown_table(rows, columns, max_rows=None):
    rows = rows[:max_rows] if max_rows is not None else rows
    if not rows:
        return '_No rows._'
    header = '| ' + ' | '.join(columns) + ' |'
    separator = '| ' + ' | '.join(['---'] * len(columns)) + ' |'
    body = []
    for row in rows:
        body.append('| ' + ' | '.join(_compact_value(row.get(column)) for column in columns) + ' |')
    return '\n'.join([header, separator, *body])


def _write_analysis_packet(
    root,
    ablation_run_id,
    project_name,
    gmm_rows,
    phase1_candidates,
    phase1b_rows,
    phase1_winner,
    phase2_rows,
    composed_row,
    phase2_summary,
    phase2_selected,
    phase3_selected,
    phase3_naive,
    final_selected,
    master_summary_rows,
    latent_stats_summary_rows,
    visualization_index_entries,
):
    phase1a_rows = _phase1a_summary_rows(gmm_rows, phase1_candidates)
    phase1b_rows_full = _phase1b_full_rows(phase1b_rows, phase1_winner)
    phase2_rows_full = _phase2_full_rows(phase2_rows, composed_row, phase2_selected)
    phase3_rows = [
        {
            'role': 'best_moe1',
            'run_name': phase3_selected['run_name'],
            FID_METRIC_KEY: phase3_selected.get(FID_METRIC_KEY),
            'valid_loss': phase3_selected.get('valid_loss'),
            LATENCY_METRIC_KEY: phase3_selected.get('inference', {}).get(LATENCY_METRIC_KEY),
            THROUGHPUT_METRIC_KEY: phase3_selected.get('inference', {}).get(THROUGHPUT_METRIC_KEY),
        },
        {
            'role': 'naive_reference',
            'run_name': phase3_naive['run_name'],
            FID_METRIC_KEY: phase3_naive.get(FID_METRIC_KEY),
            'valid_loss': phase3_naive.get('valid_loss'),
            LATENCY_METRIC_KEY: phase3_naive.get('inference', {}).get(LATENCY_METRIC_KEY),
            THROUGHPUT_METRIC_KEY: phase3_naive.get('inference', {}).get(THROUGHPUT_METRIC_KEY),
        },
    ]
    payload = {
        'meta': {
            'ablation_run_id': ablation_run_id,
            'wandb_project': project_name,
            'fid_metric_key': FID_METRIC_KEY,
            'fid_timesteps': ABLATION_FID_TIMESTEPS,
            'fid_generations': ABLATION_FID_GENERATIONS,
            'flow_viz_samples': FLOW_VIZ_SAMPLES,
            'path_plot_max_lines': PATH_PLOT_MAX_LINES,
        },
        'winners': {
            'phase1_winner': phase1_winner['run_name'],
            'phase2_selected': phase2_selected['run_name'],
            'phase3_winner': final_selected['run_name'],
            'phase2_composed_status': phase2_summary['composed_status'],
        },
        'phase1a': phase1a_rows,
        'phase1b': phase1b_rows_full,
        'phase2': phase2_rows_full,
        'phase3': phase3_rows,
        'master_summary': master_summary_rows,
        'latent_stats_summary': latent_stats_summary_rows,
        'visualization_index': visualization_index_entries,
    }
    json_path = root / 'analysis_packet.json'
    md_path = root / 'analysis_packet.md'
    _write_json(json_path, payload)

    md = [
        '# moe1-ablation analysis packet',
        '',
        'Copy this whole file into chat when you want a post-run analysis.',
        '',
        '## Quick Context',
        '',
        f'- run_id: `{ablation_run_id}`',
        f'- wandb_project: `{project_name}`',
        f'- ranking_metric: `{FID_METRIC_KEY}`',
        f'- fid_sampling: `{ABLATION_FID_TIMESTEPS}` ODE steps, `{ABLATION_FID_GENERATIONS}` images',
        f'- phase1_winner: `{phase1_winner["run_name"]}`',
        f'- phase2_selected: `{phase2_selected["run_name"]}`',
        f'- phase2_composed_status: `{phase2_summary["composed_status"]}`',
        f'- phase3_winner: `{final_selected["run_name"]}`',
        '',
        '## Master Summary',
        '',
        _markdown_table(master_summary_rows, ['stage', 'item', 'run_name', 'detail', FID_METRIC_KEY]),
        '',
        '## Phase 1A GMM Summary',
        '',
        _markdown_table(
            phase1a_rows,
            ['run_name', 'K', 'valid_nll', 'dead_components', 'max_component_fraction', 'selected_for_phase1b', 'valid'],
        ),
        '',
        '## Phase 1B Full Screen',
        '',
        _markdown_table(
            phase1b_rows_full,
            ['run_name', 'gmm_num_modes', FID_METRIC_KEY, 'valid_loss', 'max_usage', 'max_soft_usage', 'q_alpha_agreement', 'var_x0_mean', 'var_x0_min', 'winner', 'valid'],
        ),
        '',
        '## Phase 2 Full Sweep',
        '',
        _markdown_table(
            phase2_rows_full,
            ['run_name', 'phase2_group', FID_METRIC_KEY, 'valid_loss', 'max_usage', 'max_soft_usage', 'q_alpha_agreement', 'var_x0_mean', 'var_x0_min', 'group_winner', 'is_composed', 'is_selected', 'valid'],
        ),
        '',
        '## Phase 3 Final Duel',
        '',
        _markdown_table(phase3_rows, ['role', 'run_name', FID_METRIC_KEY, 'valid_loss', LATENCY_METRIC_KEY, THROUGHPUT_METRIC_KEY]),
        '',
        '## Latent Stats Summary',
        '',
        _markdown_table(
            latent_stats_summary_rows,
            ['run_name', 'split', 'sample_count', 'mean_norm', 'variance_mean', 'avg_distance_to_data', 'straightness_ratio_mean', 'final_cluster_agreement_with_data'],
        ),
        '',
        '## Visualization Index',
        '',
        _markdown_table(
            _visualization_index_rows(visualization_index_entries),
            ['stage', 'run_name', 'role', 'projection_types', 'cluster_source', 'cache_path'],
        ),
        '',
        f'Full machine-readable packet: `{json_path}`',
    ]
    md_path.write_text('\n'.join(md) + '\n', encoding='utf-8')
    return json_path, md_path


def _cleanup_run_artifacts(
    run_dir,
    keep_metrics=True,
    keep_figures=True,
    keep_checkpoint=False,
    keep_analysis_cache=False,
):
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return
    for child in run_dir.iterdir():
        if child.name == 'metrics.json' and keep_metrics:
            continue
        if child.name == 'figures' and keep_figures:
            continue
        if child.name == 'ranking_row.json':
            continue
        if child.name == 'final.pkl' and keep_checkpoint:
            continue
        if child.is_dir():
            if child.name == 'artifacts' and keep_analysis_cache:
                for nested in child.iterdir():
                    if nested.name == 'source_stats.npz':
                        continue
                    if nested.is_dir():
                        shutil.rmtree(nested, ignore_errors=True)
                    else:
                        try:
                            nested.unlink()
                        except FileNotFoundError:
                            pass
                continue
            shutil.rmtree(child, ignore_errors=True)
        else:
            if child.name == 'source_stats.npz' and keep_analysis_cache:
                continue
            try:
                child.unlink()
            except FileNotFoundError:
                pass


def _phase1a_summary_rows(rows, selected_rows):
    selected_names = {row['run_name'] for row in selected_rows}
    summary = []
    for row in rows:
        summary.append({
            'run_name': row['run_name'],
            'K': row['gmm_num_modes'],
            'valid_nll': row['valid_nll'],
            'dead_components': row['dead_component_count'],
            'max_component_fraction': row['max_component_fraction'],
            'selected_for_phase1b': 'yes' if row['run_name'] in selected_names else '',
            'valid': 'yes' if row['valid'] else 'no',
        })
    return summary


def _phase1b_summary_rows(rows, winner):
    winner_name = winner['run_name']
    summary = []
    for row in rows:
        summary.append({
            'run_name': row['run_name'],
            'gmm_num_modes': row.get('phase1b/gmm_num_modes'),
            FID_METRIC_KEY: row.get(FID_METRIC_KEY),
            'valid_loss': row.get('valid_loss'),
            'max_usage': row.get('max_usage'),
            'winner': 'yes' if row['run_name'] == winner_name else '',
            'valid': 'yes' if row['valid'] else 'no',
        })
    return summary


def _phase1b_full_rows(rows, winner):
    winner_name = winner['run_name']
    full_rows = []
    for row in rows:
        full_rows.append({
            'run_name': row['run_name'],
            'gmm_num_modes': row.get('phase1b/gmm_num_modes'),
            FID_METRIC_KEY: row.get(FID_METRIC_KEY),
            'valid_loss': row.get('valid_loss'),
            'max_usage': row.get('max_usage'),
            'max_soft_usage': row.get('max_soft_usage'),
            'q_alpha_agreement': row.get('q_alpha_agreement'),
            'var_x0_mean': row.get('var_x0_mean'),
            'var_x0_min': row.get('var_x0_min'),
            'winner': 'yes' if row['run_name'] == winner_name else '',
            'valid': 'yes' if row['valid'] else 'no',
        })
    return full_rows


def _phase2_full_rows(rows, composed_row, phase2_selected):
    selected_name = phase2_selected['run_name']
    composed_name = composed_row['run_name']
    full_rows = []
    for row in [*rows, composed_row]:
        full_rows.append({
            'run_name': row['run_name'],
            'phase2_group': row.get('phase2_group'),
            FID_METRIC_KEY: row.get(FID_METRIC_KEY),
            'valid_loss': row.get('valid_loss'),
            'max_usage': row.get('max_usage'),
            'max_soft_usage': row.get('max_soft_usage'),
            'q_alpha_agreement': row.get('q_alpha_agreement'),
            'var_x0_mean': row.get('var_x0_mean'),
            'var_x0_min': row.get('var_x0_min'),
            'group_winner': 'yes' if row.get('phase2/is_group_winner') else '',
            'is_composed': 'yes' if row['run_name'] == composed_name else '',
            'is_selected': 'yes' if row['run_name'] == selected_name else '',
            'valid': 'yes' if row.get('valid') else 'no',
        })
    return full_rows


def _visualization_index_rows(entries):
    return [
        {
            'stage': entry['stage'],
            'run_name': entry['run_name'],
            'role': entry['role'],
            'projection_types': entry['projection_types'],
            'cluster_source': entry['cluster_source'],
            'cache_path': entry['cache_path'],
        }
        for entry in entries
    ]


def _latent_stats_summary_rows(analysis_summaries):
    rows = []
    for summary in analysis_summaries:
        for row in summary.get('global_rows', []):
            rows.append({
                'run_name': row['run_name'],
                'split': row['split'],
                'sample_count': row['sample_count'],
                'mean_norm': row['mean_norm'],
                'variance_mean': row['variance_mean'],
                'avg_distance_to_data': row['avg_distance_to_data'],
                'straightness_ratio_mean': None,
                'final_cluster_agreement_with_data': None,
            })
        if summary.get('path_row') is not None:
            row = summary['path_row']
            rows.append({
                'run_name': row['run_name'],
                'split': 'path',
                'sample_count': row['sample_count'],
                'mean_norm': None,
                'variance_mean': None,
                'avg_distance_to_data': None,
                'straightness_ratio_mean': row['straightness_ratio_mean'],
                'final_cluster_agreement_with_data': row['final_cluster_agreement_with_data'],
            })
    return rows


def _master_summary_rows(run_id, project_name, phase1_winner, best_balance, best_entropy, best_variance,
                         composed_row, phase2_selected, phase3_selected, phase3_naive, final_selected):
    return [
        {
            'stage': 'phase1',
            'item': 'winning_gmm',
            'run_name': phase1_winner['run_name'],
            'detail': f"K={phase1_winner['model_overrides']['gmm_num_modes']}",
            FID_METRIC_KEY: phase1_winner.get(FID_METRIC_KEY),
        },
        {
            'stage': 'phase2',
            'item': 'best_balance',
            'run_name': best_balance['run_name'],
            'detail': f"balance={best_balance['model_overrides']['loss_balance_weight']}",
            FID_METRIC_KEY: best_balance.get(FID_METRIC_KEY),
        },
        {
            'stage': 'phase2',
            'item': 'best_entropy',
            'run_name': best_entropy['run_name'],
            'detail': f"entropy={best_entropy['model_overrides']['loss_entropy_weight']}",
            FID_METRIC_KEY: best_entropy.get(FID_METRIC_KEY),
        },
        {
            'stage': 'phase2',
            'item': 'best_variance',
            'run_name': best_variance['run_name'],
            'detail': (
                f"var_w={best_variance['model_overrides'].get('source_var_weight', 1.0)}, "
                f"target={best_variance['model_overrides'].get('source_var_target_std', 1.0)}"
            ),
            FID_METRIC_KEY: best_variance.get(FID_METRIC_KEY),
        },
        {
            'stage': 'phase2',
            'item': 'composed',
            'run_name': composed_row['run_name'],
            'detail': f"selected={phase2_selected['run_name'] == composed_row['run_name']}",
            FID_METRIC_KEY: composed_row.get(FID_METRIC_KEY),
        },
        {
            'stage': 'phase2',
            'item': 'phase2_selected',
            'run_name': phase2_selected['run_name'],
            'detail': f"project={project_name}",
            FID_METRIC_KEY: phase2_selected.get(FID_METRIC_KEY),
        },
        {
            'stage': 'phase3',
            'item': 'top1_moe',
            'run_name': phase3_selected['run_name'],
            'detail': f"run_id={run_id}",
            FID_METRIC_KEY: phase3_selected.get(FID_METRIC_KEY),
        },
        {
            'stage': 'phase3',
            'item': 'naive_reference',
            'run_name': phase3_naive['run_name'],
            'detail': '',
            FID_METRIC_KEY: phase3_naive.get(FID_METRIC_KEY),
        },
        {
            'stage': 'phase3',
            'item': 'final_winner',
            'run_name': final_selected['run_name'],
            'detail': '',
            FID_METRIC_KEY: final_selected.get(FID_METRIC_KEY),
        },
    ]


def run(flags):
    if not flags.fid_stats:
        raise ValueError('--fid_stats is required for mode=moe1-ablation.')
    ablation_run_id = _make_ablation_run_id()
    flags.wandb.project = _resolve_project_name(flags, ablation_run_id)
    root = Path('/kaggle/working/moe1_ablation') / ablation_run_id
    root.mkdir(parents=True, exist_ok=True)
    _write_json(
        root / 'ablation_context.json',
        {
            'ablation_run_id': ablation_run_id,
            'wandb_project': flags.wandb.project,
            'dataset_name': flags.dataset_name,
            'fid_stats': flags.fid_stats,
        },
    )
    cleanup_immediately = not bool(flags.wandb.offline)

    # Phase 1A
    phase1a_dir = root / 'phase1a_gmm'
    gmm_rows = [_run_gmm(flags, phase1a_dir, k) for k in PHASE1_GMM_K_LIST]
    phase1_candidates = _select_phase1_screen_gmms(gmm_rows)
    _write_json(
        phase1a_dir / 'ranking.json',
        {
            'all': gmm_rows,
            'selected_for_phase1b': phase1_candidates,
            'phase1_screen_topk': PHASE1_SCREEN_TOPK,
        },
    )
    phase1a_summary_table = _make_table_png(
        _phase1a_summary_rows(gmm_rows, phase1_candidates),
        ['run_name', 'K', 'valid_nll', 'dead_components', 'max_component_fraction', 'selected_for_phase1b', 'valid'],
        'Phase 1A GMM Summary',
        phase1a_dir / 'phase1a_summary.png',
    )

    # Phase 1B
    phase1b_dir = root / 'phase1b_screen'
    base_overrides = _base_model_overrides(flags)
    phase1b_rows = []
    for row in phase1_candidates:
        overrides = dict(base_overrides)
        overrides['gmm_stats_path'] = row['gmm_stats_path']
        overrides['gmm_num_modes'] = row['gmm_num_modes']
        phase1b_row = _run_train_screen(
                flags,
                f'P1B_GMMscreen_K{row["gmm_num_modes"]:02d}',
                PHASE1B_GROUP,
                phase1b_dir / f'K{row["gmm_num_modes"]:02d}',
                10000,
                overrides,
                dump_source_stats=True,
                save_x_render=False,
            )
        phase1b_row = _decorate_run_row(
            phase1b_row,
            phase='phase1b',
            **{
                'phase1b/gmm_num_modes': row['gmm_num_modes'],
                'phase1b/is_winner': False,
            },
        )
        phase1b_rows.append(phase1b_row)
        if cleanup_immediately:
            _cleanup_run_artifacts(
                phase1b_row['run_dir'],
                keep_metrics=True,
                keep_figures=False,
                keep_checkpoint=False,
                keep_analysis_cache=True,
            )
    phase1_winner = _select_phase1_winner(phase1b_rows)
    _decorate_run_row(phase1_winner, **{'phase1b/is_winner': True})
    _write_json(phase1b_dir / 'ranking.json', {'all': phase1b_rows, 'winner': phase1_winner})
    phase1b_summary_table = _make_table_png(
        _phase1b_summary_rows(phase1b_rows, phase1_winner),
        ['run_name', 'gmm_num_modes', FID_METRIC_KEY, 'valid_loss', 'max_usage', 'winner', 'valid'],
        'Phase 1B Downstream Summary',
        phase1b_dir / 'phase1b_summary.png',
    )
    phase1b_full_table = _make_table_png(
        _phase1b_full_rows(phase1b_rows, phase1_winner),
        [
            'run_name',
            'gmm_num_modes',
            FID_METRIC_KEY,
            'valid_loss',
            'max_usage',
            'max_soft_usage',
            'q_alpha_agreement',
            'var_x0_mean',
            'var_x0_min',
            'winner',
            'valid',
        ],
        'Phase 1B Full Screen Table',
        phase1b_dir / 'phase1b_full_table.png',
    )

    # Phase 2
    phase2_dir = root / 'phase2_sweep'
    phase2_rows = []
    default_overrides = dict(base_overrides)
    default_overrides['gmm_stats_path'] = phase1_winner['model_overrides']['gmm_stats_path']
    default_overrides['gmm_num_modes'] = phase1_winner['model_overrides']['gmm_num_modes']
    for run_name, group_name, overrides in PHASE2_CONFIGS:
        run_overrides = dict(default_overrides)
        run_overrides.update(overrides)
        row = _run_train_screen(
            flags,
            run_name,
            PHASE2_GROUP,
            phase2_dir / run_name,
            PHASE2_MAX_STEPS,
            run_overrides,
            dump_source_stats=True,
            save_x_render=False,
        )
        row = _decorate_run_row(
            row,
            phase='phase2',
            phase2_group=group_name,
            **{
                'phase2/group': group_name,
                'phase2/is_group_winner': False,
                'phase2/is_composed': False,
                'phase2/is_phase2_fallback_winner': False,
                'phase2/is_selected': False,
            },
        )
        phase2_rows.append(row)
        if cleanup_immediately:
            _cleanup_run_artifacts(
                row['run_dir'],
                keep_metrics=True,
                keep_figures=False,
                keep_checkpoint=False,
                keep_analysis_cache=True,
            )

    best_balance = _select_group_winner(phase2_rows, 'balance')
    best_entropy = _select_group_winner(phase2_rows, 'entropy')
    best_variance = _select_group_winner(phase2_rows, 'variance')
    for row in phase2_rows:
        _decorate_run_row(
            row,
            **{
                'phase2/is_group_winner': row['run_name'] in {
                    best_balance['run_name'],
                    best_entropy['run_name'],
                    best_variance['run_name'],
                },
                'phase2/is_composed': False,
                'phase2/is_phase2_fallback_winner': False,
                'phase2/is_selected': False,
            },
        )

    composed_overrides = _compose_phase2_config(default_overrides, best_balance, best_entropy, best_variance)
    composed_row = _run_train_screen(
        flags,
        'M17_Composed',
        PHASE2_COMPOSED_GROUP,
        phase2_dir / 'M17_Composed',
        PHASE2_MAX_STEPS,
        composed_overrides,
        dump_source_stats=True,
        save_x_render=False,
    )
    composed_row = _decorate_run_row(
        composed_row,
        phase='phase2',
        phase2_group='composed',
        **{
            'phase2/group': 'composed',
            'phase2/is_group_winner': False,
            'phase2/is_composed': True,
            'phase2/is_phase2_fallback_winner': False,
            'phase2/is_selected': False,
        },
    )
    if cleanup_immediately:
        _cleanup_run_artifacts(
            composed_row['run_dir'],
            keep_metrics=True,
            keep_figures=False,
            keep_checkpoint=False,
            keep_analysis_cache=True,
        )

    default_row = next(row for row in phase2_rows if row['run_name'] == 'M00_Default')
    composed_pass = (
        composed_row['valid']
        and composed_row['inference'][FID_METRIC_KEY] <= default_row['inference'][FID_METRIC_KEY] * 1.03
    )
    valid_phase2 = [row for row in phase2_rows if row['valid']]
    if not valid_phase2:
        raise ValueError('Phase 2 produced no valid single-run MoE configuration for fallback.')
    valid_phase2.sort(key=_sort_run_key)
    phase2_fallback = valid_phase2[0]
    _decorate_run_row(phase2_fallback, **{'phase2/is_phase2_fallback_winner': True})
    phase2_selected = composed_row if composed_pass else phase2_fallback
    _decorate_run_row(phase2_selected, **{'phase2/is_selected': True})

    phase2_summary = {
        'best_balance': best_balance['run_name'],
        'best_entropy': best_entropy['run_name'],
        'best_variance': best_variance['run_name'],
        'composed_status': 'pass' if composed_pass else 'fail',
        'final_selected_config': phase2_selected['run_name'],
    }
    _write_json(
        phase2_dir / 'ranking.json',
        {
            'all_phase2_runs': phase2_rows,
            'composed': composed_row,
            'group_winners': {
                'balance': best_balance,
                'entropy': best_entropy,
                'variance': best_variance,
            },
            'fallback': phase2_fallback,
            'selected': phase2_selected,
            'summary': phase2_summary,
        },
    )
    summary_table = _make_table_png(
        [
            {
                'winner_type': 'balance',
                'run_name': best_balance['run_name'],
                FID_METRIC_KEY: best_balance[FID_METRIC_KEY],
            },
            {
                'winner_type': 'entropy',
                'run_name': best_entropy['run_name'],
                FID_METRIC_KEY: best_entropy[FID_METRIC_KEY],
            },
            {
                'winner_type': 'variance',
                'run_name': best_variance['run_name'],
                FID_METRIC_KEY: best_variance[FID_METRIC_KEY],
            },
            {
                'winner_type': 'composed',
                'run_name': composed_row['run_name'],
                FID_METRIC_KEY: composed_row[FID_METRIC_KEY],
            },
            {
                'winner_type': 'selected',
                'run_name': phase2_selected['run_name'],
                FID_METRIC_KEY: phase2_selected[FID_METRIC_KEY],
            },
        ],
        ['winner_type', 'run_name', FID_METRIC_KEY],
        'Phase 2 Summary',
        phase2_dir / 'phase2_summary.png',
    )
    composed_config_table = _make_table_png(
        [
            {
                'source': 'best_balance',
                'winner_run': best_balance['run_name'],
                'parameter': 'loss_balance_weight',
                'value': composed_overrides['loss_balance_weight'],
            },
            {
                'source': 'best_entropy',
                'winner_run': best_entropy['run_name'],
                'parameter': 'loss_entropy_weight',
                'value': composed_overrides['loss_entropy_weight'],
            },
            {
                'source': 'best_variance',
                'winner_run': best_variance['run_name'],
                'parameter': 'source_var_weight',
                'value': composed_overrides['source_var_weight'],
            },
            {
                'source': 'best_variance',
                'winner_run': best_variance['run_name'],
                'parameter': 'source_var_target_std',
                'value': composed_overrides['source_var_target_std'],
            },
        ],
        ['source', 'winner_run', 'parameter', 'value'],
        'Top1_MoE_Composed Config',
        phase2_dir / 'phase2_composed_config.png',
    )
    default_vs_composed_table = _make_table_png(
        [
            {
                'run_name': default_row['run_name'],
                FID_METRIC_KEY: default_row[FID_METRIC_KEY],
                'valid_loss': default_row['valid_loss'],
                'max_usage': default_row['max_usage'],
                'max_soft_usage': default_row['max_soft_usage'],
                'q_alpha_agreement': default_row['q_alpha_agreement'],
                'var_x0_mean': default_row['var_x0_mean'],
            },
            {
                'run_name': composed_row['run_name'],
                FID_METRIC_KEY: composed_row[FID_METRIC_KEY],
                'valid_loss': composed_row['valid_loss'],
                'max_usage': composed_row['max_usage'],
                'max_soft_usage': composed_row['max_soft_usage'],
                'q_alpha_agreement': composed_row['q_alpha_agreement'],
                'var_x0_mean': composed_row['var_x0_mean'],
            },
        ],
        [
            'run_name',
            FID_METRIC_KEY,
            'valid_loss',
            'max_usage',
            'max_soft_usage',
            'q_alpha_agreement',
            'var_x0_mean',
        ],
        'M00_Default vs M17_Composed',
        phase2_dir / 'phase2_default_vs_composed.png',
    )
    phase2_full_sweep_table = _make_table_png(
        _phase2_full_rows(phase2_rows, composed_row, phase2_selected),
        [
            'run_name',
            'phase2_group',
            FID_METRIC_KEY,
            'valid_loss',
            'max_usage',
            'max_soft_usage',
            'q_alpha_agreement',
            'var_x0_mean',
            'var_x0_min',
            'group_winner',
            'is_composed',
            'is_selected',
            'valid',
        ],
        'Phase 2 Full Sweep Table',
        phase2_dir / 'phase2_full_sweep_table.png',
    )
    _log_summary_run(
        flags,
        PHASE2_COMPOSED_GROUP,
        'P2_Composed_Summary',
        {
            'phase2/best_balance': best_balance['run_name'],
            'phase2/best_entropy': best_entropy['run_name'],
            'phase2/best_variance': best_variance['run_name'],
            'phase2/composed_status': phase2_summary['composed_status'],
            'phase2/final_selected_config': phase2_selected['run_name'],
        },
        image_paths={
            'phase2_summary': str(summary_table),
            'phase2_composed_config': str(composed_config_table),
            'phase2_default_vs_composed': str(default_vs_composed_table),
            'phase2_full_sweep': str(phase2_full_sweep_table),
        },
        config={
            'best_balance': best_balance['run_name'],
            'best_entropy': best_entropy['run_name'],
            'best_variance': best_variance['run_name'],
            'composed_overrides': composed_overrides,
        },
    )

    # Phase 3
    phase3_dir = root / 'phase3_final'
    phase3_selected = _run_train_screen(
        flags,
        'P3_Final_Top1MoE',
        PHASE3_GROUP,
        phase3_dir / 'Top1_MoE',
        50000,
        phase2_selected['model_overrides'],
        dump_source_stats=True,
        final_save=True,
    )
    phase3_selected = _decorate_run_row(
        phase3_selected,
        phase='phase3',
        **{
            'phase3/model_role': 'top1_moe',
            'phase3/selected_from_phase2': phase2_selected['run_name'],
        },
    )
    naive_overrides = flags.model.to_dict()
    naive_overrides['train_type'] = 'naive'
    naive_overrides['gmm_stats_path'] = ''
    phase3_naive = _run_train_screen(
        flags,
        'P3_Final_NaiveRef',
        PHASE3_GROUP,
        phase3_dir / 'Naive_Reference',
        50000,
        naive_overrides,
        dump_source_stats=True,
        final_save=True,
    )
    phase3_naive = _decorate_run_row(
        phase3_naive,
        phase='phase3',
        **{'phase3/model_role': 'naive_reference'},
    )

    final_selected = min(
        [phase3_selected, phase3_naive],
        key=lambda row: (
            row[FID_METRIC_KEY],
            row['valid_loss'],
        ),
    )
    gmm_vis_state = load_gmm_stats(phase1_winner['model_overrides']['gmm_stats_path'])
    figures_dir = phase3_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = root / 'visualizations'
    visualization_dir.mkdir(parents=True, exist_ok=True)
    visualization_index_entries = []
    master_image_paths = {
        'phase1a_summary': str(phase1a_summary_table),
        'phase1b_summary': str(phase1b_summary_table),
        'phase1b_full_screen': str(phase1b_full_table),
        'phase2_summary': str(summary_table),
        'phase2_composed_config': str(composed_config_table),
        'phase2_default_vs_composed': str(default_vs_composed_table),
        'phase2_full_sweep': str(phase2_full_sweep_table),
    }

    for row in phase1b_rows:
        cache_path = Path(row['artifact_dir']) / 'source_stats.npz'
        if not cache_path.exists():
            continue
        key_prefix = f"phase1b_{row['run_name'].lower().replace('-', '_')}"
        render = _render_single_analysis(
            row['run_name'],
            f"Phase1B {row['run_name']}",
            _load_analysis_payload(cache_path),
            gmm_vis_state,
            visualization_dir / 'phase1b' / row['run_name'],
            key_prefix,
        )
        master_image_paths.update(render['image_paths'])
        visualization_index_entries.append({
            'stage': 'phase1b',
            'run_name': row['run_name'],
            'role': 'screen_candidate',
            'projection_types': 'pca,tsne',
            'cluster_source': phase1_winner['run_name'],
            'cache_path': str(cache_path),
        })

    phase2_visual_run_names = []
    for candidate in (default_row, best_balance, best_entropy, best_variance, composed_row, phase2_selected):
        if candidate['run_name'] not in phase2_visual_run_names:
            phase2_visual_run_names.append(candidate['run_name'])
    phase2_visual_rows = {row['run_name']: row for row in phase2_rows}
    phase2_visual_rows[composed_row['run_name']] = composed_row
    for run_name in phase2_visual_run_names:
        row = phase2_visual_rows[run_name]
        cache_path = Path(row['artifact_dir']) / 'source_stats.npz'
        if not cache_path.exists():
            continue
        key_prefix = f"phase2_{row['run_name'].lower().replace('-', '_')}"
        render = _render_single_analysis(
            row['run_name'],
            f"Phase2 {row['run_name']}",
            _load_analysis_payload(cache_path),
            gmm_vis_state,
            visualization_dir / 'phase2' / row['run_name'],
            key_prefix,
        )
        master_image_paths.update(render['image_paths'])
        visualization_index_entries.append({
            'stage': 'phase2',
            'run_name': row['run_name'],
            'role': row.get('phase2_group', 'composed'),
            'projection_types': 'pca,tsne',
            'cluster_source': phase1_winner['run_name'],
            'cache_path': str(cache_path),
        })

    image_paths = {}
    moe_source_stats = Path(phase3_selected['artifact_dir']) / 'source_stats.npz'
    naive_source_stats = Path(phase3_naive['artifact_dir']) / 'source_stats.npz'
    moe_analysis = None
    naive_analysis = None
    if moe_source_stats.exists():
        moe_analysis = _render_single_analysis(
            phase3_selected['run_name'],
            'Best MoE1',
            _load_analysis_payload(moe_source_stats),
            gmm_vis_state,
            visualization_dir / 'phase3' / 'best_moe1',
            'phase3_best_moe1',
        )
        image_paths.update(moe_analysis['image_paths'])
        master_image_paths.update(moe_analysis['image_paths'])
        visualization_index_entries.append({
            'stage': 'phase3',
            'run_name': phase3_selected['run_name'],
            'role': 'best_moe1',
            'projection_types': 'pca,tsne',
            'cluster_source': phase1_winner['run_name'],
            'cache_path': str(moe_source_stats),
        })
    if naive_source_stats.exists():
        naive_analysis = _render_single_analysis(
            phase3_naive['run_name'],
            'Naive Reference',
            _load_analysis_payload(naive_source_stats),
            gmm_vis_state,
            visualization_dir / 'phase3' / 'naive_reference',
            'phase3_naive_reference',
        )
        image_paths.update(naive_analysis['image_paths'])
        master_image_paths.update(naive_analysis['image_paths'])
        visualization_index_entries.append({
            'stage': 'phase3',
            'run_name': phase3_naive['run_name'],
            'role': 'naive_reference',
            'projection_types': 'pca,tsne',
            'cluster_source': phase1_winner['run_name'],
            'cache_path': str(naive_source_stats),
        })
    if moe_source_stats.exists() and naive_source_stats.exists():
        compare_paths = _render_phase3_compare(
            _load_analysis_payload(moe_source_stats),
            _load_analysis_payload(naive_source_stats),
            gmm_vis_state,
            visualization_dir / 'phase3' / 'compare',
        )
        image_paths.update(compare_paths)
        master_image_paths.update(compare_paths)

    phase3_compare_table = _make_table_png(
        [
            {
                'run_name': phase3_selected['run_name'],
                FID_METRIC_KEY: phase3_selected[FID_METRIC_KEY],
                'valid_loss': phase3_selected['valid_loss'],
                LATENCY_METRIC_KEY: phase3_selected['inference'].get(LATENCY_METRIC_KEY),
                THROUGHPUT_METRIC_KEY: phase3_selected['inference'].get(THROUGHPUT_METRIC_KEY),
            },
            {
                'run_name': phase3_naive['run_name'],
                FID_METRIC_KEY: phase3_naive[FID_METRIC_KEY],
                'valid_loss': phase3_naive['valid_loss'],
                LATENCY_METRIC_KEY: phase3_naive['inference'].get(LATENCY_METRIC_KEY),
                THROUGHPUT_METRIC_KEY: phase3_naive['inference'].get(THROUGHPUT_METRIC_KEY),
            },
        ],
        ['run_name', FID_METRIC_KEY, 'valid_loss', LATENCY_METRIC_KEY, THROUGHPUT_METRIC_KEY],
        'Phase 3 Final Duel',
        figures_dir / 'phase3_final_compare.png',
    )
    image_paths['phase3_final_compare'] = str(phase3_compare_table)
    master_image_paths['phase3_final_compare'] = str(phase3_compare_table)

    latent_stats_summary_rows = _latent_stats_summary_rows(
        [summary for summary in (naive_analysis, moe_analysis) if summary is not None]
    )
    latent_stats_summary_table = None
    if latent_stats_summary_rows:
        latent_stats_summary_table = _make_table_png(
            latent_stats_summary_rows,
            [
                'run_name',
                'split',
                'sample_count',
                'mean_norm',
                'variance_mean',
                'avg_distance_to_data',
                'straightness_ratio_mean',
                'final_cluster_agreement_with_data',
            ],
            'Latent Stats Summary Table',
            root / 'latent_stats_summary.png',
        )
        image_paths['latent_stats_summary'] = str(latent_stats_summary_table)
        master_image_paths['latent_stats_summary'] = str(latent_stats_summary_table)

    visualization_index_table = None
    if visualization_index_entries:
        visualization_index_table = _make_table_png(
            _visualization_index_rows(visualization_index_entries),
            ['stage', 'run_name', 'role', 'projection_types', 'cluster_source', 'cache_path'],
            'Visualization Index Table',
            root / 'visualization_index.png',
        )
        image_paths['visualization_index'] = str(visualization_index_table)
        master_image_paths['visualization_index'] = str(visualization_index_table)

    _write_json(
        phase3_dir / 'ranking.json',
        {
            'moe': phase3_selected,
            'naive': phase3_naive,
            'winner': final_selected['run_name'],
        },
    )
    master_summary_rows = _master_summary_rows(
        ablation_run_id,
        flags.wandb.project,
        phase1_winner,
        best_balance,
        best_entropy,
        best_variance,
        composed_row,
        phase2_selected,
        phase3_selected,
        phase3_naive,
        final_selected,
    )
    master_summary_table = _make_table_png(
        master_summary_rows,
        ['stage', 'item', 'run_name', 'detail', FID_METRIC_KEY],
        'moe1-ablation Master Summary',
        root / 'master_summary.png',
    )
    _write_json(
        root / 'master_summary.json',
        {
            'ablation_run_id': ablation_run_id,
            'wandb_project': flags.wandb.project,
            'phase1_winner': phase1_winner,
            'phase2_summary': phase2_summary,
            'phase3_winner': final_selected['run_name'],
            'visualization_index': visualization_index_entries,
            'rows': master_summary_rows,
        },
    )
    analysis_packet_json, analysis_packet_md = _write_analysis_packet(
        root,
        ablation_run_id,
        flags.wandb.project,
        gmm_rows,
        phase1_candidates,
        phase1b_rows,
        phase1_winner,
        phase2_rows,
        composed_row,
        phase2_summary,
        phase2_selected,
        phase3_selected,
        phase3_naive,
        final_selected,
        master_summary_rows,
        latent_stats_summary_rows,
        visualization_index_entries,
    )
    print(f'Analysis packet for copy/paste: {analysis_packet_md}')
    final_viz_keys = {
        'phase3_shared_latent_pca_endpoints',
        'phase3_shared_latent_tsne_endpoints',
        'phase3_shared_latent_pca_paths',
        'phase3_shared_latent_tsne_paths',
        'phase3_best_moe1_q_vs_alpha_heatmap',
        'phase3_best_moe1_expert_usage_hist',
    }
    summary_table_keys = {
        'phase1a_summary',
        'phase1b_summary',
        'phase1b_full_screen',
        'phase2_summary',
        'phase2_default_vs_composed',
        'phase2_full_sweep',
        'phase3_final_compare',
        'latent_stats_summary',
        'visualization_index',
        'master_summary',
    }
    p3_summary_images = _filter_image_paths(
        {
            **image_paths,
            'master_summary': str(master_summary_table),
        },
        {
            'phase3_final_compare',
            'latent_stats_summary',
            'master_summary',
            *final_viz_keys,
        },
    )
    master_summary_images = _filter_balanced_master_image_paths(
        {**master_image_paths, 'master_summary': str(master_summary_table)},
        summary_table_keys | final_viz_keys,
    )
    _log_summary_run(
        flags,
        PHASE3_GROUP,
        'P3_Final_Summary',
        {
            f'phase3/moe_{FID_METRIC_KEY}': phase3_selected[FID_METRIC_KEY],
            f'phase3/naive_{FID_METRIC_KEY}': phase3_naive[FID_METRIC_KEY],
            'phase3/winner': final_selected['run_name'],
            'ablation/run_id': ablation_run_id,
            'ablation/project_name': flags.wandb.project,
            'analysis/packet_md': str(analysis_packet_md),
            'analysis/packet_json': str(analysis_packet_json),
        },
        image_paths=p3_summary_images,
        config={'selected_phase2_run': phase2_selected['run_name']},
    )
    _log_summary_run(
        flags,
        SUMMARY_GROUP,
        'Ablation_Master_Summary',
        {
            'ablation/run_id': ablation_run_id,
            'ablation/project_name': flags.wandb.project,
            'phase1/winning_gmm': phase1_winner['run_name'],
            'phase2/final_selected_config': phase2_selected['run_name'],
            'phase3/final_winner': final_selected['run_name'],
            'analysis/packet_md': str(analysis_packet_md),
            'analysis/packet_json': str(analysis_packet_json),
        },
        image_paths=master_summary_images,
        config={
            'ablation_run_id': ablation_run_id,
            'wandb_project': flags.wandb.project,
            'phase1_winner': phase1_winner['run_name'],
            'phase2_selected': phase2_selected['run_name'],
            'phase3_winner': final_selected['run_name'],
        },
    )

    # Cleanup
    if cleanup_immediately:
        for row in phase1b_rows:
            _cleanup_run_artifacts(row['run_dir'], keep_metrics=True, keep_figures=False, keep_checkpoint=False)
        for row in phase2_rows:
            _cleanup_run_artifacts(row['run_dir'], keep_metrics=True, keep_figures=False, keep_checkpoint=False)
        _cleanup_run_artifacts(composed_row['run_dir'], keep_metrics=True, keep_figures=False, keep_checkpoint=False)
        _cleanup_run_artifacts(phase3_dir / 'Top1_MoE', keep_metrics=True, keep_figures=False, keep_checkpoint=False)
        _cleanup_run_artifacts(phase3_dir / 'Naive_Reference', keep_metrics=True, keep_figures=False, keep_checkpoint=False)
