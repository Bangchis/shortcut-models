import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb
from matplotlib.lines import Line2D

from gmm_utils import load_gmm_stats
from moe1_ablation import (
    _compute_cluster_assignments,
    _compute_path_stats,
    _flatten_embeddings,
    _load_analysis_payload,
    _make_table_png,
    _markdown_table,
    _project_chunks,
    _reshape_last_three,
    _save_cluster_endpoint_projection,
    _save_cluster_path_projection,
    _save_confusion_heatmap,
    _save_entropy_hist,
    _save_q_alpha_heatmap,
    _save_usage_bar,
)


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT = 'moe1-naive-k-ablation-celeba256'
GMM_GROUP = 'MoE1_Naive_K_GMM'
TRAIN_GROUP = 'MoE1_Naive_K_Train'
SUMMARY_GROUP = 'MoE1_Naive_K_Summary'
PATH_PLOT_MAX_LINES = 256


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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_json_ready(payload), f, indent=2, sort_keys=True)


def _load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _sanitize_flag_value(value):
    if isinstance(value, bool):
        return '1' if value else '0'
    return str(value)


def _parse_k_list(raw_value):
    values = []
    seen = set()
    for item in str(raw_value).split(','):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError('--moe1_k_list values must be positive integers.')
        if value not in seen:
            values.append(value)
            seen.add(value)
    if not values:
        raise ValueError('--moe1_k_list must contain at least one K.')
    return values


def _make_run_id():
    return datetime.now().strftime('dt-%Y%m%d-%H%M%S')


def _resolve_project_name(flags):
    project = getattr(flags.wandb, 'project', None)
    if not project or project == 'shortcut':
        return DEFAULT_PROJECT
    return project


def _run_subprocess(cmd, label):
    print(f'[{label}] running:\n  ' + ' '.join(cmd))
    subprocess.run(cmd, cwd=str(ROOT_DIR), check=True)


def _extend_config_flags(args, prefix, values):
    for key, value in values.items():
        if value is None:
            continue
        args.append(f'--{prefix}.{key}={_sanitize_flag_value(value)}')


def _append_wandb_args(args, flags, group, name):
    args.extend([
        f'--wandb.project={flags.wandb.project}',
        f'--wandb.group={group}',
        f'--wandb.name={name}',
        f'--wandb.offline={_sanitize_flag_value(flags.wandb.offline)}',
    ])
    if flags.wandb.entity:
        args.append(f'--wandb.entity={flags.wandb.entity}')
    for key in ('service_wait', 'start_method', 'disable_stats', 'save_code'):
        if hasattr(flags.wandb, key):
            args.append(f'--wandb.{key}={_sanitize_flag_value(getattr(flags.wandb, key))}')


def _flag_present(flags, name):
    try:
        return bool(flags[name].present)
    except Exception:
        return False


def _metric_value(metrics, key):
    if not isinstance(metrics, dict):
        return None
    if key in metrics:
        return metrics[key]
    for section_name in ('inference', 'train'):
        section = metrics.get(section_name, {})
        if isinstance(section, dict) and key in section:
            return section[key]
    return None


def _fid_key(flags):
    return f'fid{int(flags.inference_timesteps)}_{int(flags.inference_generations)}'


def _run_gmm(flags, root, k):
    run_name = f'GMM_K{k:02d}'
    run_dir = root / 'gmm' / f'K{k:02d}'
    run_dir.mkdir(parents=True, exist_ok=True)
    gmm_path = run_dir / 'gmm_stats.npz'
    metrics_path = run_dir / 'metrics.json'
    cmd = [
        sys.executable,
        str(ROOT_DIR / 'data_prep.py'),
        f'--dataset_name={flags.dataset_name}',
        f'--batch_size={flags.batch_size}',
        f'--seed={flags.seed}',
        f'--gmm_num_modes={k}',
        f'--gmm_save_path={gmm_path}',
        f'--metrics_output_path={metrics_path}',
        f'--gmm_fit_samples={int(flags.moe1_gmm_fit_samples)}',
        f'--gmm_valid_samples={int(flags.moe1_gmm_valid_samples)}',
        '--gmm_keep_latent_cache=0',
        '--gmm_wandb_level=summary',
    ]
    if flags.tfds_data_dir:
        cmd.append(f'--tfds_data_dir={flags.tfds_data_dir}')
    _append_wandb_args(cmd, flags, GMM_GROUP, run_name)
    _run_subprocess(cmd, run_name)
    metrics = _load_json(metrics_path)
    return {
        'K': k,
        'run_name': run_name,
        'run_dir': str(run_dir),
        'gmm_stats_path': str(gmm_path),
        'metrics_path': str(metrics_path),
        'metrics': metrics,
    }


def _base_train_args(flags, run_dir, run_name, group):
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = run_dir / 'artifacts'
    metrics_path = run_dir / 'metrics.json'
    save_interval = int(flags.save_interval)
    if not bool(flags.moe1_keep_checkpoints):
        save_interval = max(save_interval, int(flags.max_steps) + 2)
    eval_visual_level = flags.eval_visual_level if _flag_present(flags, 'eval_visual_level') else 'none'
    eval_fid_generations = (
        int(flags.eval_fid_generations)
        if _flag_present(flags, 'eval_fid_generations')
        else 512
    )
    args = [
        sys.executable,
        str(ROOT_DIR / 'train.py'),
        '--mode=train',
        f'--dataset_name={flags.dataset_name}',
        f'--batch_size={flags.batch_size}',
        f'--seed={flags.seed}',
        f'--max_steps={int(flags.max_steps)}',
        f'--log_interval={int(flags.log_interval)}',
        f'--eval_interval={int(flags.eval_interval)}',
        f'--save_interval={save_interval}',
        '--run_final_inference=1',
        '--train_metrics_level=summary',
        f'--eval_visual_level={eval_visual_level}',
        f'--eval_fid_timesteps={flags.eval_fid_timesteps}',
        f'--eval_fid_generations={eval_fid_generations}',
        f'--inference_timesteps={int(flags.inference_timesteps)}',
        f'--inference_generations={int(flags.inference_generations)}',
        f'--inference_cfg_scale={float(flags.inference_cfg_scale)}',
        '--dump_source_stats=1',
        '--dump_flow_viz=1',
        f'--source_stats_samples={int(flags.source_stats_samples)}',
        f'--flow_viz_samples={int(flags.flow_viz_samples)}',
        '--save_x_render=0',
        f'--save_dir={artifact_dir}',
        f'--metrics_output_path={metrics_path}',
    ]
    if flags.tfds_data_dir:
        args.append(f'--tfds_data_dir={flags.tfds_data_dir}')
    if flags.fid_stats:
        args.append(f'--fid_stats={flags.fid_stats}')
    _append_wandb_args(args, flags, group, run_name)
    return args, artifact_dir, metrics_path


def _run_train(flags, root, run_name, role, model_overrides, k=None):
    run_dir_name = f'K{k:02d}' if k is not None else 'naive_reference'
    run_dir = root / 'train' / run_dir_name
    args, artifact_dir, metrics_path = _base_train_args(flags, run_dir, run_name, TRAIN_GROUP)
    combined_model = flags.model.to_dict()
    combined_model.update(model_overrides)
    _extend_config_flags(args, 'model', combined_model)
    if bool(flags.moe1_keep_checkpoints):
        args.append(f'--final_save_dir={run_dir / "final.pkl"}')
    _run_subprocess(args, run_name)
    metrics = _load_json(metrics_path)
    return {
        'K': k,
        'role': role,
        'run_name': run_name,
        'run_dir': str(run_dir),
        'artifact_dir': str(artifact_dir),
        'metrics_path': str(metrics_path),
        'metrics': metrics,
        'model_overrides': model_overrides,
    }


def _arr(payload, key):
    return _reshape_last_three(payload.get(key, np.zeros((0,), dtype=np.float32))).astype(np.float32, copy=False)


def _mean_same_index_distance(a, b):
    flat_a = _flatten_embeddings(a)
    flat_b = _flatten_embeddings(b)
    if flat_a.size == 0 or flat_b.size == 0:
        return None
    take = min(flat_a.shape[0], flat_b.shape[0])
    if take == 0:
        return None
    return float(np.mean(np.linalg.norm(flat_a[:take] - flat_b[:take], axis=-1)))


def _variance_mean(arr):
    flat = _flatten_embeddings(arr)
    if flat.size == 0:
        return None
    return float(np.mean(np.var(flat, axis=0)))


def _path_stats_basic(run_name, path_states):
    states = np.asarray(path_states, dtype=np.float32)
    if states.size == 0:
        return None
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
    return {
        'run_name': run_name,
        'split': 'path',
        'sample_count': int(states.shape[0]),
        'path_length_mean': float(np.mean(path_length)),
        'endpoint_displacement_mean': float(np.mean(endpoint_displacement)),
        'straightness_ratio_mean': float(np.mean(straightness)),
        'curvature_proxy_mean': float(np.mean(curvature)),
    }


def _save_conditioned_cluster_heatmap(conditioned_mode, assigned_cluster, num_modes, output_path):
    conditioned = np.asarray(conditioned_mode, dtype=np.int32)
    assigned = np.asarray(assigned_cluster, dtype=np.int32)
    take = min(conditioned.shape[0], assigned.shape[0])
    if take == 0:
        return None
    conditioned = conditioned[:take]
    assigned = assigned[:take]
    size = int(max(num_modes, conditioned.max() + 1, assigned.max() + 1))
    confusion = np.zeros((size, size), dtype=np.float32)
    for cond, cluster in zip(conditioned, assigned):
        confusion[int(cond), int(cluster)] += 1.0
    row_sums = np.maximum(confusion.sum(axis=1, keepdims=True), 1.0)
    confusion = confusion / row_sums
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion, cmap='viridis', aspect='auto', vmin=0.0, vmax=1.0)
    ax.set_title('Conditioned Mode vs Source Cluster')
    ax.set_xlabel('GMM cluster assigned to x_source')
    ax.set_ylabel('Conditioned one-hot mode')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _save_group_endpoint_projection(group_specs, output_path, title, method='pca'):
    prepared = []
    flat_chunks = []
    for spec in group_specs:
        flat = _flatten_embeddings(spec['points'])
        if flat.size == 0:
            continue
        take = min(int(spec.get('max_points', 512)), flat.shape[0])
        prepared.append({**spec, 'flat': flat[:take]})
        flat_chunks.append(flat[:take])
    if len(prepared) < 2:
        return None
    projected = _project_chunks(flat_chunks, method)
    fig, ax = plt.subplots(figsize=(9, 7))
    for spec, coords in zip(prepared, projected):
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=float(spec.get('size', 18)),
            alpha=float(spec.get('alpha', 0.65)),
            marker=spec.get('marker', 'o'),
            color=spec.get('color', 'black'),
            label=spec['name'],
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _save_group_path_projection(anchor_specs, path_specs, output_path, title, method='pca'):
    prepared_anchors = []
    anchor_chunks = []
    for spec in anchor_specs:
        flat = _flatten_embeddings(spec['points'])
        if flat.size == 0:
            continue
        take = min(int(spec.get('max_points', 256)), flat.shape[0])
        prepared_anchors.append({**spec, 'flat': flat[:take]})
        anchor_chunks.append(flat[:take])

    prepared_paths = []
    path_chunks = []
    for spec in path_specs:
        states = np.asarray(spec['path_states'], dtype=np.float32)
        if states.size == 0:
            continue
        take = min(int(spec.get('max_paths', PATH_PLOT_MAX_LINES)), states.shape[0])
        states = states[:take]
        prepared_paths.append({**spec, 'states': states})
        path_chunks.append(states.reshape((states.shape[0] * states.shape[1], -1)))

    if not prepared_paths:
        return None

    projected = _project_chunks(anchor_chunks + path_chunks, method)
    projected_anchors = projected[:len(prepared_anchors)]
    projected_paths = projected[len(prepared_anchors):]

    fig, ax = plt.subplots(figsize=(9, 7))
    for spec, coords in zip(prepared_paths, projected_paths):
        coords = coords.reshape((spec['states'].shape[0], spec['states'].shape[1], 2))
        color = spec.get('color', 'black')
        for path_idx in range(coords.shape[0]):
            ax.plot(
                coords[path_idx, :, 0],
                coords[path_idx, :, 1],
                color=color,
                alpha=float(spec.get('line_alpha', 0.13)),
                linewidth=0.9,
                linestyle=spec.get('linestyle', '-'),
            )
        ax.scatter(coords[:, 0, 0], coords[:, 0, 1], color=color, marker='o', s=10, alpha=0.35)
        ax.scatter(coords[:, -1, 0], coords[:, -1, 1], color=color, marker='x', s=16, alpha=0.60)

    for spec, coords in zip(prepared_anchors, projected_anchors):
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=float(spec.get('size', 16)),
            alpha=float(spec.get('alpha', 0.55)),
            marker=spec.get('marker', 'o'),
            color=spec.get('color', 'black'),
            label=spec['name'],
        )
    handles = [
        Line2D([0], [0], color=spec.get('color', 'black'), linestyle=spec.get('linestyle', '-'), label=f"{spec['name']} paths")
        for spec in prepared_paths
    ]
    anchor_handles = [
        Line2D([0], [0], marker=spec.get('marker', 'o'), color='none', markerfacecolor=spec.get('color', 'black'), label=spec['name'])
        for spec in prepared_anchors
    ]
    ax.legend(handles=anchor_handles + handles, loc='best', fontsize=8)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _extract_fid(metrics, fid_key):
    value = _metric_value(metrics, fid_key)
    return None if value is None else float(value)


def _summary_row_from_metrics(run, fid_key):
    metrics = run.get('metrics', {})
    train = metrics.get('train', {}) if isinstance(metrics, dict) else {}
    return {
        'run_name': run['run_name'],
        'role': run['role'],
        'K': run.get('K'),
        fid_key: _extract_fid(metrics, fid_key),
        'valid_loss': train.get('valid_loss'),
        'max_usage': train.get('max_usage'),
        'max_soft_usage': train.get('max_soft_usage'),
        'q_alpha_agreement': train.get('q_alpha_agreement'),
        'source_var_mean': train.get('source/var_mean'),
        'source_var_min': train.get('source/var_min'),
    }


def _render_moe_run(run, gmm_state, output_dir, fid_key, max_points):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = Path(run['artifact_dir']) / 'source_stats.npz'
    payload = _load_analysis_payload(payload_path)
    prior = _arr(payload, 'x0_prior')
    posterior = _arr(payload, 'x0_posterior')
    x_data = _arr(payload, 'x1_data')
    path_states = np.asarray(payload.get('flow_path_states', np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    _, prior_clusters = _compute_cluster_assignments(prior, gmm_state)
    _, posterior_clusters = _compute_cluster_assignments(posterior, gmm_state)
    _, data_clusters = _compute_cluster_assignments(x_data, gmm_state)

    image_paths = {}
    groups = [
        {'name': 'x_source_moe_prior', 'points': prior, 'clusters': prior_clusters, 'marker': 'o', 'alpha': 0.70, 'max_points': max_points},
        {'name': 'x_source_moe_posterior', 'points': posterior, 'clusters': posterior_clusters, 'marker': 'D', 'alpha': 0.70, 'max_points': max_points},
        {'name': 'x_data', 'points': x_data, 'clusters': data_clusters, 'marker': 's', 'alpha': 0.58, 'max_points': max_points},
    ]
    for method in ('pca', 'tsne'):
        path = _save_cluster_endpoint_projection(
            groups,
            output_dir / f'{run["run_name"]}_source_vs_data_{method}.png',
            f'{run["run_name"]}: x_source_moe vs x_data {method.upper()}',
            method=method,
        )
        if path:
            image_paths[f'{run["run_name"]}_source_vs_data_{method}'] = str(path)

    path_row = None
    path_clusters = np.zeros((0, 0), dtype=np.int32)
    if path_states.size > 0:
        path_row, path_clusters = _compute_path_stats(run['run_name'], path_states, gmm_state, data_clusters)
        if path_clusters.size > 0:
            path_specs = [{
                'name': run['run_name'],
                'path_states': path_states,
                'start_clusters': path_clusters[:, 0],
                'linestyle': '-',
                'max_paths': PATH_PLOT_MAX_LINES,
                'line_alpha': 0.14,
            }]
            for method in ('pca', 'tsne'):
                path = _save_cluster_path_projection(
                    groups,
                    path_specs,
                    output_dir / f'{run["run_name"]}_ode_paths_{method}.png',
                    f'{run["run_name"]}: ODE paths {method.upper()}',
                    method=method,
                )
                if path:
                    image_paths[f'{run["run_name"]}_ode_paths_{method}'] = str(path)

    alignment = {}
    if 'q_posterior' in payload and 'alpha_posterior' in payload:
        q = np.asarray(payload['q_posterior'], dtype=np.float32)
        alpha = np.asarray(payload['alpha_posterior'], dtype=np.float32)
        q_argmax = np.argmax(q, axis=-1)
        alpha_argmax = np.argmax(alpha, axis=-1)
        take = min(q_argmax.shape[0], alpha_argmax.shape[0], posterior_clusters.shape[0])
        if take > 0:
            alignment['q_alpha_agreement_eval'] = float(np.mean(q_argmax[:take] == alpha_argmax[:take]))
            alignment['q_source_cluster_agreement'] = float(np.mean(q_argmax[:take] == posterior_clusters[:take]))
            alignment['alpha_source_cluster_agreement'] = float(np.mean(alpha_argmax[:take] == posterior_clusters[:take]))
        image_paths[f'{run["run_name"]}_q_vs_alpha_heatmap'] = str(_save_q_alpha_heatmap(
            q,
            alpha,
            output_dir / f'{run["run_name"]}_q_vs_alpha_heatmap.png',
        ))
        image_paths[f'{run["run_name"]}_router_usage_hist'] = str(_save_usage_bar(
            alpha,
            output_dir / f'{run["run_name"]}_router_usage_hist.png',
        ))
        image_paths[f'{run["run_name"]}_router_entropy_hist'] = str(_save_entropy_hist(
            alpha,
            output_dir / f'{run["run_name"]}_router_entropy_hist.png',
        ))
        if 'conditioned_mode' in payload:
            conditioned = np.asarray(payload['conditioned_mode'], dtype=np.int32)
            image_paths[f'{run["run_name"]}_condition_vs_router_confusion'] = str(_save_confusion_heatmap(
                conditioned,
                alpha,
                output_dir / f'{run["run_name"]}_condition_vs_router_confusion.png',
            ))
            path = _save_conditioned_cluster_heatmap(
                conditioned,
                posterior_clusters,
                int(run['K']),
                output_dir / f'{run["run_name"]}_condition_vs_source_cluster.png',
            )
            if path:
                image_paths[f'{run["run_name"]}_condition_vs_source_cluster'] = str(path)
                take = min(conditioned.shape[0], posterior_clusters.shape[0])
                if take > 0:
                    alignment['conditioned_source_cluster_agreement'] = float(
                        np.mean(conditioned[:take] == posterior_clusters[:take])
                    )

    summary = _summary_row_from_metrics(run, fid_key)
    summary.update({
        'source_prior_to_data_distance': _mean_same_index_distance(prior, x_data),
        'source_posterior_to_data_distance': _mean_same_index_distance(posterior, x_data),
        'source_prior_variance_mean': _variance_mean(prior),
        'source_posterior_variance_mean': _variance_mean(posterior),
        **alignment,
    })
    if path_row:
        summary.update({
            'path_length_mean': path_row.get('path_length_mean'),
            'endpoint_displacement_mean': path_row.get('endpoint_displacement_mean'),
            'straightness_ratio_mean': path_row.get('straightness_ratio_mean'),
            'curvature_proxy_mean': path_row.get('curvature_proxy_mean'),
            'cluster_switch_count_mean': path_row.get('cluster_switch_count_mean'),
            'final_cluster_agreement_with_data': path_row.get('final_cluster_agreement_with_data'),
        })

    return {
        'run': run,
        'payload': payload,
        'prior': prior,
        'posterior': posterior,
        'x_data': x_data,
        'path_states': path_states,
        'summary': summary,
        'path_row': path_row,
        'image_paths': image_paths,
    }


def _render_naive_run(run, output_dir, fid_key, max_points):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = Path(run['artifact_dir']) / 'source_stats.npz'
    payload = _load_analysis_payload(payload_path)
    prior = _arr(payload, 'x0_prior')
    x_data = _arr(payload, 'x1_data')
    path_states = np.asarray(payload.get('flow_path_states', np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    image_paths = {}
    groups = [
        {'name': 'x_source_naive', 'points': prior, 'marker': 'o', 'color': 'black', 'alpha': 0.64, 'max_points': max_points},
        {'name': 'x_data', 'points': x_data, 'marker': 's', 'color': '0.55', 'alpha': 0.54, 'max_points': max_points},
    ]
    for method in ('pca', 'tsne'):
        path = _save_group_endpoint_projection(
            groups,
            output_dir / f'{run["run_name"]}_source_vs_data_{method}.png',
            f'{run["run_name"]}: x_source_naive vs x_data {method.upper()}',
            method=method,
        )
        if path:
            image_paths[f'{run["run_name"]}_source_vs_data_{method}'] = str(path)

    path_row = _path_stats_basic(run['run_name'], path_states)
    if path_states.size > 0:
        path_specs = [{
            'name': 'x_source_naive',
            'path_states': path_states,
            'color': 'black',
            'linestyle': '--',
            'max_paths': PATH_PLOT_MAX_LINES,
            'line_alpha': 0.12,
        }]
        for method in ('pca', 'tsne'):
            path = _save_group_path_projection(
                groups,
                path_specs,
                output_dir / f'{run["run_name"]}_ode_paths_{method}.png',
                f'{run["run_name"]}: ODE paths {method.upper()}',
                method=method,
            )
            if path:
                image_paths[f'{run["run_name"]}_ode_paths_{method}'] = str(path)

    summary = _summary_row_from_metrics(run, fid_key)
    summary.update({
        'source_prior_to_data_distance': _mean_same_index_distance(prior, x_data),
        'source_prior_variance_mean': _variance_mean(prior),
    })
    if path_row:
        summary.update({
            'path_length_mean': path_row.get('path_length_mean'),
            'endpoint_displacement_mean': path_row.get('endpoint_displacement_mean'),
            'straightness_ratio_mean': path_row.get('straightness_ratio_mean'),
            'curvature_proxy_mean': path_row.get('curvature_proxy_mean'),
        })
    return {
        'run': run,
        'payload': payload,
        'prior': prior,
        'x_data': x_data,
        'path_states': path_states,
        'summary': summary,
        'path_row': path_row,
        'image_paths': image_paths,
    }


def _render_combined(moe_analyses, naive_analysis, output_dir, max_points):
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = {}
    colors = plt.cm.get_cmap('tab10', max(len(moe_analyses), 1))
    data_points = None
    if moe_analyses:
        data_points = moe_analyses[0]['x_data']
    elif naive_analysis is not None:
        data_points = naive_analysis['x_data']
    groups = []
    if data_points is not None:
        groups.append({
            'name': 'x_data',
            'points': data_points,
            'marker': 's',
            'color': '0.55',
            'alpha': 0.48,
            'max_points': max_points,
        })
    if naive_analysis is not None:
        groups.append({
            'name': 'x_source_naive',
            'points': naive_analysis['prior'],
            'marker': 'o',
            'color': 'black',
            'alpha': 0.55,
            'max_points': max_points,
        })
    for idx, analysis in enumerate(moe_analyses):
        groups.append({
            'name': f'x_source_moe_K{analysis["run"]["K"]}',
            'points': analysis['prior'],
            'marker': 'o',
            'color': colors(idx),
            'alpha': 0.62,
            'max_points': max_points,
        })
    for method in ('pca', 'tsne'):
        path = _save_group_endpoint_projection(
            groups,
            output_dir / f'combined_source_vs_data_{method}.png',
            f'Combined source distributions {method.upper()}',
            method=method,
        )
        if path:
            image_paths[f'combined_source_vs_data_{method}'] = str(path)

    path_specs = []
    if naive_analysis is not None and naive_analysis['path_states'].size > 0:
        path_specs.append({
            'name': 'naive',
            'path_states': naive_analysis['path_states'],
            'color': 'black',
            'linestyle': '--',
            'max_paths': PATH_PLOT_MAX_LINES,
            'line_alpha': 0.10,
        })
    for idx, analysis in enumerate(moe_analyses):
        if analysis['path_states'].size == 0:
            continue
        path_specs.append({
            'name': f'K{analysis["run"]["K"]}',
            'path_states': analysis['path_states'],
            'color': colors(idx),
            'linestyle': '-',
            'max_paths': PATH_PLOT_MAX_LINES,
            'line_alpha': 0.10,
        })
    if path_specs:
        for method in ('pca', 'tsne'):
            path = _save_group_path_projection(
                groups,
                path_specs,
                output_dir / f'combined_ode_paths_{method}.png',
                f'Combined ODE paths {method.upper()}',
                method=method,
            )
            if path:
                image_paths[f'combined_ode_paths_{method}'] = str(path)
    return image_paths


def _write_analysis_packet(root, run_id, summary_rows, image_paths, fid_key):
    md_path = root / 'analysis_packet.md'
    json_path = root / 'master_summary.json'
    columns = [
        'run_name',
        'role',
        'K',
        fid_key,
        'valid_loss',
        'q_alpha_agreement',
        'q_alpha_agreement_eval',
        'q_source_cluster_agreement',
        'alpha_source_cluster_agreement',
        'conditioned_source_cluster_agreement',
        'source_prior_to_data_distance',
        'source_posterior_to_data_distance',
        'straightness_ratio_mean',
        'curvature_proxy_mean',
    ]
    md = [
        '# moe1-naive-k-ablation analysis packet',
        '',
        f'- run_id: `{run_id}`',
        f'- ranking_metric: `{fid_key}`',
        '',
        '## Summary',
        '',
        _markdown_table(summary_rows, columns),
        '',
        '## Visualization Files',
        '',
        _markdown_table(
            [{'name': key, 'path': value} for key, value in sorted(image_paths.items())],
            ['name', 'path'],
        ),
        '',
        f'Machine-readable summary: `{json_path}`',
    ]
    md_path.write_text('\n'.join(md) + '\n', encoding='utf-8')
    return json_path, md_path


def _log_summary(flags, summary_rows, image_paths, root, fid_key, run_id):
    best_moe = None
    moe_rows = [row for row in summary_rows if row.get('role') == 'moe']
    if moe_rows:
        best_moe = min(
            moe_rows,
            key=lambda row: float(row.get(fid_key, float('inf')) if row.get(fid_key) is not None else float('inf')),
        )
    summary_metrics = {
        'ablation/run_id': run_id,
        'ablation/root': str(root),
        'ablation/fid_key': fid_key,
    }
    if best_moe is not None:
        summary_metrics['ablation/best_moe_run'] = best_moe['run_name']
        summary_metrics['ablation/best_moe_k'] = int(best_moe['K'])
        if best_moe.get(fid_key) is not None:
            summary_metrics[f'ablation/best_moe_{fid_key}'] = float(best_moe[fid_key])
    naive_row = next((row for row in summary_rows if row.get('role') == 'naive'), None)
    if naive_row and naive_row.get(fid_key) is not None:
        summary_metrics[f'ablation/naive_{fid_key}'] = float(naive_row[fid_key])

    os.environ.setdefault('WANDB__SERVICE_WAIT', str(getattr(flags.wandb, 'service_wait', 300)))
    run = wandb.init(
        project=flags.wandb.project,
        entity=flags.wandb.entity,
        group=SUMMARY_GROUP,
        name='moe1_naive_k_ablation_summary',
        config={
            'k_list': [row.get('K') for row in summary_rows if row.get('role') == 'moe'],
            'source_var_target_std': flags.model['source_var_target_std'],
            'fid_key': fid_key,
        },
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
    log_payload = {}
    for key, value in image_paths.items():
        if value and os.path.exists(value):
            log_payload[key] = wandb.Image(value)
    if log_payload:
        wandb.log(log_payload)
    run.finish()


def run(flags):
    if not flags.fid_stats:
        raise ValueError('--fid_stats is required for mode=moe1-naive-k-ablation.')
    flags.wandb.project = _resolve_project_name(flags)
    k_list = _parse_k_list(flags.moe1_k_list)
    run_id = _make_run_id()
    root = Path(flags.moe1_root_dir) / run_id
    root.mkdir(parents=True, exist_ok=True)
    _write_json(
        root / 'ablation_context.json',
        {
            'run_id': run_id,
            'mode': 'moe1-naive-k-ablation',
            'k_list': k_list,
            'dataset_name': flags.dataset_name,
            'tfds_data_dir': flags.tfds_data_dir,
            'fid_stats': flags.fid_stats,
            'max_steps': int(flags.max_steps),
            'keep_checkpoints': bool(flags.moe1_keep_checkpoints),
            'source_var_target_std': flags.model['source_var_target_std'],
        },
    )

    gmm_rows = []
    train_rows = []
    for k in k_list:
        gmm_row = _run_gmm(flags, root, k)
        gmm_rows.append(gmm_row)
        overrides = flags.model.to_dict()
        overrides.update({
            'train_type': 'naive-moe-source',
            'gmm_stats_path': gmm_row['gmm_stats_path'],
            'gmm_num_modes': k,
            'use_stable_vae': 1,
        })
        train_rows.append(_run_train(
            flags,
            root,
            f'moe1_naive_k{k:02d}',
            'moe',
            overrides,
            k=k,
        ))

    naive_row = None
    if bool(flags.moe1_train_naive_ref):
        naive_overrides = flags.model.to_dict()
        naive_overrides.update({
            'train_type': 'naive',
            'gmm_stats_path': '',
        })
        naive_row = _run_train(
            flags,
            root,
            'naive_reference',
            'naive',
            naive_overrides,
            k=None,
        )

    fid_key = _fid_key(flags)
    max_points = max(1, int(flags.source_stats_samples))
    image_paths = {}
    moe_analyses = []
    for train_row, gmm_row in zip(train_rows, gmm_rows):
        gmm_state = load_gmm_stats(gmm_row['gmm_stats_path'])
        analysis = _render_moe_run(
            train_row,
            gmm_state,
            root / 'visualizations' / f'K{train_row["K"]:02d}',
            fid_key,
            max_points,
        )
        moe_analyses.append(analysis)
        image_paths.update(analysis['image_paths'])

    naive_analysis = None
    if naive_row is not None:
        naive_analysis = _render_naive_run(
            naive_row,
            root / 'visualizations' / 'naive_reference',
            fid_key,
            max_points,
        )
        image_paths.update(naive_analysis['image_paths'])

    combined_paths = _render_combined(
        moe_analyses,
        naive_analysis,
        root / 'visualizations' / 'combined',
        max_points,
    )
    image_paths.update(combined_paths)

    summary_rows = [analysis['summary'] for analysis in moe_analyses]
    if naive_analysis is not None:
        summary_rows.append(naive_analysis['summary'])

    summary_columns = [
        'run_name',
        'role',
        'K',
        fid_key,
        'valid_loss',
        'max_usage',
        'source_var_mean',
        'source_var_min',
        'q_alpha_agreement',
        'q_alpha_agreement_eval',
        'q_source_cluster_agreement',
        'alpha_source_cluster_agreement',
        'conditioned_source_cluster_agreement',
        'source_prior_to_data_distance',
        'source_posterior_to_data_distance',
        'straightness_ratio_mean',
        'curvature_proxy_mean',
    ]
    summary_table = _make_table_png(
        summary_rows,
        summary_columns,
        'MoE1 naive K ablation summary',
        root / 'master_summary.png',
    )
    image_paths['master_summary'] = str(summary_table)

    path_rows = [analysis['path_row'] for analysis in moe_analyses if analysis.get('path_row')]
    if naive_analysis is not None and naive_analysis.get('path_row') is not None:
        path_rows.append(naive_analysis['path_row'])
    if path_rows:
        path_table = _make_table_png(
            path_rows,
            [
                'run_name',
                'sample_count',
                'path_length_mean',
                'endpoint_displacement_mean',
                'straightness_ratio_mean',
                'curvature_proxy_mean',
                'cluster_switch_count_mean',
                'final_cluster_agreement_with_data',
            ],
            'ODE path stats summary',
            root / 'path_stats_summary.png',
        )
        image_paths['path_stats_summary'] = str(path_table)

    _write_json(
        root / 'master_summary.json',
        {
            'run_id': run_id,
            'fid_key': fid_key,
            'gmm_rows': gmm_rows,
            'summary_rows': summary_rows,
            'image_paths': image_paths,
        },
    )
    analysis_packet_json, analysis_packet_md = _write_analysis_packet(root, run_id, summary_rows, image_paths, fid_key)
    print(f'Analysis packet for copy/paste: {analysis_packet_md}')
    _log_summary(flags, summary_rows, image_paths, root, fid_key, run_id)
    print(f'moe1-naive-k-ablation complete. Outputs: {root}')
