import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import wandb
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils.stable_vae import StableVAE


PROJECT_NAME = 'moe1-ablation-celeba256'
PHASE1_GROUP = 'Phase_1A_GMM'
PHASE1B_GROUP = 'Phase_1B_Screen'
PHASE2_GROUP = 'Phase_2_Sweep'
PHASE2_COMPOSED_GROUP = 'Phase_2_Composed'
PHASE3_GROUP = 'Phase_3_Final'
ROOT_DIR = Path(__file__).resolve().parent


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


def _extend_config_flags(args, prefix, values):
    for key, value in values.items():
        if value is None:
            continue
        args.append(f'--{prefix}.{key}={_sanitize_flag_value(value)}')


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
    })
    return overrides


def _base_train_args(flags, metrics_output_path, save_dir, max_steps, wandb_group, wandb_name):
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
        f'--eval_interval={max_steps + 1}',
        f'--save_interval={max_steps + 1}',
        f'--save_dir={save_dir}',
        f'--metrics_output_path={metrics_output_path}',
        f'--inference_timesteps=128',
        f'--inference_generations=4096',
        f'--wandb.project={PROJECT_NAME}',
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
    _extend_config_flags(args, 'model', flags.model.to_dict())
    return args


def _run_train_screen(flags, run_name, group, run_dir, max_steps, model_overrides, dump_source_stats=False, final_save=False):
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / 'metrics.json'
    save_dir = run_dir / 'artifacts'
    args = _base_train_args(flags, metrics_path, save_dir, max_steps, group, run_name)
    _extend_config_flags(args, 'model', model_overrides)
    if dump_source_stats:
        args.append('--dump_source_stats=1')
        args.append('--source_stats_samples=4096')
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
        f'--wandb.project={PROJECT_NAME}',
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
    fid = row.get('inference', {}).get('fid128_4096')
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
        float(row.get('fid128_4096', row.get('inference', {}).get('fid128_4096', float('inf')))),
        float(row.get('valid_loss', row.get('train', {}).get('valid_loss', float('inf')))),
        float(row.get('max_usage', row.get('train', {}).get('max_usage', float('inf')))),
    )


def _decorate_run_row(row, **metadata):
    train = row.get('train', {})
    inference = row.get('inference', {})
    row['fid128_4096'] = inference.get('fid128_4096')
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


def _select_top3_gmms(rows):
    survivors = [row for row in rows if row['valid']]
    if len(survivors) < 3:
        raise ValueError(f'Phase 1A produced only {len(survivors)} valid GMMs; need at least 3 to continue.')
    survivors.sort(key=lambda row: (
        row['valid_nll'],
        -row['occupancy_entropy'] / max(np.log(max(row['gmm_num_modes'], 2)), 1e-8),
        row['max_component_fraction'],
    ))
    return survivors[:3]


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


def _decode_latents(latents, num_images=16):
    latents = _reshape_last_three(latents)
    if latents.size == 0:
        return np.zeros((0,))
    vae = StableVAE.create()
    decode = jax.jit(vae.decode)
    batch = jnp.asarray(latents[:num_images])
    decoded = np.array(decode(batch))
    return decoded


def _save_source_diagnostics(moe_npz_path, naive_npz_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    moe = np.load(moe_npz_path)
    naive = np.load(naive_npz_path)

    moe_prior = _reshape_last_three(moe['x0_prior'])
    moe_post = _reshape_last_three(moe['x0_posterior']) if 'x0_posterior' in moe else np.zeros((0,))
    x1_data = _reshape_last_three(moe['x1_data']) if 'x1_data' in moe else np.zeros((0,))
    naive_prior = _reshape_last_three(naive['x0_prior'])

    figure_paths = {}
    figure_paths['naive_source_grid'] = _save_image_grid(
        _decode_latents(naive_prior),
        output_dir / 'naive_source_grid.png',
        'Naive Source Grid',
    )
    figure_paths['moe_prior_grid'] = _save_image_grid(
        _decode_latents(moe_prior),
        output_dir / 'moe_prior_grid.png',
        'MoE Prior Source Grid',
    )
    if moe_post.size > 0:
        figure_paths['moe_posterior_grid'] = _save_image_grid(
            _decode_latents(moe_post),
            output_dir / 'moe_posterior_grid.png',
            'MoE Posterior Source Grid',
        )

    groups = []
    labels = []
    for name, arr in [
        ('naive_prior', naive_prior),
        ('moe_prior', moe_prior),
        ('moe_posterior', moe_post),
        ('x1_data', x1_data),
    ]:
        if arr.size == 0:
            continue
        take = min(512, arr.shape[0])
        groups.append(arr[:take].reshape((take, -1)))
        labels.extend([name] * take)
    if groups:
        stacked = np.concatenate(groups, axis=0)
        pca = PCA(n_components=2, random_state=0)
        coords = pca.fit_transform(stacked)
        fig, ax = plt.subplots(figsize=(7, 6))
        start = 0
        unique_labels = []
        for group_name, arr in zip(['naive_prior', 'moe_prior', 'moe_posterior', 'x1_data'], [naive_prior, moe_prior, moe_post, x1_data]):
            if arr.size == 0:
                continue
            take = min(512, arr.shape[0])
            unique_labels.append(group_name)
            ax.scatter(coords[start:start + take, 0], coords[start:start + take, 1], s=8, alpha=0.6, label=group_name)
            start += take
        ax.set_title('Source/Data PCA')
        ax.legend(loc='best')
        fig.tight_layout()
        pca_path = output_dir / 'source_pca.png'
        fig.savefig(pca_path, dpi=150)
        plt.close(fig)
        figure_paths['source_pca'] = pca_path

        tsne_take = min(1024, stacked.shape[0])
        tsne = TSNE(n_components=2, random_state=0, init='pca', learning_rate='auto')
        tsne_coords = tsne.fit_transform(stacked[:tsne_take])
        fig, ax = plt.subplots(figsize=(7, 6))
        start = 0
        for group_name, arr in zip(['naive_prior', 'moe_prior', 'moe_posterior', 'x1_data'], [naive_prior, moe_prior, moe_post, x1_data]):
            if arr.size == 0 or start >= tsne_take:
                continue
            take = min(512, arr.shape[0], tsne_take - start)
            ax.scatter(tsne_coords[start:start + take, 0], tsne_coords[start:start + take, 1], s=8, alpha=0.6, label=group_name)
            start += take
        ax.set_title('Source/Data t-SNE')
        ax.legend(loc='best')
        fig.tight_layout()
        tsne_path = output_dir / 'source_tsne.png'
        fig.savefig(tsne_path, dpi=150)
        plt.close(fig)
        figure_paths['source_tsne'] = tsne_path

    if 'q_posterior' in moe and 'alpha_posterior' in moe:
        q_mean = np.mean(moe['q_posterior'], axis=0)
        alpha_mean = np.mean(moe['alpha_posterior'], axis=0)
        heatmap = np.stack([q_mean, alpha_mean], axis=0)
        fig, ax = plt.subplots(figsize=(8, 3))
        im = ax.imshow(heatmap, cmap='viridis', aspect='auto')
        ax.set_yticks([0, 1], labels=['mean q', 'mean alpha'])
        ax.set_title('q vs alpha heatmap')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        heatmap_path = output_dir / 'q_vs_alpha_heatmap.png'
        fig.tight_layout()
        fig.savefig(heatmap_path, dpi=150)
        plt.close(fig)
        figure_paths['q_vs_alpha_heatmap'] = heatmap_path

        expert_argmax = np.argmax(moe['alpha_posterior'], axis=-1)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(expert_argmax, bins=np.arange(expert_argmax.max() + 2) - 0.5, rwidth=0.8)
        ax.set_title('Expert Usage Histogram')
        ax.set_xlabel('Expert')
        usage_path = output_dir / 'expert_usage_hist.png'
        fig.tight_layout()
        fig.savefig(usage_path, dpi=150)
        plt.close(fig)
        figure_paths['expert_usage_hist'] = usage_path

        router_entropy = -np.sum(
            moe['alpha_posterior'] * np.log(np.maximum(moe['alpha_posterior'], 1e-8)),
            axis=-1,
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(router_entropy, bins=30)
        ax.set_title('Router Entropy Histogram')
        ax.set_xlabel('Entropy')
        entropy_path = output_dir / 'router_entropy_hist.png'
        fig.tight_layout()
        fig.savefig(entropy_path, dpi=150)
        plt.close(fig)
        figure_paths['router_entropy_hist'] = entropy_path

        if 'conditioned_mode' in moe:
            conditioned = moe['conditioned_mode'].astype(np.int32)
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
            confusion_path = output_dir / 'mode_vs_router_confusion.png'
            fig.tight_layout()
            fig.savefig(confusion_path, dpi=150)
            plt.close(fig)
            figure_paths['mode_vs_router_confusion'] = confusion_path

    return {k: str(v) for k, v in figure_paths.items() if v is not None}


def _log_summary_run(flags, group, name, summary_metrics, image_paths=None, config=None):
    run = wandb.init(
        project=PROJECT_NAME,
        entity=flags.wandb.entity,
        group=group,
        name=name,
        config=config or {},
        mode='offline' if flags.wandb.offline else 'online',
        save_code=False,
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


def _cleanup_run_artifacts(run_dir, keep_metrics=True, keep_figures=True, keep_checkpoint=False):
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
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except FileNotFoundError:
                pass


def run(flags):
    if not flags.fid_stats:
        raise ValueError('--fid_stats is required for mode=moe1-ablation.')
    root = Path('/kaggle/working/moe1_ablation')
    root.mkdir(parents=True, exist_ok=True)

    # Phase 1A
    phase1a_dir = root / 'phase1a_gmm'
    gmm_rows = [_run_gmm(flags, phase1a_dir, k) for k in [2, 4, 8, 16, 24, 32]]
    top3_gmms = _select_top3_gmms(gmm_rows)
    _write_json(phase1a_dir / 'ranking.json', {'all': gmm_rows, 'top3': top3_gmms})

    # Phase 1B
    phase1b_dir = root / 'phase1b_screen'
    base_overrides = _base_model_overrides(flags)
    phase1b_rows = []
    for row in top3_gmms:
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
            )
        phase1b_rows.append(
            _decorate_run_row(
                phase1b_row,
                phase='phase1b',
                **{
                    'phase1b/gmm_num_modes': row['gmm_num_modes'],
                    'phase1b/is_winner': False,
                },
            )
        )
    phase1_winner = _select_phase1_winner(phase1b_rows)
    _decorate_run_row(phase1_winner, **{'phase1b/is_winner': True})
    _write_json(phase1b_dir / 'ranking.json', {'all': phase1b_rows, 'winner': phase1_winner})

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
            10000,
            run_overrides,
        )
        phase2_rows.append(
            _decorate_run_row(
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
        10000,
        composed_overrides,
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

    default_row = next(row for row in phase2_rows if row['run_name'] == 'M00_Default')
    composed_pass = (
        composed_row['valid']
        and composed_row['inference']['fid128_4096'] <= default_row['inference']['fid128_4096'] * 1.03
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
                'fid128_4096': best_balance['fid128_4096'],
            },
            {
                'winner_type': 'entropy',
                'run_name': best_entropy['run_name'],
                'fid128_4096': best_entropy['fid128_4096'],
            },
            {
                'winner_type': 'variance',
                'run_name': best_variance['run_name'],
                'fid128_4096': best_variance['fid128_4096'],
            },
            {
                'winner_type': 'composed',
                'run_name': composed_row['run_name'],
                'fid128_4096': composed_row['fid128_4096'],
            },
            {
                'winner_type': 'selected',
                'run_name': phase2_selected['run_name'],
                'fid128_4096': phase2_selected['fid128_4096'],
            },
        ],
        ['winner_type', 'run_name', 'fid128_4096'],
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
                'fid128_4096': default_row['fid128_4096'],
                'valid_loss': default_row['valid_loss'],
                'max_usage': default_row['max_usage'],
                'max_soft_usage': default_row['max_soft_usage'],
                'q_alpha_agreement': default_row['q_alpha_agreement'],
                'var_x0_mean': default_row['var_x0_mean'],
            },
            {
                'run_name': composed_row['run_name'],
                'fid128_4096': composed_row['fid128_4096'],
                'valid_loss': composed_row['valid_loss'],
                'max_usage': composed_row['max_usage'],
                'max_soft_usage': composed_row['max_soft_usage'],
                'q_alpha_agreement': composed_row['q_alpha_agreement'],
                'var_x0_mean': composed_row['var_x0_mean'],
            },
        ],
        [
            'run_name',
            'fid128_4096',
            'valid_loss',
            'max_usage',
            'max_soft_usage',
            'q_alpha_agreement',
            'var_x0_mean',
        ],
        'M00_Default vs M17_Composed',
        phase2_dir / 'phase2_default_vs_composed.png',
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
            row['fid128_4096'],
            row['valid_loss'],
        ),
    )
    figures_dir = phase3_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    image_paths = {}
    moe_x_render = Path(phase3_selected['artifact_dir']) / 'x_render.npy'
    naive_x_render = Path(phase3_naive['artifact_dir']) / 'x_render.npy'
    if moe_x_render.exists():
        image_paths['moe_final_grid'] = str(_save_image_grid(np.load(moe_x_render), figures_dir / 'moe_final_grid.png', 'Top1 MoE Final Grid'))
    if naive_x_render.exists():
        image_paths['naive_final_grid'] = str(_save_image_grid(np.load(naive_x_render), figures_dir / 'naive_final_grid.png', 'Naive Final Grid'))
    moe_source_stats = Path(phase3_selected['artifact_dir']) / 'source_stats.npz'
    naive_source_stats = Path(phase3_naive['artifact_dir']) / 'source_stats.npz'
    if moe_source_stats.exists() and naive_source_stats.exists():
        image_paths.update(_save_source_diagnostics(moe_source_stats, naive_source_stats, figures_dir / 'source'))
    if moe_source_stats.exists():
        shutil.copy2(moe_source_stats, Path(phase3_selected['run_dir']) / 'source_stats.npz')
    if naive_source_stats.exists():
        shutil.copy2(naive_source_stats, Path(phase3_naive['run_dir']) / 'source_stats.npz')
    phase3_compare_table = _make_table_png(
        [
            {
                'run_name': phase3_selected['run_name'],
                'fid128_4096': phase3_selected['fid128_4096'],
                'valid_loss': phase3_selected['valid_loss'],
                'latency_128': phase3_selected['inference'].get('latency_128'),
                'throughput_128': phase3_selected['inference'].get('throughput_128'),
            },
            {
                'run_name': phase3_naive['run_name'],
                'fid128_4096': phase3_naive['fid128_4096'],
                'valid_loss': phase3_naive['valid_loss'],
                'latency_128': phase3_naive['inference'].get('latency_128'),
                'throughput_128': phase3_naive['inference'].get('throughput_128'),
            },
        ],
        ['run_name', 'fid128_4096', 'valid_loss', 'latency_128', 'throughput_128'],
        'Phase 3 Final Duel',
        figures_dir / 'phase3_final_compare.png',
    )
    image_paths['phase3_final_compare'] = str(phase3_compare_table)

    _write_json(
        phase3_dir / 'ranking.json',
        {
            'moe': phase3_selected,
            'naive': phase3_naive,
            'winner': final_selected['run_name'],
        },
    )
    _log_summary_run(
        flags,
        PHASE3_GROUP,
        'P3_Final_Summary',
        {
            'phase3/moe_fid128_4096': phase3_selected['fid128_4096'],
            'phase3/naive_fid128_4096': phase3_naive['fid128_4096'],
            'phase3/winner': final_selected['run_name'],
        },
        image_paths=image_paths,
        config={'selected_phase2_run': phase2_selected['run_name']},
    )

    # Cleanup
    for row in phase1b_rows:
        _cleanup_run_artifacts(row['run_dir'], keep_metrics=True, keep_figures=False, keep_checkpoint=False)
    for row in phase2_rows:
        _cleanup_run_artifacts(row['run_dir'], keep_metrics=True, keep_figures=False, keep_checkpoint=False)
    _cleanup_run_artifacts(composed_row['run_dir'], keep_metrics=True, keep_figures=False, keep_checkpoint=False)
    keep_moe_ckpt = final_selected['run_name'] == 'P3_Final_Top1MoE'
    keep_naive_ckpt = final_selected['run_name'] == 'P3_Final_NaiveRef'
    _cleanup_run_artifacts(phase3_dir / 'Top1_MoE', keep_metrics=True, keep_figures=False, keep_checkpoint=keep_moe_ckpt)
    _cleanup_run_artifacts(phase3_dir / 'Naive_Reference', keep_metrics=True, keep_figures=False, keep_checkpoint=keep_naive_ckpt)
