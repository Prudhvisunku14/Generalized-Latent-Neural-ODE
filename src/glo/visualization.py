"""Publication-style visual checks and prediction plots."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import load_split, FAMILY_TO_ID


def plot_gallery(data_root, experiment, out_file, n_examples=4):
    arrays, meta = load_split(data_root, experiment, 'train')
    selected = meta['families']
    fig, axes = plt.subplots(len(selected), n_examples, figsize=(3.3*n_examples, 3.0*len(selected)), squeeze=False)
    for row, family in enumerate(selected):
        pool = np.flatnonzero(arrays['family'] == FAMILY_TO_ID[family])[:n_examples]
        for col, idx in enumerate(pool):
            ax = axes[row, col]
            clean = arrays['xy'][idx]
            mask = arrays['default_obs_mask'][idx]
            noisy = arrays['noisy_xy'][idx]
            ax.plot(clean[:,0],clean[:,1], color='#1965a5', lw=1.9, label='Ground truth')
            ax.scatter(noisy[mask,0], noisy[mask,1], color='#e47b37', s=9, alpha=.85, label='Observed')
            ax.set_aspect('equal', adjustable='datalim')
            ax.grid(alpha=.12)
            ax.set_title(f'{family.capitalize()} / {col+1}', fontsize=10)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f'{experiment.replace("_"," ").title()} | clean trajectory + 30 noisy context observations', fontsize=13)
    fig.tight_layout(rect=(0,0,1,.965))
    out = Path(out_file); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_prediction(clean, pred, observed, time, cutoff, family_name, output_file):
    t = np.asarray(time)
    before = t <= cutoff
    future = t > cutoff
    fig, ax = plt.subplots(figsize=(6, 5.4))
    ax.plot(clean[:,0], clean[:,1], color='#2a9469', lw=2.4, label='Ground truth')
    ax.plot(pred[before,0], pred[before,1], color='#2578bc', lw=1.7, label='Reconstruction')
    ax.plot(pred[future,0], pred[future,1], color='#ed6849', lw=1.9, label='Forecast')
    ax.scatter(observed[:,0],observed[:,1], s=17, color='#27384c', alpha=.75, label='Noisy observations',zorder=3)
    ax.set_title(f'{family_name} | Latent ODE reconstruction and extrapolation')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.grid(alpha=.15)
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = Path(output_file); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=190)
    plt.close(fig)
    return out
