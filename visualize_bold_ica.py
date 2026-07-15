#!/usr/bin/env python3
"""Generate example parcel time series and ICA visualizations from a BIDS func run."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from sklearn.decomposition import FastICA


def find_sample_bold(bids_root: Path):
    func_files = list(bids_root.glob('**/*task-rest*_bold.nii*'))
    if not func_files:
        raise FileNotFoundError('No BOLD files found in BIDS root')
    return sorted(func_files)[0]


def make_grid_parcels(shape, mask, grid_shape=(4, 4, 4)):
    x_bins = np.linspace(0, shape[0], grid_shape[0] + 1, dtype=int)
    y_bins = np.linspace(0, shape[1], grid_shape[1] + 1, dtype=int)
    z_bins = np.linspace(0, shape[2], grid_shape[2] + 1, dtype=int)
    parcels = np.zeros(shape, dtype=int)
    idx = 0
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            for k in range(grid_shape[2]):
                idx += 1
                parcels[
                    x_bins[i]:x_bins[i + 1],
                    y_bins[j]:y_bins[j + 1],
                    z_bins[k]:z_bins[k + 1],
                ] = idx
    parcels[~mask] = 0
    return parcels


def average_parcel_timeseries(data, parcels):
    labels = np.unique(parcels)
    labels = labels[labels != 0]
    ts = []
    used_labels = []
    for lab in labels:
        vox = data[parcels == lab]
        if vox.shape[0] < 10:
            continue
        ts.append(vox.mean(axis=0))
        used_labels.append(lab)
    return np.vstack(ts), np.asarray(used_labels, dtype=int)


def plot_timeseries(timeseries, outpath, title, n_series=8):
    n_series = min(n_series, timeseries.shape[0])
    plt.figure(figsize=(10, 6))
    for i in range(n_series):
        plt.plot(timeseries[i], label=f'parcel {i + 1}')
    plt.xlabel('Volume')
    plt.ylabel('Signal (a.u.)')
    plt.title(title)
    plt.legend(loc='upper right', ncol=2, fontsize='small')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_ica_spatial(parcels, ica_components, outpath, component_idx=0):
    comp = ica_components[component_idx]
    spatial = np.zeros_like(parcels, dtype=float)
    labels = np.unique(parcels)
    labels = labels[labels != 0]
    for lab, weight in zip(labels, comp):
        spatial[parcels == lab] = weight
    mean_map = np.rot90(spatial[:, :, spatial.shape[2] // 2])
    plt.figure(figsize=(5, 5))
    plt.imshow(mean_map, cmap='coolwarm', origin='lower')
    plt.colorbar(label='ICA weight')
    plt.title(f'ICA component {component_idx + 1} spatial map\n(mid axial slice)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    bids_root = Path('AD_bids')
    bold_path = find_sample_bold(bids_root)
    print('Using BOLD file:', bold_path)

    bold_img = nib.load(str(bold_path))
    data = bold_img.get_fdata()
    print('data shape', data.shape)

    mean_vol = np.mean(data, axis=3)
    mask = mean_vol > np.percentile(mean_vol[mean_vol > 0], 10)
    parcels = make_grid_parcels(data.shape[:3], mask, grid_shape=(4, 4, 4))

    parcel_ts, parcel_labels = average_parcel_timeseries(data, parcels)

    print('parcels', parcel_ts.shape)
    out_dir = bids_root
    plot_timeseries(parcel_ts, out_dir / 'example_bold_parcel_timeseries.png', 'Example parcel-averaged BOLD time series', n_series=8)

    ica = FastICA(n_components=min(6, parcel_ts.shape[0]), random_state=0)
    ica_ts = ica.fit_transform(parcel_ts.T).T
    plot_timeseries(ica_ts, out_dir / 'example_bold_ica_timeseries.png', 'Example ICA component time series', n_series=6)

    plot_ica_spatial(parcels, ica.components_, out_dir / 'example_bold_ica_spatial.png', component_idx=0)

    print('Saved images:')
    print(' ', out_dir / 'example_bold_parcel_timeseries.png')
    print(' ', out_dir / 'example_bold_ica_timeseries.png')
    print(' ', out_dir / 'example_bold_ica_spatial.png')


if __name__ == '__main__':
    main()
