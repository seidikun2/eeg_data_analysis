"""CLI batch, figuras de publicação e relatórios HTML locais."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import importlib.metadata
import json
from pathlib import Path
import traceback
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy.signal import spectrogram
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from reporting import Report, overlay_plot, timeline_plot, batch_comparison, smoothed_timeline, class_trials, mean_curve
from eeg import COLORS, PROTOCOLS, fingerprint, load_run, make_epochs, scan, selection, settings

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "pdf.fonttype": 42})
mne.set_log_level("WARNING")


def save_figure(fig, directory, name, cfg, report=None, title=None):
    fig.savefig(directory / f"{name}.png", dpi=cfg["dpi"])
    fig.savefig(directory / f"{name}.pdf")
    if report is not None:
        labels = {"raw_antes_depois": "Raw · antes/depois sobrepostos", "sanity_checks": "Sanity checks · amplitude, ruído e rejeição",
                  "epocas_canais": "Épocas · mapa por canal", "pca_2d_3d": "PCA · 2D e 3D", "dashboard": "Dashboard · visão geral",
                  "tempo_frequencia": "Tempo–frequência · visão global", "topografias_mu_beta": "Topografias · mu e beta", "espectrograma": "Espectrograma · canais centrais"}
        report.add_figure(fig, title=title or labels.get(name, name.replace("_", " · ")), image_format="png")
    plt.close(fig)


def stacked(ax, data, times, names, scale, title):
    data   = data - np.median(data, axis=-1, keepdims=True)
    offset = np.arange(len(names))[::-1] * scale
    for i in range(len(names)):
        ax.plot(times, data[i] * 1e6 + offset[i], color="0.25", lw=0.45)
    ax.set(yticks=offset, yticklabels=names, xlabel="Tempo (s)", title=title)
    ax.text(1, 1.02, f"espaçamento: {scale:.0f} µV", transform=ax.transAxes, ha="right", fontsize=8)


def raw_snapshot(run):
    sfreq = run.raw.info["sfreq"]
    width = min(run.raw.n_times, int(12 * sfreq))
    first = max(0, (run.raw.n_times - width) // 2)
    picks = mne.pick_types(run.raw.info, eeg=True, exclude=[])
    return (run.raw.get_data(picks=picks, start=first, stop=first + width),
            run.clean.get_data(picks=picks, start=first, stop=first + width),
            run.raw.times[first:first + width], [run.raw.ch_names[i] + (" (excluído)" if run.raw.ch_names[i] in run.raw.info["bads"] else "") for i in picks])


def raw_figure(run, cfg):
    before, after, times, names = raw_snapshot(run)
    before                      = (before - np.median(before, axis=1, keepdims=True)) * 1e6
    after                       = (after - np.median(after, axis=1, keepdims=True)) * 1e6
    fig, axes                   = plt.subplots(len(names), 1, figsize=(13, max(8, len(names) * .75)), sharex=True, layout="constrained")
    for i, ax in enumerate(np.atleast_1d(axes)):
        ax.plot(times, before[i], color="0.5", lw=.6, label="Antes")
        ax.plot(times, after[i], color="#0072B2", lw=.7, label="Depois")
        limit = max(5, float(np.quantile(np.abs(np.r_[before[i], after[i]]), .995)) * 1.15)
        ax.set(ylim=(-limit, limit), ylabel=f"{names[i]}\nµV")
    np.atleast_1d(axes)[0].legend(ncol=2)
    np.atleast_1d(axes)[-1].set_xlabel("Tempo desde início do sinal (s)")
    fig.suptitle("Snapshot central · antes/depois sobrepostos · mediana removida só para exibição")
    return fig


def central_picks(info):
    return [i for i, name in enumerate(info["ch_names"]) if name in {"C3", "Cz", "C4"} and name not in info["bads"]]


def erp_plot(ax, epochs, cfg):
    picks = central_picks(epochs.info)
    for cls in epochs.event_id:
        values = epochs[cls].get_data(picks=picks or mne.pick_types(epochs.info, eeg=True, exclude="bads")) * 1e6
        values -= values[:, :, (epochs.times >= cfg["baseline"][0]) & (epochs.times <= cfg["baseline"][1])].mean(axis=-1, keepdims=True)
        trials = values.mean(axis=1)
        ax.plot(epochs.times, trials.mean(axis=0), color=COLORS[cls], lw=1.5, label=cls)
    ax.axvline(0, color="0.3", ls="--", lw=.8)
    ax.set(title="ERP · média C3/Cz/C4 · detalhes por canal no relatório", xlabel="Tempo desde ATTEMPT (s)", ylabel="µV")
    ax.legend(fontsize=7)


def erp_figures(epochs, cfg):
    picks = mne.pick_types(epochs.info, eeg=True, exclude="bads")
    classes = list(epochs.event_id)
    fig, axes = plt.subplots(len(picks), len(classes), figsize=(5 * len(classes), 1.8 * len(picks)),
                             squeeze=False, sharex=True, layout="constrained")
    for col, cls in enumerate(classes):
        values = epochs[cls].get_data(picks=picks) * 1e6
        values -= values[:, :, (epochs.times >= cfg["baseline"][0]) & (epochs.times <= cfg["baseline"][1])].mean(axis=-1, keepdims=True)
        for i, ch in enumerate(picks):
            ax = axes[i, col]
            for trial, trace in zip(epochs[cls].metadata.trial, values[:, i]):
                ax.plot(epochs.times, trace, color=COLORS[cls], alpha=.22, lw=.5)
            ax.plot(epochs.times, values[:, i].mean(axis=0), color="black", lw=1.6)
            ax.axvline(0, color=".5", ls=":")
            ax.set(title=f"{epochs.ch_names[ch]} · {cls} · n={len(values)}", ylabel="µV", xlabel="ATTEMPT (s)")
    fig.suptitle("ERP · trials discretos; média preta · baseline " + str(cfg["baseline"]) + " s")
    yield "erp_canais_classes", fig


def quality_checks(run, audit, cfg):
    from scipy.signal import welch
    before    = run.raw.get_data() * 1e6
    after     = run.clean.get_data() * 1e6
    std       = after.std(axis=1)
    ratio     = std / max(np.median(std), 1e-12)
    freq, psd = welch(before, fs=run.raw.info['sfreq'], nperseg=min(before.shape[-1], int(4 * run.raw.info['sfreq'])), axis=-1)
    total     = psd[:, (freq >= 1) & (freq < min(100, run.raw.info['sfreq'] / 2))].sum(axis=1)
    line      = psd[:, ((abs(freq - 50) <= 1) | (abs(freq - 60) <= 1))].sum(axis=1) / np.maximum(total, 1e-20) * 100
    rejected  = [sum(name in str(reason).split(';') for reason in audit.drop_reason) for name in run.raw.ch_names]
    frame     = pd.DataFrame(dict(channel=run.raw.ch_names, raw_dc_uv=np.median(before, axis=1),
                              raw_std_uv=before.std(axis=1), clean_std_uv=std, relative_std=ratio,
                              raw_peak_to_peak_uv=np.ptp(before, axis=1), clean_peak_to_peak_uv=np.ptp(after, axis=1),
                              line_noise_percent=line, rejected_epochs=rejected,
                              excluded=[c in run.raw.info['bads'] for c in run.raw.ch_names],
                              flag=['variação baixa' if v < .2 else ('variação alta' if v > 5 else 'sem alerta relativo') for v in ratio]))
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), layout='constrained')
    x         = np.arange(len(frame))
    axes[0].bar(x - .18, frame.raw_std_uv, .36, color='0.6', label='Antes')
    axes[0].bar(x + .18, frame.clean_std_uv, .36, color='#0072B2', label='Depois')
    axes[0].set(yscale='log', ylabel='Desvio-padrão (µV)', title='Variação por canal · escala logarítmica; offsets DC na tabela')
    axes[0].legend()
    axes[1].bar(x, line, color='0.45')
    axes[1].set(ylabel='% da potência 1–100 Hz', title='Ruído estreito em 50/60 Hz ±1 Hz · sinal antes do filtro')
    axes[2].bar(x, rejected, color='0.45')
    axes[2].set(ylabel='Épocas rejeitadas', title=f'{int((~audit.accepted).sum())}/{len(audit)} épocas rejeitadas · contribuição por canal (pode haver múltiplas causas)')
    for ax in axes:
        ax.set_xticks(x, frame.channel, rotation=30, ha='right')
    return frame, fig


def central_spectrogram(run, cfg):
    picks = central_picks(run.clean.info)
    if not picks:
        picks = mne.pick_types(run.clean.info, eeg=True, exclude='bads')[:3].tolist()
    data             = run.clean.get_data(picks=picks)
    freq, time, spec = spectrogram(data, fs=run.clean.info['sfreq'], nperseg=min(data.shape[-1], round(2 * run.clean.info['sfreq'])),
                                    noverlap=round(run.clean.info['sfreq'] * 1.5), axis=-1)
    keep             = (freq >= 1) & (freq <= cfg['h_freq'])
    values           = 10 * np.log10(np.maximum(spec[:, keep] * 1e12, 1e-20))
    low, high        = np.quantile(values, [.02, .98])
    fig, axes        = plt.subplots(len(picks), 1, figsize=(13, 3.3 * len(picks)), sharex=True, layout='constrained', squeeze=False)
    for i, ax in enumerate(axes[:, 0]):
        im = ax.pcolormesh(time, freq[keep], values[i], cmap='cividis', vmin=low, vmax=high, shading='auto')
        ax.set(title=run.clean.ch_names[picks[i]], ylabel='Hz')
    axes[-1, 0].set_xlabel('Tempo desde início do EEG (s)')
    fig.colorbar(im, ax=list(axes[:, 0]), label='dB (µV²/Hz) · escala comum')
    fig.suptitle('Espectrograma contínuo · canais centrais')
    return fig


def central_erd_figures(power, cfg):
    if power is None or not central_picks(power.info):
        return
    picks, classes = central_picks(power.info), list(power.event_id)
    fig, axes = plt.subplots(len(classes) * 2, len(picks), figsize=(5 * len(picks), 3 * len(classes) * 2),
                             squeeze=False, sharex=True, layout="constrained")
    for ci, cls in enumerate(classes):
        for bi, band in enumerate(["mu", "beta"]):
            keep = (power.freqs >= cfg[band][0]) & (power.freqs < cfg[band][1])
            for col, ch in enumerate(picks):
                traces = power[cls].data[:, ch, keep].mean(axis=1) * 100
                ax = axes[ci * 2 + bi, col]
                for trace in traces:
                    ax.plot(power.times, trace, color=COLORS[cls], alpha=.22, lw=.6)
                ax.plot(power.times, traces.mean(axis=0), color="black", lw=1.8)
                ax.axhline(0, color=".5", lw=.6)
                ax.axvline(0, color=".5", ls=":")
                ax.set(title=f"{cls} · {power.ch_names[ch]} · {band} {cfg[band]} Hz", ylabel="ERD/ERS (%)", xlabel="ATTEMPT (s)")
    fig.suptitle("ERD/ERS · trials e média preta · baseline em PAUSA auditada por trial")
    yield "erd_central", fig
    fig, axes = plt.subplots(len(classes), len(picks), figsize=(5 * len(picks), 3 * len(classes)), squeeze=False, layout="constrained")
    values = [power[c].data[:, picks].mean(axis=0) * 100 for c in classes]
    limit = max(1, float(np.quantile(np.abs(values), .98)))
    for ci, cls in enumerate(classes):
        for j, ch in enumerate(picks):
            im = axes[ci, j].pcolormesh(power.times, power.freqs, values[ci][j], cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto")
            axes[ci, j].set(title=f"{cls} · {power.ch_names[ch]}", xlabel="ATTEMPT (s)", ylabel="Hz")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="ERD/ERS (%) · escala comum")
    yield "tfr_central", fig


def erd_metrics(power, cfg):
    rows = []
    if power is None:
        return pd.DataFrame(columns=['trial', 'class', 'channel', 'band', 'erd_percent'])
    active = (power.times >= cfg['active'][0]) & (power.times <= cfg['active'][1])
    for i, trial in enumerate(power.metadata.itertuples()):
        for band in ['mu', 'beta']:
            keep = (power.freqs >= cfg[band][0]) & (power.freqs < cfg[band][1])
            for ch in central_picks(power.info):
                rows.append(dict(trial=int(trial.trial), **{'class': trial.trial_class}, channel=power.ch_names[ch], band=band,
                                 erd_percent=float(power.data[i, ch, keep][:, active].mean() * 100)))
    return pd.DataFrame(rows)


def spectral_plot(ax, epochs, cfg):
    for cls in epochs.event_id:
        spectrum = epochs[cls].compute_psd(method="welch", fmin=1, fmax=cfg["h_freq"],
                                          tmin=cfg["active"][0], tmax=cfg["active"][1],
                                          picks="eeg", n_fft=min(256, int(epochs.info["sfreq"])), verbose=False)
        power    = spectrum.get_data().mean(axis=1) * 1e12
        db       = 10 * np.log10(np.maximum(power, np.finfo(float).tiny))
        ax.plot(spectrum.freqs, db.mean(axis=0), color=COLORS[cls], label=cls)
    ax.set(title="Welch · média de potência dos canais", xlabel="Frequência (Hz)", ylabel="dB (µV²/Hz)")
    ax.legend(fontsize=7)


def baseline_intervals(run, epochs, cfg):
    intervals = []
    for row in epochs.metadata.itertuples():
        previous = run.trials[run.trials.trial < row.trial]
        pause    = previous.iloc[-1].pause_start if len(previous) else np.nan
        if cfg["erd_baseline_mode"] == "preceding_pause" and np.isfinite(pause):
            end   = row.attention_start - .25
            start = max(pause + .25, end - 1.5)
        else:
            start, end = np.asarray(cfg["baseline"]) + row.attempt_start
        valid = bool(np.isfinite(pause) and start >= max(pause, run.times[0]) and end <= row.attention_start and end - start >= .5)
        intervals.append((start - row.attempt_start, end - row.attempt_start) if valid else None)
    return intervals


def baseline_trials(run, epochs, cfg):
    intervals = baseline_intervals(run, epochs, cfg)
    valid     = np.array([v is not None for v in intervals])
    audit     = pd.DataFrame([dict(trial=int(row.trial), baseline_in_pause=ok,
                              baseline_start_s=interval[0] if ok else np.nan,
                              baseline_end_s=interval[1] if ok else np.nan,
                              reason="" if ok else "PAUSA anterior ausente/curta/fora do EEG; ERD excluído.")
                          for row, ok, interval in zip(epochs.metadata.itertuples(), valid, intervals)])
    return valid, audit


def time_frequency(run, epochs, cfg):
    intervals    = baseline_intervals(run, epochs, cfg)
    valid, audit = baseline_trials(run, epochs, cfg)
    if not valid.any():
        return None, audit
    used      = epochs[np.flatnonzero(valid)]
    intervals = {int(row.trial): interval for row, interval in zip(epochs.metadata.itertuples(), intervals) if interval is not None}
    tmin      = min(cfg["epoch_tmin"], min(v[0] for v in intervals.values()) - .5)
    used      = mne.Epochs(run.clean, events=used.events, event_id=used.event_id, tmin=tmin, tmax=cfg["epoch_tmax"],
                      baseline=None, preload=True, metadata=used.metadata, verbose=False)
    accepted  = []
    for i, row in enumerate(used.metadata.itertuples()):
        start, end = intervals[int(row.trial)]
        data       = used.get_data(picks=mne.pick_types(used.info, eeg=True, exclude="bads"))[i]
        data       = data[:, (used.times >= start) & (used.times <= end)]
        if data.shape[-1] and np.all(np.ptp(data, axis=1) <= cfg["reject_uv"] * 1e-6) and np.all(np.ptp(data, axis=1) >= cfg["flat_uv"] * 1e-6):
            accepted.append(i)
    used = used[accepted]
    keep = set(used.metadata.trial)
    for i, row in audit.iterrows():
        if row.baseline_in_pause and row.trial not in keep:
            audit.loc[i, ["baseline_in_pause", "reason"]] = [False, "Contexto indisponível ou artefato na baseline."]
    if not len(used):
        return None, audit
    freqs = np.asarray(cfg["frequencies"], float)
    if freqs.max() > cfg["h_freq"] or freqs.min() < cfg["l_freq"]:
        raise ValueError("Frequências TFR fora do filtro eletrofisiológico.")
    power = used.compute_tfr(method="morlet", freqs=freqs, n_cycles=freqs / 2, return_itc=False,
                             average=False, picks=mne.pick_types(used.info, eeg=True, exclude="bads"), decim=max(1, round(used.info["sfreq"] / 64)), n_jobs=1, verbose=False)
    for i, row in enumerate(used.metadata.itertuples()):
        mne.baseline.rescale(power.data[i], power.times, intervals[int(row.trial)], mode="percent", copy=False, verbose=False)
    power.crop(cfg["epoch_tmin"], cfg["epoch_tmax"])
    return power, audit


def erd_plot(ax, power, cfg):
    if power is None:
        ax.text(0.5, 0.5, "ERD/ERS indisponível: baseline fora de PAUSA", ha="center", wrap=True)
        ax.axis("off")
        return
    for cls in power.event_id:
        values = power[cls].data
        for band, style in [("mu", "-"), ("beta", "--")]:
            low, high = cfg[band]
            mask      = (power.freqs >= low) & (power.freqs < high)
            if not mask.any():
                continue
            traces = values[:, :, mask, :].mean(axis=(1, 2)) * 100
            ax.plot(power.times, traces.mean(axis=0), color=COLORS[cls], ls=style, label=f"{cls} · {band}")
    ax.axhline(0, color="0.5", lw=0.7)
    ax.axvline(0, color="0.3", ls=":")
    ax.set(title="ERD/ERS · normalização por trial · média dos canais", xlabel="Tempo desde ATTEMPT (s)", ylabel="Δ potência (%)")
    ax.legend(fontsize=6, ncol=2)


def tfr_figures(power, cfg):
    if power is None:
        return []
    classes   = list(power.event_id)
    fig, axes = plt.subplots(len(classes), 1, figsize=(11, 3 * len(classes)), squeeze=False, layout="constrained")
    values    = [power[c].data.mean(axis=(0, 1)) * 100 for c in classes]
    limit     = max(1, float(np.quantile(np.abs(np.concatenate([v.ravel() for v in values])), 0.98)))
    for ax, cls, val in zip(axes[:, 0], classes, values):
        im = ax.pcolormesh(power.times, power.freqs, val, cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto")
        ax.axvline(0, color="0.2", ls="--", lw=0.8)
        ax.set(title=f"{cls} · média da potência dos canais", xlabel="Tempo desde ATTEMPT (s)", ylabel="Hz")
        ax.title.set_color(COLORS[cls])
    fig.colorbar(im, ax=axes.ravel().tolist(), label="ERD/ERS (%) · azul: redução; vermelho: aumento")
    topo, tax = plt.subplots(2, len(classes), figsize=(3.5 * len(classes), 6), squeeze=False, layout="constrained")
    active    = (power.times >= cfg["active"][0]) & (power.times <= cfg["active"][1])
    maps      = []
    for band in ["mu", "beta"]:
        mask = (power.freqs >= cfg[band][0]) & (power.freqs < cfg[band][1])
        if not mask.any():
            raise ValueError(f"Grade TFR sem frequências na banda {band}.")
        maps.extend([power[c].data[:, :, mask, :][:, :, :, active].mean(axis=(0, 2, 3)) * 100 for c in classes])
    limit = max(1, float(np.max(np.abs(maps))))
    for index, (ax, values) in enumerate(zip(tax.ravel(), maps)):
        im, _ = mne.viz.plot_topomap(values, power.info, axes=ax, show=False, cmap="RdBu_r",
                                    vlim=(-limit, limit), contours=0, image_interp="linear", extrapolate="local")
        cls   = classes[index % len(classes)]
        band  = ["mu", "beta"][index // len(classes)]
        ax.set_title(f"{cls} · {band}", color=COLORS[cls])
    topo.colorbar(im, ax=tax.ravel().tolist(), label="ERD/ERS (%)")
    topo.suptitle(f"ATTEMPT {cfg['active']} s · mu {cfg['mu']} / beta {cfg['beta']} Hz\nBaseline: {cfg['erd_baseline_mode']} em PAUSA (baseline_audit.csv) · escala comum")
    return [("tempo_frequencia", fig), ("topografias_mu_beta", topo)]


def epoch_figure(epochs):
    picks     = mne.pick_types(epochs.info, eeg=True, exclude="bads")
    fig, axes = plt.subplots(int(np.ceil(len(picks) / 3)), 3, figsize=(13, 2.2 * np.ceil(len(picks) / 3)), layout="constrained", squeeze=False)
    values    = epochs.get_data(picks=picks) * 1e6
    limit     = max(1, float(np.quantile(np.abs(values), 0.99)))
    for i, ax in enumerate(axes.ravel()):
        if i >= len(picks):
            ax.axis("off")
            continue
        im = ax.imshow(values[:, i], aspect="auto", origin="lower", cmap="Greys",
                       vmin=-limit, vmax=limit, extent=[epochs.times[0], epochs.times[-1], -0.5, len(epochs) - 0.5])
        ax.axvline(0, color="0.3", ls="--", lw=0.7)
        ax.set(title=epochs.ch_names[picks[i]], xlabel="Tempo (s)", ylabel="Época aceita (ordem)")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="µV · mesma escala")
    return fig


def pca_figure(epochs, cfg):
    data = epochs.copy().crop(*cfg["active"]).get_data(picks=mne.pick_types(epochs.info, eeg=True, exclude="bads"))
    size = min(data.shape[-1], max(2, round(epochs.info["sfreq"])))
    starts = range(0, data.shape[-1] - size + 1, size)
    pieces = [(i, data[i, :, start:start + size]) for i in range(len(data)) for start in starts]
    identities = np.array([i for i, _ in pieces])
    Z    = TangentSpace(metric="riemann").fit_transform(Covariances(estimator="oas").fit_transform(np.stack([v for _, v in pieces])))
    Z    = StandardScaler().fit_transform(Z)
    n    = min(3, *Z.shape)
    if n < 2:
        raise ValueError("PCA requer pelo menos duas épocas e duas features.")
    pca    = PCA(n_components=n).fit(Z)
    pos    = pca.transform(Z)
    fig    = plt.figure(figsize=(12, 5), layout="constrained")
    left   = fig.add_subplot(121)
    right  = fig.add_subplot(122, projection="3d" if n == 3 else None)
    labels = epochs.metadata.trial_class.to_numpy()[identities]
    for cls in np.unique(labels):
        keep = labels == cls
        left.scatter(pos[keep, 0], pos[keep, 1], c=COLORS[cls], label=cls, s=25)
        if n == 3:
            right.scatter(*pos[keep].T, c=COLORS[cls], s=25)
        else:
            right.text(0.1, 0.5, "Trials insuficientes para PCA 3D")
    left.set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    for trial_index in np.unique(identities):
        keep = identities == trial_index
        left.plot(pos[keep, 0], pos[keep, 1], color=".5", alpha=.3, lw=.6)
        left.annotate(str(epochs.metadata.trial.iloc[trial_index]), (pos[keep, 0][0], pos[keep, 1][0]), fontsize=6)
    left.legend()
    if n == 3:
        right.set(xlabel="PC1", ylabel="PC2", zlabel="PC3")
    fig.suptitle("PCA exploratório · janelas reais dependentes; linhas/IDs preservam trials · não estima desempenho")
    return fig


def classifier_figure(summary, predictions, cfg):
    labels    = summary["labels"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    cm        = np.array(summary["confusion_matrix"])
    norm      = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    axes[0].imshow(norm, cmap="Greys", vmin=0, vmax=1)
    for i, j in np.ndindex(cm.shape):
        axes[0].text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", color="white" if norm[i, j] > 0.5 else "black")
    axes[0].set(xticks=range(len(labels)), xticklabels=labels, yticks=range(len(labels)), yticklabels=labels,
                   xlabel="Predito", ylabel="Real", title="Matriz por trial/fase · probabilidades médias")
    scores = [f["balanced_accuracy"] for f in summary["folds"]]
    axes[1].scatter(np.linspace(-.12, .12, len(scores)), scores, color="0.3")
    axes[1].axhline(np.mean(scores), color="#0072B2", label="Média dos folds")
    axes[1].set_xticks([0], ["Folds externos (dependentes)"])
    axes[1].axhline(summary["chance"], color="0.5", ls="--")
    axes[1].set(ylim=(0, 1), xlabel="Dispersão dos folds · sem inferência independente", ylabel="Balanced accuracy", title=f"BA agrupada: {summary['balanced_accuracy']:.3f}")
    fig.suptitle(f"{summary['protocol']} · {summary['validation']} · fonte REST: {summary['rest_source']}")
    return fig


def detailed_trials(run, before, after, report):
    # One joint panel per channel; rejected epochs remain inspectable.
    picks = mne.pick_types(before.info, eeg=True, exclude="bads")
    fig, axes = plt.subplots(int(np.ceil(len(picks) / 3)), 3, figsize=(15, 3 * np.ceil(len(picks) / 3)), layout="constrained")
    for j, ax in enumerate(axes.ravel()):
        if j >= len(picks):
            ax.axis("off")
            continue
        channel_data = before.get_data(picks=[picks[j]])[:, 0] * 1e6
        worst = int(np.argmax(np.ptp(channel_data, axis=1)))
        for i, row in enumerate(before.metadata.itertuples()):
            first = before.events[i, 0] + round(before.times[0] * before.info["sfreq"])
            raw = before.get_data(picks=[picks[j]])[i, 0] * 1e6
            clean = run.clean.get_data(picks=[picks[j]], start=first, stop=first + len(before.times))[0] * 1e6
            ax.plot(before.times, raw, color=".5", alpha=.18, lw=.5)
            ax.plot(before.times, clean, color="#0072B2", alpha=.25, lw=.5)
            if i == worst:
                peak = int(np.argmax(np.abs(raw)))
                ax.annotate(f"Trial {row.trial} · maior amplitude", (before.times[peak], raw[peak]), fontsize=6)
        ax.set(title=before.ch_names[picks[j]], xlabel="ATTEMPT (s)", ylabel="µV")
    fig.suptitle("Épocas conjuntas · antes cinza / depois azul · inclui rejeitadas; IDs e motivos em epoch_audit.csv")
    report.add_figure(fig, title="Épocas · antes/depois conjuntos")
    plt.close(fig)


def calibration_row(row, included):
    specified  = str(row.get("train_run_id", "")).strip()
    candidates = included[(included.subject == row.subject) & (included.block_type == "IM_treino")].copy()
    if specified:
        candidates = candidates[candidates.run_id == specified]
    else:
        candidates = candidates[candidates.session == row.session]
    stamp      = pd.to_datetime(row.stamp, format="%Y%m%d_%H%M%S", errors="coerce")
    dates      = pd.to_datetime(candidates.stamp, format="%Y%m%d_%H%M%S", errors="coerce")
    candidates = candidates[dates < stamp]
    if candidates.empty:
        raise ValueError("Online requer IM_treino incluído e anterior. Defina train_run_id na tabela se necessário; não usar treino futuro.")
    return candidates.sort_values("stamp").iloc[-1]


def analyze_run(root, row, cfg, directory, mode="both", detailed=False, protocols=None, included=None, compare=False):
    directory.mkdir(parents=True, exist_ok=True)
    run                   = load_run(root, row, cfg)
    before, epochs, audit = make_epochs(run, cfg)
    audit.to_csv(directory / "epoch_audit.csv", index=False)
    run.trials.to_csv(directory / "trials.csv", index=False)
    if len(epochs) < 2:
        raise ValueError("Menos de duas épocas aceitas; consultar epoch_audit.csv.")
    report     = Report(title=f"{row.subject} · {row.session} · {row.block_type} · {row.run_id}", verbose=False)
    provenance = dict(created_utc=datetime.now(timezone.utc).isoformat(), config=cfg, control=row.to_dict(), qc=run.qc,
                      files={name: fingerprint(Path(root) / row[name]) for name in ["signal_path", "marker_path"]},
                      versions={p: importlib.metadata.version(p) for p in ["mne", "numpy", "scipy", "scikit-learn", "pyriemann"]},
                      notes=["Análise offline; não reproduz automaticamente o pré-processamento do sistema de coleta.",
                             "Referência média altera o sinal; canais ruins são excluídos, sem interpolação.",
                             "Nenhum ICA automático sem EOG ou inspeção; amplitude/flatness não identificam todo artefato.",
                             "ERP é inspeção descritiva; ausência de efeito não prova ausência de resposta."])
    report.add_html(f'<div class="cards"><div class="card">Trials<strong>{len(run.trials)}</strong></div><div class="card">Épocas aceitas<strong>{len(epochs)}/{len(audit)}</strong></div><div class="card">Exclusões manuais<strong>{int(run.trials.manual_exclusion.sum())}</strong></div><div class="card">Canais excluídos<strong>{len(run.raw.info["bads"])}</strong></div></div>'
                    '<p>Escalas em µV. Centralização dos overlays é apenas visual; arquivos e amplitudes de análise não são alterados.</p>', title="Resumo · qualidade e inclusão")
    report.add_html(f"<p>Incluídos para épocas: {int(run.trials.valid.sum())}; inválidos/excluídos: {int((~run.trials.valid).sum())}; rejeitados por época: {int((~audit.accepted).sum())}. Categorias podem se sobrepor; epoch_audit.csv registra cada decisão.</p>", title="Resumo · decisões por trial")
    quality, quality_fig = quality_checks(run, audit, cfg)
    quality.to_csv(directory / "channel_quality.csv", index=False)
    flagged = quality.loc[quality.flag != "sem alerta relativo", "channel"].tolist()
    causes  = audit.loc[~audit.accepted, "drop_reason"].value_counts().to_dict()
    report.add_html(f"<p>Canais excluídos manualmente: {html.escape(', '.join(run.raw.info['bads']) or 'nenhum')}. "
                    f"Variação pós-filtro abaixo de 0,2× ou acima de 5× a mediana dos canais: {html.escape(', '.join(flagged) or 'nenhum')}. "
                    f"Causas das rejeições: {html.escape(str(causes) if causes else 'nenhuma')}.</p>"
                    "<p>Alertas relativos não excluem canais automaticamente. O gráfico de ruído é uma triagem de potência em 50/60 Hz, não um diagnóstico de origem do artefato.</p>", title="Resumo · sanity checks")

    report.add_html('<p>Tabelas completas: <a href="channel_quality.csv">qualidade por canal</a>, <a href="epoch_audit.csv">decisões por época</a>, <a href="trials.csv">trials</a>, <a href="parameters_qc.json">parâmetros e proveniência</a>.</p>', title="Auditoria · arquivos completos")
    save_figure(quality_fig, directory, "sanity_checks", cfg, report)
    processing = dict(Unidade=f"CSV × {cfg['scale_to_volts']} → volts; exibição em µV", Referência=cfg["reference"],
                      Filtro=f"FIR offline {cfg['l_freq']}–{cfg['h_freq']} Hz; notch {cfg['notch']}",
                      Amostragem=f"{run.raw.info['sfreq']:.3f} Hz", Timestamps=f"{run.qc['timing']['policy']}; reconstruídos: {run.qc['timing']['reconstructed']}; correção máxima {run.qc['timing']['max_correction_s']:.6f} s",
                      Épocas=f"{cfg['epoch_tmin']} a {cfg['epoch_tmax']} s", Baseline_ERP=str(cfg["baseline"]),
                      Baseline_ERD=cfg["erd_baseline_mode"] + " · apenas PAUSA auditada", Ativo=str(cfg["active"]),
                      Rejeição=f"pico a pico > {cfg['reject_uv']} µV ou flat < {cfg['flat_uv']} µV")
    report.add_html(pd.DataFrame(processing.items(), columns=["Processamento", "Valor"]).to_html(index=False), title="⚙️ Processamento")

    metrics, power, primary = [], None, None
    if mode in {"both", "ephys"}:
        save_figure(raw_figure(run, cfg), directory, "raw_antes_depois", cfg, report)

        for name, fig in erp_figures(epochs, cfg):
            save_figure(fig, directory, name, cfg, report)

        save_figure(pca_figure(epochs, cfg), directory, "pca_2d_3d", cfg, report)
        power, baseline_audit = time_frequency(run, epochs, cfg)
        baseline_audit.to_csv(directory / "baseline_audit.csv", index=False)
        erd_metrics(power, cfg).to_csv(directory / "erd_metrics.csv", index=False)
        report.add_html(baseline_audit.to_html(index=False), title="Validade da baseline para ERD/ERS")
        for name, fig in tfr_figures(power, cfg):
            save_figure(fig, directory, name, cfg, report if name == "topografias_mu_beta" else None)
        for name, fig in central_erd_figures(power, cfg):
            save_figure(fig, directory, name, cfg, report)
        if power is None:
            report.add_html("<p>ERD/ERS indisponível: nenhuma PAUSA anterior utilizável após inspeção de contexto e artefatos. Consulte baseline_audit.csv. Não se usa ATTENTION como referência substituta.</p>", title="Resumo · ERD/ERS indisponível")
    if mode in {"both", "classification"}:
        from classify import evaluate, save_result
        training, train_source = None, "nested_trial_cv"
        if row.block_type.startswith("online_"):
            try:
                train_row          = calibration_row(row, included)
                train_run          = load_run(root, train_row, cfg)
                _, train_epochs, _ = make_epochs(train_run, cfg)
                training           = (train_run, set(train_epochs.metadata.trial))
                train_source       = train_row.run_id
            except ValueError as exc:
                train_source = f"BLOCKED: {exc}"
        variants = cfg["comparison_pipelines"] if compare else ["nested_selection" if len(cfg["pipelines"]) > 1 else cfg["pipelines"][0]]
        for protocol in protocols or list(PROTOCOLS):
            for variant in variants:
                try:
                    if train_source.startswith("BLOCKED:"):
                        raise ValueError(train_source)
                    experiment = {**cfg, "pipelines": [variant]} if compare else cfg
                    result     = evaluate(run, experiment, protocol, set(epochs.metadata.trial), training)
                    if primary is None:
                        primary = result
                    result[0]["train_source"] = train_source
                    result[0]["pipeline"]     = variant
                    prefix                    = f"{variant}_" if compare else ""
                    report.add_html(pd.DataFrame([dict(fold=f["fold"], **f["params"], validação=result[0]["validation"], treino=train_source) for f in result[0]["folds"]]).to_html(index=False), title=f"Classificação · {variant} · {protocol} · parâmetros selecionados")
                    save_result(result, directory, prefix)
                    save_figure(classifier_figure(result[0], result[1], cfg), directory, f"classificacao_{prefix}{protocol}", cfg, report)
                    context = pd.DataFrame(result[0].get("_timeline", []))
                    if len(context):
                        report.add_plotly(timeline_plot(result[0], context, cfg, run.trials), f"Classificador · {variant} · {protocol} · zoom")
                    metrics.append(dict(protocol=protocol, pipeline=variant, status="ok", balanced_accuracy=result[0]["balanced_accuracy"],
                                        chance=result[0]["chance"], validation=result[0]["validation"], train_source=train_source,
                                        n_trials=result[0]["n_trials"]))
                except (ValueError, np.linalg.LinAlgError) as exc:
                    metrics.append(dict(protocol=protocol, pipeline=variant, status=str(exc)))
        report.add_html(pd.DataFrame(metrics).to_html(index=False), title="Classificação · resultados e impedimentos")
    fig, axes = plt.subplots(3, 4, figsize=(24, 16), layout="constrained")
    if mode in {"both", "ephys"}:
        erp_plot(axes[0, 0], epochs, cfg)
        spectral_plot(axes[0, 1], epochs, cfg)
        erd_plot(axes[0, 2], power, cfg)
    else:
        for ax in axes[0]:
            ax.axis("off")
    picks = mne.pick_types(run.clean.info, eeg=True, exclude="bads")
    stop  = min(run.clean.n_times, int(8 * run.clean.info["sfreq"]))
    stacked(axes[0, 3], run.clean.get_data(picks=picks, stop=stop), run.clean.times[:stop],
            [run.clean.ch_names[i] for i in picks], 50, "Trecho após pré-processamento")
    for ax in axes[1]:
        ax.axis("off")
    if power is not None:
        classes = [c for c in ["LEFT", "RIGHT"] if c in power.event_id] or list(power.event_id)[:2]
        active  = (power.times >= cfg["active"][0]) & (power.times <= cfg["active"][1])
        maps    = []
        for cls in classes:
            for band in ["mu", "beta"]:
                keep = (power.freqs >= cfg[band][0]) & (power.freqs < cfg[band][1])
                val  = power[cls].data[:, :, keep, :][:, :, :, active].mean(axis=(0, 2, 3)) * 100
                maps.append((cls, band, val))
        limit = max(1, float(max(np.max(np.abs(v)) for _, _, v in maps)))
        for ax, (cls, band, val) in zip(axes[1], maps):
            ax.axis("on")
            im, _ = mne.viz.plot_topomap(val, power.info, axes=ax, show=False, cmap="RdBu_r",
                                        vlim=(-limit, limit), contours=0, image_interp="linear", extrapolate="local")
            ax.set_title(f"{cls} · {band}", color=COLORS[cls])
        fig.colorbar(im, ax=list(axes[1]), label="ERD/ERS (%) · mesma escala; todas as classes no relatório")
    values   = epochs.copy().crop(*cfg["active"]).get_data(picks=mne.pick_types(epochs.info, eeg=True, exclude="bads"))
    features = TangentSpace(metric="riemann").fit_transform(Covariances(estimator="oas").fit_transform(values))
    pos      = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(features))
    for cls in epochs.event_id:
        keep = epochs.metadata.trial_class.to_numpy() == cls
        axes[2, 0].scatter(pos[keep, 0], pos[keep, 1], color=COLORS[cls], label=cls, s=20)
    axes[2, 0].set(title="PCA exploratório · um ponto por trial", xlabel="PC1", ylabel="PC2")
    axes[2, 0].legend(fontsize=7)
    passed = [m for m in metrics if m["status"] == "ok"]
    if passed:
        axes[2, 1].barh([m["protocol"] for m in passed], [m["balanced_accuracy"] for m in passed], color="0.4")
        for m in passed:
            axes[2, 1].plot(m["chance"], m["protocol"], "|", color="black", ms=18)
        axes[2, 1].set(xlim=(0, 1), title="BA por trial/fase · traço preto: acaso")
    else:
        axes[2, 1].axis("off")
        axes[2, 1].text(0, 0.5, "Classificação não executada ou indisponível.\nConsultar relatório HTML.")
    if primary is not None:
        predictions = smoothed_timeline(pd.DataFrame(primary[0]["_timeline"]), cfg["curve_smoothing_s"])
        trials      = sorted(predictions.trial.unique())
        rank        = {t: i / max(1, len(trials) - 1) for i, t in enumerate(trials)}
        for cls in primary[0]["labels"]:
            traces = []
            for trial, sub in class_trials(predictions, cls):
                sub = sub.sort_values("time_end")
                axes[2, 2].plot(sub.time_end, sub[f"p_{cls}"], color=COLORS[cls], alpha=0.08 + 0.35 * rank[trial], lw=0.7)
                traces.append(pd.Series(sub[f"p_{cls}"].to_numpy(), index=sub.time_end.round(4)))
            if traces:
                grid, mean, support = mean_curve(traces, cfg)
                axes[2, 2].plot(grid, mean, color=COLORS[cls], lw=2, label=cls)
        axes[2, 2].set(title=f"{primary[0]['protocol']} · claros: antigos → escuros: recentes", xlabel="Fim da janela desde ATTEMPT (s)", ylabel="P(classe correta)", ylim=(0, 1))
        axes[2, 2].legend(fontsize=7)
    else:
        axes[2, 2].axis("off")
    axes[2, 3].axis("off")
    axes[2, 3].text(0, 1, f"{row.subject} · {row.session} · {row.block_type}\n"
                    f"{run.qc['sfreq']:.2f} Hz · {run.qc['duration_s']:.1f} s\n"
                    f"Trials: {len(run.trials)} · épocas aceitas: {len(epochs)}\n"
                    f"Canais ruins: {run.qc['bad_channels']}\n"
                    f"Filtro: {cfg['l_freq']}–{cfg['h_freq']} Hz\n"
                    f"Referência: {cfg['reference']}\n"
                    f"Baseline: {cfg['baseline']} s\nAtivo: {cfg['active']} s\n"
                    f"PAUSAs inferidas: {run.qc['inferred_pauses']}\n"
                    "Curvas/BA offline não demonstram aprendizado causal.", va="top", fontsize=10)
    fig.suptitle(row.run_id)
    save_figure(fig, directory, "dashboard", cfg)
    if detailed:
        detailed_trials(run, before, epochs, report)
    statuses = [dict(análise="Eletrofisiologia", resultado="executada" if mode in {"both", "ephys"} else "não solicitada"),
                dict(análise="ERD/ERS", resultado="executada" if power is not None else ("sem baseline PAUSA utilizável" if mode != "classification" else "não solicitada")),
                dict(análise="Classificação", resultado="não solicitada" if mode == "ephys" else "; ".join(f"{m['pipeline']} / {m['protocol']}: {m['status']}" for m in metrics))]
    report.items.insert(1, ("Resumo · análises e disponibilidade", pd.DataFrame(statuses).to_html(index=False), True))
    provenance["classification"] = metrics
    (directory / "parameters_qc.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False))
    report.save(directory / "report.html", overwrite=True, open_browser=False, verbose=False)
    return dict(run_id=row.run_id, subject=row.subject, session=row.session, block_type=row.block_type,
                status="ok", accepted_epochs=len(epochs), classification=metrics)


def diagnose_run(root, row, cfg, protocols=None):
    """Pré-checagem sem treinar modelos; preserva todas as causas verificáveis por etapa."""
    checks = []
    result = dict(run_id=row.run_id, subject=row.subject, session=row.session,
                  block_type=row.block_type, config=cfg,
                  versions={p: importlib.metadata.version(p) for p in ["mne", "numpy", "scipy", "scikit-learn", "pyriemann"]},
                  checks=checks)

    def add(stage, status, message, action=""):
        checks.append(dict(stage=stage, status=status, message=message, action=action))

    for field, required in [("signal_path", {"lsl_time_s"}), ("marker_path", {"lsl_time_s", "label"})]:
        try:
            frame   = pd.read_csv(Path(root) / row[field])
            missing = required - set(frame)
            if missing:
                raise ValueError(f"Colunas ausentes: {sorted(missing)}")
            add(field, "ok", f"{len(frame)} linhas; {len(frame.columns)} colunas.")
            if field == "marker_path":
                result["marker_counts"] = frame.label.value_counts().to_dict()
        except Exception as exc:
            add(field, "error", f"{type(exc).__name__}: {exc}", "Conferir caminho, disponibilidade e cabeçalho do CSV.")
    run = None
    if not any(c["status"] == "error" for c in checks):
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                run = load_run(root, row, cfg)
            result["qc"] = run.qc
            add("preprocessing", "ok", f"{run.raw.n_times} amostras; {run.raw.info['sfreq']:.4f} Hz.")
            for warning in captured:
                add("preprocessing", "warning", str(warning.message), "Consultar qc/timing; não há deslocamento de marcadores.")
            invalid = run.trials[~run.trials.valid]
            if len(invalid):
                add("trials", "warning", f"{len(invalid)} trials inválidos/excluídos: {invalid.trial.tolist()}", "Conferir sequência dos marcadores e exclusões manuais.")
            durations                 = {phase: (run.trials[f"{phase}_end"] - run.trials[f"{phase}_start"]).describe().to_dict()
                         for phase in ["attention", "attempt", "pause"]}
            result["phase_durations"] = durations
        except Exception as exc:
            add("preprocessing", "error", f"{type(exc).__name__}: {exc}", "Conferir timestamps, unidades, canais e configuração; detalhes no traceback.")
            result["traceback"] = traceback.format_exc()
    if run is not None:
        try:
            _, epochs, audit          = make_epochs(run, cfg)
            result["epoch_audit"]     = audit.to_dict("records")
            result["accepted_epochs"] = len(epochs)
            add("epochs", "ok" if len(epochs) >= 2 else "error",
                f"{len(epochs)}/{len(audit)} épocas aceitas; rejeições: {audit.loc[~audit.accepted, 'drop_reason'].value_counts().to_dict()}",
                "Revisar artefatos, unidades e canais; não aumentar limiar automaticamente." if len(epochs) < len(audit) else "")
            if len(epochs):
                valid, baseline          = baseline_trials(run, epochs, cfg)
                result["baseline_audit"] = baseline.to_dict("records")
                action                   = ""
                if not valid.any():
                    intervals = []
                    for trial in epochs.metadata.itertuples():
                        previous = run.trials[run.trials.trial < trial.trial]
                        if len(previous):
                            intervals.append((previous.iloc[-1].pause_start - trial.attempt_start + 0.25,
                                              trial.attention_start - trial.attempt_start - 0.25))
                    if intervals:
                        low  = max(v[0] for v in intervals)
                        high = min(v[1] for v in intervals)
                        if high - low >= 0.5:
                            suggestion                     = dict(baseline=[round(low, 3), round(high, 3)],
                                              epoch_tmin=round(min(cfg["epoch_tmin"], low - 0.5), 3))
                            result["suggested_erd_config"] = suggestion
                            action                         = f"PAUSA comum com margem de 250 ms: {suggestion}. Sugestão não aplicada; revisar e reexecutar check."
                    if not action:
                        action = "Sem PAUSA comum suficiente: revisar protocolo ou usar baseline por trial; não substituir por ATTENTION."
                add("erd_baseline", "ok" if valid.any() else "warning", f"{int(valid.sum())}/{len(epochs)} baselines contidas em PAUSA.", action)
                from classify import prepare, trial_splits
                for protocol in protocols or list(PROTOCOLS):
                    try:
                        datasets, rejected = prepare(run, cfg, protocol, set(epochs.metadata.trial),
                                                     require_cv=not row.block_type.startswith("online_"))
                        meta               = datasets[0][2]
                        if row.block_type.startswith("online_"):
                            add(protocol, "ok", f"{len(datasets)} candidatos de teste; {meta.trial.nunique()} trials. Treino proxy verificado separadamente; modelos não ajustados.")
                        else:
                            outer = trial_splits(meta, cfg["outer_folds"], cfg["seed"])
                            for train, _ in outer:
                                trial_splits(meta.iloc[train].reset_index(drop=True), cfg["inner_folds"], cfg["seed"])
                            add(protocol, "ok", f"{len(datasets)} candidatos; {meta.trial.nunique()} trials; {len(outer)} folds externos viáveis. Ajuste numérico dos modelos ainda não executado.")
                        if rejected:
                            add(protocol, "warning", f"{len(rejected)} candidatos sem suporte temporal/classes.", "Revisar duração de janela e offset.")
                    except Exception as exc:
                        add(protocol, "warning", f"Indisponível: {type(exc).__name__}: {exc}", "Conferir classes, duração de fases e número de trials; não usar CV por janela.")
        except Exception as exc:
            add("epochs", "error", f"{type(exc).__name__}: {exc}", "Conferir parâmetros de época e sequência de eventos.")
            result["traceback"] = traceback.format_exc()
    result["status"] = "error" if any(c["status"] == "error" for c in checks) else (
        "warning" if any(c["status"] == "warning" for c in checks) else "ok")
    return result


def check_dataset(root, control, output, cfg, subjects=None, sessions=None, blocks=None, runs=None, protocols=None):
    selected = selection(control, subjects, sessions, blocks, runs)
    if selected.empty:
        raise ValueError("Nenhum registro incluído corresponde aos filtros.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for _, row in selected.iterrows():
        print(f"Check: {row.run_id}", flush=True)
        result = diagnose_run(root, row, cfg, protocols)
        if row.block_type.startswith("online_"):
            try:
                source = calibration_row(row, selection(control))
                result["checks"].append(dict(stage="online_training", status="ok", message=f"Calibração proxy anterior: {source.run_id}", action="Confirmar que corresponde à coleta; treino ainda não ajustado."))
            except ValueError as exc:
                result["checks"].append(dict(stage="online_training", status="warning", message=str(exc), action="Incluir calibração anterior na tabela ou definir train_run_id."))
                if result["status"] == "ok":
                    result["status"] = "warning"

        folder = output / row.run_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "diagnostics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        results.append(result)
        # Checkpoint por registro: diagnóstico parcial sobrevive a interrupções.
        (output / "diagnostics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    flat = [dict(run_id=r["run_id"], **c) for r in results for c in r["checks"]]
    pd.DataFrame(flat).to_csv(output / "diagnostics.csv", index=False)
    page = pd.DataFrame(flat).to_html(index=False, escape=True)
    (output / "diagnostics.html").write_text('<!doctype html><meta charset="utf-8"><title>Check EEG</title>'
        '<style>body{font:14px system-ui;margin:24px}td{padding:10px;max-width:650px;overflow-wrap:anywhere}table{border-collapse:collapse}</style>'
        '<h1>Diagnóstico por registro e etapa</h1><p>ok: etapa verificada; warning: limitação/ação recomendada; error: impedimento. Modelos não são treinados neste check.</p>' + page, encoding="utf-8")
    return results


def batch(root, control, output, cfg, subjects=None, sessions=None, blocks=None, runs=None,
          mode="both", detailed=False, protocols=None, compare=False):
    selected = selection(control, subjects, sessions, blocks, runs)
    included = selection(control)
    if selected.empty:
        raise ValueError("Nenhum registro incluído corresponde aos filtros. Edite include e block_type na tabela.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output / "selected_control.csv", index=False)
    check_dataset(root, control, output, cfg, subjects, sessions, blocks, runs, protocols)
    results = []
    for _, row in selected.iterrows():
        print(f"Analisando {row.run_id}", flush=True)
        try:
            results.append(analyze_run(root, row, cfg, output / row.run_id, mode, detailed, protocols, included, compare))
        except Exception as exc:
            traceback.print_exc()
            folder = output / row.run_id
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
            results.append(dict(run_id=row.run_id, subject=row.subject, session=row.session,
                                block_type=row.block_type, status=f"{type(exc).__name__}: {exc}"))
        finally:
            plt.close("all")
    batch_comparison(output, results)
    (output / "batch_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    flat = [{**{k: v for k, v in r.items() if k != "classification"}, **metric}
            for r in results for metric in (r.get("classification") or [dict(protocol="ephys")])]
    pd.DataFrame(flat).to_csv(output / "summary.csv", index=False)
    # The general report includes comparisons and individual links in one document.
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["index", "check", "batch", "report"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--output", type=Path, default=Path("analysis_outputs"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--sessions", nargs="+")
    parser.add_argument("--blocks", nargs="+")
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--protocols", nargs="+", choices=list(PROTOCOLS))
    parser.add_argument("--compare-pipelines", action="store_true", help="Compara pipelines separadamente, com busca aninhada em cada uma.")
    parser.add_argument("--mode", choices=["ephys", "classification", "both"], default="both")
    args    = parser.parse_args()
    control = args.control or args.root / "dataset_index.csv"
    if args.command == "index":
        frame = scan(args.root, control)
        print(f"{len(frame)} registros em {control}; novos registros começam excluídos.")
    else:
        # Subdiretório único evita confundir arquivos de execuções com parâmetros diferentes.
        output  = args.output / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        if args.command == "check":
            results = check_dataset(args.root, control, output, settings(args.config), args.subjects, args.sessions,
                                    args.blocks, args.runs, args.protocols)
            print(f"Diagnóstico: {output.resolve()}")
            raise SystemExit(int(any(r["status"] == "error" for r in results)))
        results = batch(args.root, control, output, settings(args.config), args.subjects, args.sessions,
                        args.blocks, args.runs, args.mode, args.command == "report", args.protocols, args.compare_pipelines)
        print(f"Relatórios: {output.resolve()}")
        if any(r["status"] != "ok" or any(m["status"] != "ok" for m in r.get("classification", [])) for r in results):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
