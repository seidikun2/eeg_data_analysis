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

from eeg import COLORS, PROTOCOLS, fingerprint, load_run, make_epochs, scan, selection, settings

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "pdf.fonttype": 42})
mne.set_log_level("WARNING")


def save_figure(fig, directory, name, cfg, report=None, title=None):
    fig.savefig(directory / f"{name}.png", dpi=cfg["dpi"])
    fig.savefig(directory / f"{name}.pdf")
    if report is not None:
        report.add_figure(fig, title=title or name, image_format="png")
    plt.close(fig)


def stacked(ax, data, times, names, scale, title):
    offset = np.arange(len(names))[::-1] * scale
    for i in range(len(names)):
        ax.plot(times, data[i] * 1e6 + offset[i], color="0.25", lw=0.45)
    ax.set(yticks=offset, yticklabels=names, xlabel="Tempo (s)", title=title)
    ax.text(1, 1.02, f"espaçamento: {scale:.0f} µV", transform=ax.transAxes, ha="right", fontsize=8)


def raw_figure(run, cfg):
    stop      = min(run.raw.n_times, int(20 * run.raw.info["sfreq"]))
    picks     = mne.pick_types(run.clean.info, eeg=True, exclude="bads")
    names     = [run.raw.ch_names[i] for i in picks]
    data      = run.raw.get_data(picks=picks, stop=stop)
    scale     = max(20, float(np.quantile(np.ptp(data, axis=1) * 1e6, 0.8)))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, layout="constrained")
    for ax, raw, label in zip(axes, [run.raw, run.clean], ["Antes", "Depois"]):
        stacked(ax, raw.get_data(picks=picks, stop=stop), raw.times[:stop], names, scale, label)
    return fig


def erp_plot(ax, epochs, cfg):
    times = epochs.times
    for cls in epochs.event_id:
        values = epochs[cls].get_data(picks=mne.pick_types(epochs.info, eeg=True, exclude="bads")) * 1e6
        values = values - values[:, :, (times >= cfg["baseline"][0]) & (times <= cfg["baseline"][1])].mean(axis=-1, keepdims=True)
        mean   = values.mean(axis=0)
        ax.plot(times, mean.T, color=COLORS[cls], alpha=0.35, lw=0.6)
        ax.plot([], [], color=COLORS[cls], label=f"{cls} · n={len(values)}")
    ax.axvline(0, color="0.3", ls="--", lw=0.8)
    ax.set(title="ERP: todos os canais, uma cor por classe", xlabel="Tempo desde ATTEMPT (s)", ylabel="µV")
    ax.legend(fontsize=7)


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


def baseline_trials(run, epochs, cfg):
    valid, reasons = [], []
    for row in epochs.metadata.itertuples():
        start, end = np.asarray(cfg["baseline"]) + row.attempt_start
        previous   = run.trials[run.trials.trial < row.trial]
        pause      = previous.iloc[-1].pause_start if len(previous) else np.nan
        ok         = bool(np.isfinite(pause) and start >= pause and end <= row.attention_start)
        valid.append(ok)
        reasons.append(dict(trial=int(row.trial), baseline_in_pause=ok,
                            reason="" if ok else "Baseline não contida em PAUSA anterior conhecida; ERD excluído."))
    return np.array(valid), pd.DataFrame(reasons)


def time_frequency(run, epochs, cfg):
    valid, audit = baseline_trials(run, epochs, cfg)
    if not valid.any():
        return None, audit
    used  = epochs[np.flatnonzero(valid)]
    freqs = np.array(cfg["frequencies"], float)
    if freqs.max() > cfg["h_freq"] or freqs.min() < cfg["l_freq"]:
        raise ValueError("Frequências TFR fora do filtro eletrofisiológico.")
    power = used.compute_tfr(method="morlet", freqs=freqs, n_cycles=freqs / 2,
                             return_itc=False, average=False, picks="eeg", decim=max(1, round(used.info["sfreq"] / 64)),
                             n_jobs=1, verbose=False)
    power.apply_baseline(tuple(cfg["baseline"]), mode="percent", verbose=False)
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
    topo.suptitle(f"Intervalo ativo: {cfg['active']} s · mesma escala em todos os mapas")
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
    Z    = TangentSpace(metric="riemann").fit_transform(Covariances(estimator="oas").fit_transform(data))
    Z    = StandardScaler().fit_transform(Z)
    n    = min(3, *Z.shape)
    if n < 2:
        raise ValueError("PCA requer pelo menos duas épocas e duas features.")
    pca    = PCA(n_components=n).fit(Z)
    pos    = pca.transform(Z)
    fig    = plt.figure(figsize=(12, 5), layout="constrained")
    left   = fig.add_subplot(121)
    right  = fig.add_subplot(122, projection="3d" if n == 3 else None)
    labels = epochs.metadata.trial_class.to_numpy()
    for cls in np.unique(labels):
        keep = labels == cls
        left.scatter(pos[keep, 0], pos[keep, 1], c=COLORS[cls], label=cls, s=25)
        if n == 3:
            right.scatter(*pos[keep].T, c=COLORS[cls], s=25)
        else:
            right.text(0.1, 0.5, "Trials insuficientes para PCA 3D")
    left.set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    left.legend()
    if n == 3:
        right.set(xlabel="PC1", ylabel="PC2", zlabel="PC3")
    fig.suptitle("PCA exploratório · uma observação por trial · não usado para estimar desempenho")
    return fig


def classifier_figure(summary, predictions):
    labels    = summary["labels"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    cm        = np.array(summary["confusion_matrix"])
    norm      = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    axes[0, 0].imshow(norm, cmap="Greys", vmin=0, vmax=1)
    for i, j in np.ndindex(cm.shape):
        axes[0, 0].text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", color="white" if norm[i, j] > 0.5 else "black")
    axes[0, 0].set(xticks=range(len(labels)), xticklabels=labels, yticks=range(len(labels)), yticklabels=labels,
                   xlabel="Predito", ylabel="Real", title="Matriz por trial/fase · probabilidades médias")
    scores = [f["balanced_accuracy"] for f in summary["folds"]]
    axes[0, 1].plot(range(1, len(scores) + 1), scores, "o-", color="0.3")
    axes[0, 1].axhline(summary["chance"], color="0.5", ls="--")
    axes[0, 1].set(ylim=(0, 1), xlabel="Fold externo / avaliação externa", ylabel="Balanced accuracy", title=f"BA agrupada: {summary['balanced_accuracy']:.3f}")
    trials = sorted(predictions.trial.unique())
    rank   = {t: i / max(1, len(trials) - 1) for i, t in enumerate(trials)}
    for cls in labels:
        # Não interpolar através das lacunas de fases nem prolongar trials curtos.
        sub    = predictions[predictions.target == cls]
        traces = []
        for trial, values in sub.groupby("trial", sort=True):
            values = values.sort_values("time_end")
            axes[1, 0].plot(values.time_end, values[f"p_{cls}"], color=COLORS[cls], alpha=0.08 + 0.35 * rank[trial], lw=0.7)
            trace = pd.Series(values[f"p_{cls}"].to_numpy(), index=values.time_end.round(4))
            traces.append(trace[~trace.index.duplicated()])
        if traces:
            frame = pd.concat(traces, axis=1)
            mean  = frame.mean(axis=1).sort_index()
            axes[1, 0].plot(mean.index, mean.values, color=COLORS[cls], lw=2, label=f"P({cls}) | real={cls}")
        for _, values in predictions.groupby(["trial", "phase"], sort=True):
            values = values.sort_values("start_abs")
            axes[1, 1].plot(values.start_abs - predictions.start_abs.min(), values[f"p_{cls}"], color=COLORS[cls], lw=0.7, alpha=0.7)
    for ax in axes[1]:
        ax.axhline(summary["chance"], color="0.5", ls="--", lw=0.8)
        ax.set(ylim=(0, 1), ylabel="Probabilidade calibrada")
    axes[1, 0].axvline(0, color="0.4", ls=":")
    axes[1, 0].set(title="Trials antigos: claros → recentes: escuros; média destacada", xlabel="Fim da janela relativo ao ATTEMPT (s)")
    axes[1, 0].legend(fontsize=7)
    axes[1, 1].set(title="Saídas em janelas de teste · lacunas sem inferência", xlabel="Tempo desde primeira janela (s)")
    fig.suptitle(f"{summary['protocol']} · {summary['validation']} · fonte REST: {summary['rest_source']}")
    return fig


def detailed_trials(run, before, after, report):
    accepted = {int(t): i for i, t in enumerate(after.metadata.trial)}
    picks    = mne.pick_types(before.info, eeg=True, exclude="bads")
    names    = [before.ch_names[i] for i in picks]
    for i, row in enumerate(before.metadata.itertuples()):
        fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, layout="constrained")
        data      = before.get_data(picks=picks)[i]
        scale     = max(20, float(np.quantile(np.ptp(data, axis=1) * 1e6, 0.8)))
        stacked(axes[0], data, before.times, names, scale, "Antes do pré-processamento")
        if row.trial in accepted:
            stacked(axes[1], after.get_data(picks=picks)[accepted[row.trial]], after.times, names, scale, "Depois do pré-processamento")
        else:
            first = before.events[i, 0] + int(round(before.times[0] * before.info["sfreq"]))
            data  = run.clean.get_data(picks=picks, start=first, stop=first + len(before.times))
            stacked(axes[1], data, before.times, names, scale, "Depois · REJEITADA para análises; consultar auditoria")
        fig.suptitle(f"Trial {row.trial} · {row.trial_class}", color=COLORS[row.trial_class])
        report.add_figure(fig, title=f"Trial {row.trial} · {row.trial_class}", section="Épocas individuais", image_format="png")
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


def analyze_run(root, row, cfg, directory, mode="both", detailed=False, protocols=None, included=None):
    directory.mkdir(parents=True, exist_ok=True)
    run                   = load_run(root, row, cfg)
    before, epochs, audit = make_epochs(run, cfg)
    audit.to_csv(directory / "epoch_audit.csv", index=False)
    run.trials.to_csv(directory / "trials.csv", index=False)
    if len(epochs) < 2:
        raise ValueError("Menos de duas épocas aceitas; consultar epoch_audit.csv.")
    report     = mne.Report(title=f"{row.subject} · {row.session} · {row.block_type} · {row.run_id}", verbose=False)
    provenance = dict(created_utc=datetime.now(timezone.utc).isoformat(), config=cfg, control=row.to_dict(), qc=run.qc,
                      files={name: fingerprint(Path(root) / row[name]) for name in ["signal_path", "marker_path"]},
                      versions={p: importlib.metadata.version(p) for p in ["mne", "numpy", "scipy", "scikit-learn", "pyriemann"]},
                      notes=["Análise offline; não reproduz automaticamente o pré-processamento do sistema de coleta.",
                             "Referência média altera o sinal; canais ruins são excluídos, sem interpolação.",
                             "Nenhum ICA automático sem EOG ou inspeção; amplitude/flatness não identificam todo artefato.",
                             "ERP é inspeção descritiva; ausência de efeito não prova ausência de resposta."])
    report.add_html("<pre>" + html.escape(json.dumps(provenance, indent=2, ensure_ascii=False)) + "</pre>", title="Parâmetros, proveniência e sanity check")
    report.add_html(audit.to_html(index=False), title="Aceitação/rejeição das épocas")
    metrics, power, primary = [], None, None
    if mode in {"both", "ephys"}:
        save_figure(raw_figure(run, cfg), directory, "raw_antes_depois", cfg, report)
        save_figure(epoch_figure(epochs), directory, "epocas_canais", cfg, report)
        save_figure(pca_figure(epochs, cfg), directory, "pca_2d_3d", cfg, report)
        power, baseline_audit = time_frequency(run, epochs, cfg)
        baseline_audit.to_csv(directory / "baseline_audit.csv", index=False)
        report.add_html(baseline_audit.to_html(index=False), title="Validade da baseline para ERD/ERS")
        for name, fig in tfr_figures(power, cfg):
            save_figure(fig, directory, name, cfg, report)
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), layout="constrained")
        erp_plot(axes[0], epochs, cfg)
        spectral_plot(axes[1], epochs, cfg)
        save_figure(fig, directory, "erp_espectro", cfg, report)
        data             = run.clean.get_data(picks="eeg", reject_by_annotation="omit")
        freq, time, spec = spectrogram(data, fs=run.clean.info["sfreq"], nperseg=min(256, data.shape[-1]), axis=-1)
        keep             = (freq >= 1) & (freq <= cfg["h_freq"])
        fig, ax          = plt.subplots(figsize=(12, 4), layout="constrained")
        im               = ax.pcolormesh(time, freq[keep], 10 * np.log10(np.maximum(spec[:, keep].mean(axis=0) * 1e12, 1e-20)), cmap="Greys", shading="auto")
        ax.set(title="Espectrograma contínuo · média de potência dos canais", xlabel="Tempo (s)", ylabel="Hz")
        fig.colorbar(im, ax=ax, label="dB (µV²/Hz)")
        save_figure(fig, directory, "espectrograma", cfg, report)
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
        for protocol in protocols or list(PROTOCOLS):
            try:
                if train_source.startswith("BLOCKED:"):
                    raise ValueError(train_source)
                result = evaluate(run, cfg, protocol, set(epochs.metadata.trial), training)
                if primary is None:
                    primary = result
                result[0]["train_source"] = train_source
                save_result(result, directory)
                save_figure(classifier_figure(result[0], result[1]), directory, f"classificacao_{protocol}", cfg, report)
                metrics.append(dict(protocol=protocol, status="ok", balanced_accuracy=result[0]["balanced_accuracy"],
                                    chance=result[0]["chance"], validation=result[0]["validation"], train_source=train_source))
            except (ValueError, np.linalg.LinAlgError) as exc:
                metrics.append(dict(protocol=protocol, status=str(exc)))
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
        predictions = primary[1]
        trials      = sorted(predictions.trial.unique())
        rank        = {t: i / max(1, len(trials) - 1) for i, t in enumerate(trials)}
        for cls in primary[0]["labels"]:
            traces = []
            for trial, sub in predictions[predictions.target == cls].groupby("trial"):
                sub = sub.sort_values("time_end")
                axes[2, 2].plot(sub.time_end, sub[f"p_{cls}"], color=COLORS[cls], alpha=0.08 + 0.35 * rank[trial], lw=0.7)
                traces.append(pd.Series(sub[f"p_{cls}"].to_numpy(), index=sub.time_end.round(4)))
            if traces:
                mean = pd.concat(traces, axis=1).mean(axis=1).sort_index()
                axes[2, 2].plot(mean.index, mean.values, color=COLORS[cls], lw=2, label=cls)
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
    save_figure(fig, directory, "dashboard", cfg, report)
    if detailed:
        detailed_trials(run, before, epochs, report)
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
            frame = pd.read_csv(Path(root) / row[field])
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
            durations = {phase: (run.trials[f"{phase}_end"] - run.trials[f"{phase}_start"]).describe().to_dict()
                         for phase in ["attention", "attempt", "pause"]}
            result["phase_durations"] = durations
        except Exception as exc:
            add("preprocessing", "error", f"{type(exc).__name__}: {exc}", "Conferir timestamps, unidades, canais e configuração; detalhes no traceback.")
            result["traceback"] = traceback.format_exc()
    if run is not None:
        try:
            _, epochs, audit = make_epochs(run, cfg)
            result["epoch_audit"] = audit.to_dict("records")
            result["accepted_epochs"] = len(epochs)
            add("epochs", "ok" if len(epochs) >= 2 else "error",
                f"{len(epochs)}/{len(audit)} épocas aceitas; rejeições: {audit.loc[~audit.accepted, 'drop_reason'].value_counts().to_dict()}",
                "Revisar artefatos, unidades e canais; não aumentar limiar automaticamente." if len(epochs) < len(audit) else "")
            if len(epochs):
                valid, baseline = baseline_trials(run, epochs, cfg)
                result["baseline_audit"] = baseline.to_dict("records")
                action = ""
                if not valid.any():
                    intervals = []
                    for trial in epochs.metadata.itertuples():
                        previous = run.trials[run.trials.trial < trial.trial]
                        if len(previous):
                            intervals.append((previous.iloc[-1].pause_start - trial.attempt_start + 0.25,
                                              trial.attention_start - trial.attempt_start - 0.25))
                    if intervals:
                        low = max(v[0] for v in intervals)
                        high = min(v[1] for v in intervals)
                        if high - low >= 0.5:
                            suggestion = dict(baseline=[round(low, 3), round(high, 3)],
                                              epoch_tmin=round(min(cfg["epoch_tmin"], low - 0.5), 3))
                            result["suggested_erd_config"] = suggestion
                            action = f"PAUSA comum com margem de 250 ms: {suggestion}. Sugestão não aplicada; revisar e reexecutar check."
                    if not action:
                        action = "Sem PAUSA comum suficiente: revisar protocolo ou usar baseline por trial; não substituir por ATTENTION."
                add("erd_baseline", "ok" if valid.any() else "warning", f"{int(valid.sum())}/{len(epochs)} baselines contidas em PAUSA.", action)
                from classify import prepare, trial_splits
                for protocol in protocols or list(PROTOCOLS):
                    try:
                        datasets, rejected = prepare(run, cfg, protocol, set(epochs.metadata.trial),
                                                     require_cv=not row.block_type.startswith("online_"))
                        meta = datasets[0][2]
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
          mode="both", detailed=False, protocols=None):
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
            results.append(analyze_run(root, row, cfg, output / row.run_id, mode, detailed, protocols, included))
        except Exception as exc:
            traceback.print_exc()
            folder = output / row.run_id
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
            results.append(dict(run_id=row.run_id, subject=row.subject, session=row.session,
                                block_type=row.block_type, status=f"{type(exc).__name__}: {exc}"))
        finally:
            plt.close("all")
    (output / "batch_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    flat = [{**{k: v for k, v in r.items() if k != "classification"}, **metric}
            for r in results for metric in r.get("classification", [dict(protocol="ephys")])]
    pd.DataFrame(flat).to_csv(output / "summary.csv", index=False)
    cards = []
    for row in results:
        rid     = html.escape(row["run_id"], quote=True)
        status  = html.escape(row["status"])
        content = f'<a href="{rid}/report.html">Relatório detalhado</a><img src="{rid}/dashboard.png" alt="Dashboard">' if row["status"] == "ok" else f'<a href="{rid}/diagnostics.json">Diagnóstico</a> · <a href="{rid}/error_traceback.txt">Erro completo</a>'
        cards.append(f"<article><h2>{rid}</h2><p>{status}</p>{content}</article>")
    (output / "index.html").write_text('<!doctype html><html lang="pt-BR"><meta charset="utf-8">'
        '<title>EEG · análise batch</title><style>body{font:16px system-ui;max-width:1500px;margin:32px auto;padding:20px;background:#f6f6f6}article{background:white;padding:24px;margin:24px 0}img{width:100%;height:auto}h2{overflow-wrap:anywhere}</style>'
        '<h1>EEG · análise batch</h1><p><a href="diagnostics.html">Check de todos os registros</a></p><p>Resultados descritivos offline. Consulte parâmetros, rejeições e validação em cada relatório.</p>'
        + "".join(cards) + "</html>")
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
                        args.blocks, args.runs, args.mode, args.command == "report", args.protocols)
        print(f"Relatórios: {output.resolve()}")
        if any(r["status"] != "ok" or any(m["status"] != "ok" for m in r.get("classification", [])) for r in results):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
