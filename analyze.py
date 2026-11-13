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
import time
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from reporting import Report, timeline_plot, batch_comparison
from analysis_views import stacked_snapshot, epoch_panels, tangent_pca, robust_limit, spectral_figures
from eeg import COLORS, PROTOCOLS, fingerprint, load_run, make_epochs, scan, selection, settings

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "pdf.fonttype": 42})
mne.set_log_level("WARNING")


def save_figure(fig, directory, name, cfg, report=None, title=None):
    fig.savefig(directory / f"{name}.png", dpi=cfg["dpi"])
    fig.savefig(directory / f"{name}.pdf")
    if report is not None:
        labels = {"raw_antes_depois": "Snapshot 20 s · canais em offset · cinza/vermelho", "sanity_checks": "Sanity checks · amplitude, ruído e rejeição",
                  "epocas_canais": "Épocas · mapa por canal", "pca_2d_3d": "PCA · 2D e 3D", "dashboard": "Dashboard · visão geral",
                  "tempo_frequencia": "Tempo–frequência · visão global", "topografias_mu_beta": "Topografias · mu e beta", "espectrograma": "Espectrograma · canais centrais"}
        report.add_figure(fig, title=title or labels.get(name, name.replace("_", " · ")), image_format="png")
    plt.close(fig)


def central_picks(info):
    return [i for i, name in enumerate(info["ch_names"]) if name in {"C3", "Cz", "C4"} and name not in info["bads"]]


def quality_checks(run, audit, cfg):
    from scipy.signal import welch
    before    = run.raw.get_data() * 1e6
    after     = run.clean.get_data() * 1e6
    std       = after.std(axis=1)
    amplitude = np.diff(np.quantile(before, [.25,.75], axis=1),axis=0)[0] * .7413
    spread = float(np.subtract(*np.quantile(amplitude,[.75,.25])) * .7413)
    robust_z = (amplitude-np.median(amplitude))/spread if spread > 1e-12 else np.full(len(amplitude),np.nan)
    near_constant = np.ptp(before,axis=1) < cfg['flat_uv']
    flags = ['quase constante' if flat else ('amplitude atípica; revisar' if np.isfinite(z) and abs(z)>5 else 'sem alerta relativo') for flat,z in zip(near_constant,robust_z)]
    ratio     = std / max(np.median(std), 1e-12)
    freq, psd = welch(before, fs=run.raw.info['sfreq'], nperseg=min(before.shape[-1], int(4 * run.raw.info['sfreq'])), axis=-1)
    total     = psd[:, (freq >= 1) & (freq < min(100, run.raw.info['sfreq'] / 2))].sum(axis=1)
    line      = psd[:, ((abs(freq - 50) <= 1) | (abs(freq - 60) <= 1))].sum(axis=1) / np.maximum(total, 1e-20) * 100
    if "source_line_noise_percent" in run.qc:
        line=np.array([run.qc["source_line_noise_percent"][name] for name in run.raw.ch_names],dtype=float)
    rejected  = [sum(name in str(reason).split(';') for reason in audit.drop_reason) for name in run.raw.ch_names]
    frame     = pd.DataFrame(dict(channel=run.raw.ch_names, raw_dc_uv=np.median(before, axis=1),
                              raw_std_uv=before.std(axis=1), clean_std_uv=std, relative_std=ratio,
                              raw_peak_to_peak_uv=np.ptp(before, axis=1), clean_peak_to_peak_uv=np.ptp(after, axis=1),
                              robust_amplitude_uv=amplitude, robust_deviation=robust_z, near_constant=near_constant,
                              line_noise_percent=line, rejected_epochs=rejected,
                              excluded=[c in run.raw.info['bads'] for c in run.raw.ch_names],
                              flag=flags))
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), layout='constrained')
    x         = np.arange(len(frame))
    axes[0].bar(x - .18, frame.raw_std_uv, .36, color='0.6', label='Antes')
    axes[0].bar(x + .18, frame.clean_std_uv, .36, color='#0072B2', label='Depois')
    axes[0].set(yscale='log', ylabel='Desvio-padrão (µV)', title='Variação por canal · escala logarítmica; offsets DC na tabela')
    axes[0].legend()
    axes[1].bar(x, line, color='0.45')
    axes[1].set(ylabel='% da potência 1–100 Hz', title=f'Rede 50/60 Hz ±1 Hz · aquisição {run.qc.get("source_sfreq",run.raw.info["sfreq"]):.0f} Hz · antes de antialias/low-pass')
    axes[2].bar(x, rejected, color='0.45')
    axes[2].set(ylabel='Épocas rejeitadas', title=f'{int((~audit.accepted).sum())}/{len(audit)} épocas rejeitadas · contribuição por canal (pode haver múltiplas causas)')
    for ax in axes:
        ax.set_xticks(x, frame.channel, rotation=30, ha='right')
    return frame, fig


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


def baseline_intervals(run, epochs, cfg):
    intervals = []
    for row in epochs.metadata.itertuples():
        previous = run.trials[run.trials.trial < row.trial]
        pause    = previous.iloc[-1].pause_start if len(previous) else np.nan
        if cfg["erd_baseline_mode"] == "attention":
            start, end = row.attention_start + .1, row.stimulus_start - .1
            valid = bool(np.isfinite(start) and start >= run.times[0] and end <= row.attempt_start and end-start >= .5)
            intervals.append((start-row.attempt_start,end-row.attempt_start) if valid else None)
            continue
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
    audit     = pd.DataFrame([dict(trial=int(row.trial), baseline_in_pause=ok and cfg["erd_baseline_mode"] != "attention", baseline_valid=ok, baseline_phase="ATTENTION" if cfg["erd_baseline_mode"] == "attention" else "PAUSA",
                              baseline_start_s=interval[0] if ok else np.nan,
                              baseline_end_s=interval[1] if ok else np.nan,
                              reason="" if ok else "Baseline ausente/curta/fora do EEG; ERD excluído.")
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
        if row.baseline_valid and row.trial not in keep:
            audit.loc[i, ["baseline_in_pause", "baseline_valid", "reason"]] = [False, False, "Contexto indisponível ou artefato na baseline."]
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


def tfr_figures(power, cfg):
    if power is None:
        return []
    classes   = list(power.event_id)
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
    topo.suptitle(f"ATTEMPT {cfg['active']} s · mu {cfg['mu']} / beta {cfg['beta']} Hz\nBaseline: {cfg['erd_baseline_mode']} (baseline_audit.csv) · escala comum")
    return [("topografias_mu_beta", topo)]


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
    scores = [summary["balanced_accuracy"]] if summary["validation"] == "prior_calibration_proxy" else [f["balanced_accuracy"] for f in summary["folds"]]
    if len(scores)>1:
        axes[1].boxplot([scores],positions=[0],widths=.4,showfliers=False)
    axes[1].scatter((np.linspace(-.12, .12, len(scores)) if len(scores)>1 else [0]), scores, color="0.3")
    if len(scores)>1:
        axes[1].scatter([0],[np.mean(scores)],color='#0072B2',marker='D',label='Média dos folds')
        axes[1].legend()
    axes[1].set_xticks([0], ["Teste online único" if summary["validation"] == "prior_calibration_proxy" else "CV aninhada · folds dependentes"])
    axes[1].axhline(summary["chance"], color="0.5", ls="--")
    axes[1].set(ylim=(0, 1), xlabel="Dispersão dos folds · sem inferência independente", ylabel="Balanced accuracy", title=f"BA agrupada: {summary['balanced_accuracy']:.3f}")
    fig.suptitle(f"{summary['protocol']} · {summary['validation']} · fonte REST: {summary['rest_source']}")
    return fig


def detailed_trials(run, before, after, report, directory=None):
    picks=central_picks(before.info) or list(mne.pick_types(before.info,eeg=True,exclude='bads')[:3])
    raw=before.get_data(picks=picks)*1e6
    cleaned=[]
    for i in range(len(before)):
        first=before.events[i,0]+round(before.times[0]*before.info['sfreq'])
        cleaned.append(run.clean.get_data(picks=picks,start=first,stop=first+len(before.times))*1e6)
    cleaned=np.stack(cleaned)
    limit=robust_limit(np.concatenate([raw.ravel(),cleaned.ravel()]),99)
    fig,axes=plt.subplots(len(picks),2,figsize=(14,3*len(picks)),squeeze=False,layout='constrained')
    for j,ch in enumerate(picks):
        for col,(values,label) in enumerate([(raw,'Antes'),(cleaned,'Depois')]):
            im=axes[j,col].imshow(values[:,j],origin='lower',aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit,
                extent=[before.times[0],before.times[-1],-.5,len(before)-.5])
            axes[j,col].set(title=f'{before.ch_names[ch]} · {label}',xlabel='ATTEMPT (s)',ylabel='Trial (ID original)')
            axes[j,col].set_yticks(range(len(before)),before.metadata.trial.astype(str))
    fig.colorbar(im,ax=axes.ravel().tolist(),label='µV · escala comum P99; inclui épocas rejeitadas')
    if directory is not None:
        fig.savefig(Path(directory)/'epoch_audit.png',dpi=100)
    plt.close(fig)


def calibration_row(row, included):
    specified  = str(row.get("train_run_id", "")).strip()
    candidates = included[(included.subject == row.subject) & (included.block_type == "IM_treino")].copy()
    if "project" in candidates:
        candidates = candidates[candidates.project == row.get("project", "Brasil")]
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
    from reporting import metric_table, combine_sections
    from classify import evaluate, save_result
    directory.mkdir(parents=True, exist_ok=True)
    run = load_run(root, row, cfg)
    classification_run=run
    classification_epochs=None
    training,training_epochs,train_source=None,None,'IM_treino · validação por trial'
    if row.block_type.startswith('online_'):
        try:
            source=calibration_row(row,included)
            train_run=load_run(root,source,cfg)
            # Both representations use exactly the same common usable channels.
            names=[c for c in run.raw.ch_names if c not in run.raw.info['bads'] and c in train_run.raw.ch_names and c not in train_run.raw.info['bads']]
            if len(names)<3:
                raise ValueError('Menos de três canais comuns utilizáveis entre treino e teste.')
            run.qc['transfer_channels']=names
            run.qc['transfer_excluded_channels']=[c for c in run.raw.ch_names if c not in names]
            classification_run=replace(run,raw=run.raw.copy(),clean=run.clean.copy())
            for target in [classification_run,train_run]:
                target.raw=target.raw.copy().pick(names)
                target.clean=target.raw.copy()
                notch=[f for f in cfg['notch'] if f<target.raw.info['sfreq']/2]
                if notch:
                    target.clean.notch_filter(notch,picks='eeg',verbose=False)
                target.clean.filter(cfg['l_freq'],cfg['h_freq'],picks='eeg',verbose=False)
                if cfg['reference']=='average':
                    target.clean.set_eeg_reference('average',projection=False,verbose=False)
            _,classification_epochs,_=make_epochs(classification_run,cfg)
            _,training_epochs,_=make_epochs(train_run,cfg)
            training=(train_run,set(training_epochs.metadata.trial))
            train_source=source.run_id
        except ValueError as exc:
            train_source=f'BLOCKED: {exc}'
    before, epochs, audit = make_epochs(run, cfg)
    if classification_epochs is None:
        classification_epochs=epochs
    audit.to_csv(directory / "epoch_audit.csv", index=False)
    run.trials.to_csv(directory / "trials.csv", index=False)
    if len(epochs) < 2:
        raise ValueError("Menos de duas épocas aceitas; consultar epoch_audit.csv.")
    project = row.get('project','Brasil')
    project_short = {"Brasil":"BR","Italia":"IT"}.get(project,project)
    report = Report(f"{project_short} · {row.subject} · {row.session} · {row.block_type} · {row.run_id}")
    provenance = dict(config=cfg, qc=run.qc, control=row.to_dict(),
                      files={name:fingerprint(Path(root)/row[name]) for name in ['signal_path','marker_path']},
                      versions={p:importlib.metadata.version(p) for p in ['mne','numpy','scipy','scikit-learn','pyriemann']})
    report.add_html(f"<p>Épocas totais (ATTEMPT): <strong>{len(run.trials)}</strong>; elegíveis: {len(audit)}; aceitas: {len(epochs)}; "
                    f"rejeitadas após elegibilidade: {int((~audit.accepted).sum())}; exclusões manuais: {int(run.trials.manual_exclusion.sum())}. "
                    f"Canais excluídos: {html.escape(', '.join(run.raw.info['bads']) or 'nenhum')}.</p>", 'Resumo · registro e épocas')
    processing = dict(Unidade=f"CSV × {cfg['scale_to_volts']} → V; exibição µV", Referência=cfg['reference'],
                      Amostragem=f"{run.qc['source_sfreq']:.2f} → {run.raw.info['sfreq']:.2f} Hz; decimação {run.qc['decimation']} (antialias causal antes de selecionar pontos)",
                      Filtros=f"Ephys FIR {cfg['l_freq']}–{cfg['h_freq']} Hz; notch {cfg['notch']}; classificação Butterworth causal ordem 4",
                      Timestamps=f"{run.qc['timing']['policy']}; correção máxima {run.qc['timing']['max_correction_s']:.6f} s",
                      Épocas=f"{cfg['epoch_tmin']} a {cfg['epoch_tmax']} s desde ATTEMPT", Baseline_ERP=('ATTENTION marcado, margem 100 ms; mínimo 500 ms' if cfg.get('erp_baseline_mode','attention')=='attention' else str(cfg['baseline'])),
                      Baseline_ERD=cfg['erd_baseline_mode'] + (' · ATTENTION até STIMULUS, margem 100 ms' if cfg['erd_baseline_mode']=='attention' else ' · PAUSA anterior auditada'), Ativo=str(cfg['active']),
                      Canais_transferência=str(run.qc.get('transfer_channels','mesmos canais do registro')),
                      Rejeição=f"pico a pico > {cfg['reject_uv']} µV / flat < {cfg['flat_uv']} µV")
    report.add_html(pd.DataFrame(processing.items(),columns=['Processamento','Valor']).to_html(index=False), '⚙️ Processamento')
    quality,qfig=quality_checks(run,audit,cfg)
    quality.to_csv(directory/'channel_quality.csv',index=False)
    flagged=quality[quality.flag!='sem alerta relativo'].channel.tolist()
    report.add_html(f"<p>Alertas de variação relativa: {html.escape(', '.join(flagged) or 'nenhum')}. "
                    f"Motivos de rejeição: {html.escape(str(audit.loc[~audit.accepted,'drop_reason'].value_counts().to_dict()))}. "
                    'Não há exclusão automática de canais.</p>', '✅ Qualidade e alertas')
    report.add_html(quality[['channel','robust_amplitude_uv','robust_deviation','line_noise_percent','rejected_epochs','flag']].round(2).to_html(index=False,na_rep='não mensurável')+'<p>Amplitude robusta = IQR × 0,7413 (µV), desvio relativo em unidades IQR × 0,7413 entre canais; |desvio| > 5 é sinal de revisão inspirado em PREP, não diagnóstico. Quase constante: pico a pico de aquisição abaixo de flat_uv configurado. Rede: Welch 4 s; % da potência 1–min(100, Nyquist) Hz; indisponível sem suporte para ambas as bandas ±1 Hz de 50/60 Hz. Diferenças fisiológicas e referência influenciam amplitudes; alertas nunca excluem canais.</p>','Qualidade · indicadores descritivos')
    save_figure(qfig,directory,'sanity_checks',cfg,report)
    save_figure(stacked_snapshot(run),directory,'raw_antes_depois',cfg,report)
    basic_items=list(report.items)
    metrics,power,primary=[],None,None
    failures=[]
    if mode in {'both','ephys'}:
        erp_audit=[]
        for row_epoch in epochs.metadata.itertuples():
            if cfg.get('erp_baseline_mode','attention')=='attention':
                start,end=row_epoch.attention_start-row_epoch.attempt_start+.1,row_epoch.stimulus_start-row_epoch.attempt_start-.1
            else:
                start,end=cfg['baseline']
            valid=bool(start>=epochs.times[0] and end<=epochs.times[-1] and end-start>=.5)
            erp_audit.append(dict(trial=int(row_epoch.trial),baseline_start_s=start,baseline_end_s=end,valid=valid,mode=cfg.get('erp_baseline_mode','attention')))
        pd.DataFrame(erp_audit).to_csv(directory/'erp_baseline_audit.csv',index=False)
        report.add_html(f"<p>ERP: {sum(r['valid'] for r in erp_audit)}/{len(erp_audit)} baselines utilizáveis; intervalos e exclusões em erp_baseline_audit.csv.</p>",'ERP · referência e suporte')
        save_figure(epoch_panels(epochs,cfg),directory,'erp_canais_classes',cfg,report)
        for name,fig in spectral_figures(epochs,cfg):
            save_figure(fig,directory,name,cfg,report)
        try:
            if row.block_type.startswith('online_') and training_epochs is None:
                raise ValueError('PCA de teste indisponível: IM_treino anterior não disponível; não ajustar no online.')
            pfig,pdata=tangent_pca(classification_epochs,cfg,directory,training_epochs)
            report.add_plotly(pfig,'PCA · TGSP · centroides e dispersão em teste')
            report.add_html('<p>Pontos são janelas reais deslizantes de até 1 s, passo 250 ms; não são trials independentes. Cores distinguem classes; hover mostra trial e janela. Clique na legenda para isolar uma classe. Centroide e covariância atribuem peso total igual a cada trial. Elipse de 1 DP é dispersão descritiva, não intervalo de confiança; omitida com menos de três trials ou covariância singular. Referência tangente, scaler e PCA são ajustados somente no treino; IDs em pca_windows.csv.</p>','PCA · interpretação')
            fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
            for cls,part in pdata.groupby('cls'):
                ax.scatter(part.PC1,part.PC2,color=COLORS[cls],label=cls,alpha=.6)
            ax.set(title='TGSP → PCA · janelas de teste (dependentes)',xlabel='PC1',ylabel='PC2');ax.legend()
            save_figure(fig,directory,'pca_2d_3d',cfg)
        except ValueError as exc:
            report.add_html('<p>'+html.escape(str(exc))+'</p>','PCA · indisponível')
        power,baseline_audit=time_frequency(run,epochs,cfg)
        baseline_audit.to_csv(directory/'baseline_audit.csv',index=False)
        erd_metrics(power,cfg).to_csv(directory/'erd_metrics.csv',index=False)
        if power is not None:
            save_figure(epoch_panels(power,cfg,power=True),directory,'erd_central',cfg,report)
            for name,fig in tfr_figures(power,cfg):
                if name=='topografias_mu_beta':
                    save_figure(fig,directory,name,cfg,report)
                else:
                    plt.close(fig)
        else:
            report.add_html('<p>Sem baseline utilizável após auditoria temporal e de artefatos. baseline_audit.csv registra o motivo por trial.</p>','ERD/ERS · indisponível')
    if mode in {'both','classification'} and row.block_type!='EM_treino':
        variants=cfg['comparison_pipelines'] if compare else cfg['pipelines']
        for variant in variants:
            for classifier in cfg.get('classifiers',['svm_rbf']):
                first=len(report.items)
                for protocol in protocols or list(PROTOCOLS):
                    try:
                        if train_source.startswith('BLOCKED:'):
                            raise ValueError(train_source)
                        experiment={**cfg,'pipelines':[variant],'classifiers':[classifier]}
                        result=evaluate(classification_run,experiment,protocol,set(classification_epochs.metadata.trial),training)
                        result[0].update(train_source=train_source,pipeline=variant,classifier=classifier)
                        if primary is None:
                            primary=result
                        prefix=f'{variant}_{classifier}_'
                        for record in result[0]['fixed_parameter_cv']:
                            record['cv_source_run']=train_source if training is not None else row.run_id
                        save_result(result,directory,prefix)
                        save_figure(classifier_figure(result[0],result[1],cfg),directory,f'classificacao_{prefix}{protocol}',cfg,report,title=f'{protocol} · BACC e folds externos')
                        fixed=result[0]['fixed_parameter_cv']
                        report.add_html(fixed_cv_table(fixed),f'{protocol} · CV de parâmetros fixos (seleção)')
                        report.add_html(pd.DataFrame([dict(**result[0]['deployment_params'],treino=train_source,papel=result[0]['continuous_role'])]).to_html(index=False),f'{protocol} · parâmetros selecionados')
                        if variant == 'tgsp' and classifier == 'svm_rbf':
                            continuous=pd.DataFrame(result[0]['_continuous_timeline'])
                            report.add_plotly(timeline_plot(result[0],continuous,cfg,run.trials),f'{protocol} · timeline contínua')
                        selection_scores=pd.DataFrame(fixed)
                        from classify import parameter_id
                        selected=selection_scores[selection_scores.candidate_id==parameter_id(result[0]['deployment_params'])] if len(selection_scores) else pd.DataFrame()
                        means=selected.balanced_accuracy if len(selected) and selected.status.eq('ok').all() else pd.Series(dtype=float)
                        metrics.append(dict(selection_cv_bacc=float(means.mean()) if len(means) else None,protocol=protocol,pipeline=variant,classifier=classifier,status='ok',balanced_accuracy=result[0]['balanced_accuracy'],chance=result[0]['chance'],validation=result[0]['validation'],train_source=train_source,n_trials=result[0]['n_trials']))
                    except (ValueError,np.linalg.LinAlgError) as exc:
                        from classify import phase_target
                        source='REST_STIM' if (run.trials.trial_class=='REST_STIM').any() else 'REST'
                        available={phase_target(t.trial_class,phase,protocol,source) for t in run.trials.itertuples() if t.trial in set(classification_epochs.metadata.trial) for phase in ['ATTENTION','ATTEMPT','PAUSE']}
                        missing=set(PROTOCOLS[protocol])-available
                        status='Indisponível: '+', '.join(sorted(missing))+' ausente' if missing else 'Erro de execução; consultar auditoria'
                        failures.append(dict(protocol=protocol,pipeline=variant,classifier=classifier,error=str(exc),traceback=traceback.format_exc()))
                        metrics.append(dict(protocol=protocol,pipeline=variant,classifier=classifier,status=status))
                combine_sections(report,first,f'{variant.upper()} + {classifier} · protocolos lado a lado')
        report.add_html(metric_table(pd.DataFrame(metrics)), 'Classificação · resultados e impedimentos')
    elif row.block_type=='EM_treino':
        report.add_html('<p>EM_treino é screening eletrofisiológico de movimento. O modelo é treinado no IM_treino; performance online aparece nos blocos de teste da sessão.</p>','Screening · EM_treino')
    if detailed:
        detailed_trials(run,before,epochs,report,directory)
    (directory/'classification_failures.json').write_text(json.dumps(failures,indent=2,ensure_ascii=False))
    status=[dict(análise='Eletrofisiologia',resultado='executada' if mode!='classification' else 'não solicitada'),
            dict(análise='ERD/ERS',resultado='executada' if power is not None else 'não solicitada ou baseline indisponível'),
            dict(análise='Classificação',resultado='screening; não ajustada' if row.block_type=='EM_treino' else (f"{sum(m['status']=='ok' for m in metrics)} disponíveis; {sum(m['status'].startswith('Indisponível:') for m in metrics)} indisponíveis; {sum(m['status'].startswith('Erro') for m in metrics)} erros" if metrics else 'não solicitada'))]
    report.items.insert(2,('Resumo · análises e disponibilidade',pd.DataFrame(status).to_html(index=False),True))
    report.add_html('<p>Auditoria: parameters_qc.json, trials.csv, epoch_audit.csv, channel_quality.csv, baseline_audit.csv; *_fixed_cv.csv preserva cada fold e parametrização. Decisões e valores extremos não são removidos dos arquivos.</p>','Auditoria · arquivos')
    provenance['classification']=metrics
    (directory/'parameters_qc.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False))
    report.save(directory/'report.html')
    check=Report(f'Check individual · {row.run_id}')
    check.items=basic_items
    check.add_html(pd.DataFrame(status).to_html(index=False),'Análises e disponibilidade')
    check.save(directory/'check.html')
    return dict(project=project,run_id=row.run_id,subject=row.subject,session=row.session,block_type=row.block_type,
                status='ok',total_epochs=len(run.trials),eligible_epochs=len(audit),accepted_epochs=len(epochs),
                rejected_epochs=int((~audit.accepted).sum()),excluded_trials=int((~run.trials.valid).sum()),
                bad_channels=run.qc['bad_channels'],quality_flags=flagged,analyses=status,classification=metrics)


def fixed_cv_table(records):
    if not records:
        return '<p>CV de seleção não executada.</p>'
    frame=pd.DataFrame(records)
    rows=[]
    for cid,part in frame.groupby('candidate_id',sort=False):
        scores=part.loc[part.status=='ok','balanced_accuracy'].astype(float)
        rows.append(dict(candidate_id=cid,**part.params.iloc[0],balanced_accuracy=scores.mean(),sd=scores.std(ddof=0),folds=len(scores),falhas=int((part.status!='ok').sum())))
    from reporting import metric_table
    return '<p>Mesmos folds por trial; uma parametrização fixa por CV. Média/DP servem à seleção, não são desempenho independente. A BACC aninhada e o teste online permanecem separados.</p>'+metric_table(pd.DataFrame(rows))


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
            result["total_epochs"] = len(run.trials)
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
                add("erd_baseline", "ok" if valid.any() else "warning", f"{int(valid.sum())}/{len(epochs)} baselines válidas; referência {cfg['erd_baseline_mode']}.", action)
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


def estimate_seconds(count,cfg,mode,protocols=None,compare=False):
    from classify import candidates
    variants=cfg["comparison_pipelines"] if compare else cfg["pipelines"]
    configs=sum(len(candidates({**cfg,"pipelines":[p],"classifiers":[c]})) for p in variants for c in cfg.get("classifiers",["svm_rbf"]))
    classification=0 if mode=='ephys' else configs*len(protocols or PROTOCOLS)*cfg['outer_folds']*(cfg['inner_folds']+1)*.4
    return count*(20 if mode!='classification' else 2)+count*classification


def batch(root, control, output, cfg, subjects=None, sessions=None, blocks=None, runs=None,
          mode="both", detailed=False, protocols=None, compare=False, progress=None, projects=None):
    selected = selection(control, subjects, sessions, blocks, runs, projects)
    included = selection(control)
    if selected.empty:
        raise ValueError("Nenhum registro incluído corresponde aos filtros. Edite include e block_type na tabela.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output / "selected_control.csv", index=False)
    check_dataset(root, control, output, cfg, subjects, sessions, blocks, selected.run_id.tolist(), protocols)
    results = []
    started=time.monotonic()
    print(f"Estimativa inicial: ~{estimate_seconds(len(selected),cfg,mode,protocols,compare)/60:.1f} min (heurística; duração e canais alteram o custo).",flush=True)
    for position, (_, row) in enumerate(selected.iterrows()):
        if progress:
            eta=(time.monotonic()-started)/position*(len(selected)-position) if position else estimate_seconds(len(selected),cfg,mode,protocols,compare)
            progress(position,len(selected),eta,row.run_id)
        print(f"Analisando {row.run_id}", flush=True)
        try:
            results.append(analyze_run(root, row, cfg, output / row.run_id, mode, detailed, protocols, included, compare))
        except Exception as exc:
            traceback.print_exc()
            folder = output / row.run_id
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
            results.append(dict(project=row.get("project","Brasil"), run_id=row.run_id, subject=row.subject, session=row.session,
                                block_type=row.block_type, status=f"{type(exc).__name__}: {exc}"))
        finally:
            plt.close("all")
    batch_comparison(output, results)
    (output / "batch_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    flat = [{**{k: v for k, v in r.items() if k != "classification"}, **metric}
            for r in results for metric in (r.get("classification") or [dict(protocol="ephys")])]
    pd.DataFrame(flat).to_csv(output / "summary.csv", index=False)
    if progress:
        progress(len(selected),len(selected),0,"Concluído")
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
