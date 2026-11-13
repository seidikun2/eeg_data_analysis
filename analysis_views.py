"""Shared scales, trial-isolated tangent PCA and compact report panels."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.signal import welch, spectrogram
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from eeg import COLORS


def robust_limit(values, percentile=99):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return max(1., float(np.percentile(np.abs(finite), percentile))) if len(finite) else 1.


def stacked_snapshot(run, seconds=20):
    picks = mne.pick_types(run.raw.info, eeg=True, exclude=[])
    width = min(run.raw.n_times, round(seconds * run.raw.info['sfreq']))
    start = (run.raw.n_times - width) // 2
    times = run.raw.times[start:start + width]
    arrays = [r.get_data(picks=picks, start=start, stop=start + width) * 1e6 for r in [run.raw, run.clean]]
    arrays = [a - np.median(a, axis=1, keepdims=True) for a in arrays]
    limit = robust_limit(arrays, 99.5)
    offset = np.arange(len(picks))[::-1] * 1.3 * limit
    fig, ax = plt.subplots(figsize=(16, max(6, .48 * len(picks))), layout='constrained')
    stride = max(1, len(times) // 3500)
    for data, color, label in zip(arrays, ['#777777', '#D62728'], ['Antes · cinza', 'Depois · vermelho']):
        for i, ch in enumerate(picks):
            # Only the display omits extremes; CSV/QC retain original amplitudes.
            trace = np.where(np.abs(data[i]) <= limit, data[i], np.nan)
            ax.plot(times[::stride], trace[::stride] + offset[i], color=color, lw=.7, label=label if i == 0 else None)
    names = [run.raw.ch_names[i] + (' [excluído]' if run.raw.ch_names[i] in run.raw.info['bads'] else '') for i in picks]
    ax.set(yticks=offset, yticklabels=names, ylim=(-limit, offset[0] + limit), xlabel='Tempo desde início do bloco (s)',
           title=f'Snapshot central · {width/run.raw.info["sfreq"]:.1f} s · escala comum ±{limit:.1f} µV por canal')
    ax.legend(loc='upper right')
    ax.text(0, -.12, f'Offsets entre canais: {1.3*limit:.1f} µV. Mediana removida só na exibição; extremos acima de P99,5 omitidos no desenho.', transform=ax.transAxes, fontsize=11)
    return fig


def epoch_panels(epochs, cfg, power=False):
    """Class columns; stacked channel means and separate trial heatmaps, no trace pile-up."""
    classes = list(epochs.event_id)
    picks = [i for i, c in enumerate(epochs.ch_names) if c in {'C3','Cz','C4'}] if power else list(mne.pick_types(epochs.info,eeg=True,exclude='bads'))
    if not picks:
        picks = list(range(min(3, len(epochs.ch_names))))
    bands = ['mu','beta'] if power else [None]
    fig, axes = plt.subplots(len(bands) * (1 if power else 2), len(classes), figsize=(max(9, 6 * len(classes)), (4 if power else 7) * len(bands)), squeeze=False, layout='constrained')
    for bi, band in enumerate(bands):
        matrices = []
        for cls in classes:
            part = epochs[cls]
            if power:
                mask = (part.freqs >= cfg[band][0]) & (part.freqs < cfg[band][1])
                data = part.data[:, picks][:, :, mask].mean(axis=2) * 100
            else:
                data = part.get_data(picks=picks) * 1e6
                if cfg.get('erp_baseline_mode','attention')=='attention':
                    for i,row in enumerate(part.metadata.itertuples()):
                        start,end=row.attention_start-row.attempt_start+.1,row.stimulus_start-row.attempt_start-.1
                        mask=(part.times>=start)&(part.times<=end)
                        if mask.sum()/part.info['sfreq']<.5:
                            data[i]=np.nan
                        else:
                            data[i]-=data[i,:,mask].mean(axis=0)[:,None]
                else:
                    baseline = (part.times >= cfg['baseline'][0]) & (part.times <= cfg['baseline'][1])
                    data -= data[:, :, baseline].mean(axis=-1, keepdims=True)
            matrices.append(data)
        limit = robust_limit([np.nanmean(m,axis=0) for m in matrices], 99)
        trial_limit = robust_limit(np.concatenate([m[:, 0].ravel() for m in matrices]),99)
        offsets = np.arange(len(picks))[::-1] * limit * 2.5
        for ci, (cls, data) in enumerate(zip(classes, matrices)):
            ax = axes[bi if power else bi * 2, ci]
            heat = None if power else axes[bi * 2 + 1, ci]
            for j,ch in enumerate(picks):
                mean = np.nanmean(data[:, j],axis=0)
                ax.axhline(offsets[j], color='.65', lw=.7, ls=':')
                ax.plot(epochs.times, np.where(np.abs(mean)<=limit,mean,np.nan) + offsets[j], color=COLORS[cls], lw=1.8)
            unit = '%' if power else 'µV'
            ax.set(yticks=offsets,yticklabels=[epochs.ch_names[i] for i in picks], ylim=(-limit, offsets[0]+limit),
                   title=f'{cls} · {band or "ERP"} · média · escala comum ±{limit:.1f} {unit}',xlabel='ATTEMPT (s)')
            ax.axvline(0,color='.5',ls=':')
            if heat is not None:
                im=heat.imshow(data[:,0], origin='lower',aspect='auto',cmap='RdBu_r',vmin=-trial_limit,vmax=trial_limit,
                             extent=[epochs.times[0],epochs.times[-1],-.5,len(data)-.5])
                heat.set(title=f'Trials individuais · {epochs.ch_names[picks[0]]} · n={len(data)}',xlabel='ATTEMPT (s)',ylabel='Trial (ID original)')
                heat.set_yticks(np.arange(len(data)),epochs[cls].metadata.trial.astype(str))
                fig.colorbar(im,ax=heat,label=unit)
    fig.suptitle(f'ERD/ERS · zero em cada offset · escala comum · baseline {cfg["erd_baseline_mode"]}' if power else f'ERP · classes separadas; canais em offset; escala comum P99 · baseline {cfg.get("erp_baseline_mode","attention")}')
    return fig


def tangent_pca(epochs, cfg, directory, training_epochs=None):
    """Fit reference/scaler/PCA only on train trials, project real sliding test windows."""
    def windows(part):
        crop = part.copy().crop(*cfg['active'])
        picks = mne.pick_types(crop.info,eeg=True,exclude='bads')
        data = crop.get_data(picks=picks)
        size = min(data.shape[-1],max(2,round(part.info['sfreq'])))
        step = max(1,round(part.info['sfreq']*.25))
        chunks, rows = [], []
        for i,row in enumerate(part.metadata.itertuples()):
            for start in range(0,data.shape[-1]-size+1,step):
                chunks.append(data[i,:,start:start+size])
                rows.append(dict(trial=int(row.trial),cls=row.trial_class,window_start=cfg['active'][0]+start/part.info['sfreq']))
        return np.stack(chunks),pd.DataFrame(rows)
    X,meta=windows(epochs)
    if training_epochs is None:
        unique=meta[['trial','cls']].drop_duplicates()
        counts=unique.cls.value_counts()
        if len(unique)<4 or counts.min()<2:
            raise ValueError('PCA teste requer ao menos dois trials por classe para separar treino/teste.')
        train_ids,test_ids=train_test_split(unique.trial,test_size=max(len(counts),int(np.ceil(len(unique)*.33))),stratify=unique.cls,random_state=cfg['seed'])
        train_mask=meta.trial.isin(train_ids).to_numpy()
        train,test=X[train_mask],X[~train_mask]
        train_meta,test_meta=meta[train_mask].copy(),meta[~train_mask].copy()
        source='trials separados do mesmo registro (exploratório)'
    else:
        names=[c for c in epochs.ch_names if c not in epochs.info['bads'] and c in training_epochs.ch_names and c not in training_epochs.info['bads']]
        if len(names)<3:
            raise ValueError('PCA sem três canais comuns entre IM_treino e teste.')
        epochs=epochs.copy().pick(names)
        training_epochs=training_epochs.copy().pick(names)
        test,test_meta=windows(epochs)
        train,train_meta=windows(training_epochs)
        source='IM_treino anterior → bloco atual (conjuntos distintos)'
    cov=Covariances(estimator='oas')
    tangent=TangentSpace(metric='riemann').fit(cov.transform(train))
    scaler=StandardScaler().fit(tangent.transform(cov.transform(train)))
    ztrain=scaler.transform(tangent.transform(cov.transform(train)))
    pca=PCA(n_components=2).fit(ztrain)
    projected=pca.transform(scaler.transform(tangent.transform(cov.transform(test))))
    test_meta=test_meta.reset_index(drop=True).assign(PC1=projected[:,0],PC2=projected[:,1],role='test')
    train_meta=train_meta.assign(role='train')
    pd.concat([train_meta,test_meta],ignore_index=True).to_csv(Path(directory)/'pca_windows.csv',index=False)
    (Path(directory)/'pca_model.json').write_text(json.dumps(dict(
        source=source,representation='OAS covariance → TangentSpace(metric=riemann) → StandardScaler → PCA',
        fit_role='train_only',train_trials=sorted(train_meta.trial.unique().tolist()),test_trials=sorted(test_meta.trial.unique().tolist()),
        tangent_reference=tangent.reference_.tolist(),scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist(),
        pca_mean=pca.mean_.tolist(),pca_components=pca.components_.tolist(),explained_variance=pca.explained_variance_ratio_.tolist()),indent=2))
    fig=go.Figure()
    for cls,part in test_meta.groupby('cls'):
        color=COLORS.get(cls,'#444444')
        fig.add_trace(go.Scatter(x=part.PC1,y=part.PC2,mode='markers',name=f'{cls} · teste',legendgroup=cls,
            marker=dict(color=color,size=7),
            customdata=part[['trial','window_start']].to_numpy(),hovertemplate=f'{cls}<br>Trial %{{customdata[0]}}<br>Janela %{{customdata[1]:.2f}} s<extra>teste</extra>'))
        center, ellipse = trial_dispersion(part)
        fig.add_trace(go.Scatter(x=[center[0]],y=[center[1]],mode='markers',name=f'{cls} · centroide',
            marker=dict(color=color,size=16,symbol='x'),legendgroup=cls,showlegend=False))
        if ellipse is not None:
            fig.add_trace(go.Scatter(x=ellipse[:,0],y=ellipse[:,1],mode='lines',line=dict(color=color,width=2),
                name='Dispersão descritiva · 1 DP',legendgroup=cls,showlegend=False,hoverinfo='name'))
    fig.update_layout(height=650,title=('TGSP → PCA · teste online; ajuste no IM anterior' if training_epochs is not None else 'TGSP → PCA · teste por trials separados; ajuste no treino'),
                      xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} treino)',yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} treino)',
                      legend=dict(groupclick='togglegroup'))
    return fig,test_meta


def trial_dispersion(part):
    """Equal total weight per trial; one-SD descriptive ellipse, never a CI."""
    xy = part[['PC1','PC2']].to_numpy(float)
    weights = (1 / part.groupby('trial').trial.transform('size')).to_numpy(float)
    weights /= weights.sum()
    center = np.sum(xy * weights[:,None], axis=0)
    delta = xy - center
    cov = (delta * weights[:,None]).T @ delta
    if part.trial.nunique() < 3 or np.linalg.matrix_rank(cov) < 2:
        return center, None
    values, vectors = np.linalg.eigh(cov)
    angle = np.linspace(0, 2*np.pi, 101)
    ellipse = np.column_stack([np.cos(angle),np.sin(angle)]) @ (vectors * np.sqrt(values)).T + center
    return center, ellipse


def spectral_figures(epochs, cfg):
    """Welch PSD and event-aligned STFT density; average linear power first."""
    picks = [c for c in ['C3','Cz','C4'] if c in epochs.ch_names and c not in epochs.info['bads']]
    if not picks:
        picks = [epochs.ch_names[i] for i in mne.pick_types(epochs.info,eeg=True,exclude='bads')[:3]]
    classes = list(epochs.event_id)
    sfreq = epochs.info['sfreq']
    size = min(len(epochs.times), round(sfreq))
    psdfig, pa = plt.subplots(len(picks),len(classes),squeeze=False,figsize=(5*len(classes),3*len(picks)),layout='constrained')
    tffig, ta = plt.subplots(len(picks),len(classes),squeeze=False,figsize=(5*len(classes),3*len(picks)),layout='constrained')
    densities=[]
    for ci, cls in enumerate(classes):
        data=epochs[cls].get_data(picks=picks)*1e6
        f, density=welch(data,fs=sfreq,window='hann',nperseg=size,noverlap=size//2,scaling='density',axis=-1)
        ft, t, tf=spectrogram(data,fs=sfreq,window='hann',nperseg=size,noverlap=3*size//4,scaling='density',mode='psd',axis=-1)
        mask=(ft>=cfg['l_freq'])&(ft<=cfg['h_freq'])
        db=10*np.log10(np.maximum(tf.mean(axis=0)[:,mask],np.finfo(float).tiny))
        densities.append((ft[mask],t+epochs.times[0],db))
        for pi,ch in enumerate(picks):
            pa[pi,ci].semilogy(f,density.mean(axis=0)[pi],color=COLORS[cls])
            pa[pi,ci].set(xlim=(cfg['l_freq'],cfg['h_freq']),xlabel='Hz',ylabel='µV²/Hz',title=f'{ch} · {cls}')
    low,high=np.quantile(np.concatenate([x[2].ravel() for x in densities]),[.01,.99])
    for ci,(f,t,db) in enumerate(densities):
        for pi,ch in enumerate(picks):
            im=ta[pi,ci].pcolormesh(t,f,db[pi],shading='auto',vmin=low,vmax=high,cmap='viridis')
            ta[pi,ci].axvline(0,color='white',ls=':')
            ta[pi,ci].set(xlabel='ATTEMPT (s)',ylabel='Hz',title=f'{ch} · {classes[ci]}')
    tffig.colorbar(im,ax=ta.ravel().tolist(),label='dB re 1 µV²/Hz · média linear antes de log10')
    psdfig.suptitle('PSD absoluta · Welch · Hann 1 s, sobreposição 50% · média de épocas')
    tffig.suptitle('Espectrograma absoluto · Hann 1 s, sobreposição 75% · alinhado a ATTEMPT')
    return [('psd_welch',psdfig),('espectrograma',tffig)]
