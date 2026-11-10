"""Relatórios HTML autocontidos, figuras ampliáveis e gráficos com zoom nos eixos."""

import base64
import html
from io import BytesIO
from pathlib import Path
from PIL import Image

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from eeg import COLORS


class Report:
    def __init__(self, title, **kwargs):
        self.title = title
        self.items = []

    def add_html(self, content, title, **kwargs):
        self.items.append((title, content, title.startswith("Resumo")))

    def add_figure(self, fig, title, **kwargs):
        stream = BytesIO()
        fig.savefig(stream, format="png", dpi=150, bbox_inches="tight")
        compressed = BytesIO()
        Image.open(stream).convert("RGB").save(compressed, format="WEBP", quality=95)
        data = base64.b64encode(compressed.getvalue()).decode()
        self.items.append((title, f'<button class="enlarge" aria-label="Ampliar {html.escape(title, quote=True)}"><img loading="lazy" src="data:image/webp;base64,{data}" alt="{html.escape(title, quote=True)}"></button>', False))

    def add_plotly(self, fig, title):
        fig.update_layout(template="plotly_white", font=dict(size=13), margin=dict(l=65, r=30, t=60, b=55))
        self.items.append((title, fig.to_html(full_html=False, include_plotlyjs=False,
                          config=dict(scrollZoom=True, displaylogo=False, responsive=True)), True))

    def save(self, path, **kwargs):
        cards = [f'<details id="section-{i}" {"open" if opened else ""}><summary>{html.escape(title)}</summary><div class="content">{content}</div></details>'
                 for i, (title, content, opened) in enumerate(self.items)]
        page  = '''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font:16px system-ui;color:#20252b;background:#f2f4f7;margin:0}main{max-width:1280px;margin:auto;padding:24px}h1{font-size:24px}details{background:white;border:1px solid #d6dbe1;border-radius:8px;margin:20px 0}summary{padding:18px;cursor:pointer;font-weight:600}.content{padding:8px 20px 24px;overflow:auto}table{border-collapse:collapse;font-size:14px}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;font-size:12px}img{max-width:100%;height:auto}.enlarge{border:0;background:white;cursor:zoom-in;width:100%}.hint{color:#505b66}.viewer{position:fixed;inset:0;background:#18202cfa;z-index:9999;display:none}.toolbar{height:52px;display:flex;gap:16px;align-items:center;padding:0 20px;color:white}.toolbar button{padding:8px 16px;cursor:pointer}.viewport{height:calc(100% - 52px);overflow:auto;cursor:grab}.viewport img{max-width:none;display:block}.cards{display:flex;gap:20px;flex-wrap:wrap}.card{padding:16px;background:#f4f6f8;border-radius:6px;min-width:130px}.card strong{display:block;font-size:25px}</style>
<title>''' + html.escape(self.title) + '</title><script>' + get_plotlyjs() + '''</script><main><h1>''' + html.escape(self.title) + '''</h1>
<p class="hint">Gráficos interativos: arraste para selecionar zoom; duplo clique restaura; legenda oculta/mostra traços. Figuras: clique para ampliar, use a roda para zoom e arraste para navegar. Conteúdo técnico fica recolhido.</p>''' + "".join(cards) + '''</main>
<div class="viewer" role="dialog" aria-label="Figura ampliada"><div class="toolbar"><button id="close">Fechar (Esc)</button><button id="fit">Ajustar à tela</button><button id="plus">+</button><button id="minus">−</button><span id="zoom"></span></div><div class="viewport"><img alt="Figura ampliada"></div></div>
<script>const viewer=document.querySelector('.viewer'),vp=document.querySelector('.viewport'),pic=vp.querySelector('img');let scale=1,drag=null;
function resize(){pic.style.width=(pic.naturalWidth*scale)+'px';document.querySelector('#zoom').textContent=Math.round(scale*100)+'%'}
function fit(){scale=Math.min(vp.clientWidth/pic.naturalWidth,vp.clientHeight/pic.naturalHeight);resize()}
document.querySelectorAll('.enlarge').forEach(b=>b.onclick=()=>{viewer.style.display='block';pic.onload=fit;pic.src=b.querySelector('img').src;document.querySelector('#close').focus()});
function close(){viewer.style.display='none'}document.querySelector('#close').onclick=close;document.addEventListener('keydown',e=>{if(e.key==='Escape')close()});document.querySelector('#fit').onclick=fit;document.querySelector('#plus').onclick=()=>{scale=Math.min(6,scale*1.3);resize()};document.querySelector('#minus').onclick=()=>{scale=Math.max(.1,scale/1.3);resize()};
vp.addEventListener('wheel',e=>{e.preventDefault();let x=(vp.scrollLeft+e.clientX)/scale,y=(vp.scrollTop+e.clientY-52)/scale;scale=Math.max(.1,Math.min(6,scale*Math.exp(-e.deltaY*.001)));resize();vp.scrollLeft=x*scale-e.clientX;vp.scrollTop=y*scale-e.clientY+52},{passive:false});
vp.onpointerdown=e=>{drag=[e.clientX,e.clientY,vp.scrollLeft,vp.scrollTop];vp.setPointerCapture(e.pointerId);e.preventDefault()};vp.onpointermove=e=>{if(drag){vp.scrollLeft=drag[2]+drag[0]-e.clientX;vp.scrollTop=drag[3]+drag[1]-e.clientY}};vp.onpointerup=()=>drag=null;
document.querySelectorAll('details').forEach(d=>d.addEventListener('toggle',()=>{if(d.open)d.querySelectorAll('.plotly-graph-div').forEach(g=>Plotly.Plots.resize(g))}));</script></html>'''
        Path(path).write_text(page, encoding="utf-8")


def overlay_plot(before, after, times, names, title):
    # Centralização apenas visual, sem modificar o sinal usado nas análises.
    before = (before - np.median(before, axis=-1, keepdims=True)) * 1e6
    after  = (after - np.median(after, axis=-1, keepdims=True)) * 1e6
    stride = max(1, len(times) // 4000)
    fig    = make_subplots(rows=len(names), cols=1, shared_xaxes=True, vertical_spacing=0.006)
    for i, name in enumerate(names):
        limit = max(5, float(np.quantile(np.abs(np.r_[before[i], after[i]]), 0.995)) * 1.15)
        for values, color, label in [(before, "#777777", "Antes · cinza"), (after, "#0072B2", "Depois · azul")]:
            fig.add_trace(go.Scatter(x=times[::stride], y=values[i, ::stride], name=label,
                                    legendgroup=label, showlegend=i == 0, line=dict(color=color, width=1),
                                    hovertemplate=f"{name}<br>%{{x:.3f}} s<br>%{{y:.2f}} µV<extra>{label}</extra>"), row=i + 1, col=1)
        fig.update_yaxes(title_text=name, range=[-limit, limit], row=i + 1, col=1)
    fig.update_xaxes(title_text="Tempo (s)", row=len(names), col=1)
    fig.update_layout(height=max(500, 90 * len(names)), title=title + " · µV; mediana removida só para exibição", hovermode="x")
    return fig


def smoothed_timeline(timeline, seconds):
    frame = timeline.sort_values(["trial", "time_end"]).copy()
    for _, indices in frame.groupby(["trial", "fold"]).groups.items():
        part  = frame.loc[indices].sort_values("time_end")
        step  = np.median(np.diff(part.time_end)) if len(part) > 1 else seconds
        width = max(1, int(round(seconds / max(step, 1e-6))))
        for col in [c for c in frame if c.startswith("p_")]:
            frame.loc[part.index, col] = part[col].rolling(width, min_periods=1).mean().to_numpy()
    return frame


def class_trials(values, cls):
    for trial, sub in values.groupby("trial"):
        condition = sub.trial_class.iloc[0]
        if cls in {"LEFT", "RIGHT", "REST_STIM"} and condition != cls:
            continue
        if cls == "MOVE" and condition not in {"LEFT", "RIGHT", "BOTH"}:
            continue
        yield trial, sub.sort_values("time_end")


def mean_curve(traces, cfg):
    grid    = np.arange(cfg["epoch_tmin"], cfg["epoch_tmax"] + cfg["step"] / 2, cfg["step"])
    aligned = [np.interp(grid, t.index, t.values, left=np.nan, right=np.nan) for t in traces]
    table   = pd.DataFrame(aligned)
    return grid, table.mean(axis=0).to_numpy(), table.notna().sum(axis=0).to_numpy()


def timeline_plot(summary, timeline, cfg):
    values = smoothed_timeline(timeline, cfg["curve_smoothing_s"])
    labels = summary["labels"]
    fig    = make_subplots(rows=2, cols=1, subplot_titles=["Linha temporal · somente trials de teste", "Épocas alinhadas · ATTEMPT · média em destaque"], vertical_spacing=0.2)
    origin = values.start_abs.min()
    ranks  = {t: i / max(1, values.trial.nunique() - 1) for i, t in enumerate(sorted(values.trial.unique()))}
    for cls in labels:
        traces = []
        for trial, sub in values.groupby("trial"):
            fig.add_trace(go.Scatter(x=sub.start_abs - origin, y=sub[f"p_{cls}"], mode="lines", name=cls,
                                    legendgroup=cls, showlegend=bool(trial == min(ranks)), line=dict(color=COLORS[cls], width=1.3)), row=1, col=1)
            condition = sub.trial_class.iloc[0]
            if (cls in {"LEFT", "RIGHT", "REST_STIM"} and condition != cls) or (cls == "MOVE" and condition not in {"LEFT", "RIGHT", "BOTH"}):
                continue
            fig.add_trace(go.Scatter(x=sub.time_end, y=sub[f"p_{cls}"], mode="lines", name=f"{cls} · trial {trial}",
                                    showlegend=False, line=dict(color=COLORS[cls], width=0.8), opacity=0.08 + 0.35 * ranks[trial]), row=2, col=1)
            traces.append(pd.Series(sub[f"p_{cls}"].to_numpy(), index=sub.time_end.round(3)))
        if traces:
            grid, mean, support = mean_curve(traces, cfg)
            fig.add_trace(go.Scatter(x=grid, y=mean, mode="lines", name=f"Média {cls}", legendgroup=cls,
                                    line=dict(color=COLORS[cls], width=3), customdata=support,
                                    hovertemplate="%{x:.2f} s<br>P=%{y:.3f}<br>n=%{customdata} trials<extra></extra>"), row=2, col=1)
    for row in [1, 2]:
        fig.add_hline(y=summary["chance"], line_dash="dot", line_color="#666666", row=row, col=1)
        fig.update_yaxes(range=[0, 1], title_text="Probabilidade", row=row, col=1)
    fig.update_xaxes(title_text="Tempo desde primeira janela disponível (s)", row=1, col=1)
    fig.update_xaxes(title_text="Fim da janela relativo ao ATTEMPT (s)", range=[cfg["epoch_tmin"], cfg["epoch_tmax"]], row=2, col=1)
    fig.add_vline(x=0, line_dash="dot", row=2, col=1)
    fig.update_layout(height=750, title=f"{summary['protocol']} · média móvel causal {cfg['curve_smoothing_s']} s · não altera métricas")
    return fig


def batch_comparison(output, results):
    """Uma linha por sujeito/bloco/protocolo/pipeline; runs não viram sujeitos extras."""
    rows, erd = [], []
    for result in results:
        meta = {k: result[k] for k in ['run_id', 'subject', 'session', 'block_type']}
        for metric in result.get('classification', []):
            if metric.get('status') == 'ok':
                rows.append({**meta, **metric})
        path = Path(output) / result['run_id'] / 'erd_metrics.csv'
        if path.exists():
            frame = pd.read_csv(path)
            if len(frame):
                erd.append(frame.assign(**meta))
    report = Report('Batch · comparação entre sujeitos e pipelines')
    report.add_html('<p>Replicações são agregadas dentro de cada sujeito antes da comparação. As figuras são descritivas; não se tratam janelas ou folds como sujeitos independentes. Comparar blocos e protocolos equivalentes. CV dentro do bloco e teste online proxy são separados.</p>', title='Resumo · unidade de comparação')
    if rows:
        frame = pd.DataFrame(rows)
        frame.to_csv(Path(output) / 'comparison_runs.csv', index=False)
        keys    = ['subject', 'block_type', 'protocol', 'pipeline', 'validation']
        grouped = frame.groupby(keys, as_index=False).agg(balanced_accuracy=('balanced_accuracy', 'mean'), n_runs=('run_id', 'nunique'), chance=('chance', 'first'))
        grouped.to_csv(Path(output) / 'comparison_subjects.csv', index=False)
        for (block, protocol, validation), sub in grouped.groupby(['block_type', 'protocol', 'validation']):
            fig = go.Figure()
            for i, (pipeline, values) in enumerate(sub.groupby('pipeline')):
                values = values.sort_values('subject')
                fig.add_trace(go.Scatter(x=values.subject, y=values.balanced_accuracy, name=pipeline, mode='markers+lines',
                                        line=dict(color=['#0072B2','#D55E00','#009E73'][i % 3], dash=['solid','dash','dot'][i % 3]),
                                        marker=dict(size=10, symbol=['circle','diamond','square'][i % 3]), customdata=values.n_runs,
                                        hovertemplate='%{x}<br>BA=%{y:.3f}<br>%{customdata} registro(s)<extra>%{fullData.name}</extra>'))
            fig.add_hline(y=sub.chance.iloc[0], line_dash='dot', line_color='#555555')
            fig.update_layout(height=450, title=f'{block} · {protocol} · {validation}', yaxis=dict(range=[0,1], title='BA média por sujeito'), xaxis_title='Sujeito')
            report.add_plotly(fig, f'Classificação · {block} · {protocol} · {validation}')
        report.add_html(grouped.to_html(index=False, float_format=lambda v: f'{v:.3f}'), title='Tabela · resultados por sujeito')
        # Diferenças pareadas de pipelines, com cada sujeito contribuindo uma vez.
        paired = []
        for keys, sub in grouped.groupby(['block_type','protocol','validation']):
            pivot = sub.pivot(index='subject', columns='pipeline', values='balanced_accuracy')
            for a in pivot:
                for b in pivot:
                    if a >= b:
                        continue
                    common = pivot[[a,b]].dropna()
                    for subject, values in common.iterrows():
                        paired.append(dict(subject=subject, block_type=keys[0], protocol=keys[1], validation=keys[2],
                                           pipeline_a=a, pipeline_b=b, delta_ba=float(values[b]-values[a])))
        pd.DataFrame(paired, columns=['subject','block_type','protocol','validation','pipeline_a','pipeline_b','delta_ba']).to_csv(Path(output) / 'pipeline_paired_differences.csv', index=False)
    else:
        report.add_html('<p>Sem classificação válida nesta execução. Selecione classificação e comparação de pipelines para produzir a comparação.</p>', title='Resumo · classificação não disponível')
    if erd:
        trials   = pd.concat(erd, ignore_index=True)
        runs     = trials.groupby(['run_id','subject','block_type','class','channel','band'], as_index=False).erd_percent.mean()
        subjects = runs.groupby(['subject','block_type','class','channel','band'], as_index=False).erd_percent.mean()
        subjects.to_csv(Path(output) / 'erd_subjects.csv', index=False)
        for (block, band, channel), sub in subjects.groupby(['block_type','band','channel']):
            fig = go.Figure()
            for cls, values in sub.groupby('class'):
                fig.add_trace(go.Scatter(x=values.subject, y=values.erd_percent, mode='markers', name=cls, marker=dict(color=COLORS[cls],size=11)))
            fig.add_hline(y=0, line_dash='dot', line_color='#555555')
            fig.update_layout(title=f'{block} · {channel} · {band}', height=400, yaxis_title='ERD/ERS (%) · média por sujeito', xaxis_title='Sujeito')
            report.add_plotly(fig, f'ERD/ERS · {block} · {channel} · {band}')
    report.save(Path(output) / 'comparison.html')


