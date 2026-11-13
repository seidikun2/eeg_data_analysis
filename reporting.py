"""Relatórios HTML autocontidos, figuras ampliáveis e gráficos com zoom nos eixos."""

import base64
import html
import json
import re
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
        fig.update_layout(template="plotly_white", font=dict(size=16), margin=dict(l=65, r=30, t=60, b=55))
        self.items.append((title, fig.to_html(full_html=False, include_plotlyjs=False,
                          config=dict(scrollZoom=True, displaylogo=False, responsive=True)), True))

    def save(self, path, **kwargs):
        cards = [f'<section id="section-{i}"><h2>{html.escape(title)}</h2><div class="content">{content}</div></section>'
                 for i, (title, content, opened) in enumerate(self.items)]
        page  = '''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font:18px system-ui;color:#20252b;background:#f2f4f7;margin:0}main{max-width:1280px;margin:auto;padding:24px}h1{font-size:24px}section{background:white;border:1px solid #d6dbe1;border-radius:8px;margin:12px 0}h2{padding:12px 16px;font-size:20px;margin:0}.content{padding:6px 16px 14px;overflow:auto}table{border-collapse:collapse;font-size:16px}th,td{padding:4px 8px;line-height:1.3;max-width:28em;overflow-wrap:break-word;border-bottom:1px solid #ddd;text-align:left}th{white-space:nowrap}pre{white-space:pre-wrap;font-size:12px}img{max-width:100%;height:auto}.enlarge{border:0;background:white;cursor:zoom-in;width:100%}.hint{color:#505b66}.viewer{position:fixed;inset:0;background:#18202cfa;z-index:9999;display:none}.toolbar{height:52px;display:flex;gap:16px;align-items:center;padding:0 20px;color:white}.toolbar button{padding:8px 16px;cursor:pointer}.viewport{height:calc(100% - 52px);overflow:auto;cursor:grab}.viewport img{max-width:none;display:block}.pipeline-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:20px}.pipeline-grid .wide{grid-column:1/-1}.cards{display:flex;gap:20px;flex-wrap:wrap}.card{padding:16px;background:#f4f6f8;border-radius:6px;min-width:130px}.card strong{display:block;font-size:25px}</style>
<title>''' + html.escape(self.title) + '</title><script>' + get_plotlyjs() + '''</script><main><h1>''' + html.escape(self.title) + '''</h1>
<p class="hint">Gráficos interativos: arraste para selecionar zoom; duplo clique restaura; legenda oculta/mostra traços. Figuras: clique para ampliar, use a roda para zoom e arraste para navegar. Seções visíveis; detalhes extensos nos arquivos CSV/JSON de auditoria.</p>''' + "".join(cards) + '''</main>
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


def metric_table(frame):
    if frame.empty:
        return '<p>Sem resultados disponíveis.</p>'
    frame=frame.drop(columns=['params','traceback','error','deployment_params','candidate_id'],errors='ignore')
    if 'balanced_accuracy' not in frame:
        return frame.to_html(index=False,escape=True)
    def cell(value):
        if pd.isna(value):
            return ''
        score=float(value)
        # Text always carries the value; colour is redundant support.
        return f'background-color:rgba(0,114,178,{.08+.40*max(0,min(1,score)):.2f});font-weight:700'
    return frame.style.map(cell,subset=['balanced_accuracy']).format(precision=3,na_rep='—').hide(axis='index').to_html()


def combine_sections(report,first,title):
    items=report.items[first:]
    report.items=report.items[:first]
    def order(item):
        name=item[0]
        return 3 if 'timeline' in name else (2 if 'parâmetros selecionados' in name else (1 if 'CV de parâmetros' in name else 0))
    items=sorted(items,key=order)
    cells=[]
    for name,content,_ in items:
        cls='wide' if 'timeline' in name else ''
        cells.append(f'<div class="{cls}"><h3>{html.escape(name)}</h3>{content}</div>')
    report.add_html('<div class="pipeline-grid">'+''.join(cells)+'</div>',title)


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
    if "prediction_role" in timeline:
        frame=timeline.sort_values("start_abs").copy()
        step=np.median(np.diff(frame.start_abs)) if len(frame)>1 else seconds
        width=max(1,round(seconds/max(step,1e-6)))
        cols=[c for c in frame if c.startswith('p_')]
        frame[cols]=frame[cols].rolling(width,min_periods=1).mean()
        return frame
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


def timeline_plot(summary, timeline, cfg, trials=None):
    values = smoothed_timeline(timeline, cfg["curve_smoothing_s"])
    labels = summary["labels"]
    fig    = make_subplots(rows=2, cols=1, subplot_titles=["Sinal completo · modelo fixo · papel indicado no título", "Épocas extraídas da timeline · ATTEMPT · média em destaque"], vertical_spacing=0.2)
    origin = values.start_abs.min()
    ranks  = {t: i / max(1, values.trial.nunique() - 1) for i, t in enumerate(sorted(values.trial.unique()))}
    if trials is not None:
        for trial in trials.itertuples():
            if trial.trial not in set(values.trial):
                continue
            fig.add_vrect(x0=trial.attempt_start - origin, x1=trial.attempt_end - origin,
                          fillcolor=COLORS.get(trial.trial_class, "#777777"), opacity=.09, line_width=1,
                          annotation_text=f"{trial.trial_class} · {trial.trial}", row=1, col=1, exclude_empty_subplots=False)
    continuous='prediction_role' in values
    for cls in labels:
        traces = []
        if continuous:
            sub=values.sort_values('start_abs')
            fig.add_trace(go.Scatter(x=sub.start_abs-origin,y=sub[f'p_{cls}'],mode='lines',name=cls,legendgroup=cls,line=dict(color=COLORS[cls],width=1.6)),row=1,col=1)
        for trial, sub in values.groupby("trial"):
            if trial < 0:
                continue
            if not continuous:
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
    role=summary.get('continuous_role','out-of-fold por trial')
    role={'in_sample_IM_treino_not_validation':'IM_treino · ajustado neste bloco (não é validação)', 'online_test_prior_IM_treino':'Teste online · modelo do IM_treino anterior'}.get(role,role)
    fig.update_layout(height=850, title=f"{summary['protocol']} · {role}<br>Média móvel causal {cfg['curve_smoothing_s']} s · não altera métricas")
    return fig


def batch_comparison(output, results):
    """Reports never pool people, sessions or repeated blocks into one observation."""
    import matplotlib.pyplot as plt
    output=Path(output)
    report=Report('Relatório geral · projetos, sujeitos, sessões e repetições')
    report.add_html('<p>Cada projeto/sujeito/sessão/registro permanece separado. EM_treino: screening; IM_treino: ajuste e CV; online: teste com calibração anterior identificada. CV de seleção não é teste independente. Folds e janelas não são réplicas independentes.</p>','Resumo · interpretação')
    columns=['project','subject','session','run_id','block_type','status','total_epochs','eligible_epochs','accepted_epochs','rejected_epochs','excluded_trials','bad_channels','quality_flags']
    summary=pd.DataFrame([{k:r.get(k,'Brasil' if k=='project' else '') for k in columns} for r in results])
    display_summary=summary.copy()
    display_summary['project']=display_summary.project.replace({'Brasil':'BR','Italia':'IT'})
    report.add_html(record_table(display_summary),'Resumo · épocas totais, aceitas e qualidade')
    rows,erd,cv=[],[],[]
    for result in results:
        meta={k:result.get(k,'Brasil' if k=='project' else '') for k in ['project','subject','session','run_id','block_type']}
        for metric in result.get('classification',[]):
            rows.append({**meta,**metric})
        directory=output/result['run_id']
        path=directory/'erd_metrics.csv'
        if path.exists():
            frame=pd.read_csv(path)
            if len(frame):
                erd.append(frame.assign(**meta))
        for path in directory.glob('*_fixed_cv.csv'):
            part=pd.read_csv(path)
            if len(part):
                part['protocol']=path.stem.removesuffix('_fixed_cv').split('_',2)[-1]
                # Derive protocol from paired JSON, rather than parsing classifier underscores.
                sidecar=path.with_name(path.name.replace('_fixed_cv.csv','.json'))
                if sidecar.exists():
                    part['protocol']=json.loads(sidecar.read_text())['protocol']
                cv.append(part.assign(**meta))
    performance=pd.DataFrame(rows)
    performance.to_csv(output/'comparison_runs.csv',index=False)
    # Retained compatibility filenames now keep each repetition, never a pooled subject mean.
    performance.to_csv(output/'comparison_subjects.csv',index=False)
    pd.DataFrame(columns=['project','subject','session','run_id','pipeline_a','pipeline_b','delta_ba']).to_csv(output/'pipeline_paired_differences.csv',index=False)
    fixed=pd.concat(cv,ignore_index=True) if cv else pd.DataFrame()
    if len(fixed):
        fixed.to_csv(output/'fixed_parameter_cv.csv',index=False)
    trials=pd.concat(erd,ignore_index=True) if erd else pd.DataFrame()
    if len(trials):
        trials.to_csv(output/'erd_subjects.csv',index=False)
    for (project,subject),person in summary.groupby(['project','subject'],sort=False):
        project_label={'Brasil':'BR','Italia':'IT'}.get(project,project)
        person_report=Report(f'{project_label} · {subject} · sessões')
        for session,session_rows in person.groupby('session',sort=False):
            session_report=Report(f'{project_label} · {subject} · sessão {session}')
            session_report.add_html(record_table(session_rows),'Resumo · blocos, repetições, qualidade e performance')
            session_performance=performance[performance.run_id.isin(session_rows.run_id)] if len(performance) else pd.DataFrame()
            if len(session_performance):
                session_report.add_html(metric_table(session_performance.merge(session_rows[['run_id','total_epochs','accepted_epochs','bad_channels','quality_flags']],on='run_id',how='left')), 'Performance junto à qualidade por registro')
            session_report.add_html('<p>EM_treino identifica a resposta ao movimento; IM_treino ajusta o modelo; online avalia transferência. Repetições são mantidas individualmente, inclusive falhas e perda de canais. Compare qualidade e BACC antes de atribuir mudanças ao usuário.</p>','Leitura da sessão')
            for rid in session_rows.run_id:
                path=output/rid/'report.html'
                if path.exists():
                    page=path.read_text()
                    content=page.split('<main>',1)[1].split('</main>',1)[0]
                    session_report.add_html(content,f'Registro · {rid}')
                else:
                    session_report.add_html(session_rows[session_rows.run_id==rid].to_html(index=False),f'Registro · {rid} · indisponível')
            safe=re.sub(r'[^a-zA-Z0-9_-]','_',f'{project}_{subject}_{session}')
            session_report.save(output/f'session_{safe}.html')
            person_report.add_html(f'<p><a href="session_{safe}.html">Abrir sessão {html.escape(session)}</a></p>'+record_table(session_rows),f'Sessão {session}')
            report.add_html(f'<p><a href="session_{safe}.html">Relatório da sessão completo</a></p>',f'{project_label} · {subject} · {session}')
        if len(performance):
            part=performance[(performance.project==project)&(performance.subject==subject)]
            for rid,run_metrics in part.groupby('run_id',sort=False):
                valid=run_metrics[run_metrics.status=='ok']
                if len(valid):
                    protocols=list(valid.protocol.unique())
                    fig,axes=plt.subplots(1,len(protocols),figsize=(max(8,6*len(protocols)),5),squeeze=False,layout='constrained')
                    for ax,protocol in zip(axes[0],protocols):
                        values=valid[valid.protocol==protocol]
                        names=values.pipeline+' + '+values.get('classifier',pd.Series('svm_rbf',index=values.index))
                        ax.scatter(names,values.balanced_accuracy,s=65)
                        ax.axhline(values.chance.iloc[0],color='.5',ls=':')
                        ax.set(title=protocol,ylim=(0,1),ylabel='BACC por trial/fase')
                        ax.tick_params(axis='x',labelrotation=20)
                    fig.suptitle('Performance · features e classificadores')
                    report.add_figure(fig,f'{subject} · {rid} · protocolos lado a lado');plt.close(fig)
                report.add_html(metric_table(run_metrics.drop(columns=['project','subject','session','run_id','block_type'],errors='ignore')),f'{subject} · {rid} · performance e impedimentos')
        if len(fixed):
            person_cv=fixed[(fixed.project==project)&(fixed.subject==subject)]
            for (rid,protocol),part in person_cv.groupby(['run_id','protocol'],sort=False):
                valid=part[part.status=='ok'].copy()
                if valid.empty:
                    continue
                for classifier,group in valid.groupby('classifier',sort=False):
                    fig=parameter_distribution(group,protocol,classifier)
                    report.add_figure(fig,f'{subject} · {rid} · {protocol} · {classifier} · CV de seleção');plt.close(fig)
        person_report.save(output/f'subject_{project}_{subject}.html')
    if len(fixed):
        columns=[c for c in ['candidate_id','pipeline','classifier','window','offset','bands','C','gamma'] if c in fixed]
        params=fixed[columns].drop_duplicates()
        params.to_csv(output/'parameter_combinations.csv',index=False)
        report.add_html('<details><summary>Combinações · auditoria comum</summary>'+params.to_html(index=False)+'</details>','Apêndice · parâmetros')
    links=[]
    for result in results:
        rid=html.escape(result['run_id'],quote=True)
        target=('check.html' if (output/result['run_id']/'check.html').exists() else 'report.html') if result.get('status','ok')=='ok' else 'diagnostics.json'
        links.append(f'<li><a href="{rid}/{target}">{rid}</a> · {html.escape(result.get("status","ok"))}</li>')
    report.add_html('<ul>'+''.join(links)+'</ul>','Relatórios individuais')
    report.save(output/'index.html')
    (output/'comparison.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=index.html"><a href="index.html">Relatório geral</a>',encoding='utf-8')


def parameter_distribution(frame, protocol, classifier):
    """Separate offsets and fixed band/feature/hyperparameter conditions; no connecting lines."""
    import matplotlib.pyplot as plt
    offsets=sorted(frame.offset.unique())
    fig,axes=plt.subplots(1,len(offsets),squeeze=False,figsize=(max(8,5*len(offsets)),5),layout='constrained')
    def label(row):
        import ast
        bank=ast.literal_eval(row.bands) if isinstance(row.bands,str) else row.bands
        bands=' + '.join(f'{lo:g}–{hi:g}' for lo,hi in bank)+' Hz'
        parts=[str(row.pipeline).upper(),bands]
        if classifier!='lda' and 'C' in row and pd.notna(row['C']):
            parts.append(f'C={row["C"]:g}')
        if classifier=='svm_rbf' and 'gamma' in row and pd.notna(row.gamma):
            parts.append(f'γ={row.gamma}')
        return ' · '.join(parts)
    frame=frame.copy()
    frame['condition']=frame.apply(label,axis=1)
    conditions=list(dict.fromkeys(frame.condition))
    colors=plt.get_cmap('tab10')
    windows=sorted(frame.window.unique())
    for ax,offset in zip(axes[0],offsets):
        part=frame[frame.offset==offset]
        for ci,condition in enumerate(conditions):
            subset=part[part.condition==condition]
            for window,scores in subset.groupby('window'):
                values=scores.balanced_accuracy.to_numpy(float)
                x=windows.index(window)+(ci-(len(conditions)-1)/2)*(.7/max(1,len(conditions)))
                color=colors(ci%10)
                if len(values)>1:
                    boxes=ax.boxplot([values],positions=[x],widths=.55/max(1,len(conditions)),showfliers=False,patch_artist=True,manage_ticks=False)
                    boxes['boxes'][0].set(facecolor=color,alpha=.25)
                ax.scatter(x+np.linspace(-.05,.05,len(values)),values,color=color,s=18)
                ax.scatter([x],[values.mean()],color=color,marker='D',s=45,label=condition if window==subset.window.min() else None)
        ax.set(xticks=range(len(windows)),xticklabels=[str(w) for w in windows],xlabel='Janela (s)',ylabel='BACC · seleção',ylim=(-.02,1.03),title=f'Offset {offset:g} s')
    # Include conditions that only occur in other offsets.
    mapping={}
    for ax in axes[0]:
        h,l=ax.get_legend_handles_labels();mapping.update(zip(l,h))
    fig.legend(mapping.values(),mapping.keys(),loc='outside lower center',ncols=min(2,max(1,len(mapping))),fontsize=10)
    fig.suptitle(f'{protocol} · {classifier} · folds dependentes; losango = média')
    return fig


def record_table(frame):
    """Compact identity and essential QC; full rows remain in CSV/JSON."""
    names={'project':'Projeto','subject':'Sujeito','session':'Sessão','block_type':'Bloco',
           'run_id':'Registro','total_epochs':'Total','accepted_epochs':'Aceitas',
           'rejected_epochs':'Rejeitadas','bad_channels':'Excluídos','quality_flags':'Alertas','status':'Status'}
    view=frame[[c for c in names if c in frame]].copy()
    if 'project' in view:
        view['project']=view.project.replace({'Brasil':'BR','Italia':'IT'})
    if 'run_id' in view:
        view['run_id']=view.run_id.map(lambda rid:str(rid).rsplit('__',1)[-1])
    for column in ['bad_channels','quality_flags']:
        if column in view:
            view[column]=view[column].map(lambda value: ', '.join(map(str,value)) or '—' if isinstance(value,(list,tuple)) else value)
    return view.rename(columns=names).to_html(index=False,escape=True)
