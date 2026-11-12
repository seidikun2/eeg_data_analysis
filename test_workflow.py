"""Dados sintéticos no formato de coleta; nenhum dado real ou identificação pessoal."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eeg import load_run, make_epochs, scan, selection, settings, trials_from_markers
from classify import evaluate, prepare, trial_splits


def synthetic(root, n=16, classes=("LEFT", "RIGHT")):
    path = root / "AB01" / "S1" / "IM_treino"
    path.mkdir(parents=True)
    sfreq = 128
    times = np.arange(0, n * 10 + 12, 1 / sfreq)
    rng   = np.random.default_rng(24)
    data  = rng.normal(0, 2, (len(times), 15))
    rows  = []
    for i in range(n):
        attempt = 6 + i * 10
        cls     = classes[i % len(classes)]
        for label, time in [("ATTENTION", attempt - 1.5), (f"{cls}_MI_STIM", attempt - 0.5),
                            ("ATTEMPT", attempt), ("PAUSE", attempt + 3.75)]:
            rows.append(dict(label=label, lsl_time_s=1000 + time))
        mask = (times >= attempt) & (times < attempt + 3.75)
        ch   = 3 if cls == "LEFT" else 9
        data[mask, ch] += 8 * np.sin(2 * np.pi * 10 * times[mask])
    signal = pd.DataFrame(data, columns=[f"ch{i}" for i in range(1, 16)])
    signal.insert(0, "lsl_time_s", times + 1000)
    signal.to_csv(path / "test_signal_20260101_120000.csv", index=False)
    pd.DataFrame(rows).to_csv(path / "test_markers_20260101_120000.csv", index=False)
    control          = root / "dataset_index.csv"
    index            = scan(root, control)
    index["include"] = True
    index.to_csv(control, index=False)
    return control


def quick_config():
    cfg = settings()
    cfg.update(windows=[1.0], offsets=[0.5], bands=[[[8, 30]]], svm_c=[1.0],
               outer_folds=2, inner_folds=2, search_budget=1, classifiers=["svm_rbf"], dpi=72)
    return cfg


def test_control_preserved_and_relocated(tmp_path):
    control                 = synthetic(tmp_path)
    frame                   = pd.read_csv(control)
    frame["exclude_trials"] = "1,3"
    frame["bad_channels"]   = "CPz"
    frame.to_csv(control, index=False)
    indexed = scan(tmp_path, control)
    assert indexed.include.all()
    assert indexed.exclude_trials.iloc[0] == "1,3"
    assert not Path(indexed.signal_path.iloc[0]).is_absolute()
    cfg              = quick_config()
    run              = load_run(tmp_path, indexed.iloc[0], cfg)
    _, epochs, audit = make_epochs(run, cfg)
    assert not {1, 3} & set(epochs.metadata.trial)
    assert run.clean.info["bads"] == ["CPz"]
    assert audit.accepted.all()


def test_missing_pause_and_invalid_sequence():
    markers = pd.DataFrame(dict(label=["ATTENTION", "LEFT_MI_STIM", "ATTEMPT", "ATTENTION", "ATTEMPT"],
                                lsl_time_s=[1, 2, 3, 8, 10]))
    trials  = trials_from_markers(markers, 20, 3.75)
    assert trials.pause_start.iloc[0] == 6.75
    assert trials.pause_source.iloc[0] == "inferred"
    assert not trials.valid.iloc[1]


def test_nested_cv_no_trial_leakage_and_probabilities(tmp_path):
    control      = synthetic(tmp_path)
    cfg          = quick_config()
    row          = selection(control).iloc[0]
    run          = load_run(tmp_path, row, cfg)
    _, epochs, _ = make_epochs(run, cfg)
    accepted     = set(epochs.metadata.trial)
    data, _      = prepare(run, cfg, "left_right", accepted)
    meta         = data[0][2]
    for tr, te in trial_splits(meta, 2, 42):
        assert not set(meta.trial.iloc[tr]) & set(meta.trial.iloc[te])
    summary, predictions, phase = evaluate(run, cfg, "left_right", accepted)
    assert predictions.groupby("trial").fold.nunique().eq(1).all()
    assert np.allclose(predictions[["p_LEFT", "p_RIGHT"]].sum(axis=1), 1)
    assert summary["n_trials"] == len(accepted)
    assert len(phase) == len(accepted)
    for fold in summary["folds"]:
        assert not set(fold["train_trials"]) & set(fold["test_trials"])


def test_irregular_timestamps_fail(tmp_path):
    control                     = synthetic(tmp_path, n=4)
    row                         = selection(control).iloc[0]
    path                        = tmp_path / row.signal_path
    frame                       = pd.read_csv(path)
    frame.loc[10, "lsl_time_s"] = frame.loc[9, "lsl_time_s"]
    frame.to_csv(path, index=False)
    cfg = quick_config()
    cfg["timestamp_policy"] = "strict"
    with pytest.raises(ValueError, match="duplicados"):
        load_run(tmp_path, row, cfg)
    cfg["timestamp_policy"] = "sample_clock"
    with pytest.warns(RuntimeWarning, match="reconstruído"):
        run = load_run(tmp_path, row, cfg)
    assert run.qc["timing"]["duplicate_intervals"] == 1
    assert run.raw.n_times == len(frame)
    assert np.all(np.diff(run.times) > 0)
    assert np.allclose(run.raw.get_data()[0], frame.ch1 * cfg["scale_to_volts"])



@pytest.mark.parametrize("pipeline", ["tgsp", "csp"])
def test_all_protocols_filterbank_and_csp(tmp_path, pipeline):
    control      = synthetic(tmp_path, n=24, classes=("LEFT", "RIGHT", "REST_STIM"))
    run          = load_run(tmp_path, selection(control).iloc[0], quick_config())
    _, epochs, _ = make_epochs(run, quick_config())
    accepted     = set(epochs.metadata.trial)
    cfg          = quick_config()
    cfg.update(bands=[[[8, 13], [13, 30]]], pipelines=[pipeline], offsets=[0.0])
    for protocol in ["reststim_move", "attention_move", "rest_left_right"]:
        summary, predictions, _ = evaluate(run, cfg, protocol, accepted)
        assert summary["n_trials"] >= 16
        assert summary["rest_source"] == "REST_STIM"
        assert predictions.groupby("trial").fold.nunique().eq(1).all()


def test_report_and_interface(tmp_path):
    from analyze import batch
    from streamlit.testing.v1 import AppTest
    control = synthetic(tmp_path, n=8)
    output  = tmp_path / "reports"
    results = batch(tmp_path, control, output, quick_config(), mode="ephys", detailed=True)
    assert results[0]["status"] == "ok"
    folder = output / results[0]["run_id"]
    for name in ["report.html", "dashboard.png", "topografias_mu_beta.pdf", "pca_2d_3d.png", "baseline_audit.csv"]:
        assert (folder / name).stat().st_size > 100
    assert "Épocas · antes/depois conjuntos" in (folder / "report.html").read_text()
    at = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run()
    at.sidebar.text_input[0].set_value(str(tmp_path)).run()
    assert not list(at.exception)
    assert [t.label for t in at.tabs] == ["Controle dos dados", "Verificar qualidade", "Produzir relatório", "Consultar resultados"]


def test_online_requires_prior_included_calibration(tmp_path):
    from analyze import calibration_row
    control              = synthetic(tmp_path, n=4)
    included             = selection(control)
    online               = included.iloc[0].copy()
    online["stamp"]      = "20260101_130000"
    online["block_type"] = "online_com_feedback"
    assert calibration_row(online, included).run_id == included.iloc[0].run_id
    online["stamp"] = "20260101_110000"
    with pytest.raises(ValueError, match="anterior"):
        calibration_row(online, included)


def test_clock_reset_refused():
    from eeg import sample_clock
    times = np.arange(2000) / 128
    times[1000:] -= 2
    with pytest.raises(ValueError, match="Relógio inconsistente"):
        sample_clock(times, quick_config())


def test_legacy_online_is_feedback(tmp_path):
    control = synthetic(tmp_path, n=4)
    frame = pd.read_csv(control)
    frame["block_type"] = "online_definir"
    frame.to_csv(control, index=False)
    assert selection(control).block_type.eq("online_com_feedback").all()
    assert scan(tmp_path, control).block_type.eq("online_com_feedback").all()


@pytest.mark.parametrize("chunks", [False, True])
def test_clock_chunks_and_small_reversal(chunks):
    from eeg import sample_clock
    expected = np.arange(1280) / 128 + 1000
    times = expected.copy()
    if chunks:
        times = expected[(np.arange(len(times)) // 8) * 8]
    else:
        times[50] = times[48]
    with pytest.warns(RuntimeWarning, match="reconstruído"):
        corrected, qc = sample_clock(times, quick_config())
    assert len(corrected) == len(times)
    assert np.all(np.diff(corrected) > 0)
    assert qc["sfreq"] == pytest.approx(128, rel=0.001)
    assert qc["max_correction_s"] < 0.1


def test_marker_edges_do_not_discard_recording(tmp_path):
    control = synthetic(tmp_path, n=8)
    row = selection(control).iloc[0]
    path = tmp_path / row.marker_path
    frame = pd.read_csv(path)
    frame.loc[frame.index[frame.label == 'ATTENTION'][0], 'lsl_time_s'] = 999.995
    extra = pd.DataFrame({'label': ['BASELINE', 'BLOCK_END'], 'lsl_time_s': [999.99, 1100.01]})
    pd.concat([extra, frame]).to_csv(path, index=False)
    with pytest.warns(RuntimeWarning, match='bordas'):
        run = load_run(tmp_path, row, quick_config())
    assert run.qc['marker_bounds']['outside_count'] == 3
    assert run.trials.valid.all()
    assert len(make_epochs(run, quick_config())[1]) == 8


def test_check_collects_errors_from_both_files(tmp_path):
    from analyze import check_dataset
    control = synthetic(tmp_path, n=4)
    row = selection(control).iloc[0]
    (tmp_path / row.signal_path).unlink()
    pd.DataFrame({'wrong_column': [1]}).to_csv(tmp_path / row.marker_path, index=False)
    output = tmp_path / 'diagnostics'
    results = check_dataset(tmp_path, control, output, quick_config())
    assert results[0]['status'] == 'error'
    assert {c['stage'] for c in results[0]['checks'] if c['status'] == 'error'} == {'signal_path', 'marker_path'}
    for name in ['diagnostics.json', 'diagnostics.csv', 'diagnostics.html']:
        assert (output / name).exists()


def test_infeasible_long_window_does_not_poison_short_candidate(tmp_path):
    control = synthetic(tmp_path, n=16)
    row = selection(control).iloc[0]
    path = tmp_path / row.marker_path
    markers = pd.read_csv(path)
    markers.loc[markers.index[markers.label == 'ATTENTION'][0], 'lsl_time_s'] = 1002.5
    markers.to_csv(path, index=False)
    cfg = quick_config()
    cfg.update(windows=[1.0, 1.5], offsets=[0.0], search_budget=2)
    run = load_run(tmp_path, row, cfg)
    _, epochs, _ = make_epochs(run, cfg)
    data, errors = prepare(run, cfg, 'attention_move', set(epochs.metadata.trial), require_cv=True)
    assert len(data) == 1 and len(errors) == 1
    assert data[0][0]['window'] == 1.0
    assert data[0][2].trial.nunique() == 16


def test_legacy_mne_montage_name(tmp_path, monkeypatch):
    import mne
    control = synthetic(tmp_path, n=4)
    monkeypatch.setattr(mne.channels, 'get_builtin_montages', lambda **kwargs: ['standard_1020'])
    run = load_run(tmp_path, selection(control).iloc[0], quick_config())
    assert run.qc['montage'] == 'standard_1020'
    assert all(channel['loc'][:3].any() for channel in run.raw.info['chs'])
    assert len(make_epochs(run, quick_config())[1]) == 4


def test_comparison_timeline_and_causal_smoothing(tmp_path):
    from analyze import batch
    from reporting import smoothed_timeline
    control = synthetic(tmp_path, n=16)
    cfg = quick_config()
    output = tmp_path / 'comparison'
    results = batch(tmp_path, control, output, cfg, mode='classification', protocols=['left_right'], compare=True)
    assert results[0]['status'] == 'ok'
    assert {m['pipeline'] for m in results[0]['classification'] if m['status']=='ok'} == {'tgsp','csp'}
    assert (output / 'comparison.html').exists()
    scores = pd.read_csv(output / 'comparison_subjects.csv')
    assert len(scores) == 2 and scores.run_id.nunique() == 1
    paired = pd.read_csv(output / 'pipeline_paired_differences.csv')
    assert len(paired) == 0  # no pooling across repetitions
    folder = output / results[0]['run_id']
    timeline = pd.read_csv(folder / 'tgsp_svm_rbf_left_right_timeline.csv')
    evaluation = pd.read_csv(folder / 'tgsp_svm_rbf_left_right_predictions.csv')
    assert timeline.groupby('trial').fold.nunique().eq(1).all()
    assert timeline.groupby('trial').fold.first().equals(evaluation.groupby('trial').fold.first())
    assert set(timeline.phase) >= {'STIMULUS','ATTEMPT','PAUSE'}
    assert np.allclose(timeline[['p_LEFT','p_RIGHT']].sum(axis=1),1)
    smooth = smoothed_timeline(timeline, .6)
    changed = timeline.copy()
    last = changed.index[-1]
    changed.loc[last,'p_LEFT'] = .123
    again = smoothed_timeline(changed,.6)
    assert np.allclose(smooth.p_LEFT.iloc[:-1],again.p_LEFT.iloc[:-1])
    page = (folder/'report.html').read_text()
    assert 'plotly' in page.lower() and 'Raw antes/depois · zoom por canal' not in page  # modo classificação
    assert 'viewport' in page and 'svm_rbf' in page


def test_context_trials_not_used_for_training_or_metrics(tmp_path):
    control = synthetic(tmp_path, n=24, classes=('LEFT','RIGHT','REST_STIM'))
    cfg = quick_config()
    run = load_run(tmp_path, selection(control).iloc[0], cfg)
    _, epochs, _ = make_epochs(run,cfg)
    summary, evaluation, _ = evaluate(run,cfg,'left_right',set(epochs.metadata.trial))
    context = pd.DataFrame(summary['_timeline'])
    assert context.trial.nunique() == 24
    assert evaluation.trial.nunique() == 16
    assert set(context.loc[context.context_only,'trial_class']) == {'REST_STIM'}
    assert context.groupby('trial').fold.nunique().eq(1).all()
    for fold in summary['folds']:
        assert not set(context.loc[context.fold==fold['fold'],'trial']) & set(fold['train_trials'])


def test_batch_comparison_keeps_subjects_and_repetitions_separate(tmp_path):
    from reporting import batch_comparison
    records=[]
    for subject in ['AB01','AB02']:
        for index, score in enumerate([.4,.8]):
            records.append(dict(run_id=f'{subject}_{index}',subject=subject,session='S1',block_type='IM_treino',
                                classification=[dict(protocol='left_right',pipeline='tgsp',validation='nested_trial_cv',
                                                     balanced_accuracy=score,chance=.5,status='ok')]))
    batch_comparison(tmp_path,records)
    frame=pd.read_csv(tmp_path/'comparison_subjects.csv')
    assert len(frame)==4
    assert frame.run_id.nunique()==4
    assert sorted(frame.balanced_accuracy)==[.4,.4,.8,.8]


def test_general_report_two_subjects_and_pipelines(tmp_path):
    """Exercise actual batch generation, not a one-subject comparison mock."""
    import shutil
    from analyze import batch
    control = synthetic(tmp_path, n=8)
    shutil.copytree(tmp_path / "AB01", tmp_path / "CD02")
    indexed = scan(tmp_path, control)
    indexed["include"] = True
    indexed.to_csv(control, index=False)
    output = tmp_path / "output"
    results = batch(tmp_path, control, output, quick_config(), mode="both",
                    protocols=["left_right"], compare=True)
    assert len(results) == 2
    assert all(r["status"] == "ok" for r in results)
    assert all(len(r["classification"]) == 2 and all(m["status"] == "ok" for m in r["classification"]) for r in results)
    general = (output / "index.html").read_text()
    assert "AB01" in general and "CD02" in general
    assert "evolução nas tentativas" in general
    assert "Relatórios individuais" in general
    assert "<details" not in general
    comparisons = pd.read_csv(output / "comparison_runs.csv")
    assert set(comparisons.subject) == {"AB01", "CD02"}
    assert set(comparisons.pipeline) == {"tgsp", "csp"}
    assert set(comparisons.session) == {"S1"}
    for result in results:
        folder = output / result["run_id"]
        for name in ["tgsp_svm_rbf_left_right", "csp_svm_rbf_left_right"]:
            import json
            summary = json.loads((folder / f"{name}.json").read_text())
            for fold in summary["folds"]:
                assert not set(fold["train_trials"]) & set(fold["test_trials"])


def test_timeline_background_and_gaps(tmp_path):
    from reporting import timeline_plot
    control = synthetic(tmp_path)
    cfg = quick_config()
    run = load_run(tmp_path, selection(control).iloc[0], cfg)
    _, epochs, _ = make_epochs(run, cfg)
    summary, _, _ = evaluate(run, cfg, "left_right", set(epochs.metadata.trial))
    timeline = pd.DataFrame(summary["_timeline"])
    fig = timeline_plot(summary, timeline, cfg, run.trials)
    regions = [s for s in fig.layout.shapes if s.type == "rect"]
    assert len(regions) == timeline.trial.nunique()
    for trace in fig.data:
        assert trace.connectgaps is not True
    assert fig.layout.xaxis.domain == (0., 1.)


def test_italia_mapping_projects_and_antialias_decimation(tmp_path):
    import shutil
    from scipy.signal import resample_poly
    from eeg import ITALIA_CHANNELS
    control=synthetic(tmp_path,n=4)
    brazil=tmp_path/'Projeto Brasil'
    brazil.mkdir()
    shutil.move(str(tmp_path/'AB01'),brazil/'AB01')
    italy=tmp_path/'Projeto Italia'
    italy.mkdir()
    shutil.copytree(brazil/'AB01',italy/'AB01')
    path=next(italy.rglob('*_signal_*.csv'))
    data=pd.read_csv(path)
    source=resample_poly(data[[f'ch{i}' for i in range(1,16)]].to_numpy(),2,1)
    frame=pd.DataFrame(source,columns=[f'ch{i}' for i in range(1,16)])
    rng=np.random.default_rng(37)
    for ch in range(16,len(ITALIA_CHANNELS)+1):
        frame[f'ch{ch}']=rng.normal(0,2,len(frame))
    frame[f'ch{len(ITALIA_CHANNELS)+1}']=0
    frame.insert(0,'lsl_time_s',1000+np.arange(len(frame))/256)
    frame.to_csv(path,index=False)
    # Fresh index: legacy missing paths remain auditable but are not selected.
    new_control=tmp_path/'new_index.csv'
    indexed=scan(tmp_path,new_control)
    indexed['include']=True
    indexed.to_csv(new_control,index=False)
    assert len(indexed)==2 and indexed.run_id.nunique()==2
    row=selection(new_control,projects=['Italia']).iloc[0]
    run=load_run(tmp_path,row,quick_config())
    assert run.raw.ch_names==ITALIA_CHANNELS
    assert run.qc['ignored_zero_trigger']
    assert run.qc['decimation']==2
    assert np.isclose(run.raw.info['sfreq'],128)
    assert run.trials.attempt_start.iloc[0]==1006
    assert len(run.times)==len(frame.iloc[::2])
    assert run.raw.get_data().shape[0]==53
    # An unidentified, nonzero extra channel must never be silently discarded.
    frame.iloc[0,-1]=1
    frame.to_csv(path,index=False)
    with pytest.raises(ValueError,match='Canais sem mapa'):
        load_run(tmp_path,row,quick_config())


def test_fixed_cv_same_trial_folds_for_all_classifiers(tmp_path):
    from classify import fixed_parameter_cv
    control=synthetic(tmp_path,n=8)
    cfg=quick_config()
    cfg.update(classifiers=['lda','svm_linear','svm_rbf'],windows=[1.,1.5],search_budget=6)
    run=load_run(tmp_path,selection(control).iloc[0],cfg)
    _,epochs,_=make_epochs(run,cfg)
    datasets,_=prepare(run,cfg,'left_right',set(epochs.metadata.trial),require_cv=True)
    records=fixed_parameter_cv(datasets,cfg)
    assert len(records)==12
    assert all(r['status']=='ok' for r in records)
    by_fold={}
    for r in records:
        assert not set(r['train_trials']) & set(r['test_trials'])
        split=(r['train_trials'],r['test_trials'])
        assert r['fold'] not in by_fold or by_fold[r['fold']]==split
        by_fold[r['fold']]=split
    assert {r['params']['classifier'] for r in records}=={'lda','svm_linear','svm_rbf'}


def test_online_continuous_inference_uses_previous_training_and_covers_intervals(tmp_path):
    from classify import evaluate
    from reporting import timeline_plot
    from analyze import calibration_row
    trainroot,testroot=tmp_path/'train',tmp_path/'test'
    traincontrol=synthetic(trainroot,n=8)
    testcontrol=synthetic(testroot,n=8)
    cfg=quick_config()
    train=load_run(trainroot,selection(traincontrol).iloc[0],cfg)
    test=load_run(testroot,selection(testcontrol).iloc[0],cfg)
    _,trepochs,_=make_epochs(train,cfg)
    _,teepochs,_=make_epochs(test,cfg)
    summary,_,_=evaluate(test,cfg,'left_right',set(teepochs.metadata.trial),(train,set(trepochs.metadata.trial)))
    timeline=pd.DataFrame(summary['_continuous_timeline'])
    assert summary['continuous_role']=='online_test_prior_IM_treino'
    assert summary['validation']=='prior_calibration_proxy'
    gaps=np.diff(timeline.start_abs)
    assert np.max(gaps)-np.min(gaps)<1e-8
    assert timeline.start_abs.iloc[0]<test.trials.attention_start.iloc[0]
    assert timeline.start_abs.iloc[-1]>test.trials.attempt_end.iloc[-1]
    assert np.allclose(timeline[['p_LEFT','p_RIGHT']].sum(axis=1),1)
    fig=timeline_plot(summary,timeline,cfg,test.trials)
    full=[t for t in fig.data if t.xaxis=='x']
    assert len(full)==2 and all(len(t.x)==len(timeline) for t in full)
    online=selection(testcontrol).iloc[0].copy()
    online['project']='Italia'
    online['stamp']='20260101_130000'
    included=selection(traincontrol)
    with pytest.raises(ValueError,match='anterior'):
        calibration_row(online,included)


def test_attention_baseline_and_pca_test_trials(tmp_path):
    from analyze import baseline_intervals,baseline_trials
    from analysis_views import tangent_pca
    control=synthetic(tmp_path,n=8)
    cfg=quick_config()
    cfg['erd_baseline_mode']='attention'
    run=load_run(tmp_path,selection(control).iloc[0],cfg)
    _,epochs,_=make_epochs(run,cfg)
    intervals=baseline_intervals(run,epochs,cfg)
    for row,interval in zip(epochs.metadata.itertuples(),intervals):
        assert interval is not None
        start,end=np.asarray(interval)+row.attempt_start
        assert row.attention_start<start<end<row.stimulus_start
    valid,audit=baseline_trials(run,epochs,cfg)
    assert valid.all() and not audit.baseline_in_pause.any()
    assert audit.baseline_phase.eq('ATTENTION').all()
    fig,_=tangent_pca(epochs,cfg,tmp_path)
    windows=pd.read_csv(tmp_path/'pca_windows.csv')
    train=windows[windows.role=='train']
    test=windows[windows.role=='test']
    assert not set(train.trial)&set(test.trial)
    assert test.groupby('trial').size().min()>1
    assert any(t.type=='contour' for t in fig.data)
    assert all(t.legendgroup for t in fig.data)


def test_session_screening_repetitions_and_online_channel_loss(tmp_path):
    import shutil
    from analyze import batch
    control=synthetic(tmp_path,n=8)
    source=tmp_path/'AB01'/'S1'/'IM_treino'
    shutil.copytree(source,source.with_name('EM_treino'))
    online=source.with_name('online_com_feedback')
    shutil.copytree(source,online)
    for path in list(source.glob('*.csv')):
        shutil.copyfile(path,path.with_name(path.name.replace('120000','123000')))
    for path in list(online.glob('*.csv')):
        path.rename(path.with_name(path.name.replace('120000','130000')))
    indexed=scan(tmp_path,control)
    indexed['include']=True
    indexed.loc[indexed.block_type=='online_com_feedback','bad_channels']='C3'
    indexed.to_csv(control,index=False)
    cfg=quick_config()
    cfg['classifiers']=['lda','svm_linear']
    progress=[]
    output=tmp_path/'reports'
    results=batch(tmp_path,control,output,cfg,mode='classification',protocols=['left_right'],
                  progress=lambda done,total,eta,name: progress.append((done,total,eta)))
    assert len(results)==4 and all(r['status']=='ok' for r in results)
    assert progress[-1]==(4,4,0)
    assert all(p[2]>=0 for p in progress)
    screening=next(r for r in results if r['block_type']=='EM_treino')
    assert not screening['classification']
    test=next(r for r in results if r['block_type']=='online_com_feedback')
    assert all(m['status']=='ok' and m['validation']=='prior_calibration_proxy' for m in test['classification'])
    assert all('123000' in m['train_source'] for m in test['classification'])
    assert test['bad_channels']==['C3']
    assert all(r['total_epochs']==8 for r in results)
    session=output/'session_Brasil_AB01_S1.html'
    assert session.exists() and 'Performance junto à qualidade' in session.read_text()
    comparison=pd.read_csv(output/'comparison_runs.csv')
    assert comparison.run_id.nunique()==3  # both IM repetitions and online, no EM classifier
    for result in results:
        assert (output/result['run_id']/'check.html').exists()
    import json
    qc=json.loads((output/test['run_id']/'parameters_qc.json').read_text())['qc']
    assert 'C3' not in qc['transfer_channels']
    assert 'C3' in qc['channels']  # original record is preserved for quality/snapshot
