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
               outer_folds=2, inner_folds=2, search_budget=1, dpi=72)
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
    assert "Trial 0" in (folder / "report.html").read_text()
    at = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run()
    at.sidebar.text_input[0].set_value(str(tmp_path)).run()
    assert not list(at.exception)
    assert [t.label for t in at.tabs] == ["Controle dos dados", "Produzir relatório", "Consultar resultados"]


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
    assert len(scores) == 2 and scores.n_runs.eq(1).all()
    paired = pd.read_csv(output / 'pipeline_paired_differences.csv')
    assert len(paired) == 1
    folder = output / results[0]['run_id']
    timeline = pd.read_csv(folder / 'tgsp_left_right_timeline.csv')
    evaluation = pd.read_csv(folder / 'tgsp_left_right_predictions.csv')
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
    assert 'viewport' in page and 'Classificador' in page


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


def test_batch_comparison_aggregates_subjects_not_runs(tmp_path):
    from reporting import batch_comparison
    records=[]
    for subject in ['AB01','AB02']:
        for index, score in enumerate([.4,.8]):
            records.append(dict(run_id=f'{subject}_{index}',subject=subject,session='S1',block_type='IM_treino',
                                classification=[dict(protocol='left_right',pipeline='tgsp',validation='nested_trial_cv',
                                                     balanced_accuracy=score,chance=.5,status='ok')]))
    batch_comparison(tmp_path,records)
    frame=pd.read_csv(tmp_path/'comparison_subjects.csv')
    assert len(frame)==2
    assert frame.n_runs.eq(2).all()
    assert np.allclose(frame.balanced_accuracy,.6)
