"""TGSP/SVM, busca aninhada por trial e probabilidades calibradas sem sobreposição."""

from __future__ import annotations

from itertools import product
import hashlib
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from eeg import PROTOCOLS


class BandFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, kind="tgsp"):
        self.kind = kind

    def fit(self, X, y=None):
        self.transforms_ = []
        for band in range(X.shape[1]):
            if self.kind == "tgsp":
                model = make_pipeline(Covariances(estimator="oas"), TangentSpace(metric="riemann"))
            elif self.kind == "csp":
                model = mne.decoding.CSP(n_components=min(4, X.shape[2]), reg="oas", log=True)
            else:
                raise ValueError(f"Pipeline desconhecida: {self.kind}")
            self.transforms_.append(model.fit(X[:, band], y))
        return self

    def transform(self, X):
        return np.concatenate([model.transform(X[:, i]) for i, model in enumerate(self.transforms_)], axis=1)


def candidates(cfg):
    params = []
    seen = set()
    for w, o, b, p, classifier, c, g in product(cfg["windows"], cfg["offsets"], cfg["bands"],
                      cfg["pipelines"], cfg.get("classifiers", ["svm_rbf"]), cfg["svm_c"], cfg["svm_gamma"]):
        if classifier not in {"lda", "svm_linear", "svm_rbf"}:
            raise ValueError(f"Classificador desconhecido: {classifier}")
        param = dict(window=w, offset=o, bands=b, pipeline=p, classifier=classifier,
                     C=c if classifier != "lda" else None, gamma=g if classifier == "svm_rbf" else None)
        key = json.dumps(param, sort_keys=True)
        if key not in seen:
            params.append(param)
            seen.add(key)
    budget = cfg["search_budget"]
    if budget < 1:
        raise ValueError("search_budget deve ser positivo.")
    if len(params) > budget:
        rng    = np.random.default_rng(cfg["seed"])
        chosen = np.r_[0, rng.choice(np.arange(1, len(params)), budget - 1, replace=False)]
        params = [params[i] for i in chosen]
    return params


def phase_target(cls, phase, protocol, rest_source):
    motor = cls in {"LEFT", "RIGHT", "BOTH"}
    if protocol == "left_right":
        return cls if phase == "ATTEMPT" and cls in {"LEFT", "RIGHT"} else None
    if protocol == "reststim_move":
        return ("MOVE" if motor else "REST_STIM") if phase == "ATTEMPT" else None
    if protocol == "rest_move":
        return "REST" if phase == "PAUSE" else ("MOVE" if phase == "ATTEMPT" and motor else None)
    if protocol == "attention_move":
        return "ATTENTION" if phase == "ATTENTION" else ("MOVE" if phase == "ATTEMPT" and motor else None)
    if protocol == "rest_left_right":
        if phase == "ATTEMPT" and cls in {"LEFT", "RIGHT", rest_source}:
            return "REST" if cls == "REST_STIM" else cls
        return "REST" if rest_source == "REST" and phase == "PAUSE" else None
    raise ValueError(protocol)


def trial_signal(run, trial, cfg, params, picks):
    sfreq = run.raw.info["sfreq"]
    lo = max(0, int(round((trial.attention_start - run.times[0]) * sfreq)))
    hi = min(run.raw.n_times, int(round((trial.pause_end - run.times[0]) * sfreq)))
    segment = run.raw.get_data(picks=picks, start=lo, stop=hi)
    for frequency in cfg["notch"]:
        if frequency < sfreq / 2:
            segment = mne.filter.notch_filter(segment, sfreq, [frequency], method="iir", phase="forward", verbose=False)
    if cfg["reference"] == "average":
        segment = segment - segment.mean(axis=0, keepdims=True)
    filtered = np.stack([mne.filter.filter_data(segment, sfreq, low, high, method="iir", phase="forward",
                         iir_params=dict(order=4, ftype="butter"), verbose=False) for low, high in params["bands"]])
    return filtered, lo, hi


def continuous_windows(run, cfg, params, trials):
    sfreq = run.raw.info["sfreq"]
    picks = mne.pick_types(run.raw.info, eeg=True, exclude="bads")
    size  = int(round(params["window"] * sfreq))
    data, rows = [], []
    for trial in run.trials.itertuples():
        if not trial.valid or trial.trial not in trials:
            continue
        signal, lo, hi = trial_signal(run, trial, cfg, params, picks)
        for first in range(0, hi - lo - size + 1, max(1, round(cfg["step"] * sfreq))):
            end = run.times[0] + (lo + first + size) / sfreq
            phase = "ATTENTION" if end < trial.stimulus_start else ("STIMULUS" if end < trial.attempt_start else ("ATTEMPT" if end < trial.pause_start else "PAUSE"))
            data.append(signal[:, :, first:first + size])
            rows.append(dict(trial=trial.trial, trial_class=trial.trial_class, phase=phase,
                             start_abs=end, time_end=end - trial.attempt_start))
    if not rows:
        return None, pd.DataFrame()
    return np.array(data), pd.DataFrame(rows)


def windows(run, cfg, params, protocol, accepted, rest_source=None):
    sfreq = run.clean.info["sfreq"]
    picks = mne.pick_types(run.clean.info, eeg=True, exclude="bads")
    if any(not cfg["l_freq"] <= lo < hi <= cfg["h_freq"] for lo, hi in params["bands"]):
        raise ValueError("Bandas da classificação precisam caber no filtro eletrofisiológico.")
    rest_source = rest_source or ("REST_STIM" if (run.trials.trial_class == "REST_STIM").any() else "REST")
    n_samples   = int(round(params["window"] * sfreq))
    data, meta  = [], []
    for trial in run.trials.itertuples():
        if not trial.valid or trial.trial not in accepted:
            continue
        filtered, lo, hi = trial_signal(run, trial, cfg, params, picks)
        if hi - lo < n_samples:
            continue
        phases   = [("ATTENTION", trial.attention_start, trial.attention_end),
                  ("ATTEMPT", trial.attempt_start, trial.attempt_end),
                  ("PAUSE", trial.pause_start, trial.pause_end)]
        for phase, start, end in phases:
            target = phase_target(trial.trial_class, phase, protocol, rest_source)
            if target is None:
                continue
            start = max(start + params["offset"], run.times[0])
            end    = min(end, start + cfg["attempt_duration"] - params["offset"])
            for time in np.arange(start, end - params["window"] + 1e-8, cfg["step"]):
                first = int(round((time - run.times[0]) * sfreq)) - lo
                piece = filtered[:, :, first:first + n_samples]
                if first < 0 or piece.shape[-1] != n_samples:
                    continue
                if np.any(np.ptp(piece, axis=-1) > cfg["reject_uv"] * 1e-6):
                    continue
                data.append(piece)
                meta.append(dict(trial=trial.trial, stratum=trial.trial_class,
                                 target=target, phase=phase, start_abs=time,
                                 time_end=time + params["window"] - trial.attempt_start,
                                 rest_source=rest_source))
    if not data:
        raise ValueError("Nenhuma janela disponível para este protocolo/parâmetro.")
    return np.asarray(data), pd.DataFrame(meta)


def trial_splits(meta, max_splits, seed):
    trials   = meta[["trial", "stratum"]].drop_duplicates("trial").reset_index(drop=True)
    maximum  = min(max_splits, int(trials.stratum.value_counts().min()))
    expected = set(meta.target)
    for k in range(maximum, 1, -1):
        cv     = StratifiedKFold(k, shuffle=True, random_state=seed)
        splits = []
        for train, test in cv.split(trials.trial, trials.stratum):
            tr = np.flatnonzero(meta.trial.isin(trials.trial.iloc[train]))
            te = np.flatnonzero(meta.trial.isin(trials.trial.iloc[test]))
            if set(meta.target.iloc[tr]) != expected or set(meta.target.iloc[te]) != expected:
                break
            splits.append((tr, te))
        if len(splits) == k:
            return splits
    raise ValueError("Trials independentes insuficientes para folds com todas as classes.")


def estimator(params, cfg):
    kind = params.get("classifier", "svm_rbf")
    if kind == "lda":
        classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    else:
        classifier = SVC(C=params["C"], gamma=params.get("gamma") or "scale",
                         kernel="linear" if kind == "svm_linear" else "rbf",
                         class_weight="balanced", random_state=cfg["seed"])
    return make_pipeline(BandFeatures(params["pipeline"]), StandardScaler(), classifier)


def parameter_id(params):
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]


def fixed_parameter_cv(datasets, cfg):
    """Same grouped folds for every fixed candidate; selection scores, not test estimates."""
    base = datasets[0][2]
    splits = trial_splits(base, cfg["outer_folds"], cfg["seed"])
    trial_folds = [(set(base.trial.iloc[tr]), set(base.trial.iloc[te])) for tr, te in splits]
    records = []
    for params, X, meta in datasets:
        for fold, (train_trials, test_trials) in enumerate(trial_folds):
            tr, te = np.flatnonzero(meta.trial.isin(train_trials)), np.flatnonzero(meta.trial.isin(test_trials))
            row = dict(candidate_id=parameter_id(params), fold=fold, params=params,
                       train_trials=sorted(train_trials), test_trials=sorted(test_trials),
                       role="selection_cv_not_independent_test")
            try:
                model = estimator(params, cfg).fit(X[tr], meta.target.iloc[tr])
                row.update(status="ok", balanced_accuracy=phase_score(meta.iloc[te], model.predict(X[te])))
            except (ValueError, np.linalg.LinAlgError) as exc:
                row.update(status=str(exc), balanced_accuracy=None)
            records.append(row)
    return records


def block_windows(run, cfg, params):
    """Causal full-block inference, including inter-trial intervals; no edge extrapolation."""
    sfreq = run.raw.info["sfreq"]
    picks = mne.pick_types(run.raw.info, eeg=True, exclude="bads")
    segment = run.raw.get_data(picks=picks)
    for frequency in cfg["notch"]:
        if frequency < sfreq / 2:
            segment = mne.filter.notch_filter(segment, sfreq, [frequency], method="iir", phase="forward", verbose=False)
    if cfg["reference"] == "average":
        segment -= segment.mean(axis=0, keepdims=True)
    filtered = np.stack([mne.filter.filter_data(segment, sfreq, lo, hi, method="iir", phase="forward",
                  iir_params=dict(order=4, ftype="butter"), verbose=False) for lo, hi in params["bands"]])
    size, step = round(params["window"] * sfreq), max(1, round(cfg["step"] * sfreq))
    starts = np.arange(0, segment.shape[-1] - size + 1, step)
    # Chunk prediction below avoids a giant materialized window tensor on long recordings.
    return filtered, starts, size


def full_block_prediction(model, run, cfg, params, role):
    filtered, starts, size = block_windows(run, cfg, params)
    records = []
    sfreq = run.raw.info["sfreq"]
    for chunk in range(0, len(starts), 256):
        selected = starts[chunk:chunk + 256]
        X = np.stack([filtered[:, :, first:first + size] for first in selected])
        probabilities = model.predict_proba(X)
        for first, probs in zip(selected, probabilities):
            end = run.times[0] + (first + size) / sfreq
            matches = run.trials[(run.trials.attention_start <= end) & (run.trials.pause_end > end)]
            trial = matches.iloc[-1] if len(matches) else None
            records.append(dict(trial=int(trial.trial) if trial is not None else -1,
                                trial_class=trial.trial_class if trial is not None else "INTERVAL",
                                phase=("ATTEMPT" if trial is not None and trial.attempt_start <= end < trial.attempt_end else "CONTEXT"),
                                start_abs=end, time_end=end - trial.attempt_start if trial is not None else np.nan,
                                fold=-1, prediction_role=role, context_only=trial is None,
                                **{f"p_{cls}": float(v) for cls, v in zip(model.classes_, probs)}))
    return pd.DataFrame(records)


def final_model(train_run, train_accepted, cfg, protocol, params, rest_source=None):
    # Match deployment preprocessing: continuous causal filtering, but supervised windows
    # are wholly within eligible trial phases. They never cross trial boundaries.
    X, meta = windows(train_run, cfg, params, protocol, train_accepted, rest_source)
    filtered, _, size = block_windows(train_run, cfg, params)
    sfreq = train_run.raw.info["sfreq"]
    starts = np.rint((meta.start_abs.to_numpy() - train_run.times[0]) * sfreq).astype(int)
    X = np.stack([filtered[:, :, first:first + size] for first in starts])
    return calibrated(params, X, meta, cfg)


def phase_score(meta, predictions):
    frame         = meta[["trial", "target"]].copy()
    frame["pred"] = predictions
    phase         = frame.groupby(["trial", "target"], sort=False).pred.agg(lambda s: s.mode().iloc[0]).reset_index()
    return float(balanced_accuracy_score(phase.target, phase.pred))


def prepare(run, cfg, protocol, accepted, rest_source=None, require_cv=False):
    datasets, errors = [], []
    cache            = {}
    for params in candidates(cfg):
        try:
            key = json.dumps({k: params[k] for k in ["window", "offset", "bands"]}, sort_keys=True)
            if key not in cache:
                cache[key] = windows(run, cfg, params, protocol, accepted, rest_source)
            X, meta = cache[key]
            if set(meta.target) != set(PROTOCOLS[protocol]):
                raise ValueError("Classes ausentes.")
            if require_cv:
                trial_splits(meta, cfg["outer_folds"], cfg["seed"])
            datasets.append((params, X, meta))
        except ValueError as exc:
            errors.append(dict(params=params, error=str(exc)))
    if not datasets:
        raise ValueError(f"Nenhum candidato viável: {errors[:2]}")
    # Todos os candidatos comparam exatamente as mesmas unidades trial/classe.
    units = set.intersection(*[set(zip(m.trial, m.target)) for _, _, m in datasets])
    if not units:
        raise ValueError("Não há unidades trial/classe comuns aos candidatos.")
    result, selected = [], {}
    for params, X, meta in datasets:
        key = id(X)
        if key not in selected:
            keep          = np.array([(t, y) in units for t, y in zip(meta.trial, meta.target)])
            selected[key] = X[keep], meta.loc[keep].reset_index(drop=True)
            if set(selected[key][1].target) != set(PROTOCOLS[protocol]):
                raise ValueError("A interseção dos candidatos removeu uma classe; revise janelas/offsets e qualidade.")
        result.append((params, *selected[key]))
    return result, errors


def choose(datasets, train_trials, cfg):
    searches = []
    best     = None
    for index, (params, X, meta) in enumerate(datasets):
        idx = np.flatnonzero(meta.trial.isin(train_trials))
        sub = meta.iloc[idx].reset_index(drop=True)
        try:
            splits = trial_splits(sub, cfg["inner_folds"], cfg["seed"])
            scores = []
            for tr, te in splits:
                model = estimator(params, cfg).fit(X[idx[tr]], sub.target.iloc[tr])
                scores.append(phase_score(sub.iloc[te], model.predict(X[idx[te]])))
            score = float(np.mean(scores))
            searches.append(dict(params=params, score=score, status="ok"))
            if best is None or score > best[0]:
                best = score, index
        except (ValueError, np.linalg.LinAlgError) as exc:
            searches.append(dict(params=params, score=None, status=str(exc)))
    if best is None:
        raise ValueError("Busca interna inviável: aumentar trials ou reduzir folds; não será usado CV de janelas.")
    return best[1], searches


def calibrated(params, X, meta, cfg):
    cv    = trial_splits(meta, cfg["inner_folds"], cfg["seed"])
    model = CalibratedClassifierCV(estimator(params, cfg), method="sigmoid", cv=cv, ensemble=False)
    return model.fit(X, meta.target)


def evaluate(run, cfg, protocol, accepted, training=None):
    datasets, invalid        = prepare(run, cfg, protocol, accepted, require_cv=training is None)
    records, folds, searches, timelines = [], [], [], []
    if training is None:
        base_meta  = datasets[0][2]
        splits     = trial_splits(base_meta, cfg["outer_folds"], cfg["seed"])
        partitions = [(set(base_meta.trial.iloc[tr]), set(base_meta.trial.iloc[te])) for tr, te in splits]
        train_sets = datasets
    else:
        train_run, train_accepted = training
        source                    = datasets[0][2].rest_source.iloc[0]
        train_sets, invalid_train = prepare(train_run, cfg, protocol, train_accepted, source, require_cv=True)
        invalid += invalid_train
        partitions = [(set(train_sets[0][2].trial), set(datasets[0][2].trial))]
    fixed_cv = fixed_parameter_cv(train_sets, cfg) if cfg.get("fixed_parameter_cv", True) else []
    if fixed_cv:
        scores={}
        for record in fixed_cv:
            scores.setdefault(record['candidate_id'],[]).append(record)
        valid={cid:np.mean([r['balanced_accuracy'] for r in part]) for cid,part in scores.items() if all(r['status']=='ok' for r in part)}
        if not valid:
            raise ValueError('Nenhuma parametrização completou todos os folds da CV fixa.')
        best_id=max(valid,key=valid.get)
        deployment_index=next(i for i,(params,_,_) in enumerate(train_sets) if parameter_id(params)==best_id)
    else:
        deployment_index,_=choose(train_sets,set(train_sets[0][2].trial),cfg)
    evaluated_trials = set(datasets[0][2].trial)
    extra_context = sorted(set(accepted) - evaluated_trials)
    for fold, (train_trials, test_trials) in enumerate(partitions):
        index, search    = choose(train_sets, train_trials, cfg) if training is None else (deployment_index, fixed_cv)
        params, Xtr, mtr = train_sets[index]
        tr               = np.flatnonzero(mtr.trial.isin(train_trials))
        model            = (calibrated(params, Xtr[tr], mtr.iloc[tr].reset_index(drop=True), cfg) if training is None
                            else final_model(train_run, train_accepted, cfg, protocol, params, train_sets[0][2].rest_source.iloc[0]))
        if training is None:
            Xte, mte = Xtr, mtr
        else:
            Xte, mte = windows(run, cfg, params, protocol, accepted,
                               rest_source=datasets[0][2].rest_source.iloc[0])
            filtered,_,size = block_windows(run,cfg,params)
            starts=np.rint((mte.start_abs.to_numpy()-run.times[0])*run.raw.info['sfreq']).astype(int)
            Xte=np.stack([filtered[:,:,first:first+size] for first in starts])
        te            = np.flatnonzero(mte.trial.isin(test_trials))
        proba         = model.predict_proba(Xte[te])
        pred          = model.classes_[proba.argmax(axis=1)]
        frame         = mte.iloc[te].copy()
        frame["pred"] = pred
        frame["fold"] = fold
        for j, cls in enumerate(model.classes_):
            frame[f"p_{cls}"] = proba[:, j]
        records.append(frame)
        context_trials = set(test_trials) | {trial for i, trial in enumerate(extra_context) if i % len(partitions) == fold}
        context, context_meta = continuous_windows(run, cfg, params, context_trials)
        if context is not None:
            context_meta["context_only"] = ~context_meta.trial.isin(test_trials)
            context_proba = model.predict_proba(context)
            context_meta["fold"] = fold
            for j, cls in enumerate(model.classes_):
                context_meta[f"p_{cls}"] = context_proba[:, j]
            timelines.append(context_meta)

        pcols      = [f"p_{cls}" for cls in model.classes_]
        phase      = frame.groupby(["trial", "target"], as_index=False)[pcols].mean()
        phase_pred = model.classes_[phase[pcols].to_numpy().argmax(axis=1)]
        folds.append(dict(fold=fold, balanced_accuracy=float(balanced_accuracy_score(phase.target, phase_pred)), params=params,
                          train_trials=sorted(train_trials), test_trials=sorted(test_trials)))
        searches.append(dict(fold=fold, candidates=search))
    predictions   = pd.concat(records, ignore_index=True)
    labels        = PROTOCOLS[protocol]
    pcols         = [f"p_{c}" for c in labels]
    phase         = predictions.groupby(["trial", "target"], as_index=False)[pcols].mean()
    phase["pred"] = np.asarray(labels)[phase[pcols].to_numpy().argmax(axis=1)]
    summary       = dict(
        protocol=protocol, labels=labels, chance=1 / len(labels),
        balanced_accuracy=float(balanced_accuracy_score(phase.target, phase.pred)),
        window_balanced_accuracy=float(balanced_accuracy_score(predictions.target, predictions.pred)),
        confusion_matrix=confusion_matrix(phase.target, phase.pred, labels=labels).tolist(),
        n_trial_phases=len(phase), n_trials=int(phase.trial.nunique()), n_windows=len(predictions),
        rest_source=predictions.rest_source.iloc[0], folds=folds, search=searches,
        invalid_candidates=invalid,
        validation="nested_trial_cv" if training is None else "prior_calibration_proxy",
        curve_time="fim da janela causal; timeline inferida somente em trials de teste; suavização apenas visual",
        _timeline=pd.concat(timelines, ignore_index=True).to_dict("records") if timelines else [],
        preprocessing="causal Butterworth ordem 4 por trial; referência conforme config; canais ruins excluídos",
    )
    # A fixed deployment model is selected using training data only.
    deployed_params = train_sets[deployment_index][0]
    source_run, source_accepted = training if training is not None else (run, accepted)
    deployed_model = model if training is not None else final_model(source_run, source_accepted, cfg, protocol, deployed_params, train_sets[0][2].rest_source.iloc[0])
    role = "online_test_prior_IM_treino" if training is not None else "in_sample_IM_treino_not_validation"
    continuous = full_block_prediction(deployed_model, run, cfg, deployed_params, role)
    summary.update(fixed_parameter_cv=fixed_cv, deployment_params=deployed_params,
                   continuous_role=role, _continuous_timeline=continuous.to_dict("records"))
    return summary, predictions, phase


def save_result(result, directory, prefix=""):
    directory = Path(directory)
    summary, predictions, phase = result
    name                        = prefix + summary["protocol"]
    summary                     = summary.copy()
    timeline                    = summary.pop("_timeline", [])
    continuous                  = summary.pop("_continuous_timeline", [])
    fixed_cv                    = summary.pop("fixed_parameter_cv", [])
    if continuous:
        pd.DataFrame(continuous).to_csv(directory / f"{name}_continuous_timeline.csv", index=False)
    if fixed_cv:
        pd.DataFrame([{**{k: v for k, v in r.items() if k != "params"}, **r["params"]} for r in fixed_cv]).to_csv(directory / f"{name}_fixed_cv.csv", index=False)
    if timeline:
        pd.DataFrame(timeline).to_csv(directory / f"{name}_timeline.csv", index=False)
    (directory / f"{name}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    predictions.to_csv(directory / f"{name}_predictions.csv", index=False)
    phase.to_csv(directory / f"{name}_trial_scores.csv", index=False)
