"""TGSP/SVM, busca aninhada por trial e probabilidades calibradas sem sobreposição."""

from __future__ import annotations

from itertools import product
import json

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
    params = [dict(window=w, offset=o, bands=b, pipeline=p, C=c, gamma=g)
              for w, o, b, p, c, g in product(cfg["windows"], cfg["offsets"], cfg["bands"],
                                            cfg["pipelines"], cfg["svm_c"], cfg["svm_gamma"])]
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
        lo = max(0, int(round((trial.attention_start - run.times[0]) * sfreq)))
        hi = min(run.raw.n_times, int(round((trial.pause_end - run.times[0]) * sfreq)))
        if hi - lo < n_samples:
            continue
        # Cada trial é filtrado isoladamente: não há mistura temporal entre folds.
        segment = run.raw.get_data(picks=picks, start=lo, stop=hi)
        for frequency in cfg["notch"]:
            if frequency < sfreq / 2:
                segment = mne.filter.notch_filter(segment, sfreq, [frequency], method="iir",
                                                  phase="forward", verbose=False)
        if cfg["reference"] == "average":
            segment = segment - segment.mean(axis=0, keepdims=True)
        filtered = np.stack([mne.filter.filter_data(
            segment, sfreq, low, high, method="iir", phase="forward",
            iir_params=dict(order=4, ftype="butter"), verbose=False,
        ) for low, high in params["bands"]])
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
    return make_pipeline(BandFeatures(params["pipeline"]), StandardScaler(),
                         SVC(C=params["C"], gamma=params["gamma"], kernel="rbf",
                             class_weight="balanced", random_state=cfg["seed"]))


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
    records, folds, searches = [], [], []
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
    for fold, (train_trials, test_trials) in enumerate(partitions):
        index, search    = choose(train_sets, train_trials, cfg)
        params, Xtr, mtr = train_sets[index]
        tr               = np.flatnonzero(mtr.trial.isin(train_trials))
        model            = calibrated(params, Xtr[tr], mtr.iloc[tr].reset_index(drop=True), cfg)
        if training is None:
            Xte, mte = Xtr, mtr
        else:
            Xte, mte = windows(run, cfg, params, protocol, accepted,
                               rest_source=datasets[0][2].rest_source.iloc[0])
        te            = np.flatnonzero(mte.trial.isin(test_trials))
        proba         = model.predict_proba(Xte[te])
        pred          = model.classes_[proba.argmax(axis=1)]
        frame         = mte.iloc[te].copy()
        frame["pred"] = pred
        frame["fold"] = fold
        for j, cls in enumerate(model.classes_):
            frame[f"p_{cls}"] = proba[:, j]
        records.append(frame)
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
        curve_time="fim da janela causal; sem suavização/interpolação entre fases",
        preprocessing="causal Butterworth ordem 4 por trial; referência conforme config; canais ruins excluídos",
    )
    return summary, predictions, phase


def save_result(result, directory):
    summary, predictions, phase = result
    name                        = summary["protocol"]
    (directory / f"{name}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    predictions.to_csv(directory / f"{name}_predictions.csv", index=False)
    phase.to_csv(directory / f"{name}_trial_scores.csv", index=False)
