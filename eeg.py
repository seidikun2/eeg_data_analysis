"""Leitura local, tabela de controle e reconstrução auditável das fases."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd

CHANNEL_MAP = dict(zip(
    [f"ch{i}" for i in range(1, 16)],
    "FC3 FCz C5 C3 FC4 C1 CP3 Cz C2 C4 CP1 CPz CP4 CP2 C6".split(),
))
COLORS      = {
    "LEFT": "#0072B2", "RIGHT": "#D55E00", "BOTH": "#332288",
    "REST": "#4D4D4D", "ATTENTION": "#009E73",
    "REST_STIM": "#CC79A7", "MOVE": "#E69F00",
}
BLOCKS      = ["IM_treino", "EM_treino", "online_com_feedback", "online_sem_feedback"]
PROTOCOLS   = {
    "left_right": ["LEFT", "RIGHT"],
    "rest_move": ["REST", "MOVE"],
    "attention_move": ["ATTENTION", "MOVE"],
    "reststim_move": ["REST_STIM", "MOVE"],
    "rest_left_right": ["LEFT", "RIGHT", "REST"],
}


def settings(path=None):
    base = json.loads(Path(__file__).with_name("config.json").read_text())
    if path:
        supplied = json.loads(Path(path).read_text())
        unknown  = set(supplied) - set(base)
        if unknown:
            raise ValueError(f"Parâmetros desconhecidos: {sorted(unknown)}")
        base.update(supplied)
    if not 0 < base["l_freq"] < base["h_freq"]:
        raise ValueError("Exigir 0 < l_freq < h_freq.")
    for name in ["baseline", "active"]:
        lo, hi = base[name]
        if not base["epoch_tmin"] <= lo < hi <= base["epoch_tmax"]:
            raise ValueError(f"{name} precisa estar dentro da época.")
    if base["step"] <= 0 or min(base["windows"]) <= 0 or min(base["offsets"]) < 0:
        raise ValueError("Janelas/passo positivos; offsets não negativos.")
    return base


def truth(series):
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "sim"])


def save_table(frame, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def scan(root, control):
    root = Path(root).resolve()
    rows = []
    for signal in sorted(root.rglob("*_signal_*.csv")):
        marker  = signal.with_name(signal.name.replace("_signal_", "_markers_"))
        parts   = signal.relative_to(root).parts
        subject = next((p for p in parts if re.fullmatch(r"[A-Za-z]{2}\d+", p)), "UNKNOWN")
        session = next((p for p in parts if re.fullmatch(r"S\d+", p, re.I)), "UNKNOWN")
        block   = next((p for p in parts if p.lower() in {"im_treino", "em_treino", "online"}), "UNKNOWN")
        text    = str(signal).lower()
        kind    = {"im_treino": "IM_treino", "em_treino": "EM_treino"}.get(block.lower(), "online_definir")
        if kind == "online_definir":
            if any(s in text for s in ["sem_feedback", "no_feedback", "nofeedback", "without_feedback"]):
                kind = "online_sem_feedback"
            elif "feedback" in text:
                kind = "online_com_feedback"
        stamp = re.search(r"(\d{8}_\d{6})$", signal.stem)
        stage = "train" if "train" in [p.lower() for p in parts] or "_train_" in signal.stem.lower() else ""
        stamp = stamp.group(1) if stamp else signal.stem.split("_")[-1]
        rows.append(dict(
            run_id=f"{subject}_{session}_{block}_{stage}_{stamp}", include=False,
            group="UNASSIGNED", subject=subject, session=session, block=block,
            block_type=kind, stage=stage, stamp=stamp,
            signal_path=signal.relative_to(root).as_posix(),
            marker_path=marker.relative_to(root).as_posix(), available=marker.exists(),
            bad_channels="", exclude_trials="", reason="", train_run_id="",
        ))
    if not rows:
        raise ValueError(f"Nenhum *_signal_*.csv encontrado em {root}.")
    current = pd.DataFrame(rows)
    if current.run_id.duplicated().any():
        raise ValueError("run_id duplicado: nomes/timestamps ambíguos. Corrija a organização antes de indexar.")
    control = Path(control)
    if control.exists():
        old = pd.read_csv(control, keep_default_na=False)
        if old.run_id.duplicated().any():
            raise ValueError("run_id duplicado na tabela de controle.")
        old = old.set_index("run_id")
        for column in ["include", "group", "block_type", "bad_channels", "exclude_trials", "reason", "train_run_id"]:
            if column in old:
                current[column] = current.run_id.map(old[column]).fillna(current[column])
        missing = old.loc[~old.index.isin(current.run_id)].reset_index()
        if len(missing):
            missing["available"] = False
            current              = pd.concat([current, missing], ignore_index=True)
    current["include"] = truth(current.include)
    current            = current.sort_values(["subject", "session", "stamp"]).reset_index(drop=True)
    control.parent.mkdir(parents=True, exist_ok=True)
    save_table(current, control)
    return current


def selection(control, subjects=None, sessions=None, blocks=None, runs=None):
    frame = pd.read_csv(control, keep_default_na=False)
    if frame.run_id.duplicated().any() or frame.run_id.astype(str).str.contains(r"[/\\]").any():
        raise ValueError("run_id deve ser único e não conter separadores de caminho.")
    frame = frame[truth(frame.include) & frame.block_type.isin(BLOCKS)].copy()
    for column, values in [("subject", subjects), ("session", sessions), ("block_type", blocks), ("run_id", runs)]:
        if values:
            frame = frame[frame[column].isin(values)]
    return frame.sort_values(["subject", "session", "stamp"]).reset_index(drop=True)


def trials_from_markers(markers, signal_end, duration):
    trials    = []
    attention = stimulus = np.nan
    cls       = "UNKNOWN"
    for row in markers.itertuples():
        label = str(row.label)
        time  = float(row.lsl_time_s)
        if label == "ATTENTION":
            attention, stimulus, cls = time, np.nan, "UNKNOWN"
        elif label in {"REST_STIM", "REST_MI_STIM"} or label.endswith("_MI_STIM"):
            cls      = "REST_STIM" if label in {"REST_STIM", "REST_MI_STIM"} else label.removesuffix("_MI_STIM")
            stimulus = time
        elif label == "ATTEMPT":
            trials.append(dict(trial=len(trials), trial_class=cls, attention_start=attention,
                               stimulus_start=stimulus, attempt_start=time, pause_start=np.nan,
                               pause_source="inferred"))
            attention = stimulus = np.nan
            cls       = "UNKNOWN"
        elif label in {"REST", "PAUSE", "PAUSA"} and trials and np.isnan(trials[-1]["pause_start"]):
            trials[-1].update(pause_start=time, pause_source="marker")
    if not trials:
        raise ValueError("Nenhum ATTEMPT encontrado nos marcadores.")
    frame                  = pd.DataFrame(trials)
    frame["attention_end"] = frame.stimulus_start
    frame["stimulus_end"]  = frame.attempt_start
    frame["pause_start"]   = frame.pause_start.fillna(frame.attempt_start + duration)
    frame["attempt_end"]   = frame.pause_start
    frame["pause_end"]     = frame.attention_start.shift(-1).fillna(signal_end)
    frame["valid"]         = (
        frame.trial_class.isin(["LEFT", "RIGHT", "BOTH", "REST_STIM"])
        & (frame.attention_start < frame.stimulus_start)
        & (frame.stimulus_start < frame.attempt_start)
        & (frame.attempt_start < frame.attempt_end)
        & (frame.attempt_end <= frame.pause_end)
    )
    return frame


@dataclass
class Run:
    row: pd.Series
    raw: mne.io.BaseRaw
    clean: mne.io.BaseRaw
    times: np.ndarray
    trials: pd.DataFrame
    qc: dict


def load_run(root, row, cfg):
    signal  = pd.read_csv(Path(root) / row.signal_path)
    markers = pd.read_csv(Path(root) / row.marker_path)
    if not {"lsl_time_s", "label"}.issubset(markers):
        raise ValueError("Markers requer lsl_time_s e label.")
    cols  = sorted([c for c in signal if re.fullmatch(r"ch\d+", c)], key=lambda x: int(x[2:]))
    if not cols or "lsl_time_s" not in signal:
        raise ValueError("Signal requer lsl_time_s e colunas ch1, ch2, ...")
    times = signal.lsl_time_s.to_numpy(float)
    data  = signal[cols].to_numpy(float).T * cfg["scale_to_volts"]
    if len(times) < 10 or not np.isfinite(data).all() or not np.isfinite(times).all():
        raise ValueError("Sinal curto ou valores NaN/Inf: corrigir o arquivo, sem imputação silenciosa.")
    delta = np.diff(times)
    if np.any(delta <= 0):
        raise ValueError("Timestamps do sinal duplicados ou fora de ordem.")
    dt     = float(np.median(delta))
    sfreq  = 1 / dt
    jitter = float(np.quantile(np.abs(delta / dt - 1), 0.99))
    drift  = float(np.max(np.abs(times - (times[0] + np.arange(len(times)) * dt))))
    if jitter > cfg["jitter_tolerance"] or np.max(delta) > 1.5 * dt or drift > 2 * dt:
        raise ValueError(f"Amostragem irregular: jitter p99={jitter:.3f}, gap={max(delta):.4f}s, drift={drift:.4f}s. Reamostrar explicitamente antes de analisar.")
    if cfg["h_freq"] >= sfreq / 2:
        raise ValueError("h_freq precisa ser menor que Nyquist.")
    if not np.isfinite(markers.lsl_time_s.to_numpy(float)).all():
        raise ValueError("Timestamp de marcador inválido.")
    markers = markers.sort_values("lsl_time_s").reset_index(drop=True)
    if (markers.lsl_time_s < times[0]).any() or (markers.lsl_time_s > times[-1]).any():
        raise ValueError("Marcadores fora do intervalo do sinal; conferir relógios LSL.")
    names = [CHANNEL_MAP.get(c, c.upper()) for c in cols]
    raw   = mne.io.RawArray(data, mne.create_info(names, sfreq, "eeg"), verbose=False)
    raw.set_montage("colin27_1020", on_missing="raise", verbose=False)
    bads = [v.strip() for v in str(row.get("bad_channels", "")).split(",") if v.strip()]
    if set(bads) - set(names):
        raise ValueError(f"Canais ruins desconhecidos: {set(bads) - set(names)}")
    if len(names) - len(bads) < 3:
        raise ValueError("Exigir pelo menos três canais utilizáveis.")
    raw.info["bads"] = bads
    clean            = raw.copy()
    notch            = [float(f) for f in cfg["notch"] if float(f) < sfreq / 2]
    if notch:
        clean.notch_filter(notch, picks="eeg", verbose=False)
    clean.filter(cfg["l_freq"], cfg["h_freq"], picks="eeg", verbose=False)
    if cfg["reference"] == "average":
        clean.set_eeg_reference("average", projection=False, verbose=False)
    elif cfg["reference"] != "original":
        raise ValueError("reference deve ser average ou original.")
    trials   = trials_from_markers(markers, times[-1], cfg["attempt_duration"])
    excluded = {int(v.strip()) for v in str(row.get("exclude_trials", "")).split(",") if v.strip()}
    if excluded - set(trials.trial):
        raise ValueError("exclude_trials contém IDs inexistentes (IDs começam em zero).")
    trials["manual_exclusion"] = trials.trial.isin(excluded)
    trials["valid"] &= ~trials.manual_exclusion
    trials["valid"] &= trials.attention_start >= times[0]
    qc = dict(sfreq=sfreq, n_samples=len(times), duration_s=float(times[-1] - times[0]),
              jitter_p99=jitter, timestamp_drift_s=drift, channels=names, bad_channels=bads,
              peak_to_peak_uv=dict(zip(names, np.ptp(data, axis=1) * 1e6)),
              std_uv=dict(zip(names, np.std(data, axis=1) * 1e6)),
              flat_step_fraction=dict(zip(names, np.mean(np.abs(np.diff(data, axis=1)) < 1e-12, axis=1))),
              n_trials=len(trials), invalid_or_excluded=int((~trials.valid).sum()),
              inferred_pauses=int((trials.pause_source == "inferred").sum()),
              marker_counts=markers.label.value_counts().to_dict())
    return Run(row, raw, clean, times, trials, qc)


def make_epochs(run, cfg):
    rows   = run.trials[run.trials.valid].copy()
    codes  = {name: i + 1 for i, name in enumerate(["LEFT", "RIGHT", "BOTH", "REST_STIM"])}
    events = np.array([[round((r.attempt_start - run.times[0]) * run.raw.info["sfreq"]),
                        0, codes[r.trial_class]] for r in rows.itertuples()], int)
    if not len(events):
        raise ValueError("Nenhum trial válido após controle de dados.")
    kwargs = dict(events=events, event_id={k: v for k, v in codes.items() if v in events[:, 2]},
                  tmin=cfg["epoch_tmin"], tmax=cfg["epoch_tmax"], baseline=None,
                  metadata=rows.reset_index(drop=True), preload=True, verbose=False)
    before = mne.Epochs(run.raw, **kwargs)
    after  = mne.Epochs(run.clean, **kwargs)
    after.drop_bad(reject={"eeg": cfg["reject_uv"] * 1e-6}, flat={"eeg": cfg["flat_uv"] * 1e-6}, verbose=False)
    after.event_id = {name: code for name, code in after.event_id.items() if code in after.events[:, 2]}
    accepted             = set(after.metadata.trial) if len(after) else set()
    audit                = rows[["trial", "trial_class"]].copy()
    audit["accepted"]    = audit.trial.isin(accepted)
    audit["drop_reason"] = [";".join(after.drop_log[i]) for i in range(len(rows))]
    return before, after, audit


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
