"""Leitura local, tabela de controle e reconstrução auditável das fases."""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd

CHANNEL_MAP = dict(zip(
    [f"ch{i}" for i in range(1, 16)],
    "FC3 FCz C5 C3 FC4 C1 CP3 Cz C2 C4 CP1 CPz CP4 CP2 C6".split(),
))
ITALIA_CHANNELS = "Fp1 Fpz Fp2 F7 F3 Fz F4 F8 FC5 FC1 FC2 FC6 M1 T7 C3 Cz C4 T8 M2 CP5 CP1 CP2 CP6 P7 P3 Pz P4 P8 AF7 AF3 AF4 AF8 F5 F1 F2 F6 FC3 FCz FC4 C5 C1 C2 C6 CP3 CP4 P5 P1 P2 P6 FT7 FT8 TP7 TP8".split()


def project_name(path):
    parts = [p.casefold() for p in Path(path).parts]
    return "Italia" if any(p in {"projeto italia", "projeto itália"} for p in parts) else "Brasil"

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
    if base["erd_baseline_mode"] not in {"preceding_pause", "fixed", "attention"}:
        raise ValueError("erd_baseline_mode deve ser preceding_pause ou fixed.")
    if base["curve_smoothing_s"] <= 0:
        raise ValueError("curve_smoothing_s deve ser positivo.")
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
        project = project_name(signal)
        subject = next((p for p in parts if re.fullmatch(r"[A-Za-z]{2,}\d+", p)), "UNKNOWN")
        session = next((p for p in parts if re.fullmatch(r"S\d+", p, re.I)), "UNKNOWN")
        block   = next((p for p in parts if re.match(r"^(im_treino|em_treino|online)(?:$|[_ -])",p,re.I)), "UNKNOWN")
        text    = str(signal).lower()
        kind    = "IM_treino" if block.lower().startswith("im_treino") else ("EM_treino" if block.lower().startswith("em_treino") else "online_definir")
        if kind == "online_definir":
            if any(s in text for s in ["sem_feedback", "no_feedback", "nofeedback", "without_feedback"]):
                kind = "online_sem_feedback"
            elif "feedback" in text:
                kind = "online_com_feedback"
        stamp = re.search(r"(\d{8}_\d{6})$", signal.stem)
        stage = "train" if "train" in [p.lower() for p in parts] or "_train_" in signal.stem.lower() else ""
        stamp = stamp.group(1) if stamp else signal.stem.split("_")[-1]
        rows.append(dict(
            run_id=(f"Italia_{subject}_{session}_{block}_{stage}_{stamp}" if project == "Italia" else f"{subject}_{session}_{block}_{stage}_{stamp}"), include=False,
            project=project,
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
        if "project" not in old:
            old["project"]=old.signal_path.map(project_name)
        old = old.set_index("run_id")
        for column in ["include", "group", "block_type", "bad_channels", "exclude_trials", "reason", "train_run_id"]:
            if column in old:
                current[column] = current.run_id.map(old[column]).fillna(current[column])
        missing = old.loc[~old.index.isin(current.run_id)].reset_index()
        if len(missing):
            missing["available"] = False
            current              = pd.concat([current, missing], ignore_index=True)
    current["block_type"] = current.block_type.replace("online_definir", "online_com_feedback")
    current["include"] = truth(current.include)
    current            = current.sort_values(["subject", "session", "stamp"]).reset_index(drop=True)
    control.parent.mkdir(parents=True, exist_ok=True)
    save_table(current, control)
    return current


def selection(control, subjects=None, sessions=None, blocks=None, runs=None, projects=None):
    frame = pd.read_csv(control, keep_default_na=False)
    if frame.run_id.duplicated().any() or frame.run_id.astype(str).str.contains(r"[/\\]").any():
        raise ValueError("run_id deve ser único e não conter separadores de caminho.")
    if "project" not in frame:
        frame["project"] = frame.signal_path.map(project_name)
    frame["block_type"] = frame.block_type.replace("online_definir", "online_com_feedback")
    frame = frame[truth(frame.include) & frame.block_type.isin(BLOCKS)].copy()
    for column, values in [("subject", subjects), ("session", sessions), ("block_type", blocks), ("run_id", runs), ("project", projects)]:
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


def sample_clock(times, cfg):
    """Preserva as linhas EEG; ajusta somente o relógio sob hipótese de aquisição uniforme."""
    delta = np.diff(times)
    positive = delta[delta > 0]
    if not len(positive):
        raise ValueError("Sem progressão temporal para estimar a taxa de amostragem.")
    dt = float(np.median(positive))
    jitter = float(np.quantile(np.abs(delta / dt - 1), 0.99))
    drift  = float(np.max(np.abs(times - (times[0] + np.arange(len(times)) * dt))))
    irregular = bool(np.any(delta <= 0) or jitter > cfg["jitter_tolerance"]
                     or np.max(delta) > 1.5 * dt or drift > 2 * dt)
    mode = cfg["timestamp_policy"]
    if mode not in {"sample_clock", "strict"}:
        raise ValueError("timestamp_policy deve ser sample_clock ou strict.")
    if irregular and mode == "strict":
        raise ValueError("Timestamps do sinal duplicados ou fora de ordem, ou amostragem irregular (modo strict).")
    corrected = times.copy()
    if irregular:
        # Ajuste centrado evita perda de precisão em timestamps LSL grandes.
        x = np.arange(len(times), dtype=float)
        y = times - times[0]
        keep = np.ones(len(times), dtype=bool)
        for _ in range(4):
            slope, intercept = np.polyfit(x[keep], y[keep], 1)
            residual = y - (intercept + slope * x)
            center = np.median(residual)
            mad = 1.4826 * np.median(np.abs(residual - center))
            keep = np.abs(residual - center) <= max(4 * mad, 2 * abs(slope))
            if keep.sum() < 10:
                raise ValueError("Suporte insuficiente para reconstruir relógio de amostragem.")
        corrected = times[0] + intercept + slope * x
        correction = float(np.max(np.abs(corrected - times)))
        if slope <= 0 or correction > cfg["max_timestamp_correction_s"]:
            raise ValueError(f"Relógio inconsistente: ajuste máximo {correction:.4f}s. "
                             "Conferir gaps, reinício do relógio ou ordem real das amostras; não reordenar EEG automaticamente.")
        dt = float(slope)
        warnings.warn(f"Relógio EEG reconstruído sem remover/reordenar amostras: "
                      f"{sum(delta == 0)} intervalos repetidos, {sum(delta < 0)} reversões; "
                      f"ajuste máximo {correction:.6f}s. Conferir timing no relatório.", RuntimeWarning)
    timing = dict(sfreq=1 / dt, policy=mode, reconstructed=irregular,
                  duplicate_intervals=int(sum(delta == 0)), reversed_intervals=int(sum(delta < 0)),
                  max_correction_s=float(np.max(np.abs(corrected - times))),
                  jitter_p99=jitter, timestamp_drift_s=drift,
                  assumption="Ordem das linhas representa ordem de aquisição; taxa uniforme; sem perda de amostras.")
    return corrected, timing


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
    times, timing = sample_clock(times, cfg)
    sfreq = timing["sfreq"]
    jitter = timing["jitter_p99"]
    drift  = timing["timestamp_drift_s"]
    if cfg["h_freq"] >= sfreq / 2:
        raise ValueError("h_freq precisa ser menor que Nyquist.")
    if not np.isfinite(markers.lsl_time_s.to_numpy(float)).all():
        raise ValueError("Timestamp de marcador inválido.")
    markers = markers.sort_values("lsl_time_s").reset_index(drop=True)
    outside = (markers.lsl_time_s < times[0]) | (markers.lsl_time_s > times[-1])
    marker_bounds = dict(outside_count=int(outside.sum()),
                         before_s=float(max(0, times[0] - markers.lsl_time_s.min())),
                         after_s=float(max(0, markers.lsl_time_s.max() - times[-1])))
    if not markers.lsl_time_s.between(times[0], times[-1]).any():
        raise ValueError("Nenhum marcador sobrepõe o sinal; conferir relógios LSL.")
    if outside.any():
        warnings.warn("Marcadores nas bordas/fora do sinal: registrados no QC, sem deslocar ou recortar timestamps.", RuntimeWarning)
    project = str(row.get("project", project_name(row.signal_path)))
    configured = cfg.get("channel_names")
    mapping = configured or (ITALIA_CHANNELS if project == "Italia" else list(CHANNEL_MAP.values()))
    ignored_trigger = False
    if len(cols) == len(mapping) + 1 and np.all(data[-1] == 0):
        data, cols, ignored_trigger = data[:-1], cols[:-1], True
    if len(cols) > len(mapping):
        raise ValueError("Canais sem mapa: configure channel_names; canal extra só é ignorado se totalmente zerado (TRIGGER).")
    names = list(mapping[:len(cols)])
    source_sfreq, source_samples = sfreq, len(times)
    from scipy.signal import welch
    frequency,spectrum=welch(data,fs=sfreq,nperseg=min(len(times),round(4*sfreq)),axis=-1)
    total=spectrum[:,(frequency>=1)&(frequency<min(100,sfreq/2))].sum(axis=1)
    line=spectrum[:,(np.abs(frequency-50)<=1)|(np.abs(frequency-60)<=1)].sum(axis=1)/np.maximum(total,1e-30)*100
    source_line_noise=dict(zip(names,line.tolist()))
    factor = cfg.get("analysis_decimation", "auto")
    factor = (2 if sfreq >= 250 else 1) if factor == "auto" else int(factor)
    if factor not in {1, 2}:
        raise ValueError("analysis_decimation deve ser auto, 1 ou 2.")
    if cfg["h_freq"] >= sfreq / factor / 2:
        raise ValueError("Decimação deixaria h_freq acima de Nyquist.")
    if factor > 1:
        # Causal antialiasing precedes simple stride; the same signal feeds train/test.
        data = mne.filter.filter_data(data, sfreq, None, min(cfg["h_freq"], sfreq / factor * .4),
                                    method="iir", phase="forward", iir_params=dict(order=8, ftype="butter"), verbose=False)[:, ::factor]
        times, sfreq = times[::factor], sfreq / factor

    raw   = mne.io.RawArray(data, mne.create_info(names, sfreq, "eeg"), verbose=False)
    available = mne.channels.get_builtin_montages()
    montage   = "colin27_1020" if "colin27_1020" in available else "standard_1020"
    raw.set_montage(montage, on_missing="raise", verbose=False)
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
    trials["valid"] &= trials.attempt_start.between(times[0], times[-1])
    qc = dict(source_line_noise_percent=source_line_noise, project=project, source_sfreq=source_sfreq, source_samples=source_samples, decimation=factor, ignored_zero_trigger=ignored_trigger, sfreq=sfreq, n_samples=len(times), duration_s=float(times[-1] - times[0]),
              jitter_p99=jitter, timestamp_drift_s=drift, timing=timing, montage=montage, marker_bounds=marker_bounds, channels=names, bad_channels=bads,
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
