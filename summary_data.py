"""Read-only summary data, aligned on the instruments' local clock."""
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data_io import BatteryReader, StrainReader, check_cancel, snapshot
from processing import infer_rate, make_plot_data
from tof_analysis import analyze_tof_files, mat_recorded_time, path_label


@dataclass
class Trace:
    time: np.ndarray
    values: dict[str, np.ndarray]
    sample_mask: dict[str, np.ndarray] | None = None


@dataclass(frozen=True)
class WorkStep:
    number: int
    name: str
    first_row: int
    past_last_row: int
    start: float
    last: float
    windows: tuple[tuple[float, float], ...]


@dataclass
class SummaryData:
    origin: float
    traces: dict[str, Trace]
    steps: list[WorkStep]
    strain_label: str
    battery_clock_rebuilt: bool = False
    strain_rate_note: str = ""
    ultrasound_note: str = ""


def _local_clock_seconds(value):
    stamp = datetime.fromisoformat(str(value).strip().replace("/", "-"))
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone()
    # Match data_io.parse_times: represent the local clock on a common numeric axis.
    return stamp.replace(tzinfo=timezone.utc).timestamp()


def _temperature_column(frame, *names):
    name = next((candidate for candidate in names if candidate in frame), None)
    return frame[name] if name is not None else pd.Series("", index=frame.index)


def read_temperature(path, start_time="", cancel=None):
    """Read the recorder CSV and retain explicit missing samples as plot gaps."""
    with snapshot(path, cancel) as copy:
        frame = pd.read_csv(copy, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    check_cancel(cancel)
    frame.columns = frame.columns.str.strip()

    values = {}
    for label, names in (
        ("CVD 温度", ("Temperature_CVD_C",)),
        ("查表温度", ("Temperature_Table_C",)),
        ("温度", ("Temperature_C", "温度(℃)", "温度(°C)", "温度")),
    ):
        column = pd.to_numeric(_temperature_column(frame, *names), errors="coerce").to_numpy(float, copy=True)
        column[~np.isfinite(column)] = np.nan
        if np.isfinite(column).any():
            values[label] = column
    if not values:
        raise ValueError("温度 CSV 没有有效温度值。请选择温度监测软件保存的记录。")

    valid = np.logical_or.reduce([np.isfinite(column) for column in values.values()])
    absolute = _temperature_column(frame, "PC_Time", "AbsoluteTime", "绝对时间").str.strip()
    scheduled = _temperature_column(frame, "ScheduledPC_Time")
    elapsed = pd.to_numeric(
        _temperature_column(frame, "Elapsed_s", "相对时间(s)", "相对时间 / s", "相对时间"),
        errors="coerce",
    ).to_numpy(float)
    timestamps = np.full(len(frame), np.nan)
    for i, text in enumerate(absolute):
        if i % 5000 == 0:
            check_cancel(cancel)
        if text:
            try:
                timestamps[i] = _local_clock_seconds(text)
            except ValueError:
                raise ValueError(f"温度 CSV 第 {i + 2} 行的绝对时间无法识别。") from None

    anchors = np.flatnonzero(np.isfinite(timestamps) & np.isfinite(elapsed))
    missing = ~np.isfinite(timestamps) & np.isfinite(elapsed)
    if missing.any() and (len(anchors) or start_time):
        if len(anchors):
            i = anchors[0]
            zero = timestamps[i] - elapsed[i]
        else:
            try:
                zero = _local_clock_seconds(start_time)
            except ValueError:
                raise ValueError("请按 YYYY-MM-DD HH:MM:SS 填写温度的相对 0 秒时刻。") from None
        # Scheduled times mark skipped attempts, not measured samples.
        needs_anchor = missing & (valid | scheduled.str.strip().eq("").to_numpy())
        timestamps[needs_anchor] = zero + elapsed[needs_anchor]
    for i in np.flatnonzero(~np.isfinite(timestamps)):
        if str(scheduled.iloc[i]).strip():
            try:
                timestamps[i] = _local_clock_seconds(scheduled.iloc[i])
            except ValueError:
                raise ValueError(f"温度 CSV 第 {i + 2} 行的计划时间无法识别。") from None

    if not np.isfinite(timestamps[valid]).all():
        if np.isfinite(elapsed[valid]).all():
            raise ValueError("温度 CSV 只有相对时间。请填写温度相对 0 秒对应的绝对时间后再读取。")
        raise ValueError("温度 CSV 缺少可对齐的时间。请选择有绝对时间或相对秒数的记录。")
    measured = timestamps[valid]
    if np.any(np.diff(measured) < 0):
        raise ValueError("温度记录的绝对时间有倒退，请核对采集时钟。")
    keep = np.isfinite(timestamps) & (timestamps >= measured[0]) & (timestamps <= measured[-1])
    indices = np.flatnonzero(keep)
    order = indices[np.argsort(timestamps[indices], kind="stable")]
    return Trace(timestamps[order], {label: column[order] for label, column in values.items()})


def work_steps(battery):
    time = battery.time
    starts = battery.step_starts
    if starts is None:
        starts = np.r_[0, np.flatnonzero(battery.stages[1:] != battery.stages[:-1]) + 1]

    diff = np.diff(time)
    positive = diff[diff > 0]
    gaps = np.flatnonzero(diff > 5 * np.median(positive)) + 1 if len(positive) else np.array([], dtype=int)
    gap_rows = set(gaps)
    steps = []
    for number, (first, end) in enumerate(zip(starts, np.r_[starts[1:], len(time)]), 1):
        inside = gaps[np.searchsorted(gaps, first, side="right"):np.searchsorted(gaps, end)]
        edges = np.r_[first, inside, end]
        windows = tuple((float(time[a]), float(time[b]) if b < len(time) and b not in gap_rows
                         else float(np.nextafter(time[b - 1], np.inf)))
                        for a, b in zip(edges[:-1], edges[1:]))
        steps.append(WorkStep(number, str(battery.stages[first]), int(first), int(end),
                              float(time[first]), float(time[end - 1]), windows))
    return steps


def read_ultrasound(folder, selection, progress, cancel):
    if selection is None:
        raise ValueError("请选择超声采集模式和通道。")
    mode, pairs = selection
    files = [path for path in Path(folder).glob("*.mat") if path.is_file()]
    if not files:
        raise ValueError("所选超声文件夹中没有 MAT 文件。")
    timestamps = {}
    for path in files:
        check_cancel(cancel)
        try:
            timestamps[path.name] = mat_recorded_time(path)
        except ValueError:
            continue
    if not timestamps:
        raise ValueError("超声 MAT 文件名中没有可识别的时间戳。")
    ordered = sorted((path for path in files if path.name in timestamps),
                     key=lambda path: (timestamps[path.name], path.name.lower()))

    def before_file(index):
        check_cancel(cancel)
        if index == 1 or index % 50 == 0:
            progress(f"计算超声 ToF · {index} / {len(ordered)}")

    paths = analyze_tof_files(ordered, mode, pairs, before_file=before_file)
    rows = {pair: records for pair, records in paths.items()
            if any(np.isfinite(row["DeltaToF_us"]) for row in records)}
    if not rows:
        labels = "、".join(map(path_label, pairs))
        raise ValueError(f"没有可绘制的超声 ToF。请确认 MAT 保存了 {mode} 模式的所选路径（{labels}）。")
    # Use file positions even when two files have the same timestamp.
    names = sorted({row["FileName"] for records in rows.values() for row in records},
                   key=lambda name: (timestamps[name], name.lower()))
    time = np.array([timestamps[name] for name in names])
    positions = {name: i for i, name in enumerate(names)}
    values, sampled = {}, {}
    valid_files = np.zeros(time.size, dtype=bool)
    for pair, records in rows.items():
        name = path_label(pair)
        indices = [positions[row["FileName"]] for row in records]
        tof = np.array([row["DeltaToF_us"] for row in records])
        values[name] = np.full(time.size, np.nan)
        values[name][indices] = tof
        sampled[name] = np.zeros(time.size, dtype=bool)
        sampled[name][indices] = True
        valid_files[indices] |= np.isfinite(tof)
    valid = np.flatnonzero(valid_files)
    span = slice(valid[0], valid[-1] + 1)
    trace = Trace(time[span], {name: series[span] for name, series in values.items()},
                  {name: mask[span] for name, mask in sampled.items()})
    note = f"{mode} 模式超声 ToF 有效 {valid.size} / {len(files)} 个 MAT"
    missing = [path_label(pair) for pair in pairs if pair not in rows]
    if missing:
        note += " · 无有效数据：" + "、".join(missing)
    return trace, note


def load_summary(paths, strain_settings=None, temperature_start="", progress=lambda _: None, cancel=None,
                 *, tof_selection=None):
    """Read checked sources and start at the latest first valid timestamp."""
    traces = {}
    steps = []
    strain_label = "应变 / με"
    strain_rate_note = ""
    battery_clock_rebuilt = False
    if "strain" in paths:
        cfg = replace(strain_settings, include_battery=False)
        reader = StrainReader(paths["strain"], cfg.selected_channels)
        absolute, raw = reader.update(progress, cancel)
        if cfg.strain_hz:
            try:
                detected, _ = infer_rate(absolute)
            except ValueError:
                pass
            else:
                if abs(cfg.strain_hz - detected) / detected > .25:
                    strain_rate_note = f"应变按文件时间戳 {detected:g} Hz 处理（原设 {cfg.strain_hz:g} Hz）"
                    cfg = replace(cfg, strain_hz=detected)
        progress("处理应变通道")
        data = make_plot_data(absolute, raw, None, cfg, cancel)
        traces["strain"] = Trace(data.strain_t + data.origin, data.strains)
        if cfg.delta:
            strain_label = "Δ 微应变 / με"
    if "battery" in paths:
        battery = BatteryReader().update(paths["battery"], "prefer_absolute", progress, cancel)
        battery_clock_rebuilt = battery.used_elapsed
        steps = work_steps(battery)
        traces["battery"] = Trace(battery.time, {"电压": battery.voltage, "电流": battery.current})
    if "temperature" in paths:
        progress("读取温度 CSV")
        traces["temperature"] = read_temperature(paths["temperature"], temperature_start, cancel)
    ultrasound_note = ""
    if "ultrasound" in paths:
        progress("扫描超声 MAT 文件夹")
        traces["ultrasound"], ultrasound_note = read_ultrasound(paths["ultrasound"], tof_selection, progress, cancel)
    check_cancel(cancel)
    if not traces:
        raise ValueError("请至少勾选一类数据。")
    origin = max(float(trace.time[0]) for trace in traces.values())
    if origin > min(float(trace.time[-1]) for trace in traces.values()):
        raise ValueError("所选数据的绝对时间没有共同区间，请核对实验日期和时间。")
    return SummaryData(origin, traces, steps, strain_label, battery_clock_rebuilt, strain_rate_note, ultrasound_note)


def latest_temperature_file(folder):
    if not folder:
        return ""
    try:
        # Prefer the monitor's own files when unrelated CSV exports share the directory.
        files = [path for path in Path(folder).expanduser().glob("*.csv") if path.is_file()]
        recorded = [path for path in files if path.name.lower().startswith("temperature_")]
        latest = max(recorded or files, key=lambda path: (path.stat().st_mtime_ns, path.name), default=None)
        return str(latest) if latest is not None else ""
    except OSError:
        return ""


def _join_contiguous(intervals):
    merged = []
    for first, last in intervals:
        if merged and first <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(last, merged[-1][1]))
        else:
            merged.append((first, last))
    return merged


def visible_traces(data, selected_numbers):
    """Use row bounds for battery and time intervals for the other instruments."""
    selected = [step for step in data.steps if step.number in selected_numbers]
    filtered = len(selected) != len(data.steps)
    windows = _join_contiguous(window for step in selected for window in step.windows) if filtered else None
    rows = _join_contiguous((step.first_row, step.past_last_row) for step in selected) if filtered else None
    output = {}
    for key, trace in data.traces.items():
        time = trace.time
        lower = np.searchsorted(time, data.origin)
        parts = [(lower, len(time))]
        if filtered:
            if key == "battery":
                parts = [(max(lower, a), b) for a, b in rows]
            else:
                parts = [(max(lower, np.searchsorted(time, a)), np.searchsorted(time, b)) for a, b in windows]
        parts = [(a, b) for a, b in parts if a < b]
        if not parts:
            continue
        if len(parts) == 1:
            indices = slice(*parts[0])
            breaks = np.array([], dtype=int)
        else:
            pieces = [np.arange(a, b) for a, b in parts]
            indices = np.concatenate(pieces)
            breaks = np.cumsum([len(piece) for piece in pieces[:-1]], dtype=int)
        values = {label: column[indices] for label, column in trace.values.items()
                  if np.isfinite(column[indices]).any()}
        if values:
            times = time[indices]
            masks = ({label: trace.sample_mask[label][indices] for label in values}
                     if trace.sample_mask is not None else None)
            if breaks.size:
                times = np.insert(times, breaks, times[breaks])
                values = {label: np.insert(column, breaks, np.nan) for label, column in values.items()}
                if masks is not None:
                    masks = {label: np.insert(mask, breaks, True) for label, mask in masks.items()}
            output[key] = Trace(times, values, masks)
    return output, selected
