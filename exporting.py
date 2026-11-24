"""Export processed data at the instruments' recorded times."""
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from plotting import clock_text
from summary_data import summary_plot_start, visible_traces
from summary_plot import SCALES, elapsed_clock

EXCEL_DATA_ROWS = 1_048_575


def export_frame(data):
    frames = []
    strains = data.strains or {data.settings.channel: data.strain}
    for channel, values in strains.items():
        frames.append(pd.DataFrame({
            "series": "strain", "elapsed_s": data.strain_t, "value": values,
            "unit": "delta_microstrain" if data.settings.delta else "microstrain",
            "strain_channel": channel,
            "strain_baseline_microstrain": data.baselines.get(channel, data.baseline),
        }))
    if data.settings.include_battery:
        for series, values, unit in [("voltage", data.voltage, "V"), ("current", data.current, "A")]:
            frames.append(pd.DataFrame({"series": series, "elapsed_s": data.battery_t,
                                        "value": values, "unit": unit,
                                        "strain_channel": "", "strain_baseline_microstrain": float("nan")}))
    result = pd.concat(frames, ignore_index=True)
    result["alignment"] = data.settings.alignment if data.settings.include_battery else "strain_start"
    result["origin"] = clock_text(data.origin) if data.origin else "first_sample"
    return result


def _summary_columns(key, trace, strain_label):
    if key == "strain":
        prefix = "Δ" if strain_label.startswith("Δ") else ""
        return {f"{prefix}{name} / με": values for name, values in trace.values.items()}
    if key == "battery":
        return {"电压 / V": trace.values["电压"], "电流 / A": trace.values["电流"]}
    if key == "temperature":
        return {f"{name} / °C": values for name, values in trace.values.items()}
    return {f"{name} 相对 ToF / μs": values for name, values in trace.values.items()}


def summary_export_frame(data, selection, time_axis):
    traces, steps = visible_traces(data, selection, line_breaks=False)
    if not traces:
        raise ValueError("所选工步没有可导出的绘图数据。")
    tables = {}
    for key, trace in traces.items():
        time = trace.time
        columns = _summary_columns(key, trace, data.strain_label)
        if key == "strain":
            # Strain processing inserts a NaN at the next segment's first sample to break the line.
            samples = np.logical_or.reduce([np.isfinite(values) for values in columns.values()])
            time = time[samples]
            columns = {name: values[samples] for name, values in columns.items()}
        # Pair repeated timestamps by occurrence without multiplying rows or losing samples.
        occurrences = pd.Series(time).groupby(time, sort=False).cumcount()
        index = pd.MultiIndex.from_arrays([time, occurrences])
        tables[key] = pd.DataFrame(columns, index=index)
    result = pd.concat(tables.values(), axis=1).sort_index()
    time = result.index.get_level_values(0).to_numpy()
    elapsed = time - summary_plot_start(data, traces, steps)

    if "battery" in tables and steps:
        numbers = np.zeros(time.size, dtype=int)
        for step in steps:
            for start, stop in step.windows:
                a, b = np.searchsorted(time, (start, stop))
                numbers[a:b] = step.number
        # The battery can record the end and start of two steps at the same time.
        # Those rows belong to their recorded steps, even when the time windows touch.
        lower = np.searchsorted(data.traces["battery"].time, data.origin)
        battery_numbers = np.concatenate([
            np.full(max(0, step.past_last_row - max(lower, step.first_row)), step.number, dtype=int)
            for step in steps
        ])
        numbers[result.index.get_indexer(tables["battery"].index)] = battery_numbers
        labels = {step.number: step.name for step in steps}
        result.insert(0, "工步类型", pd.Series(numbers).map(labels).to_numpy())
        result.insert(0, "工步序号", pd.array(np.where(numbers, numbers, np.nan), dtype="Int64"))

    if time_axis in ("min", "h"):
        result.insert(0, f"相对时间 / {time_axis}", elapsed / SCALES[time_axis])
    elif time_axis == "相对时间":
        result.insert(0, "相对时间（时:分:秒）", [elapsed_clock(value) for value in elapsed])
    result.insert(0, "相对时间 / s", elapsed)
    stamps = pd.to_datetime(np.rint(time * 1e6).astype(np.int64), unit="us")
    result.insert(0, "绝对时间", stamps.strftime("%Y-%m-%d %H:%M:%S.%f"))
    return result.reset_index(drop=True)


def write_summary_export(path, data, selection, time_axis):
    """Write a complete CSV or workbook before replacing the requested output."""
    path = Path(path)
    extension = path.suffix.lower()
    if extension not in (".csv", ".xlsx"):
        raise ValueError("请选择 CSV（.csv）或 Excel（.xlsx）格式。")
    result = summary_export_frame(data, selection, time_axis)
    with NamedTemporaryFile(dir=path.parent, suffix=extension, delete=False) as file:
        temporary = Path(file.name)
    try:
        if extension == ".csv":
            result.to_csv(temporary, index=False, encoding="utf-8-sig", chunksize=10_000)
        else:
            _write_summary_workbook(temporary, result)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return len(result)


def _write_summary_workbook(path, frame):
    from openpyxl import Workbook
    from openpyxl.cell import WriteOnlyCell

    book = Workbook(write_only=True)
    for start in range(0, len(frame), EXCEL_DATA_ROWS):
        page = start // EXCEL_DATA_ROWS + 1
        sheet = book.create_sheet("汇总数据" if page == 1 else f"汇总数据 {page}")
        sheet.append(frame.columns.tolist())
        for row in frame.iloc[start:start + EXCEL_DATA_ROWS].itertuples(index=False, name=None):
            cells = []
            for value in row:
                if isinstance(value, str) and value.startswith("="):
                    cell = WriteOnlyCell(sheet, value=value)
                    cell.data_type = "s"
                    cells.append(cell)
                else:
                    cells.append(None if pd.isna(value) else value)
            sheet.append(cells)
    book.save(path)
