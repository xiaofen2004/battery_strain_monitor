"""Draw a shared time axis for the summary inputs."""
from datetime import datetime, timezone

import numpy as np
from matplotlib import colormaps
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

from plotting import STAGE_COLORS, battery_line, clock_text, display_points
from processing import stage_kind
from summary_data import visible_traces

TIME_AXES = ("s", "min", "h", "绝对时间", "相对时间")
PLOT_LAYOUTS = ("分开绘制", "同图叠加")
SCALES = {"s": 1, "min": 60, "h": 3600}
STEP_COLORS = {**STAGE_COLORS, "未知": "#edf1fa"}
TOF_COLORS = colormaps["tab20"].colors[::2] + colormaps["tab20"].colors[1:4:2]


def elapsed_clock(seconds):
    hours, remainder = divmod(max(0, round(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def draw_summary(figure, data, selection, time_axis, layout):
    figure.clear()
    figure.set_layout_engine("none")
    traces, selected = visible_traces(data, selection)
    if not traces:
        figure.text(.5, .5, "所选工步在共同起点后没有可绘制的数据",
                    ha="center", color="#64748b")
        return False

    scale = SCALES.get(time_axis, 1)
    lower = min(trace.time[0] for trace in traces.values()) if len(selected) != len(data.steps) else data.origin
    upper = max(trace.time[-1] for trace in traces.values())
    origin = data.origin if time_axis == "绝对时间" else lower
    origin_label = "共同起点" if origin == data.origin else "绘图起点"
    split = layout == "分开绘制"
    five_scales = not split and len(traces) == 4
    has_ultrasound = "ultrasound" in traces
    axes = list(figure.subplots(len(traces) if split else 1, 1, sharex=True, squeeze=False)[:, 0])
    curves = []
    curve_labels = []
    series_count = 0

    def append_curves(base, key, label, values, colors, offset):
        nonlocal series_count
        idx = series_count if not split else offset
        if idx == 0:
            axis = base
        else:
            axis = base.twinx()
            if idx == 2:
                axis.spines["right"].set_position(("outward", 26 if five_scales else 64))
            elif idx == 3:
                axis.yaxis.set_label_position("left")
                axis.yaxis.tick_left()
                axis.spines["left"].set_visible(True)
                axis.spines["right"].set_visible(False)
                axis.spines["left"].set_position(("outward", 26 if five_scales else 67))
            elif idx == 4:
                axis.spines["right"].set_position(("outward", 52))
        color = colors[0]
        axis.set_ylabel(label, color=color, fontsize=9)
        if five_scales:
            # Units in the legend keep the five scales readable on a narrow plot.
            axis.yaxis.label.set_visible(False)
        axis.tick_params(axis="y", colors=color, labelsize=7 if five_scales else 8, length=3)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4 if five_scales else 5 if split else 7, min_n_ticks=2))
        axis.margins(y=.14)
        time = traces[key].time
        for index, (name, series) in enumerate(values.items()):
            t, y = battery_line(time, series) if key == "battery" else (time, series)
            if traces[key].sample_mask is not None:
                sampled = traces[key].sample_mask[name]
                t, y = t[sampled], y[sampled]
            t, y = display_points(t, y)
            marker, markevery = ".", None
            if key not in ("temperature", "ultrasound"):
                finite = np.isfinite(y)
                isolated = finite & ~np.r_[False, finite[:-1]] & ~np.r_[finite[1:], False]
                markevery = np.flatnonzero(isolated).tolist()
                marker = "." if markevery else None
            line, = axis.plot((t - origin) / scale, y, label=name, color=colors[index],
                              linewidth=1.1, marker=marker, markersize=3, markevery=markevery, zorder=3)
            curves.append(line)
            curve_labels.append(f"{name} / {label.rsplit(' / ', 1)[-1]}" if five_scales else name)
        series_count += 1

    for i, (key, trace) in enumerate(traces.items()):
        base = axes[i] if split else axes[0]
        if key == "strain":
            append_curves(base, key, data.strain_label, trace.values,
                          ["#2563ad", "#108875", "#5263ae"], 0)
        elif key == "battery":
            for offset, (name, unit, color) in enumerate((("电压", "V", "#99620f"), ("电流", "A", "#8850aa"))):
                append_curves(base, key, f"{name} / {unit}", {name: trace.values[name]}, [color], offset)
        elif key == "temperature":
            append_curves(base, key, "温度 / °C", trace.values, ["#bc4225", "#e07c30", "#8f4222"], 0)
        else:
            append_curves(base, key, "相对 ToF / μs", trace.values, TOF_COLORS, 0)

    plotted_kinds = set()
    for axis in axes:
        for step in selected:
            kind = stage_kind(step.name)
            for start, stop in step.windows:
                left, right = max(lower, start), min(upper, stop)
                if right > left:
                    axis.axvspan((left - origin) / scale, (right - origin) / scale, color=STEP_COLORS[kind],
                                 linewidth=0, zorder=0)
                    plotted_kinds.add(kind)
        axis.grid(color="#d7e0ea", alpha=.65, linewidth=.6)
        axis.tick_params(axis="x", colors="#475569", labelsize=8)
    for axis in figure.axes:
        axis.spines["top"].set_visible(False)
        for side in ("left", "right", "bottom"):
            axis.spines[side].set_color("#cbd5e1")
            axis.spines[side].set_linewidth(.7)

    xaxis = axes[-1]
    xaxis.set_xlim((lower - origin) / scale, max(upper - origin, lower - origin + .001) / scale)
    width_pt = figure.get_figwidth() * 72
    bins = 6
    if has_ultrasound:
        scale_space = 132 if five_scales else 65
        bins = min(6, max(2, int((width_pt - scale_space) / (48 if time_axis == "绝对时间" else 30))))
    xaxis.xaxis.set_major_locator(MaxNLocator(nbins=bins))
    if time_axis == "绝对时间":
        xaxis.xaxis.set_major_formatter(FuncFormatter(
            lambda value, _: datetime.fromtimestamp(origin + value, timezone.utc).strftime("%Y-%m-%d\n%H:%M:%S")
        ))
        xaxis.set_xlabel("绝对时间")
    elif time_axis == "相对时间":
        xaxis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: elapsed_clock(value)))
        xaxis.set_xlabel("相对时间（时:分:秒）")
    else:
        xaxis.set_xlabel(f"与{origin_label}的时间差 / {time_axis}")

    figure.suptitle(f"{origin_label}  {clock_text(origin)}", x=.5, y=.98, fontsize=10, color="#334155")
    kinds = [kind for kind in STEP_COLORS if kind in plotted_kinds]
    legend = curves + [Patch(facecolor=STEP_COLORS[kind], label=kind) for kind in kinds]
    columns = min(5, len(legend))
    if has_ultrasound:
        columns = min(columns, max(1, int(width_pt / (90 if five_scales else 75))))
    figure.legend(handles=legend, labels=curve_labels + kinds, ncol=columns, loc="upper center",
                  bbox_to_anchor=(.5, .94), frameon=False, fontsize=7 if five_scales else 8, columnspacing=1.0)
    if has_ultrasound:
        rows = (len(legend) + columns - 1) // columns
        top = min(.89, max(.5, .94 - 12 * rows / (figure.get_figheight() * 72)))
        figure.set_layout_engine("tight", pad=.6, h_pad=.6, rect=(0, .01, 1, top))
        return True

    left = .19 if not split and series_count >= 4 else .105
    right = .82 if not split and series_count >= 3 else (.91 if "battery" in traces or series_count > 1 and not split else .965)
    figure.subplots_adjust(left=left, right=right, top=.81 if len(legend) > 5 else .86,
                           bottom=.15 if time_axis == "绝对时间" else .115, hspace=.28)
    return True
