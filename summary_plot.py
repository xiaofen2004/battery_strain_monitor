"""Draw a shared time axis for the summary inputs."""
from datetime import datetime, timezone

import numpy as np
from matplotlib import colormaps
from matplotlib.artist import Artist
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.transforms import Bbox, IdentityTransform

from plotting import STAGE_COLORS, battery_line, clock_text, display_points
from processing import stage_kind
from summary_data import summary_plot_start, visible_traces

TIME_AXES = ("s", "min", "h", "绝对时间", "相对时间")
PLOT_LAYOUTS = ("分开绘制", "同图叠加")
SCALES = {"s": 1, "min": 60, "h": 3600}
STEP_COLORS = {**STAGE_COLORS, "未知": "#edf1fa"}
TOF_COLORS = colormaps["tab20"].colors[::2] + colormaps["tab20"].colors[1:4:2]
STEP_POINT_MARKERS = ("o", "s", "v", "^")
STEP_POINT_HEADINGS = ("○ 起点", "□ 终点", "▽ 最小值", "△ 最大值")


def _point_value(source, name, value):
    decimals = {"strain": 1, "temperature": 2, "ultrasound": 3}.get(source, 3 if name == "电压" else 2)
    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


class StepValueLabels(Artist):
    """Lay out small value callouts in display space, with shared room for twin axes."""

    def __init__(self, pane, labels):
        super().__init__()
        self.pane = pane
        self.labels = sorted(labels, key=lambda item: item[-1] < 2)
        self.set_in_layout(False)
        self.set_zorder(10)

    def draw(self, renderer):
        if not self.get_visible():
            return
        font = FontProperties(size=7)
        unit = renderer.points_to_pixels(1)
        bounds = self.pane.bbox.padded(-2.5 * unit)
        transform = IdentityTransform()
        placed = []
        shown = {}
        for axis, x, y, text, color, role in self.labels:
            px, py = axis.transData.transform((x, y))
            if not self.pane.bbox.contains(px, py):
                continue
            group = shown.setdefault((axis, text), [])
            if any(abs(px - other_x) < 8 * unit and abs(py - other_y) < 6 * unit for other_x, other_y in group):
                continue
            width, height, _ = renderer.get_text_width_height_descent(text, font, ismath=False)
            half_w = width / 2 + 2.5 * unit
            half_h = max(height / 2, 3.5 * unit) + 1.5 * unit
            if bounds.width < 2 * half_w or bounds.height < 2 * half_h:
                continue
            direction = -1 if role == 2 else 1
            # The same positions are reconsidered on resize, zoom and image export.
            for dx, dy in ((0, 10), (0, -10), (15, 10), (-15, 10), (15, -10), (-15, -10),
                           (0, 21), (0, -21), (26, 16), (-26, 16), (26, -16), (-26, -16),
                           (42, 13), (-42, 13), (42, -13), (-42, -13), (0, 32), (0, -32)):
                cx = min(max(px + dx * unit, bounds.x0 + half_w), bounds.x1 - half_w)
                cy = min(max(py + direction * dy * unit, bounds.y0 + half_h), bounds.y1 - half_h)
                box = Bbox.from_extents(cx - half_w, cy - half_h, cx + half_w, cy + half_h)
                if box.contains(px, py) or any(box.overlaps(other) for other in placed):
                    continue
                line = Line2D((px, cx), (py, cy), transform=transform, color=color, linewidth=.6, alpha=.65)
                line.set_figure(self.figure)
                line.draw(renderer)
                label = Text(cx, cy, text, color=color, fontproperties=font, transform=transform,
                             horizontalalignment="center", verticalalignment="center",
                             bbox={"boxstyle": "round,pad=.14", "facecolor": "white", "edgecolor": "none",
                                   "alpha": .94})
                label.set_figure(self.figure)
                label.draw(renderer)
                placed.append(box)
                group.append((px, py))
                break


def elapsed_clock(seconds):
    hours, remainder = divmod(max(0, round(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def draw_summary(figure, data, selection, time_axis, layout, *, visible=None, special_points=()):
    figure.clear()
    figure.set_layout_engine("none")
    traces, selected = visible if visible is not None else visible_traces(data, selection)
    if not traces:
        figure.text(.5, .5, "所选工步在共同起点后没有可绘制的数据",
                    ha="center", color="#64748b")
        return False

    scale = SCALES.get(time_axis, 1)
    lower = summary_plot_start(data, traces, selected)
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
    marked = {}
    labels_by_pane = {}
    for record in special_points:
        marked.setdefault((record.source, record.series), []).append(record.samples)

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
            samples = marked.get((key, name))
            if samples:
                labels = labels_by_pane.setdefault(base, [])
                for role, (marker, group) in enumerate(zip(STEP_POINT_MARKERS, np.asarray(samples).transpose(1, 0, 2))):
                    x_points, y_points = (group[:, 0] - origin) / scale, group[:, 1]
                    axis.scatter(x_points, y_points, marker=marker, s=22,
                                 facecolors="none", edgecolors=line.get_color(), linewidths=1, zorder=4)
                    labels.extend((axis, x, y, _point_value(key, name, y), line.get_color(), role)
                                  for x, y in zip(x_points, y_points))
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

    for pane, labels in labels_by_pane.items():
        figure.add_artist(StepValueLabels(pane, labels))

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
