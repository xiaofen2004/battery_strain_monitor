from datetime import datetime, timezone
from math import ceil, floor
import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import ScaledTranslation

matplotlib.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                            "axes.unicode_minus": False, "font.size": 10})
STAGE_COLORS = {"放电": "#fbe1e1", "充电": "#dff2e1", "搁置": "#eeeeee"}


def chart_export_dpi(figure):
    """Render sharply even when the current figure is small on a high-DPI screen."""
    width, height = figure.get_size_inches()
    return max(1, min(max(600, ceil(3200 / width)), floor(10000 / max(width, height))))


def clock_text(epoch):
    # 文件是本地无时区时间，存储为 UTC 形式的数值只是为了运算。
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def display_points(x, y, max_points=16000):
    """显示压缩保留每块极值及 NaN 断点，不改变处理数据或导出值。"""
    if len(x) <= max_points:
        return x, y
    indices = [0, len(x) - 1]
    for chunk in np.array_split(np.arange(len(x)), max_points // 4):
        finite = chunk[np.isfinite(y[chunk])]
        if len(finite):
            indices.extend([chunk[0], chunk[-1], finite[np.argmin(y[finite])], finite[np.argmax(y[finite])]])
        indices.extend(chunk[~np.isfinite(y[chunk])])
    indices = np.unique(indices)
    return x[indices], y[indices]


def battery_line(t, y):
    d = np.diff(t)
    positive = d[d > 0]
    if not len(positive):
        return t, y
    breaks = np.flatnonzero(d > 5 * np.median(positive)) + 1
    return np.insert(t, breaks, t[breaks]), np.insert(y, breaks, np.nan)


def fit_bottom_margin(figure):
    """保持时间刻度和轴标题的实际留白，窗口变矮时不被工具栏裁掉。"""
    if figure.axes:
        height = figure.get_figheight()
        bottom = min(.22, max(.09, .68 / height))
        figure.subplots_adjust(bottom=bottom, top=max(bottom + .1, 1 - getattr(figure, "header_inches", .9) / height),
                               left=min(.22, max(.105, getattr(figure, "left_inches", .85) / figure.get_figwidth())))


def from_top(figure, points):
    return figure.transFigure + ScaledTranslation(0, -points / 72, figure.dpi_scale_trans)


def draw_plot(data, figure=None):
    combined = data.settings.include_battery
    split = data.settings.plot_layout == "subplots"
    strains = data.strains or {data.settings.channel: data.strain}
    count = len(strains) + int(combined) if split else 1
    fig = figure if figure is not None else Figure(figsize=(12, max(6, 2 * count + 1.5)), dpi=110)
    fig.clear()
    fig.set_facecolor("white")
    axes = list(fig.subplots(count, 1, sharex=True, squeeze=False)[:, 0])
    colors = ["#2563ad", "#108875", "#d45a3c"]
    curves = []
    ylabel = "Δ 微应变 / με" if data.settings.delta else "微应变 / με"
    unit = data.settings.time_unit
    scale = 60 if unit == "min" else 1

    def curve(axis, t, y, color, label, battery=False):
        if battery:
            t, y = battery_line(t, y)
        tx, yy = display_points(t, y)
        line, = axis.plot(tx / scale, yy, color=color, lw=1.25, label=label, zorder=3)
        curves.append(line)
        return line

    for i, (channel, values) in enumerate(strains.items()):
        axis = axes[i] if split else axes[0]
        color = colors[i]
        curve(axis, data.strain_t, values, color, channel)
        compact_ylabel = "Δε / με" if data.settings.delta else "ε / με"
        axis.set_ylabel(f"{channel}\n{compact_ylabel}" if split else ylabel, fontsize=10, labelpad=8)
        if split or len(strains) == 1:
            axis.tick_params(axis="y", colors=color)
            axis.yaxis.label.set_color(color)
        if split:
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))

    if combined:
        voltage = axes[-1] if split else axes[0].twinx()
        current = voltage.twinx() if split else axes[0].twinx()
        if not split:
            current.spines["right"].set_position(("axes", 1.13))
        for axis, values, color, label, unit in [(voltage, data.voltage, "#b67713", "电压", "V"),
                                                (current, data.current, "#8850aa", "电流", "A")]:
            curve(axis, data.battery_t, values, color, label, battery=True)
            axis.set_ylabel(f"{label} / {unit}", color=color, fontsize=10, labelpad=10)
            axis.tick_params(axis="y", colors=color)
            if split:
                axis.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))

    end = max(data.strain_t[-1] if len(data.strain_t) else 0,
              data.battery_t[-1] if len(data.battery_t) else 0) / scale
    axes[-1].set_xlim(0, max(end, 1 / scale))
    axes[-1].set_xlabel(("对齐后时间" if combined else "采集经过时间") + f" / {unit}", color="#475569", labelpad=8)
    for axis in axes:
        axis.set_facecolor("#ffffff")
        for start, stop, kind in data.spans:
            if kind in STAGE_COLORS:
                axis.axvspan(start / scale, stop / scale, color=STAGE_COLORS[kind], zorder=0)
        axis.grid(True, color="#d7e0ea", linewidth=.6, alpha=.65)
        axis.margins(y=.13)
        axis.tick_params(axis="x", labelsize=9, colors="#64748b", length=3)
        axis.tick_params(axis="y", labelsize=9, length=3)
    for axis in fig.axes:
        axis.spines["top"].set_visible(False)
        for side in ("bottom", "left", "right"):
            axis.spines[side].set_color("#cbd5e1")
            axis.spines[side].set_linewidth(.7)
    title = "应变与电参数" if combined else "应变–时间"
    subtitle = ((f"共同起点 {clock_text(data.origin)}" if data.settings.alignment == "absolute"
                 else "两文件首点对齐") if combined else f"采集起点 {clock_text(data.origin)}")
    if split:
        subtitle += "   ·   " + ("Δ 微应变" if data.settings.delta else "原始微应变")
    fig.text(.09, 1, title, va="top", fontsize=14, weight="bold", color="#20324b", transform=from_top(fig, 8))
    fig.text(.09, 1, subtitle, va="top", fontsize=9, color="#64748b", transform=from_top(fig, 27))
    # 独立一行图例，不与标题争用横向空间。
    legend = curves + ([Patch(facecolor=color, label=name) for name, color in STAGE_COLORS.items()] if combined else [])
    cols = min(len(legend), 4 if combined else 3)
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(.085, 1), bbox_transform=from_top(fig, 40), ncol=cols,
               frameon=False, fontsize=9, borderaxespad=0, columnspacing=1.8)
    right = .89 if split and combined else (.77 if combined else .965)
    fig.header_inches = (85 if combined else 66) / 72
    fig.left_inches = 1.05 if split else .85
    if split:
        fig.align_ylabels(axes)
    fig.subplots_adjust(left=.105, right=right, hspace=.20)
    fit_bottom_margin(fig)
    return fig
