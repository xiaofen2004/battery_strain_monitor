"""左侧设置、右侧图表的自适应桌面布局。"""
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from ui_helpers import ScrollArea, ScrollPanel
from plotting import fit_bottom_margin

TOOL_TIPS = {"Home": "回到完整视图", "Back": "上一个视图", "Forward": "下一个视图",
             "Pan": "平移：左键拖动平移，右键拖动缩放坐标轴", "Zoom": "缩放：左键框选放大，右键框选缩小",
             "Subplots": "调整边距与子图间距", "Save": "按当前屏幕显示保存图像"}


class ChartToolbar(NavigationToolbar2Tk):
    toolitems = [(name, TOOL_TIPS.get(name, tip), image, method)
                 for name, tip, image, method in NavigationToolbar2Tk.toolitems]


def build_ui(app, parent):
    p = app.px
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(0, weight=1)
    def min_size():
        rows = app.plot_rows()
        plot = p(220) if rows == 1 else rows * p(120)
        controls = app.right_panel.winfo_reqheight() if hasattr(app, "right_panel") else p(260)
        return p(600) + (p(342) if app.config_visible else 0), controls + p(24) + p(200) + plot

    app.scroll_area = ScrollArea(parent, min_size)
    app.scroll_area.grid(row=0, column=0, sticky="nsew")
    app.scroll_area.body.columnconfigure(0, weight=1)
    app.scroll_area.body.rowconfigure(0, weight=1)
    app.content = ttk.Frame(app.scroll_area.body, padding=p(12))
    app.content.grid(row=0, column=0, sticky="nsew")
    app.content.columnconfigure(1, weight=1)
    app.content.rowconfigure(0, weight=1)
    app.sidebar = ttk.Frame(app.content, style="Card.TFrame", width=p(330))
    app.sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, p(12)))
    app.sidebar.grid_propagate(False)
    app.sidebar.columnconfigure(0, weight=1)
    app.sidebar.rowconfigure(1, weight=1)
    ttk.Label(app.sidebar, text="采集设置", style="Section.TLabel", padding=p(16)).grid(row=0, column=0, sticky="ew")
    tabs = ttk.Notebook(app.sidebar)
    tabs.grid(row=1, column=0, sticky="nsew")
    first, second = ScrollPanel(tabs, p), ScrollPanel(tabs, p)
    tabs.add(first, text="数据与通道")
    tabs.add(second, text="滤波与时间")
    data, process = first.body, second.body

    def label(parent, text):
        ttk.Label(parent, text=text).pack(anchor="w", pady=(p(12), p(5)))

    def control(parent, text, key, default, values=None, editable=False):
        label(parent, text)
        var = app.var(key, default)
        if values:
            widget = ttk.Combobox(parent, textvariable=var, values=values,
                                  state="normal" if editable else "readonly", width=18)
        else:
            widget = ttk.Entry(parent, textvariable=var, width=18)
        widget.pack(fill="x")
        return widget

    mode = control(data, "绘图内容", "plot_mode", "应变＋电压电流", ["应变＋电压电流", "仅应变–时间"])
    mode.bind("<<ComboboxSelected>>", app._mode_changed)
    app.battery_widgets = []
    for key, title in (("strain_file", "应变文件"), ("battery_file", "电池文件")):
        label(data, title)
        row = ttk.Frame(data, style="Card.TFrame")
        row.pack(fill="x")
        row.columnconfigure(0, weight=1)
        entry = ttk.Entry(row, textvariable=app.var(key), width=12)
        entry.grid(row=0, column=0, sticky="ew", padx=(0, p(5)))
        button = ttk.Button(row, text="浏览", command=lambda name=key: app._browse(name))
        button.grid(row=0, column=1)
        if key == "battery_file":
            app.battery_widgets.extend([entry, button])
    label(data, "应变通道  ·  最多 3 个")
    channels = ttk.Frame(data, style="Card.TFrame")
    channels.pack(fill="x")
    app.channel_boxes = []
    for i, key in enumerate(("channel", "channel2", "channel3")):
        channels.columnconfigure(i, weight=1, uniform="channel")
        box = ttk.Combobox(channels, textvariable=app.var(key, "CH05" if i == 0 else "不选"),
                          values=(["不选"] if i else []) + [f"CH{n:02d}" for n in range(1, 21)], width=5)
        box.grid(row=0, column=i, sticky="ew", padx=(0, p(4) if i < 2 else 0))
        app.channel_boxes.append(box)
    control(data, "应变数值", "strain_display", "Δ 微应变（减首值）", ["Δ 微应变（减首值）", "原始微应变（保留初值）"])
    ttk.Label(data, text="Δ 微应变：每个通道分别减去自己的初值。\n不同量程可在图表上方切换为分通道子图。",
              style="Muted.TLabel", wraplength=p(270), justify="left").pack(anchor="w", pady=p(14))
    ttk.Separator(data).pack(fill="x", pady=p(4))
    ttk.Label(data, text="修改文件、通道或滤波参数后，\n点击「读取并绘图」应用。",
              style="Muted.TLabel", justify="left").pack(anchor="w", pady=p(12))

    control(process, "输入采样率 / Hz", "strain_rate", "自动", ["自动", "100", "1"], editable=True)
    control(process, "输出采样率 / Hz", "output_rate", "1")
    control(process, "低通截止 / Hz  ·  0 关闭平滑", "cutoff", "0.2")
    control(process, "中值窗口 / 点  ·  奇数，1 关闭", "median", "5")
    ttk.Label(process, text="降采样前保留抗混叠处理。\n因果滤波会引入延迟。", style="Muted.TLabel").pack(anchor="w", pady=p(12))
    ttk.Separator(process).pack(fill="x", pady=p(4))
    app.alignment_box = control(process, "联合绘图的时间对齐", "align", "实际时间，共同起点", ["实际时间，共同起点", "两文件首点对齐"])
    app.offset_entry = control(process, "应变时间偏移 / s", "offset", "0")
    app.battery_time_box = control(process, "电池时间依据", "battery_time", "自动（优先总时间）",
                                   ["自动（优先总时间）", "表内绝对时间", "固定 100 Hz 逐行重建"])
    first.bind_scrolling()
    second.bind_scrolling()
    app.scroll_area.bind_scrolling()

    right = app.right_panel = ttk.Frame(app.content, style="Card.TFrame", padding=p(14))
    right.grid(row=0, column=1, sticky="nsew")
    right.columnconfigure(0, weight=1)
    right.rowconfigure(3, weight=1)
    display = ttk.Frame(right, style="Card.TFrame")
    display.grid(row=0, column=0, sticky="ew", pady=(0, p(12)))
    app.config_visible = True
    app.config_button = ttk.Button(display, text="隐藏设置", command=app.toggle_config)
    app.config_button.pack(side="left", padx=(0, p(14)))
    ttk.Label(display, text="通道布局").pack(side="left", padx=(0, p(8)))
    layout = ttk.Combobox(display, textvariable=app.var("plot_layout", "同图叠加"),
                          values=["同图叠加", "分通道子图"], state="readonly", width=12)
    layout.pack(side="left")
    layout.bind("<<ComboboxSelected>>", app._layout_changed)
    app.layout_hint = tk.StringVar(value="共用应变纵轴")
    ttk.Label(display, textvariable=app.layout_hint, style="Muted.TLabel").pack(side="left", padx=p(10))
    ttk.Label(display, text="时间单位").pack(side="left", padx=(p(8), p(8)))
    unit = ttk.Combobox(display, textvariable=app.var("time_unit", "min"), values=["min", "s"], state="readonly", width=4)
    unit.pack(side="left")
    unit.bind("<<ComboboxSelected>>", app._layout_changed)

    actions = ttk.Frame(right, style="Card.TFrame")
    actions.grid(row=1, column=0, sticky="ew", pady=(0, p(12)))
    actions.columnconfigure(1, weight=1)
    run = ttk.Frame(actions, style="Card.TFrame")
    run.grid(row=0, column=0, sticky="w")
    app.refresh_button = ttk.Button(run, text="读取并绘图", style="Primary.TButton", command=app.refresh)
    app.refresh_button.pack(side="left")
    app.cancel_button = ttk.Button(run, text="取消", command=app.cancel_event.set, state="disabled")
    app.cancel_button.pack(side="left", padx=p(5))
    app.auto = tk.BooleanVar(value=False)
    ttk.Checkbutton(run, text="自动刷新", variable=app.auto).pack(side="left", padx=(p(6), p(4)))
    ttk.Entry(run, textvariable=app.var("interval", "10"), width=4).pack(side="left")
    ttk.Label(run, text="秒").pack(side="left", padx=p(5))
    outputs = ttk.Frame(actions, style="Card.TFrame")
    outputs.grid(row=0, column=2, sticky="e")
    ttk.Button(outputs, text="保存图片", command=app.save_image).pack(side="left", padx=(0, p(5)))
    ttk.Button(outputs, text="导出 CSV", command=app.export_data).pack(side="left")

    app.summary = tk.StringVar(value="选择应变文件和通道，开始查看采集数据。")
    summary = ttk.Label(right, textvariable=app.summary, style="Muted.TLabel", padding=(p(5), p(7)))
    summary.grid(row=2, column=0, sticky="ew")
    chart = ttk.Frame(right, style="Card.TFrame")
    chart.grid(row=3, column=0, sticky="nsew")
    chart.grid_propagate(False)
    chart.columnconfigure(0, weight=1)
    chart.rowconfigure(0, weight=1)
    app.figure = Figure(figsize=(6, 4), dpi=100, facecolor="white")
    app.figure.text(.5, .55, "准备查看采集曲线", ha="center", fontsize=18, color="#334e70", weight="bold")
    app.figure.text(.5, .45, "选择左侧文件与通道，然后点击“读取并绘图”", ha="center", fontsize=10, color="#8292a6")
    app.canvas = FigureCanvasTkAgg(app.figure, master=chart)
    app.canvas.mpl_connect("resize_event", lambda event: fit_bottom_margin(app.figure))
    app.canvas.get_tk_widget().configure(highlightthickness=0, borderwidth=0)
    app.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
    app.toolbar = ChartToolbar(app.canvas, chart, pack_toolbar=False)
    app.toolbar.configure(background="white")
    app.toolbar.grid(row=1, column=0, sticky="ew")
    for widget in app.toolbar.winfo_children():
        for option, value in (("background", "white"), ("activebackground", "#e2eaf5"),
                              ("highlightthickness", 0), ("relief", "flat"), ("overrelief", "flat"), ("offrelief", "flat")):
            try:
                widget.configure(**{option: value})
            except tk.TclError:
                pass

    footer = ttk.Frame(right, style="Card.TFrame")
    footer.grid(row=4, column=0, sticky="ew", pady=(p(8), 0))
    footer.columnconfigure(0, weight=1)
    app.status = tk.StringVar(value="就绪 · 仅读取磁盘上的完整记录")
    status = ttk.Label(footer, textvariable=app.status, style="Muted.TLabel")
    status.grid(row=0, column=0, sticky="ew", padx=(0, p(8)))
    app.details_button = ttk.Button(footer, text="处理详情", command=app.toggle_details)
    app.details_button.grid(row=0, column=1)
    app.log = tk.Text(right, height=5, wrap="word", relief="flat", bg="#f1f5fa", fg="#425570",
                      borderwidth=0, padx=p(10), pady=p(8), state="disabled", font=("Microsoft YaHei UI", 9))
    app.details_visible = False
    wrapped = [False]

    def reflow(event):
        summary.configure(wraplength=max(p(200), event.width - p(40)))
        status.configure(wraplength=max(p(200), event.width - p(150)))
        wrap = event.width - p(28) < run.winfo_reqwidth() + outputs.winfo_reqwidth() + p(16)
        if wrap != wrapped[0]:
            wrapped[0] = wrap
            if wrap:
                outputs.grid(row=1, column=0, columnspan=3, sticky="w", pady=(p(8), 0))
            else:
                outputs.grid(row=0, column=2, columnspan=1, sticky="e", pady=0)
    right.bind("<Configure>", reflow)
    app.scroll_area.watch(display, actions, summary, footer, app.log)
