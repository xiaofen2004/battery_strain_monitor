"""Offline summary page for completed acquisition files."""
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from plotting import clock_text
from summary_data import latest_temperature_file, load_summary
from summary_plot import PLOT_LAYOUTS, TIME_AXES, draw_summary
from ui_helpers import ScrollArea
from ui_layout import ChartToolbar


SOURCE_LABELS = {"strain": "应变", "battery": "电池充放电", "temperature": "温度"}


class SummaryPage:
    def __init__(self, parent, px, source_locations, strain_settings):
        self.parent = parent
        self.px = px
        self.source_locations = source_locations
        self.strain_settings = strain_settings
        self.files = {key: tk.StringVar(master=parent) for key in SOURCE_LABELS}
        self.included = {key: tk.BooleanVar(master=parent, value=True) for key in SOURCE_LABELS}
        self.time_axis = tk.StringVar(master=parent, value="min")
        self.layout = tk.StringVar(master=parent, value=PLOT_LAYOUTS[0])
        self.temperature_start = tk.StringVar(master=parent)
        self.status = tk.StringVar(master=parent, value="选择已保存的数据文件，读取后可筛选电池工步。")
        self.defaults = {}
        self.data = None
        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.draw_job = None
        self._build_ui()

    def _build_ui(self):
        p = self.px
        self.scroll = ScrollArea(self.parent, lambda: (p(900), p(605)))
        self.scroll.pack(fill="both", expand=True)
        self.scroll.bind_scrolling()
        outer = ttk.Frame(self.scroll.body, padding=p(12))
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        controls = ttk.Frame(outer)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, p(9)))
        controls.columnconfigure(0, weight=3)
        controls.columnconfigure(1, weight=2)
        files = ttk.Frame(controls, style="Card.TFrame", padding=p(10))
        files.grid(row=0, column=0, sticky="nsew", padx=(0, p(9)))
        files.columnconfigure(1, weight=1)
        ttk.Label(files, text="数据来源", style="Section.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, p(5))
        )
        for row, (key, name) in enumerate(SOURCE_LABELS.items(), 1):
            ttk.Checkbutton(files, text=name, variable=self.included[key]).grid(row=row, column=0, sticky="w")
            ttk.Entry(files, textvariable=self.files[key], width=18).grid(
                row=row, column=1, sticky="ew", padx=p(7), pady=p(2)
            )
            ttk.Button(files, text="浏览", command=lambda name=key: self._browse(name)).grid(row=row, column=2)

        options = ttk.Frame(controls, style="Card.TFrame", padding=p(10))
        options.grid(row=0, column=1, sticky="nsew")
        options.columnconfigure(1, weight=1)
        ttk.Label(options, text="绘图设置", style="Section.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, p(5))
        )
        for row, (label, variable, values) in enumerate((
            ("横坐标", self.time_axis, TIME_AXES),
            ("绘制方式", self.layout, PLOT_LAYOUTS),
        ), 1):
            ttk.Label(options, text=label).grid(row=row, column=0, sticky="w", padx=(0, p(8)))
            box = ttk.Combobox(options, textvariable=variable, values=values, width=13, state="readonly")
            box.grid(row=row, column=1, sticky="ew", pady=p(2))
            box.bind("<<ComboboxSelected>>", self._request_render)
        ttk.Label(options, text="温度相对 0 秒的时刻（可选）", style="Muted.TLabel").grid(
            row=3, column=0, sticky="w", padx=(0, p(8))
        )
        ttk.Entry(options, textvariable=self.temperature_start, width=19).grid(
            row=3, column=1, sticky="ew", pady=p(2)
        )
        self.read_button = ttk.Button(options, text="读取并绘图", style="Primary.TButton", command=self.read_files)
        self.read_button.grid(row=4, column=0, columnspan=2, sticky="e", pady=(p(5), 0))

        split = ttk.Panedwindow(outer, orient="horizontal")
        split.grid(row=1, column=0, sticky="nsew")
        steps = ttk.Frame(split, style="Card.TFrame", padding=p(10), width=p(465))
        chart = ttk.Frame(split, style="Card.TFrame", padding=p(8))
        split.add(steps, weight=0)
        split.add(chart, weight=1)
        steps.columnconfigure(0, weight=1)
        steps.rowconfigure(2, weight=1)
        step_header = ttk.Frame(steps, style="Card.TFrame")
        step_header.grid(row=0, column=0, sticky="ew", pady=(0, p(5)))
        ttk.Label(step_header, text="电池工步", style="Section.TLabel").pack(side="left")
        self.clear_button = ttk.Button(step_header, text="取消全选", command=self._clear_steps, state="disabled")
        self.clear_button.pack(side="right")
        self.all_button = ttk.Button(step_header, text="全选", command=self._select_steps, state="disabled")
        self.all_button.pack(side="right", padx=p(5))
        ttk.Label(steps, text="点击工步切换选中状态。", style="Muted.TLabel").grid(
            row=1, column=0, sticky="w", pady=(0, p(5))
        )

        ttk.Style(steps).configure("Summary.Treeview", rowheight=p(24))
        self.steps = ttk.Treeview(steps, columns=("number", "kind", "start", "end"),
                                  show="headings", selectmode="extended", style="Summary.Treeview")
        for key, heading, width in (("number", "序号", 41), ("kind", "工步类型", 103),
                                     ("start", "开始时间", 148), ("end", "结束时间", 148)):
            self.steps.heading(key, text=heading)
            self.steps.column(key, width=p(width), minwidth=p(width), stretch=False,
                              anchor="center" if key == "number" else "w")
        self.steps.grid(row=2, column=0, sticky="nsew")
        ybar = ttk.Scrollbar(steps, orient="vertical", command=self.steps.yview)
        xbar = ttk.Scrollbar(steps, orient="horizontal", command=self.steps.xview)
        ybar.grid(row=2, column=1, sticky="ns")
        xbar.grid(row=3, column=0, sticky="ew")
        self.steps.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        self.steps.tag_configure("before", foreground="#8191a5")
        self.steps.bind("<Button-1>", self._toggle_step)
        self.steps.bind("<<TreeviewSelect>>", self._request_render)

        chart.grid_propagate(False)
        chart.columnconfigure(0, weight=1)
        chart.rowconfigure(0, weight=1)
        self.figure = Figure(figsize=(6, 4), dpi=100, facecolor="white")
        self.figure.text(.5, .5, "读取应变、电池或温度文件后绘图", ha="center", color="#64748b")
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.toolbar = ChartToolbar(self.canvas, chart, pack_toolbar=False)
        self.toolbar.grid(row=1, column=0, sticky="ew")

        status = ttk.Label(outer, textvariable=self.status, style="Muted.TLabel", background="#edf2f7",
                           justify="left", wraplength=p(840))
        status.grid(row=2, column=0, sticky="ew", pady=(p(6), 0))

    def sync_sources(self):
        strain, battery, temperature_folder = self.source_locations()
        defaults = {"strain": strain, "battery": battery,
                    "temperature": latest_temperature_file(temperature_folder)}
        for key, variable in self.files.items():
            if variable.get() in ("", self.defaults.get(key)):
                variable.set(defaults[key])
        self.defaults = defaults

    def _browse(self, key):
        current = self.files[key].get()
        directory = (Path(current).expanduser().parent if current else
                     self.source_locations()[2] if key == "temperature" else None)
        filetypes = {
            "strain": [("应变文件", "*.xls *.xlsx *.txt *.tsv")],
            "battery": [("电池工作簿", "*.xls *.xlsx")],
            "temperature": [("温度 CSV", "*.csv")],
        }
        path = filedialog.askopenfilename(parent=self.parent, title=f"选择{SOURCE_LABELS[key]}文件",
                                          initialdir=directory, filetypes=filetypes[key] + [("所有文件", "*.*")])
        if path:
            self.files[key].set(path)
            self.included[key].set(True)

    def read_files(self):
        self.sync_sources()
        try:
            paths = {}
            for key, button in self.included.items():
                if button.get():
                    path = Path(self.files[key].get().strip()).expanduser()
                    if not path.is_file():
                        raise ValueError(f"请选择有效的{SOURCE_LABELS[key]}文件。")
                    paths[key] = path
            if not paths:
                raise ValueError("请至少勾选一类数据。")
            cfg = self.strain_settings() if "strain" in paths else None
        except (OSError, ValueError) as exc:
            self.status.set(str(exc))
            return

        self.read_button.state(["disabled"])
        self.status.set("读取采集文件…")
        threading.Thread(target=self._read, args=(paths, cfg, self.temperature_start.get().strip()),
                         daemon=True).start()

    def _read(self, paths, cfg, start):
        def progress(message):
            self.events.put(("progress", message))

        try:
            data = load_summary(paths, cfg, start, progress, self.cancel)
        except Exception as exc:
            self.events.put(("error", str(exc)))
        else:
            self.events.put(("loaded", data))

    def poll(self):
        while True:
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                return
            if kind == "progress":
                self.status.set(value)
                continue
            self.read_button.state(["!disabled"])
            if kind == "error":
                self.status.set(f"读取失败：{value}")
                continue
            self.data = value
            for item in self.steps.get_children():
                self.steps.delete(item)
            for step in value.steps:
                self.steps.insert("", "end", iid=str(step.number),
                                  values=(step.number, step.name, clock_text(step.start)[:19],
                                          clock_text(step.last)[:19]),
                                  tags=("before",) if step.last < value.origin else ())
            control_state = "normal" if value.steps else "disabled"
            self.all_button.configure(state=control_state)
            self.clear_button.configure(state=control_state)
            self._select_steps()
            self._request_render()

    def _toggle_step(self, event):
        if event.state & 0x5:  # Keep shift/control range selection.
            return None
        row = self.steps.identify_row(event.y)
        if row and self.steps.identify_region(event.x, event.y) == "cell":
            self.steps.focus(row)
            self.steps.selection_toggle(row)
            return "break"
        return None

    def _select_steps(self):
        self.steps.selection_set(self.steps.get_children())

    def _clear_steps(self):
        self.steps.selection_remove(self.steps.selection())

    def _request_render(self, _event=None):
        if self.data is not None and self.draw_job is None:
            self.draw_job = self.parent.after_idle(self._render)

    def _render(self):
        self.draw_job = None
        selected = {int(row) for row in self.steps.selection()}
        drawn = draw_summary(self.figure, self.data, selected, self.time_axis.get(), self.layout.get())
        self.toolbar.update()
        self.canvas.draw_idle()
        start = f"共同起点 {clock_text(self.data.origin)}"
        if self.data.steps:
            count = f"已选 {len(selected)} / {len(self.data.steps)} 个工步"
            start += f"  ·  {count}" if drawn else "  ·  所选工步无可绘制数据"
        if self.data.battery_clock_rebuilt:
            start += "  ·  电池绝对时间有倒退，已按首条时间＋总时间重建"
        self.status.set(start)

    def get_preferences(self):
        return {
            "paths": {key: field.get() for key, field in self.files.items()
                      if field.get() and field.get() != self.defaults.get(key)},
            "included": {key: field.get() for key, field in self.included.items()},
            "axis": self.time_axis.get(),
            "layout": self.layout.get(),
            "temperature_start": self.temperature_start.get(),
        }

    def restore_preferences(self, values):
        if not isinstance(values, dict):
            return
        for key, fields, value_type in (("paths", self.files, str), ("included", self.included, bool)):
            items = values.get(key)
            if isinstance(items, dict):
                for name, field in fields.items():
                    if isinstance(items.get(name), value_type):
                        field.set(items[name])
        if values.get("axis") in TIME_AXES:
            self.time_axis.set(values["axis"])
        if values.get("layout") in PLOT_LAYOUTS:
            self.layout.set(values["layout"])
        if isinstance(values.get("temperature_start"), str):
            self.temperature_start.set(values["temperature_start"])

    def shutdown(self):
        self.cancel.set()
        if self.draw_job is not None:
            self.parent.after_cancel(self.draw_job)
