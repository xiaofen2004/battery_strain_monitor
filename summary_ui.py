"""Offline summary page for completed acquisition files."""
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from exporting import write_summary_export
from plotting import clock_text
from summary_data import latest_temperature_file, load_summary
from summary_plot import PLOT_LAYOUTS, TIME_AXES, draw_summary, elapsed_clock
from ui_helpers import ScrollArea
from ui_layout import ChartToolbar


SOURCE_LABELS = {"strain": "应变", "battery": "电池充放电", "temperature": "温度", "ultrasound": "超声 ToF"}


class SummaryPage:
    def __init__(self, parent, px, source_locations, strain_settings, tof_selection):
        self.parent = parent
        self.px = px
        self.source_locations = source_locations
        self.strain_settings = strain_settings
        self.tof_selection = tof_selection
        self.files = {key: tk.StringVar(master=parent) for key in SOURCE_LABELS}
        self.included = {key: tk.BooleanVar(master=parent, value=key != "ultrasound") for key in SOURCE_LABELS}
        self.time_axis = tk.StringVar(master=parent, value="min")
        self.layout = tk.StringVar(master=parent, value=PLOT_LAYOUTS[0])
        self.temperature_start = tk.StringVar(master=parent)
        self.status = tk.StringVar(master=parent, value="选择已保存的数据文件或超声 MAT 文件夹，读取后可筛选电池工步。")
        self.defaults = {}
        self.data = None
        self.loaded_paths = ()
        self.checked_steps = set()
        self.has_plot = False
        self.step_icons = parent.winfo_toplevel().checkbutton_images[:2]
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
            ttk.Button(files, text="选择文件夹" if key == "ultrasound" else "浏览",
                       command=lambda name=key: self._browse(name)).grid(row=row, column=2, sticky="ew")

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
        ttk.Label(options, text="温度时间原点（可选）", style="Muted.TLabel").grid(
            row=3, column=0, sticky="w", padx=(0, p(8))
        )
        ttk.Entry(options, textvariable=self.temperature_start, width=19).grid(
            row=3, column=1, sticky="ew", pady=p(2)
        )
        ttk.Label(options, text="仅用于只有相对秒数的旧温度 CSV", style="Muted.TLabel").grid(
            row=4, column=1, sticky="w", pady=(p(2), 0)
        )
        actions = ttk.Frame(options, style="Card.TFrame")
        actions.grid(row=5, column=0, columnspan=2, sticky="e", pady=(p(5), 0))
        self.export_button = ttk.Button(actions, text="导出绘图数据", command=self.export_data, state="disabled")
        self.export_button.pack(side="left", padx=(0, p(6)))
        self.read_button = ttk.Button(actions, text="读取并绘图", style="Primary.TButton", command=self.read_files)
        self.read_button.pack(side="left")

        split = ttk.Panedwindow(outer, orient="horizontal")
        split.grid(row=1, column=0, sticky="nsew")
        steps = ttk.Frame(split, style="Card.TFrame", padding=p(10), width=p(570))
        chart = ttk.Frame(split, style="Card.TFrame", padding=p(8))
        split.add(steps, weight=0)
        split.add(chart, weight=1)
        steps.columnconfigure(0, weight=1)
        steps.rowconfigure(2, weight=1)
        step_header = ttk.Frame(steps, style="Card.TFrame")
        step_header.grid(row=0, column=0, sticky="ew", pady=(0, p(5)))
        ttk.Label(step_header, text="电池工步", style="Section.TLabel").pack(side="left")
        self.clear_button = ttk.Button(step_header, text="取消全选", command=lambda: self._set_all_steps(False), state="disabled")
        self.clear_button.pack(side="right")
        self.all_button = ttk.Button(step_header, text="全选", command=lambda: self._set_all_steps(True), state="disabled")
        self.all_button.pack(side="right", padx=p(5))
        ttk.Label(steps, text="勾选要绘制的工步。时长按完整工步计算。", style="Muted.TLabel").grid(
            row=1, column=0, sticky="w", pady=(0, p(5))
        )

        ttk.Style(steps).configure("Summary.Treeview", rowheight=p(24))
        self.steps = ttk.Treeview(steps, columns=("number", "kind", "start", "end", "duration"),
                                  show="tree headings", selectmode="browse", style="Summary.Treeview")
        self.steps.heading("#0", text="绘制")
        self.steps.column("#0", width=p(42), minwidth=p(42), stretch=False)
        for key, heading, width in (("number", "序号", 41), ("kind", "工步类型", 103),
                                     ("start", "开始时间", 148), ("end", "结束时间", 148),
                                     ("duration", "时长", 68)):
            self.steps.heading(key, text=heading)
            self.steps.column(key, width=p(width), minwidth=p(width), stretch=False,
                              anchor="center" if key in ("number", "duration") else "w")
        self.steps.grid(row=2, column=0, sticky="nsew")
        ybar = ttk.Scrollbar(steps, orient="vertical", command=self.steps.yview)
        xbar = ttk.Scrollbar(steps, orient="horizontal", command=self.steps.xview)
        ybar.grid(row=2, column=1, sticky="ns")
        xbar.grid(row=3, column=0, sticky="ew")
        self.steps.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        self.steps.tag_configure("before", foreground="#8191a5")
        self.steps.bind("<Button-1>", self._on_step_click)
        self.steps.bind("<space>", self._on_step_space)

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
        strain, battery, temperature_folder, ultrasound_folder = self.source_locations()
        defaults = {"strain": strain, "battery": battery,
                    "temperature": latest_temperature_file(temperature_folder), "ultrasound": ultrasound_folder}
        for key, variable in self.files.items():
            if variable.get() in ("", self.defaults.get(key)):
                variable.set(defaults[key])
        self.defaults = defaults

    def _browse(self, key):
        current = self.files[key].get().strip()
        if key == "ultrasound":
            directory = current or self.source_locations()[3]
            path = filedialog.askdirectory(parent=self.parent, title="选择超声 MAT 文件夹",
                                           initialdir=directory, mustexist=True)
        else:
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
                    value = self.files[key].get().strip()
                    path = Path(value).expanduser().resolve()
                    if key == "ultrasound":
                        if not value or not path.is_dir():
                            raise ValueError("请选择有效的超声 MAT 文件夹。")
                    elif not path.is_file():
                        raise ValueError(f"请选择有效的{SOURCE_LABELS[key]}文件。")
                    paths[key] = path
            if not paths:
                raise ValueError("请至少勾选一类数据。")
            cfg = self.strain_settings() if "strain" in paths else None
            selection = self.tof_selection() if "ultrasound" in paths else None
        except (OSError, ValueError) as exc:
            self.status.set(str(exc))
            return

        self.read_button.state(["disabled"])
        self._update_export_button()
        self.status.set("读取采集文件…")
        threading.Thread(target=self._read, args=(paths, cfg, self.temperature_start.get().strip(), selection),
                         daemon=True).start()

    def _read(self, paths, cfg, start, selection):
        def progress(message):
            self.events.put(("progress", message))

        try:
            data = load_summary(paths, cfg, start, progress, self.cancel, tof_selection=selection)
        except Exception as exc:
            self.events.put(("error", str(exc)))
        else:
            self.events.put(("loaded", (data, tuple(paths.values()))))

    def _update_export_button(self):
        enabled = self.has_plot and self.read_button.instate(["!disabled"])
        self.export_button.state(["!disabled" if enabled else "disabled"])

    def export_data(self):
        if self.export_button.instate(["disabled"]):
            return
        csv_type, excel_type = "CSV 文件", "Excel 工作簿"
        file_type = tk.StringVar(master=self.parent, value=csv_type)
        filename = filedialog.asksaveasfilename(
            parent=self.parent, title="导出汇总绘图数据", initialfile="汇总绘图数据",
            filetypes=[(csv_type, "*.csv"), (excel_type, "*.xlsx")], typevariable=file_type,
        )
        if not filename:
            return
        path = Path(filename)
        add_extension = not path.suffix
        if add_extension:
            path = path.with_suffix(".xlsx" if file_type.get() == excel_type else ".csv")
        try:
            resolved = path.resolve()
            inputs = {Path(field.get().strip()).expanduser().resolve()
                      for field in self.files.values() if field.get().strip()}
            if resolved in inputs or resolved in self.loaded_paths:
                raise ValueError("不能覆盖源数据文件。请选择其他文件名。")
            if add_extension and path.exists() and not messagebox.askyesno(
                "替换文件", f"是否替换已有的 {path.name}？", parent=self.parent
            ):
                return
        except (OSError, ValueError) as exc:
            self.status.set(f"导出失败：{exc}")
            return
        self.read_button.state(["disabled"])
        self._update_export_button()
        self.status.set("正在导出汇总绘图数据…")
        threading.Thread(target=self._export, args=(path, self.data, set(self.checked_steps), self.time_axis.get()),
                         daemon=True).start()

    def _export(self, path, data, selection, time_axis):
        try:
            rows = write_summary_export(path, data, selection, time_axis)
        except Exception as exc:
            self.events.put(("export_error", str(exc)))
        else:
            self.events.put(("exported", f"已导出 {rows:,} 行 · {path}"))

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
            self._update_export_button()
            if kind == "exported":
                self.status.set(value)
                continue
            if kind == "export_error":
                self.status.set(f"导出失败：{value}")
                continue
            if kind == "error":
                self.status.set(f"读取失败：{value}")
                continue
            self.data, self.loaded_paths = value
            for item in self.steps.get_children():
                self.steps.delete(item)
            for step in self.data.steps:
                self.steps.insert("", "end", iid=str(step.number),
                                  values=(step.number, step.name, clock_text(step.start)[:19],
                                          clock_text(step.last)[:19], elapsed_clock(step.last - step.start)),
                                  tags=("before",) if step.last < self.data.origin else ())
            control_state = "normal" if self.data.steps else "disabled"
            self.all_button.configure(state=control_state)
            self.clear_button.configure(state=control_state)
            self._set_all_steps(True)

    def _on_step_click(self, event):
        if self.steps.identify_region(event.x, event.y) != "tree":
            return None
        if self.steps.identify_column(event.x) != "#0":
            return None
        row = self.steps.identify_row(event.y)
        if row:
            self.steps.focus(row)
            self.steps.focus_set()
            return self._toggle_step(row)
        return None

    def _on_step_space(self, _event):
        return self._toggle_step(self.steps.focus())

    def _toggle_step(self, row):
        if row:
            number = int(row)
            checked = number not in self.checked_steps
            if checked:
                self.checked_steps.add(number)
            else:
                self.checked_steps.remove(number)
            self.steps.item(row, image=self.step_icons[checked])
            self._request_render()
        return "break"

    def _set_all_steps(self, checked):
        rows = self.steps.get_children()
        self.checked_steps = {int(row) for row in rows} if checked else set()
        for row in rows:
            self.steps.item(row, image=self.step_icons[checked])
        self._request_render()

    def _request_render(self, _event=None):
        if self.data is not None and self.draw_job is None:
            self.draw_job = self.parent.after_idle(self._render)

    def _render(self):
        self.draw_job = None
        selected = self.checked_steps
        drawn = draw_summary(self.figure, self.data, selected, self.time_axis.get(), self.layout.get())
        self.has_plot = drawn
        self._update_export_button()
        self.toolbar.update()
        self.canvas.draw_idle()
        start = f"共同起点 {clock_text(self.data.origin)}"
        if self.data.steps:
            count = f"已选 {len(selected)} / {len(self.data.steps)} 个工步"
            start += f"  ·  {count}" if drawn else "  ·  所选工步无可绘制数据"
        if self.data.strain_rate_note:
            start += f"  ·  {self.data.strain_rate_note}"
        if self.data.battery_clock_rebuilt:
            start += "  ·  电池时间倒退已分段校正，保留记录间隔"
        if self.data.ultrasound_note:
            start += f"  ·  {self.data.ultrasound_note}"
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
