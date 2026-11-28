"""运行：python app.py；无需 Excel/Office 或 MATLAB。"""
from __future__ import annotations
import json
import queue
import threading
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from data_io import StrainReader, BatteryReader
from processing import Settings, make_plot_data
from plotting import chart_export_dpi, draw_plot
from exporting import export_frame
from ui_helpers import ScrollArea, enable_high_dpi, apply_theme
from ui_layout import build_ui
from ugw_host import UGWHost
from summary_ui import SummaryPage

ROOT = Path(__file__).resolve().parent


class MonitorApp(tk.Tk):
    def __init__(self):
        enable_high_dpi()
        super().__init__()
        self.title("电池实验监测")
        screen_w, screen_h = self.winfo_screenwidth(), self.winfo_screenheight()
        self.ui_scale = self.winfo_fpixels("1i") / 96
        self.px = apply_theme(self, self.ui_scale)
        width, height = min(self.px(1500), int(screen_w * .94)), min(self.px(930), int(screen_h * .88))
        self.geometry(f"{width}x{height}+{max(0, (screen_w-width)//2)}+{self.px(20)}")
        self.minsize(min(self.px(640), int(screen_w*.6)), min(self.px(460), int(screen_h*.6)))
        self.events = queue.Queue()
        self.cancel_event = threading.Event()
        self.busy = False
        self.busy_job = None
        self.progress_text = ""
        self.reader = None
        self.battery_reader = BatteryReader()
        self.plot_data = None
        self.last_context = None
        self.last_done = 0.
        self.fields = {}
        self._ui()
        self._load_settings()
        self._mode_changed()
        self._layout_changed()
        self.narrow = False
        self.config_auto_hidden = False
        self.bind("<Configure>", self._window_resized)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.poll_job = self.after(120, self._poll)

    def var(self, name, default=""):
        self.fields[name] = tk.StringVar(value=default)
        return self.fields[name]

    def _ui(self):
        self.monitor_tabs = ttk.Notebook(self)
        self.monitor_tabs.pack(fill="both", expand=True)
        strain_tab = ttk.Frame(self.monitor_tabs)
        serial_tab = ttk.Frame(self.monitor_tabs)
        summary_tab = ttk.Frame(self.monitor_tabs)
        self.monitor_tabs.add(strain_tab, text="应变与电参数")
        self.monitor_tabs.add(serial_tab, text="温度与超声")
        self.monitor_tabs.add(summary_tab, text="汇总绘图")
        build_ui(self, strain_tab)

        ugw_area = ScrollArea(serial_tab, lambda: self.ugw.minimum_size())
        ugw_area.pack(fill="both", expand=True)
        self.ugw = UGWHost(ugw_area.body, self.px)
        ugw_area.watch(ugw_area.body)
        ugw_area.bind_scrolling()

        def resize_acquisition(event):
            reflowed = self.ugw.reflow_controls(ugw_area.winfo_width())
            if reflowed or event.widget is ugw_area.body:
                ugw_area.schedule_refresh()

        ugw_area.bind("<Configure>", resize_acquisition, add="+")
        ugw_area.body.bind("<<MonitorLayoutChanged>>", resize_acquisition)
        self.summary_page = SummaryPage(summary_tab, self.px, self._summary_locations,
                                        self._summary_strain_settings, self.ugw.tof_selection)
        self.monitor_tabs.bind("<<NotebookTabChanged>>", self._tab_changed)

    def _tab_changed(self, _event):
        if self.monitor_tabs.select() == str(self.summary_page.parent):
            self.summary_page.sync_sources()

    def _summary_locations(self):
        return (self.fields["strain_file"].get().strip(), self.fields["battery_file"].get().strip(),
                self.ugw.temp_save_dir_var.get().strip(),
                self.ugw.tof_dir.strip() or self.ugw.save_dir_var.get().strip())

    def _summary_strain_settings(self):
        return self._strain_config({key: variable.get().strip() for key, variable in self.fields.items()})

    def _browse(self, name):
        path = filedialog.askopenfilename(title="选择" + ("应变" if name == "strain_file" else "电池") + "数据",
                                          filetypes=[("采集文件", "*.xls *.xlsx *.txt *.tsv"), ("所有文件", "*.*")])
        if path:
            self.fields[name].set(path)

    def _mode_changed(self, event=None):
        combined = self.fields["plot_mode"].get() == "应变＋电压电流"
        for widget in self.battery_widgets + [self.offset_entry]:
            widget.configure(state="normal" if combined else "disabled")
        for widget in (self.alignment_box, self.battery_time_box):
            widget.configure(state="readonly" if combined else "disabled")

    def _window_resized(self, event):
        if event.widget is not self or self.narrow == (event.width < self.px(900)):
            return
        self.narrow = not self.narrow
        if self.narrow and self.config_visible:
            self.toggle_config()
            self.config_auto_hidden = True
        elif not self.narrow and self.config_auto_hidden and not self.config_visible:
            self.toggle_config()

    def toggle_config(self):
        self.config_auto_hidden = False
        if self.config_visible:
            self.sidebar.grid_remove()
            self.config_button.configure(text="显示设置")
        else:
            self.sidebar.grid()
            self.config_button.configure(text="隐藏设置")
        self.config_visible = not self.config_visible
        self.scroll_area.refresh()

    def toggle_details(self):
        if self.details_visible:
            self.log.grid_remove()
            self.details_button.configure(text="处理详情")
        else:
            self.log.grid(row=5, column=0, sticky="ew", pady=(self.px(8), 0))
            self.details_button.configure(text="收起详情")
        self.details_visible = not self.details_visible
        self.update_idletasks()
        self.scroll_area.refresh()

    def _layout_changed(self, event=None):
        layout = "subplots" if self.fields["plot_layout"].get() == "分通道子图" else "overlay"
        self.layout_hint.set("独立纵轴 · 共用时间轴" if layout == "subplots" else "共用应变纵轴")
        if self.plot_data is not None:
            self.plot_data.settings = replace(self.plot_data.settings, plot_layout=layout,
                                              time_unit=self.fields["time_unit"].get())
            self._show_summary()
            draw_plot(self.plot_data, self.figure)
            self.canvas.draw_idle()
            self.toolbar.update()
        self.scroll_area.refresh()

    def plot_rows(self):
        data = self.plot_data
        if data is None or data.settings.plot_layout != "subplots":
            return 1
        return len(data.strains or [data.settings.channel]) + int(data.settings.include_battery)

    def _show_summary(self):
        data = self.plot_data
        unit = data.settings.time_unit
        scale = 60 if unit == "min" else 1
        strain_end = data.strain_t[-1] if len(data.strain_t) else 0
        battery_end = data.battery_t[-1] if len(data.battery_t) else 0
        baselines = "；".join(f"{ch} {v:.6f}" for ch, v in data.baselines.items())
        summary = f"各通道首值 / με：{baselines}  |  应变末点 {strain_end / scale:.2f} {unit}"
        if data.settings.include_battery:
            summary += f"  |  电池末点 {battery_end / scale:.2f} {unit}"
        self.summary.set(summary)

    @staticmethod
    def _strain_config(values):
        channel = ",".join(values[k] for k in ("channel", "channel2", "channel3") if values[k] not in ("", "不选"))
        return Settings(channel=channel,
                        strain_hz=0 if values["strain_rate"] == "自动" else float(values["strain_rate"]),
                        output_hz=float(values["output_rate"]), cutoff_hz=float(values["cutoff"]),
                        median_points=int(values["median"]), delta=values["strain_display"].startswith("Δ"))

    def _settings(self):
        values = {k: v.get().strip() for k, v in self.fields.items()}
        combined = values["plot_mode"] == "应变＋电压电流"
        cfg = replace(self._strain_config(values),
                       alignment="absolute" if not combined or values["align"].startswith("实际") else "first",
                       offset_s=float(values["offset"]) if combined else 0,
                       battery_mode={"自动（优先总时间）": "auto", "表内绝对时间": "absolute", "固定 100 Hz 逐行重建": "fixed"}[values["battery_time"]] if combined else "auto",
                       include_battery=combined,
                       plot_layout="subplots" if values["plot_layout"] == "分通道子图" else "overlay",
                       time_unit=values["time_unit"])
        cfg.validate()
        for key in (("strain_file", "battery_file") if combined else ("strain_file",)):
            if not Path(values[key]).is_file():
                raise ValueError("请选择有效的" + ("应变" if key == "strain_file" else "电池") + "文件。")
        return values, cfg

    def refresh(self):
        if self.busy:
            return
        try:
            values, cfg = self._settings()
        except (ValueError, KeyError) as exc:
            self.auto.set(False)
            messagebox.showerror("检查设置", str(exc))
            return
        self._save_settings(values)
        self.busy = True
        self.cancel_event.clear()
        self.progress_text = "正在读取文件；窗口可继续操作。"
        self.busy_job = self.after(400, self._show_busy)
        threading.Thread(target=self._worker, args=(values, cfg), daemon=True).start()

    def _show_busy(self):
        self.busy_job = None
        self.cancel_button.configure(state="normal")
        self.status.set(self.progress_text)

    def _worker(self, values, cfg):
        progress = lambda text: self.events.put(("progress", text))
        try:
            path = Path(values["strain_file"]).resolve()
            if self.reader is None or self.reader.path.resolve() != path or self.reader.selected != cfg.selected_channels:
                self.reader = StrainReader(path, cfg.selected_channels)
            t, y = self.reader.update(progress, self.cancel_event)
            battery = (self.battery_reader.update(values["battery_file"], cfg.battery_mode, progress, self.cancel_event)
                       if cfg.include_battery else None)
            progress("滤波、降采样并对齐时间")
            data = make_plot_data(t, y, battery, cfg, self.cancel_event)
            if self.cancel_event.is_set():
                raise InterruptedError("已取消，保留上一张图。")
            self.events.put(("done", (data, self.reader.channels, values)))
        except Exception as exc:
            # 不复用一次失败读取中可能已经部分更新的缓存。
            self.reader = None
            self.events.put(("error", (str(exc), traceback.format_exc())))

    def _poll(self):
        import time
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "progress":
                    self.progress_text = payload
                    if self.busy_job is None:
                        self.status.set(payload)
                elif kind == "done":
                    data, channels, values = payload
                    self.plot_data, self.last_context = data, values
                    for i, box in enumerate(self.channel_boxes):
                        box.configure(values=(["不选"] if i else []) + channels)
                    self._layout_changed()
                    self.status.set("已更新 " + datetime.now().strftime("%H:%M:%S") + " · 源文件只读")
                    self._log("\n".join(data.notes))
                    self._finish()
                elif kind == "error":
                    self.status.set("未更新图像：" + payload[0])
                    self._log(payload[0] + "\n若采集软件正在保存或独占文件，请等待后再次刷新。上一张图仍为上次成功的数据。")
                    (ROOT / "last_error.log").write_text(payload[1], encoding="utf-8")
                    self._finish()
        except queue.Empty:
            pass
        if self.auto.get() and not self.busy:
            try:
                interval = float(self.fields["interval"].get())
                if not 1 <= interval <= 86400:
                    raise ValueError()
                if time.monotonic() - self.last_done >= interval:
                    self.refresh()
            except ValueError:
                self.auto.set(False)
                self.status.set("自动刷新间隔须为 1～86400 秒。")
        self.summary_page.poll()
        self.poll_job = self.after(120, self._poll)

    def _finish(self):
        import time
        self.busy = False
        self.last_done = time.monotonic()
        if self.busy_job is not None:
            self.after_cancel(self.busy_job)
            self.busy_job = None
        self.cancel_button.configure(state="disabled")

    def _log(self, text):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.insert("1.0", text)
        self.log.configure(state="disabled")

    def _safe_output(self, path):
        resolved = Path(path).resolve()
        sources = {Path(self.fields[k].get()).resolve() for k in ("strain_file", "battery_file") if self.fields[k].get()}
        if self.last_context:
            sources.update(Path(self.last_context[k]).resolve() for k in ("strain_file", "battery_file") if self.last_context.get(k))
        if resolved in sources:
            raise ValueError("不能把输出保存到源数据文件。请选择其他文件名。")

    def save_image(self):
        if self.plot_data is None:
            return messagebox.showinfo("保存图片", "请先读取并绘图。")
        file_type = tk.StringVar(self, value="高清 PNG 图片")
        filename = filedialog.asksaveasfilename(parent=self, initialfile="电池应变_电压_电流",
                                               filetypes=[("高清 PNG 图片", "*.png"), ("PDF 矢量图", "*.pdf")],
                                               typevariable=file_type)
        if not filename:
            return
        try:
            path = Path(filename)
            add_suffix = not path.suffix
            if add_suffix:
                path = path.with_suffix(".pdf" if file_type.get() == "PDF 矢量图" else ".png")
            self._safe_output(path)
            if add_suffix and path.exists() and not messagebox.askyesno("替换文件", f"是否替换已有的 {path.name}？", parent=self):
                return
            export_figure = draw_plot(self.plot_data)
            for source_axis, target_axis in zip(self.figure.axes, export_figure.axes):
                target_axis.set_xlim(source_axis.get_xlim())
                target_axis.set_ylim(source_axis.get_ylim())
            export_figure.savefig(path, dpi=chart_export_dpi(export_figure), facecolor="white")
            self.status.set(f"图片已保存：{path}")
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))

    def export_data(self):
        if self.plot_data is None:
            return messagebox.showinfo("导出数据", "请先读取并绘图。")
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="电池监测_处理数据.csv",
                                           filetypes=[("CSV 文件", "*.csv")])
        if not path:
            return
        try:
            self._safe_output(path)
            result = export_frame(self.plot_data)
            result.to_csv(path, index=False, encoding="utf-8-sig")
            self.status.set("处理数据已导出：" + path)
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))

    def _save_settings(self, values):
        settings = values | {
            "ugw_save_dir": self.ugw.save_dir_var.get().strip(),
            "temperature_save_dir": self.ugw.temp_save_dir_var.get().strip(),
            "ugw_tof_dir": self.ugw.tof_dir,
            "serial_preferences": self.ugw.get_preferences(),
            "summary_preferences": self.summary_page.get_preferences(),
        }
        try:
            (ROOT / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _load_settings(self):
        try:
            data = json.loads((ROOT / "settings.json").read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return
            for k, value in data.items():
                if k in self.fields:
                    self.fields[k].set(value)
            for key, field in (
                ("ugw_save_dir", self.ugw.save_dir_var),
                ("temperature_save_dir", self.ugw.temp_save_dir_var),
            ):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    field.set(value)
            tof_dir = data.get("ugw_tof_dir")
            if isinstance(tof_dir, str):
                self.ugw.tof_dir = tof_dir
            self.ugw.restore_preferences(data.get("serial_preferences"))
            self.summary_page.restore_preferences(data.get("summary_preferences"))
        except (OSError, ValueError):
            pass

    def _close(self):
        self._save_settings({k: v.get() for k, v in self.fields.items()})
        self.destroy()

    def destroy(self):
        self.cancel_event.set()
        self.after_cancel(self.poll_job)
        if self.busy_job is not None:
            self.after_cancel(self.busy_job)
        self.ugw.shutdown()
        self.summary_page.shutdown()
        self.update_idletasks()
        super().destroy()


if __name__ == "__main__":
    MonitorApp().mainloop()
