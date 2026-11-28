"""窗口控件与读取流程检查；不保存用户配置，不占用实际采集设备。"""
import json
import tempfile
import time
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, call, patch

import app as monitor
from app import MonitorApp


class AppModeTests(unittest.TestCase):
    def test_chart_save_controls_use_high_resolution_and_keep_pdf(self):
        import matplotlib
        import numpy as np
        from PIL import Image
        from plotting import draw_plot
        from processing import Settings, make_plot_data

        with tempfile.TemporaryDirectory() as folder, patch.object(MonitorApp, "_load_settings"):
            app = MonitorApp()
            app.withdraw()
            try:
                root = Path(folder)
                cfg = Settings(strain_hz=1, output_hz=1, cutoff_hz=0, median_points=1, include_battery=False)
                app.plot_data = make_plot_data(np.arange(7.), np.arange(7.) ** 2, None, cfg)
                draw_plot(app.plot_data, app.figure)
                app.figure.axes[0].set_xlim(.02, .07)
                main_png = root / "应变.png"
                with patch("tkinter.filedialog.asksaveasfilename", return_value=str(main_png)):
                    app.save_image()
                with Image.open(main_png) as saved:
                    self.assertEqual(saved.size, (7200, 3600))
                    self.assertAlmostEqual(saved.info["dpi"][0], 600, places=1)
                self.assertEqual(app.figure.axes[0].get_xlim(), (.02, .07))

                def choose_pdf(**kwargs):
                    kwargs["typevariable"].set("PDF 矢量图")
                    return str(root / "矢量")

                with patch("tkinter.filedialog.asksaveasfilename", side_effect=choose_pdf):
                    app.save_image()
                self.assertEqual((root / "矢量.pdf").read_bytes()[:5], b"%PDF-")

                summary = app.summary_page
                summary.figure.clear()
                summary.figure.set_size_inches(3, 2, forward=False)
                summary.figure.add_subplot().plot([0, 1], [1, 2])
                with matplotlib.rc_context():
                    initial_dpi = matplotlib.rcParams["savefig.dpi"]
                    for extension in ("png", "pdf"):
                        output = root / f"汇总.{extension}"
                        with patch("tkinter.filedialog.asksaveasfilename", return_value=str(output)):
                            self.assertEqual(summary.toolbar.save_figure(), str(output))
                    self.assertEqual(matplotlib.rcParams["savefig.dpi"], initial_dpi)
                    self.assertEqual(Path(matplotlib.rcParams["savefig.directory"]), root)
                with Image.open(root / "汇总.png") as saved:
                    self.assertGreaterEqual(saved.width, 3200)
                    self.assertGreaterEqual(saved.info["dpi"][0], 600)
                self.assertEqual((root / "汇总.pdf").read_bytes()[:5], b"%PDF-")
            finally:
                app.destroy()

    def test_strain_only_does_not_require_or_read_battery(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(MonitorApp, "_load_settings"):
            path = Path(folder) / "strain.xls"
            path.write_text("\tCH05-\tCH08-\tCH17-\n"
                            "26/11/10 18:17:58.000\t728\t850\t-9000\n"
                            "26/11/10 18:17:59.000\t729\t852\t-8997\n"
                            "26/11/10 18:18:00.000\t730\t854\t-8994\n", encoding="ascii")
            app = MonitorApp()
            app.withdraw()
            try:
                for key, value in {"plot_mode": "仅应变–时间", "strain_file": str(path),
                                   "battery_file": "nonexistent.xlsx", "channel2": "8", "channel3": "17",
                                   "offset": "unused", "strain_rate": "1", "cutoff": "0", "median": "1"}.items():
                    app.fields[key].set(value)
                app._mode_changed()
                self.assertEqual(str(app.battery_widgets[0]["state"]), "disabled")
                values, cfg = app._settings()
                with patch.object(app.battery_reader, "update", side_effect=AssertionError("must not read battery")):
                    app._worker(values, cfg)
                events = []
                while not app.events.empty():
                    events.append(app.events.get_nowait())
                self.assertFalse([e for e in events if e[0] == "error"], events)
                data = [e[1][0] for e in events if e[0] == "done"][0]
                self.assertEqual(list(data.strains), ["CH05", "CH08", "CH17"])
                app.plot_data = data
                app.fields["plot_layout"].set("分通道子图")
                with patch.object(app.reader, "update", side_effect=AssertionError("layout must not read file")):
                    app._layout_changed()
                self.assertEqual(len(app.figure.axes), 3)
                self.assertEqual(app.plot_data.settings.plot_layout, "subplots")
                app.fields["plot_layout"].set("同图叠加")
                app._layout_changed()
                self.assertEqual(len(app.figure.axes), 1)
                app.toggle_config()
                self.assertFalse(app.config_visible)
                app.toggle_config()
                self.assertTrue(app.config_visible)
                app.fields["plot_mode"].set("应变＋电压电流")
                app.fields["offset"].set("0")
                app._mode_changed()
                self.assertEqual(str(app.battery_widgets[0]["state"]), "normal")
                with self.assertRaises(ValueError):
                    app._settings()
            finally:
                app.destroy()

    def test_visible_window_shows_content(self):
        with patch.object(MonitorApp, "_load_settings"):
            app = MonitorApp()
            try:
                app.update()
                app.after(500, app.quit)
                app.mainloop()
                canvas = app.scroll_area.canvas
                self.assertGreater(canvas.winfo_width(), app.winfo_width() // 2)
                self.assertGreater(canvas.winfo_height(), app.winfo_height() // 2)
            finally:
                app.destroy()


    def test_ugw_power_buttons_are_visible_and_use_original_serial_commands(self):
        link = Mock(is_open=True)
        link.write.side_effect = len
        ports = [SimpleNamespace(device="COM3")]
        with (patch.object(MonitorApp, "_load_settings"),
              patch("ugw_host.list_ports.comports", return_value=ports),
              patch("ugw_host.serial.Serial", return_value=link)):
            app = MonitorApp()
            host = app.ugw
            try:
                app.monitor_tabs.select(1)
                app.update()
                self.assertTrue(host.power_on_button.winfo_ismapped())
                self.assertTrue(host.power_off_button.winfo_ismapped())
                self.assertLess(host.power_panel.winfo_y(), host.control_grid.winfo_y())
                self.assertTrue(host.power_on_button.instate(["disabled"]))
                self.assertIn("连接 STM32", host.power_status_var.get())

                host.connect_button.invoke()
                host.auto_var.set(False)
                host.update_auto()
                self.assertTrue(host.temp_once_button.instate(["!disabled"]))
                self.assertTrue(host.once_button.instate(["disabled"]))
                host.temp_auto_running = True
                host.update_buttons()
                self.assertTrue(host.power_on_button.instate(["disabled"]))
                self.assertIn("先停止温度定时采集", host.power_status_var.get())
                host.power_on_button.invoke()
                link.write.assert_not_called()
                host.temp_auto_running = False
                host.update_buttons()

                def wait_for_power():
                    deadline = time.monotonic() + 3

                    def check():
                        if host.power_busy and time.monotonic() < deadline:
                            app.after(20, check)
                        else:
                            app.quit()

                    app.after(20, check)
                    app.mainloop()
                    self.assertFalse(host.power_busy)

                host.power_on_button.invoke()
                wait_for_power()
                self.assertIs(host.power_enabled, True)
                self.assertIn("已发送 v", host.power_status_var.get())
                self.assertTrue(host.once_button.instate(["!disabled"]))
                with patch("ugw_host.messagebox.askyesno", return_value=False):
                    host.power_off_button.invoke()
                link.write.assert_called_once_with(b"v")
                with patch("ugw_host.messagebox.askyesno", return_value=True):
                    host.power_off_button.invoke()
                    wait_for_power()
                self.assertEqual(link.write.call_args_list, [call(b"v"), call(b"0")])
                self.assertIs(host.power_enabled, False)
                self.assertTrue(host.once_button.instate(["disabled"]))
                self.assertTrue(host.temp_once_button.instate(["!disabled"]))

                link.write.side_effect = lambda _: 0
                with patch("ugw_host.messagebox.showerror") as error:
                    host.power_on_button.invoke()
                    wait_for_power()
                self.assertIsNone(host.power_enabled)
                error.assert_called_once_with("电源控制错误", "串口未完整发送电源指令。")
                self.assertTrue(host.once_button.instate(["disabled"]))
            finally:
                app.destroy()

    def test_serial_monitor_restores_settings_and_independent_output_directories(self):
        ports = [SimpleNamespace(device="COM3"), SimpleNamespace(device="COM5")]
        with (tempfile.TemporaryDirectory() as folder, patch.object(monitor, "ROOT", Path(folder)),
              patch("ugw_host.list_ports.comports", return_value=ports)):
            root = Path(folder)
            temp_dir, ugw_dir, tof_dir = (root / name for name in ("温度记录", "超声采集", "历史超声"))
            summary_dir = root / "汇总超声"
            for directory in (temp_dir, ugw_dir, tof_dir, summary_dir):
                directory.mkdir()
            (root / "settings.json").write_text(
                json.dumps({"channel": "CH17"}, ensure_ascii=False), encoding="utf-8"
            )
            app = MonitorApp()
            app.withdraw()
            try:
                self.assertEqual(app.fields["channel"].get(), "CH17")
                self.assertEqual(len(app.monitor_tabs.tabs()), 3)
                host = app.ugw
                self.assertEqual(float(host.temp_interval_var.get()), 10)
                self.assertEqual(float(host.interval_var.get()), 15)
                host.save_dir_var.set(str(ugw_dir))
                host.temp_save_dir_var.set(str(temp_dir))
                host.tof_dir = str(tof_dir)
                page = app.summary_page
                page.sync_sources()
                self.assertFalse(page.included["ultrasound"].get())
                self.assertEqual(page.files["ultrasound"].get(), str(tof_dir))
                with patch("summary_ui.filedialog.askdirectory", return_value=str(summary_dir)) as dialog:
                    page._browse("ultrasound")
                    self.assertEqual(dialog.call_args.kwargs["initialdir"], str(tof_dir))
                self.assertTrue(page.included["ultrasound"].get())
                host.port_var.set("COM5")
                host.baud_var.set("921600")
                host.mode_var.set("R")
                host.channel_vars[2].set(False)
                host.r_pair_vars[2, 4].set(True)
                self.assertIn((2, 4), page.tof_selection()[1])
                self.assertEqual(page.tof_selection()[0], "R")
                host.mode_var.set("M")
                self.assertEqual(page.tof_selection(), ("M", ((1, 4), (3, 4))))
                host.mode_var.set("R")
                host.interval_var.set("21.5")
                host.temp_interval_var.set("13.25")
                host.auto_var.set(True)
                host.temp_save_var.set(True)
                host.plot_voltage_var.set(True)
                host.time_axis_var.set(True)
                host.display_fs_var.set("4000000")
                host.timeout_var.set("4.5")
                saved_preferences = host.get_preferences()
            except Exception:
                app.destroy()
                raise
            app._close()

            app = MonitorApp()
            app.withdraw()
            try:
                self.assertEqual(app.fields["channel"].get(), "CH17")
                self.assertEqual(app.ugw.save_dir_var.get(), str(ugw_dir))
                self.assertEqual(app.ugw.temp_save_dir_var.get(), str(temp_dir))
                app.summary_page.sync_sources()
                self.assertTrue(app.summary_page.included["ultrasound"].get())
                self.assertEqual(app.summary_page.files["ultrasound"].get(), str(summary_dir))
                app.monitor_tabs.select(1)
                app.update_idletasks()
                host = app.ugw
                self.assertEqual(host.get_preferences(), saved_preferences)
                self.assertEqual(host.r_options.winfo_manager(), "pack")
                self.assertTrue(host.save_check.instate(("disabled",)))
                self.assertIsNone(host.ser)
                self.assertIsNone(host.worker)
                self.assertFalse(host.temp_auto_running)
                with patch("ugw_host.filedialog.askdirectory", return_value="") as dialog:
                    for action, expected in (
                        (app.ugw.choose_folder, ugw_dir),
                        (app.ugw.choose_temp_folder, temp_dir),
                        (app.ugw.analyze_tof_folder, tof_dir),
                    ):
                        action()
                        self.assertEqual(dialog.call_args.kwargs["initialdir"], str(expected))
            finally:
                app.destroy()


if __name__ == "__main__":
    unittest.main(verbosity=2)
