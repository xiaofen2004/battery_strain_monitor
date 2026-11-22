"""窗口控件与读取流程检查；不保存用户配置，不占用实际采集设备。"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as monitor
from app import MonitorApp


class AppModeTests(unittest.TestCase):
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


    def test_serial_monitor_restores_independent_output_directories(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(monitor, "ROOT", Path(folder)):
            root = Path(folder)
            temp_dir, ugw_dir, tof_dir = (root / name for name in ("温度记录", "超声采集", "历史超声"))
            for directory in (temp_dir, ugw_dir, tof_dir):
                directory.mkdir()
            (root / "settings.json").write_text(
                json.dumps({"channel": "CH17"}, ensure_ascii=False), encoding="utf-8"
            )
            app = MonitorApp()
            app.withdraw()
            try:
                self.assertEqual(app.fields["channel"].get(), "CH17")
                self.assertEqual(len(app.monitor_tabs.tabs()), 2)
                app.ugw.save_dir_var.set(str(ugw_dir))
                app.ugw.temp_save_dir_var.set(str(temp_dir))
                app.ugw.tof_dir = str(tof_dir)
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
