import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
import numpy as np

from data_io import StrainReader, BatteryData, parse_times, snapshot
from processing import Settings, infer_rate, preprocess, make_plot_data, stage_spans


class ProcessingTests(unittest.TestCase):
    def test_subplots_share_time_but_scale_independently(self):
        from plotting import draw_plot
        t = 1700000000 + np.arange(10.)
        y = np.column_stack([np.arange(10.), 100 * np.arange(10.), -1000 * np.arange(10.)])
        b = BatteryData(t, np.ones(10) * 4, np.ones(10) * 30, np.array(["恒流充电"] * 10), [], 10)
        cfg = Settings(channel="1,2,3", strain_hz=1, cutoff_hz=0, median_points=1,
                       plot_layout="subplots", include_battery=False)
        for combined in (False, True):
            data = make_plot_data(t, y, b if combined else None, replace(cfg, include_battery=combined))
            fig = draw_plot(data)
            self.assertEqual(len(fig.axes), 5 if combined else 3)
            first = fig.axes[0]
            for axis in fig.axes[1:]:
                self.assertTrue(first.get_shared_x_axes().joined(first, axis))
            self.assertFalse(first.get_shared_y_axes().joined(first, fig.axes[1]))
            self.assertLess(first.get_ylim()[1], fig.axes[1].get_ylim()[1] / 10)
            first.set_xlim(.02, .08)
            for axis in fig.axes:
                np.testing.assert_allclose(axis.get_xlim(), [.02, .08])
            if combined:
                self.assertTrue(all(len(axis.patches) > 0 for axis in fig.axes[:4]))

    def test_three_channels_strain_only_baselines_axes_and_export(self):
        from plotting import draw_plot
        from exporting import export_frame
        t = 1700000000 + np.arange(10.)
        y = np.column_stack([728 + np.arange(10.), -9000 + 2 * np.arange(10.), 851 - np.arange(10.)])
        cfg = Settings(channel="5,17,8", strain_hz=1, cutoff_hz=0, median_points=1,
                       include_battery=False, alignment="first", offset_s=999)
        data = make_plot_data(t, y, None, cfg)
        self.assertEqual(data.baselines, {"CH05": 728, "CH17": -9000, "CH08": 851})
        np.testing.assert_allclose(data.strain_t, np.arange(10.))
        for channel, expected in zip(data.strains, (np.arange(10.), 2 * np.arange(10.), -np.arange(10.))):
            np.testing.assert_allclose(data.strains[channel], expected)
        self.assertEqual(len(data.battery_t), 0)
        fig = draw_plot(data)
        self.assertEqual(len(fig.axes), 1)
        self.assertEqual(len(fig.axes[0].lines), 3)
        self.assertEqual(len(fig.axes[0].patches), 0)
        exported = export_frame(data)
        self.assertEqual(len(exported), 30)
        self.assertEqual(set(exported.series), {"strain"})
        self.assertEqual(set(exported.strain_channel), {"CH05", "CH08", "CH17"})
        raw = make_plot_data(t, y, None, replace(cfg, delta=False))
        for channel in data.strains:
            np.testing.assert_allclose(raw.strains[channel] - data.strains[channel], data.baselines[channel])

    def test_three_channels_combined_crop(self):
        from plotting import draw_plot
        t = 1700000000 + np.arange(10.)
        y = np.column_stack([np.arange(10.) + n for n in (10, 20, 30)])
        b = BatteryData(t[3:], np.ones(7), np.ones(7), np.array(["搁置"] * 7), [], 7)
        cfg = Settings(channel="CH01,CH02,CH03", strain_hz=1, cutoff_hz=0, median_points=1)
        data = make_plot_data(t, y, b, cfg)
        for values in data.strains.values():
            np.testing.assert_allclose(values, np.arange(3., 10.))
        fig = draw_plot(data)
        self.assertEqual(len(fig.axes), 3)
        self.assertEqual(len(fig.axes[0].lines), 3)
        with self.assertRaises(ValueError):
            make_plot_data(t, y, None, cfg)

    def test_channel_selection_validation(self):
        for value in ("", "CH05,5", "1,2,3,4", "CH21"):
            with self.assertRaises(ValueError):
                Settings(channel=value).validate()
    def test_detect_1_and_100_hz(self):
        for rate in (1, 100):
            t = 1700000000 + np.arange(1000) / rate
            self.assertEqual(infer_rate(t)[0], rate)

    def test_delta_is_original_first_before_crop(self):
        t = 1700000000 + np.arange(10.)
        y = 728 + np.arange(10.)
        cfg = Settings(strain_hz=1, cutoff_hz=0, median_points=1)
        b = BatteryData(t[3:], np.ones(7), np.ones(7), np.array(["搁置"] * 7), [], 7)
        result = make_plot_data(t, y, b, cfg)
        self.assertEqual(result.baseline, 728)
        self.assertEqual(result.strain[0], 3)
        original = make_plot_data(t, y, b, replace(cfg, delta=False))
        np.testing.assert_allclose(original.strain - result.strain, 728)

    def test_first_alignment_and_offset(self):
        t = 1700000000 + np.arange(10.)
        b = BatteryData(t + 100, np.ones(10), np.ones(10), np.array(["搁置"] * 10), [], 10)
        cfg = Settings(strain_hz=1, cutoff_hz=0, median_points=1, alignment="first", offset_s=2)
        result = make_plot_data(t, np.arange(10.), b, cfg)
        self.assertEqual(result.strain_t[0], 2)
        self.assertEqual(result.battery_t[0], 0)

    def test_antialias_and_dc(self):
        t = np.arange(20000) / 100
        y = 728 + np.sin(2 * np.pi * 10.1 * t)
        cfg = Settings(strain_hz=100, output_hz=1, cutoff_hz=0, median_points=1, delta=False)
        tt, yy, _, _ = preprocess(t, y, cfg)
        self.assertEqual(len(tt), 200)
        self.assertLess(np.std(yy[30:]), .001)
        self.assertAlmostEqual(np.mean(yy[30:]), 728, places=3)

    def test_causal_prefix_stability(self):
        t = np.arange(2000) / 100
        y = np.sin(t) + 20
        cfg = Settings(strain_hz=100)
        first = preprocess(t[:1000], y[:1000], cfg)[1]
        full = preprocess(t, y, cfg)[1]
        np.testing.assert_allclose(first, full[:len(first)])

    def test_gap_and_bounds(self):
        t = np.r_[np.arange(10.), 20 + np.arange(10.)]
        result = preprocess(t, np.ones(20), Settings(strain_hz=1))
        self.assertTrue(np.isnan(result[1]).any())
        with self.assertRaises(ValueError):
            preprocess(t, np.ones(20), Settings(strain_hz=1, output_hz=100))

    def test_stage_backgrounds_and_gap(self):
        spans = stage_spans(np.array([0, 1, 2, 3, 4, 10, 11.]),
                            ["搁置", "恒流放电", "恒流放电", "恒流恒压充电", "恒流恒压充电", "搁置", "搁置"])
        self.assertEqual(spans, [(0, 1, "搁置"), (1, 3, "放电"), (3, 4, "充电"), (10, 11, "搁置")])


class ReaderTests(unittest.TestCase):
    def test_multiple_channels_order_increment_and_reset(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "strain.xls"
            header = b"\tCH01-\tCH05-\tCH08-\n"
            line1 = b"26/11/10 18:17:58.000\t1\t728\t850\n"
            line2 = b"26/11/10 18:17:59.000\t2\t730\t851\n"
            path.write_bytes(header + line1)
            reader = StrainReader(path, ("CH08", "CH01", "CH05"))
            self.assertEqual(reader.update()[1].tolist(), [[850, 1, 728]])
            with path.open("ab") as f:
                f.write(line2)
            self.assertEqual(reader.update()[1].tolist(), [[850, 1, 728], [851, 2, 730]])
            path.write_bytes(header + line1.replace(b"728", b"900"))
            self.assertEqual(reader.update()[1].tolist(), [[850, 1, 900]])
            with self.assertRaises(ValueError):
                StrainReader(path, ("CH17", "CH05")).update()
    def test_incremental_partial_row_rewrite_and_channel(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "strain.xls"
            header = b"\tCH01-\tCH05-\n"
            line1 = b"26/11/10 18:17:58.000\t1\t728\n"
            line2 = b"26/11/10 18:17:59.000\t2\t730\n"
            path.write_bytes(header + line1 + line2[:12])
            reader = StrainReader(path, "CH05")
            t, y = reader.update()
            self.assertEqual(y.tolist(), [728])
            with path.open("ab") as f:
                f.write(line2[12:])
            t, y = reader.update()
            self.assertEqual(y.tolist(), [728, 730])
            np.testing.assert_equal(reader.update()[1], y)
            self.assertEqual(StrainReader(path, "CH01").update()[1].tolist(), [1, 2])
            path.write_bytes(header + line1.replace(b"728", b"900"))
            self.assertEqual(reader.update()[1].tolist(), [900])

    def test_snapshot_and_backward_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "strain.xls"
            data = b"\tCH01-\tCH05-\n26/11/10 18:17:59.000\t1\t2\n26/11/10 18:17:58.000\t1\t2\n"
            path.write_bytes(data)
            with snapshot(path) as copy:
                self.assertEqual(copy.read_bytes(), data)
            self.assertFalse(copy.exists())
            with self.assertRaises(ValueError):
                StrainReader(path, "CH05").update()
            self.assertEqual(path.read_bytes(), data)

    def test_timestamp_units(self):
        t = parse_times(["2026-11-10 18:10:27", "2026-11-10 18:10:28"])
        self.assertAlmostEqual(t[1] - t[0], 1)
        self.assertGreater(t[0], 1_700_000_000)



class SummaryTests(unittest.TestCase):
    def test_three_clocks_and_step_selection_use_real_intervals(self):
        from datetime import datetime, timedelta
        import openpyxl
        from matplotlib.figure import Figure
        from data_io import parse_times
        from summary_data import load_summary, visible_traces
        from summary_plot import draw_summary

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            start = datetime(2026, 11, 22, 12, 0, 0)

            def at(seconds):
                return start + timedelta(seconds=seconds)

            strain = root / "strain.xls"
            records = [f"{at(i).strftime('%y/%m/%d %H:%M:%S')}.000\t{700+i}" for i in range(20)]
            strain.write_text("\tCH05-\n" + "\n".join(records) + "\n", encoding="utf-8")
            battery = root / "battery.xlsx"
            book = openpyxl.Workbook()
            record = book.active
            record.title = "record"
            record.append(("绝对时间", "电压(V)", "电流(A)", "工步类型", "时间", "工步开始结束标识", "总时间"))
            for seconds in range(1, 20):
                first, kind = ((1, "搁置") if seconds < 6 else (6, "搁置") if seconds < 10
                               else (10, "恒流放电") if seconds < 15 else (15, "恒流充电"))
                absolute = at(1) if seconds == 3 else at(seconds)
                record.append((absolute, 3.7, 0., kind, f"00:00:{seconds-first:02d}",
                               0 if seconds == first else None, f"00:00:{seconds-1:02d}"))
            book.save(battery)

            temperature = root / "temperature.csv"
            lines = ["PC_Time,ScheduledPC_Time,Temperature_CVD_C,Elapsed_s"]
            for i in range(4, 19):
                absolute = at(i).astimezone().isoformat()
                if i == 7:
                    lines.append(f",{absolute},,{i-4}")
                else:
                    lines.append(f"{absolute},{absolute},{20+i/10},{i-4}")
            temperature.write_text("\n".join(lines), encoding="utf-8-sig")

            cfg = Settings(strain_hz=100, median_points=1, cutoff_hz=0)
            summary = load_summary({"strain": strain, "battery": battery, "temperature": temperature}, cfg)
            epoch = parse_times([at(0)])[0]
            self.assertEqual(summary.origin, epoch + 4)
            self.assertTrue(summary.battery_clock_rebuilt)
            self.assertEqual([step.name for step in summary.steps], ["搁置", "搁置", "恒流放电", "恒流充电"])
            self.assertEqual([step.start - epoch for step in summary.steps], [1, 6, 10, 15])
            complete, _ = visible_traces(summary, {1, 2, 3, 4})
            self.assertEqual(complete["strain"].time[0], epoch + 4)
            self.assertIn("1 Hz", summary.strain_rate_note)
            self.assertTrue(np.isfinite(complete["strain"].values["CH05"]).all())
            self.assertTrue(np.isnan(complete["temperature"].values["CVD 温度"]).any())
            filtered, selected = visible_traces(summary, {1, 3})
            self.assertEqual([step.number for step in selected], [1, 3])
            for trace in filtered.values():
                self.assertFalse(((trace.time > epoch + 5) & (trace.time < epoch + 10)).any())
                self.assertTrue(any(np.isnan(values).any() for values in trace.values.values()))

            figure = Figure()
            self.assertTrue(draw_summary(figure, summary, {1, 3}, "h", "分开绘制"))
            self.assertEqual(len(figure.axes), 4)
            self.assertEqual(figure.axes[2].get_xlabel(), "与共同起点的时间差 / h")
            self.assertEqual(len(figure.axes[2].patches), 2)
            self.assertTrue(draw_summary(figure, summary, {1, 3}, "绝对时间", "同图叠加"))
            self.assertEqual(len(figure.axes), 4)
            self.assertEqual(figure.axes[0].get_xlabel(), "绝对时间")

            for unit, scale in (("s", 1), ("min", 60), ("h", 3600), ("相对时间", 1)):
                with self.subTest(unit=unit):
                    self.assertTrue(draw_summary(figure, summary, {2, 4}, unit, "分开绘制"))
                    axis = figure.axes[0]
                    line = axis.lines[0]
                    self.assertEqual(axis.get_xlim()[0], 0)
                    self.assertIn(0, axis.get_xticks())
                    np.testing.assert_allclose(line.get_xdata()[np.isfinite(line.get_ydata())],
                                               np.r_[0:4, 9:14] / scale)
                    self.assertTrue(np.isnan(line.get_ydata()).any())
            self.assertEqual(figure.axes[2].xaxis.get_major_formatter()(0), "00:00:00")
            self.assertTrue(draw_summary(figure, summary, {2, 4}, "绝对时间", "同图叠加"))
            axis = figure.axes[0]
            self.assertEqual(axis.get_xlim()[0], 2)
            self.assertEqual(axis.xaxis.get_major_formatter()(2), "2026-11-22\n12:00:06")
            self.assertFalse(draw_summary(figure, summary, set(), "min", "分开绘制"))
            temp_only = load_summary({"temperature": temperature})
            self.assertTrue(draw_summary(figure, temp_only, set(), "相对时间", "同图叠加"))
            self.assertEqual(len(figure.axes[0].patches), 0)


    def test_summary_shows_isolated_strain_points_without_connecting_gaps(self):
        from matplotlib.figure import Figure
        from summary_data import SummaryData, Trace
        from summary_plot import draw_summary

        trace = Trace(np.array([0., 1., 1., 2., 2.]),
                      {"CH05": np.array([2., np.nan, 3., np.nan, 5.])})
        data = SummaryData(0., {"strain": trace}, [], "应变 / με")
        figure = Figure()
        self.assertTrue(draw_summary(figure, data, set(), "s", "分开绘制"))
        line = figure.axes[0].lines[0]
        self.assertEqual(line.get_marker(), ".")
        self.assertEqual(line.get_markevery(), [0, 2, 4])
        self.assertTrue(np.isnan(line.get_ydata()[1]))

    def test_temperature_legacy_relative_seconds_require_real_start(self):
        from data_io import parse_times
        from summary_data import read_temperature

        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "old.csv"
            file.write_text("Elapsed_s,Temperature_C\n2,21.5\n12,22.5\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "只有相对时间"):
                read_temperature(file)
            trace = read_temperature(file, "2026-11-22 12:00:00")
            np.testing.assert_allclose(trace.time - parse_times(["2026-11-22 12:00:00"])[0], [2, 12])
            np.testing.assert_allclose(trace.values["温度"], [21.5, 22.5])


if __name__ == "__main__":
    unittest.main(verbosity=2)
