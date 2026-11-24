"""只读读取仪器文本和电池工作簿。源文件从不以写入模式打开。"""
from __future__ import annotations

import ctypes
import io
import os
import re
import tempfile
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd


@contextmanager
def shared_read(path):
    """Windows 允许采集端继续读、写、替换；无法绕过采集端的独占锁。"""
    if os.name != "nt":
        with open(path, "rb") as handle:
            yield handle
        return
    import msvcrt
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                       wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    handle = create(str(Path(path).resolve()), 0x80000000, 7, None, 3, 0x80, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except Exception:
        close = kernel.CloseHandle
        close.argtypes = [wintypes.HANDLE]
        close(handle)
        raise
    with os.fdopen(fd, "rb") as stream:
        yield stream


def stamp(path):
    s = Path(path).stat()
    return s.st_size, s.st_mtime_ns, s.st_ino


def check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise InterruptedError("已取消，保留上一张图。")


@contextmanager
def snapshot(path, cancel=None):
    """仅复制一次完整、稳定的磁盘版本，临时文件退出时删除。"""
    before = stamp(path)
    with tempfile.TemporaryDirectory(prefix="battery_strain_monitor_") as folder:
        dest = Path(folder) / ("snapshot" + Path(path).suffix)
        with shared_read(path) as source, dest.open("wb") as target:
            remaining = before[0]
            while remaining:
                check_cancel(cancel)
                block = source.read(min(4 * 1024**2, remaining))
                if not block:
                    raise OSError("源文件正在重写，请稍后再次刷新。")
                target.write(block)
                remaining -= len(block)
        if stamp(path) != before:
            raise OSError("源文件正在保存，本次未采用不完整快照；请再次刷新。")
        yield dest


def excel_rows(path, sheet=None):
    """支持真 XLS/XLSX；不会根据错误的 A1:A1 范围截断数据。"""
    with open(path, "rb") as f:
        magic = f.read(8)
    if magic.startswith(b"PK"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Workbook contains no default style.*")
            # 使用流以兼容被误命名为 .xls 的 OOXML 文件。
            with open(path, "rb") as stream:
                book = openpyxl.load_workbook(stream, read_only=True, data_only=True)
                try:
                    ws = book[sheet] if sheet and sheet in book.sheetnames else book.worksheets[0]
                    if sheet and sheet not in book.sheetnames:
                        raise ValueError(f"电池文件没有 {sheet} 工作表。")
                    ws.reset_dimensions()
                    yield from ws.values
                finally:
                    book.close()
    elif magic.startswith(b"\xd0\xcf\x11\xe0"):
        import xlrd
        book = xlrd.open_workbook(path, on_demand=True)
        try:
            ws = book.sheet_by_name(sheet) if sheet else book.sheet_by_index(0)
            for row in range(ws.nrows):
                yield [xlrd.xldate_as_datetime(c.value, book.datemode)
                       if c.ctype == xlrd.XL_CELL_DATE else c.value for c in ws.row(row)]
        finally:
            book.release_resources()
    else:
        raise ValueError("此文件不是真正的 Excel 工作簿。")


def parse_times(values):
    values = pd.Series(values)
    if pd.api.types.is_numeric_dtype(values):
        result = pd.to_datetime(values, unit="D", origin="1899-12-30", errors="coerce")
    else:
        text = values.astype(str).str.strip()
        fmt = "%y/%m/%d %H:%M:%S.%f" if re.match(r"^\d{2}/", text.iloc[0]) else "mixed"
        result = pd.to_datetime(text, format=fmt, errors="coerce")
    if result.isna().any():
        raise ValueError("存在无法识别的日期时间，请检查文件中的完整时间列。")
    # 避免 pandas 新版默认微秒精度造成数量级错误。
    return result.to_numpy(dtype="datetime64[ns]").astype(np.int64) / 1e9


def channel_header(row):
    return {re.match(r"^(CH\d+)", str(v).strip(), re.I).group(1).upper(): i
            for i, v in enumerate(row) if re.match(r"^CH\d+", str(v).strip(), re.I)}


class StrainReader:
    """文本增量缓存仅保存所选通道。改变通道会重新读取源文件。"""
    def __init__(self, path, channel):
        self.path = Path(path)
        self.single = isinstance(channel, str)
        self.selected = (channel.upper(),) if self.single else tuple(c.upper() for c in channel)
        if not 1 <= len(self.selected) <= 3 or len(set(self.selected)) != len(self.selected):
            raise ValueError("请选择 1～3 个不同的应变通道。")
        self.channel = self.selected[0] if self.single else self.selected
        self.offset = 0
        self.parts_t, self.parts_y = [], []
        self.signature = None
        self.head = self.guard = b""
        self.channels = []
        self.last_stamp = None
        self.kind = ""
        self.header_end = 0
        self.column = 0
        self.ncolumns = 0

    def update(self, progress=lambda _: None, cancel=None):
        check_cancel(cancel)
        current = stamp(self.path)
        if current == self.last_stamp and self.parts_t:
            return self.arrays()
        with shared_read(self.path) as f:
            start = f.read(65536)
            is_excel = start.startswith((b"PK", b"\xd0\xcf\x11\xe0"))
            if is_excel:
                self.kind = "excel"
            else:
                reset = bool(self.last_stamp and (
                    current[0] < self.offset or current[2] != self.last_stamp[2]
                    or (current[0] == self.last_stamp[0] and current != self.last_stamp)
                    or not start.startswith(self.head)))
                if self.guard and not reset:
                    f.seek(self.offset - len(self.guard))
                    reset = f.read(len(self.guard)) != self.guard
                if reset:
                    self.__init__(self.path, self.channel)
                    progress("检测到应变文件被替换或重写，正在重新读取")
                self.kind = "text"
                if not self.offset:
                    used = 0
                    for line in start.splitlines(keepends=True):
                        used += len(line)
                        fields = line.decode("gb18030", errors="replace").rstrip("\r\n").split("\t")
                        columns = channel_header(fields)
                        if columns and (len(columns) > 1 or fields[0].strip().lower() in ("", "time", "时间")):
                            self.channels = list(columns)
                            missing = [c for c in self.selected if c not in columns]
                            if missing:
                                raise ValueError(f"找不到 {', '.join(missing)}；可选通道：{', '.join(columns)}")
                            self.columns = [columns[c] for c in self.selected]
                            self.ncolumns = len(fields)
                            self.offset = self.header_end = used
                            self.head = start[:used]
                            break
                    else:
                        raise ValueError("未找到应变通道表头。需要时间列及 CH01…CH20 通道列。")
                limit = current[0]  # 点击刷新时的文件大小；之后新增的数据留到下一次。
                f.seek(self.offset)
                while self.offset < limit:
                    check_cancel(cancel)
                    block = f.read(min(8 * 1024**2, limit - self.offset))
                    end = block.rfind(b"\n") + 1
                    if not end:
                        break  # 未完成的末行保留到下一次。
                    data = pd.read_csv(io.BytesIO(block[:end]), sep="\t", header=None,
                                       usecols=[0, *self.columns], encoding="gb18030")
                    t = parse_times(data.iloc[:, 0])
                    # read_csv 按源文件列序返回，显式按选择顺序取列，避免通道错配。
                    y = data[self.columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
                    if self.single:
                        y = y[:, 0]
                    self._validate(t, y)
                    self.parts_t.append(t)
                    self.parts_y.append(y)
                    self.offset += end
                    f.seek(self.offset)
                    progress(f"读取应变 {100 * self.offset / max(limit, 1):.0f}%")
                f.seek(max(0, self.offset - 256))
                self.guard = f.read(min(256, self.offset))
        if is_excel:
            progress("读取应变 Excel 快照")
            with snapshot(self.path, cancel) as copy:
                rows = iter(excel_rows(copy))
                for row in rows:
                    cols = channel_header(row)
                    if len(cols) > 1:
                        break
                else:
                    raise ValueError("应变工作簿未找到多通道表头。")
                self.channels = list(cols)
                missing = [c for c in self.selected if c not in cols]
                if missing:
                    raise ValueError(f"文件中没有 {', '.join(missing)}")
                pairs = []
                for i, row in enumerate(rows):
                    if i % 10000 == 0:
                        check_cancel(cancel)
                    if not row or row[0] in (None, ""):
                        continue
                    pairs.append((row[0], [row[cols[c]] for c in self.selected]))
                if not pairs:
                    raise ValueError("应变文件尚无数据。")
                t = parse_times([p[0] for p in pairs])
                y = np.asarray([p[1] for p in pairs], float)
                if self.single:
                    y = y[:, 0]
                self.parts_t, self.parts_y = [], []
                self._validate(t, y)
                self.parts_t, self.parts_y = [t], [y]
        self.last_stamp = current
        return self.arrays()

    def _validate(self, t, y):
        if not len(t) or not np.isfinite(y).all():
            raise ValueError("应变记录含空值或无效数值，未跳过这些记录。")
        if np.any(np.diff(t) < 0) or (self.parts_t and t[0] < self.parts_t[-1][-1]):
            raise ValueError("应变时间发生倒退，请检查文件；上一张图保留。")

    def arrays(self):
        if not self.parts_t:
            raise ValueError("应变文件尚无完整数据行。")
        return np.concatenate(self.parts_t), np.concatenate(self.parts_y)


@dataclass
class BatteryData:
    time: np.ndarray
    voltage: np.ndarray
    current: np.ndarray
    stages: np.ndarray
    notes: list[str]
    rows: int
    step_starts: np.ndarray | None = None
    used_elapsed: bool = False


def battery_step_starts(frame, stages):
    """Return row indices of distinct work-step occurrences."""
    starts = np.zeros(len(stages), dtype=bool)
    starts[0] = True
    starts[1:] = stages[1:] != stages[:-1]
    for names in (("循环序号", "循环号", "循环编号"), ("工步序号", "工步号", "工步编号")):
        column = next((name for name in names if name in frame), None)
        if column is not None:
            ids = frame[column].ffill()
            starts[1:] |= (ids.notna() & ids.shift().notna() & ids.ne(ids.shift())).to_numpy()[1:]
    if "工步开始结束标识" in frame:
        marker = frame["工步开始结束标识"]
        begins = pd.to_numeric(marker, errors="coerce").eq(0) | marker.astype(str).str.strip().isin(("开始", "start"))
        starts |= (begins & ~begins.shift(fill_value=False)).to_numpy()
    if "时间" in frame:
        elapsed = pd.to_timedelta(frame["时间"].astype(str), errors="coerce").dt.total_seconds().to_numpy()
        starts[1:] |= np.diff(elapsed) < 0
    return np.flatnonzero(starts)


def battery_elapsed_clock(absolute, elapsed, *, prefer_absolute):
    """Rebuild from local anchors without collapsing pauses between recordings."""
    absolute_steps = np.diff(absolute)
    elapsed_steps = np.diff(elapsed)
    # Timestamp rounding can move a sample by one second. Split only when
    # the wall clock jumps well beyond the elapsed interval and sampling cadence.
    cadences = [np.median(steps[steps > 0]) for steps in (absolute_steps, elapsed_steps)
                if np.any(steps > 0)]
    threshold = max(1., 5 * min(cadences, default=0.))
    extra = absolute_steps - elapsed_steps
    pauses = extra > threshold
    # A single timestamp that jumps forward and immediately returns is a bad
    # reading within a segment, not the first row of a resumed recording.
    pauses[:-1] &= extra[:-1] + extra[1:] > threshold
    gaps = np.flatnonzero(pauses) + 1
    edges = np.r_[0, gaps, absolute.size]
    result = absolute.copy()
    for a, b in zip(edges[:-1], edges[1:]):
        if not prefer_absolute or np.any(absolute_steps[a:b - 1] < 0):
            result[a:b] = absolute[a] + (elapsed[a:b] - elapsed[a])
        if a and result[a] < result[a - 1]:
            # Keep the records ordered if a reconstructed segment reaches
            # past the timestamp at the start of the next segment.
            result[a:b] += result[a - 1] + elapsed_steps[a - 1] - result[a]
    return result, gaps.size


class BatteryReader:
    def __init__(self):
        self.key = None
        self.data = None

    def update(self, path, mode="auto", progress=lambda _: None, cancel=None):
        key = (str(Path(path).resolve()), stamp(path), mode)
        if key == self.key:
            return self.data
        progress("读取电池工作簿快照")
        with snapshot(path, cancel) as copy:
            rows = iter(excel_rows(copy, "record"))
            header = next(rows)
            cols = {str(v).strip(): i for i, v in enumerate(header)}
            required = ["绝对时间", "电压(V)", "电流(A)", "工步类型"]
            missing = [k for k in required if k not in cols]
            if missing:
                raise ValueError("record 工作表缺少列：" + ", ".join(missing))
            records = []
            for i, row in enumerate(rows):
                if i % 5000 == 0:
                    check_cancel(cancel)
                if not row or all(x is None for x in row):
                    continue
                records.append(tuple(row) + (None,) * (len(header) - len(row)))
            if not records:
                raise ValueError("电池文件尚无数据。")
        frame = pd.DataFrame(records, columns=header)
        absolute = parse_times(frame["绝对时间"])
        dif = np.diff(absolute)
        notes = []
        positive = dif[dif > 0]
        interval = float(np.median(positive)) if len(positive) else float("nan")
        notes.append(f"电池表记录 {len(frame):,} 行，正时间间隔中位数 {interval:g} s；仪器标称 100 Hz")
        backwards = np.any(dif < 0)
        if backwards:
            notes.append(f"电池绝对时间存在 {int((dif < 0).sum())} 处倒退")
        use_elapsed = "总时间" in frame and (mode == "auto" or (mode == "prefer_absolute" and backwards))
        if mode == "fixed":
            t = absolute[0] + np.arange(len(frame)) / 100
            notes.append("已强制每行按 100 Hz 重建时间；仅适用于每行确为一个采样点的文件")
        elif use_elapsed:
            elapsed = pd.to_timedelta(frame["总时间"].astype(str), errors="coerce").dt.total_seconds().to_numpy()
            if not np.isfinite(elapsed).all() or np.any(np.diff(elapsed) < 0):
                raise ValueError("总时间无效或倒退，请检查源文件或选择绝对时间模式。")
            t, gaps = battery_elapsed_clock(absolute, elapsed, prefer_absolute=mode == "prefer_absolute")
            notes.append(f"电池按各段起始绝对时间与总时间校正，保留 {gaps} 处记录间隔。"
                         f"与表内绝对时间的最大偏差 {np.max(np.abs(t - absolute)):.3f} 秒")
        else:
            t = absolute
            if backwards:
                raise ValueError("电池绝对时间倒退；请核对表内时间或使用有效的总时间列。")
        voltage = pd.to_numeric(frame["电压(V)"], errors="coerce").to_numpy(float)
        current = pd.to_numeric(frame["电流(A)"], errors="coerce").to_numpy(float)
        if not (np.isfinite(voltage).all() and np.isfinite(current).all()):
            raise ValueError("电池电压/电流存在空值或非数值。")
        stages = frame["工步类型"].fillna("未知").astype(str).to_numpy()
        self.data = BatteryData(t, voltage, current, stages, notes, len(frame),
                                battery_step_starts(frame, stages), bool(use_elapsed))
        self.key = key
        return self.data
