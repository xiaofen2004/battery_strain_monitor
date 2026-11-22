"""Relative time of flight for recorded ultrasonic paths."""
from datetime import datetime, timezone
import re

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, correlate, detrend, lfilter

# Actual STM32 ADC sampling rate
ADC_FS_HZ = 80e6 / (6.5 + 12.5)

# ToF parameters
TOF_HP_HZ = 170e3
TOF_LP_HZ = 300e3
TOF_INTERP_FACTOR = 10

TOF_SEARCH_US = (5.0, 100.0)
TOF_MAX_LAG_US = 2.0
TOF_MIN_CORR = 0.35
MAT_TIMESTAMP = re.compile(r"(?<!\d)(\d{8}_\d{6})(?:_(\d{1,6}))?$")


def mat_recorded_time(path):
    """Return the filename's local clock on the summary's common numeric axis."""
    match = MAT_TIMESTAMP.search(path.stem)
    if match is None:
        raise ValueError(f"无法从文件名读取时间：{path.name}")
    stamp = datetime.strptime(match[1], "%Y%m%d_%H%M%S")
    if match[2]:
        stamp = stamp.replace(microsecond=int(match[2].ljust(6, "0")))
    return stamp.replace(tzinfo=timezone.utc).timestamp()


def _mat_field(obj, name, default=None):

    if isinstance(obj, dict):
        return obj.get(name, default)

    return getattr(
        obj,
        name,
        default
    )


def _mat_ints(value):

    if value is None:
        return []

    try:

        arr = np.asarray(value).squeeze()

        if arr.size == 0:
            return []

        return [
            int(round(float(v)))
            for v in np.asarray(arr).reshape(-1)
        ]

    except Exception:

        return []


def _mat_string(value):

    if value is None:
        return ""

    try:

        arr = np.asarray(value).squeeze()

        if arr.size == 1:
            value = arr.item()

        if isinstance(value, bytes):

            return value.decode(
                "utf-8",
                errors="ignore"
            ).strip()

        return str(value).strip()

    except Exception:

        return str(value).strip()


def _preprocess_tof_wave(raw):

    fs = ADC_FS_HZ

    x = detrend(
        np.asarray(raw, dtype=np.float64),
        type="linear"
    )

    bh, ah = butter(
        1,
        TOF_HP_HZ / (fs / 2),
        btype="high"
    )

    bl, al = butter(
        2,
        TOF_LP_HZ / (fs / 2),
        btype="low"
    )

    x = lfilter(
        bh,
        ah,
        x
    )

    x = lfilter(
        bl,
        al,
        x
    )

    t_old = (
        np.arange(x.size, dtype=np.float64)
        / fs
    )

    n_new = (
        (x.size - 1)
        * TOF_INTERP_FACTOR
        + 1
    )

    t_new = np.linspace(
        t_old[0],
        t_old[-1],
        n_new
    )

    y = np.interp(
        t_new,
        t_old,
        x
    )

    return (
        y,
        fs * TOF_INTERP_FACTOR
    )


def _relative_tof_xcorr(
    waveform,
    reference,
    fs_interp
):

    if waveform.size != reference.size:

        return np.nan, np.nan

    t_us = (
        np.arange(reference.size, dtype=np.float64)
        / fs_interp
        * 1e6
    )

    mask = (
        (t_us >= TOF_SEARCH_US[0])
        &
        (t_us <= TOF_SEARCH_US[1])
    )

    if np.count_nonzero(mask) < 20:

        return np.nan, np.nan

    a = reference[mask].copy()

    b = waveform[mask].copy()

    a -= np.mean(a)

    b -= np.mean(b)

    na = np.linalg.norm(a)

    nb = np.linalg.norm(b)

    if na < 1e-12 or nb < 1e-12:

        return np.nan, np.nan

    c = correlate(
        b,
        a,
        mode="full",
        method="auto"
    )

    lags = np.arange(
        -a.size + 1,
        b.size
    )

    max_lag = int(
        round(
            TOF_MAX_LAG_US
            * 1e-6
            * fs_interp
        )
    )

    keep = (
        np.abs(lags) <= max_lag
    )

    c = c[keep] / (na * nb)

    lags = lags[keep]

    idx = int(
        np.argmax(c)
    )

    quality = float(
        c[idx]
    )

    lag_us = float(
        lags[idx] / fs_interp * 1e6
    )

    if quality < TOF_MIN_CORR:

        return np.nan, quality

    return lag_us, quality


def path_label(pair):
    return f"Tx{pair[0]} → Rx{pair[1]}"


def _recorded_paths(ugw, path, mode, columns):
    saved_mode = _mat_string(_mat_field(ugw, "Mode", "")).upper()
    command = _mat_string(_mat_field(ugw, "Command", "")).upper()
    if not command:
        match = re.match(r"UGW_(M|R[1-4][1-4])_", path.name.upper())
        if match:
            command = match[1]
    command_mode = command[:1] if command.startswith(("M", "R")) else ""
    if (saved_mode or command_mode or "M") != mode or (command_mode and command_mode != mode):
        raise ValueError(f"文件的采集模式与所选 {mode} 模式不同")

    tx = _mat_ints(_mat_field(ugw, "TxChannels", None))
    rx = _mat_ints(_mat_field(ugw, "RxChannels", None))
    fixed_rx = _mat_ints(_mat_field(ugw, "FixedRxChannel", None))
    if mode == "M":
        channel_numbers = _mat_ints(_mat_field(ugw, "ChannelNumbers", None))
        if len(channel_numbers) == columns:
            tx = channel_numbers
        if not tx and not channel_numbers and columns == 3:
            tx = [1, 2, 3]
        if len(tx) == columns and fixed_rx in ([], [4]):
            return tuple((ch, 4) for ch in tx)
    else:
        pair = re.fullmatch(r"R([1-4])([1-4])", command)
        if pair:
            sent, received = int(pair[1]), int(pair[2])
            if columns == 1 and tx in ([], [sent]) and rx in ([], [received]) and fixed_rx in ([], [received]):
                return ((sent, received),)
        elif len(tx) == columns:
            rx = rx or fixed_rx * columns
            if len(rx) == columns:
                return tuple(zip(tx, rx))
    raise ValueError("无法从 MAT 确认各列对应的收发路径")


def _read_tof_waves(path, mode, pairs):
    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    if "UGW_result" in mat:
        ugw = mat["UGW_result"]
    elif "UGW_Struct" in mat:
        ugw = mat["UGW_Struct"]
    else:
        raise ValueError("缺少 UGW_result / UGW_Struct")
    raw_value = _mat_field(ugw, "ADC_raw", None)
    if raw_value is None:
        raise ValueError("缺少 ADC_raw 字段")
    raw = np.asarray(raw_value, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if raw.ndim != 2:
        raise ValueError("ADC_raw 不是二维矩阵")
    if raw.shape[0] <= 12 and raw.shape[1] >= 100:
        raw = raw.T
    recorded = _recorded_paths(ugw, path, mode, raw.shape[1])
    waves = {
        pair: raw[:, column] for column, pair in enumerate(recorded)
        if pair in pairs and raw.shape[0] >= 100 and np.isfinite(raw[:, column]).all()
    }
    if not waves:
        raise ValueError("没有所选路径的有效波形")
    return waves


def analyze_tof_files(files, mode, pairs, log=lambda _: None, before_file=lambda _: None):
    """Keep separate references and acquisition indices for each physical path."""
    rows = {pair: [] for pair in pairs}
    references = {}
    for index, path in enumerate(files, 1):
        before_file(index)
        try:
            waves = _read_tof_waves(path, mode, rows)
        except Exception as exc:
            log(f"ToF 跳过 {path.name}：{exc}")
            continue

        for pair, raw in waves.items():
            try:
                wave, fs = _preprocess_tof_wave(raw)
            except (ValueError, FloatingPointError) as exc:
                log(f"ToF 跳过 {path.name} 的 {path_label(pair)}：{exc}")
                continue
            reference, reference_fs = references.setdefault(pair, (wave, fs))
            if not np.isclose(fs, reference_fs) or wave.size != reference.size:
                delta, quality = np.nan, np.nan
                status = "长度/采样率不一致"
            else:
                delta, quality = _relative_tof_xcorr(wave, reference, reference_fs)
                if not np.isfinite(quality):
                    status = "相关计算失败"
                elif quality < TOF_MIN_CORR:
                    status = "相关质量不足"
                elif not np.isfinite(delta):
                    status = "ToF 无效"
                else:
                    status = "有效"
            rows[pair].append({
                "ContinuousIndex": len(rows[pair]) + 1,
                "FileName": path.name,
                "DeltaToF_us": delta,
                "XCorrCoefficient": quality,
                "Status": status,
            })
    return {pair: records for pair, records in rows.items() if records}
