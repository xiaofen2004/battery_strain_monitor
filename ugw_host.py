# -*- coding: utf-8 -*-
"""
STM32 UGW + Temperature Python Host

M:
    Tx1/Rx4, Tx2/Rx4, Tx3/Rx4
    3 x 800 uint16 = 4800 Bytes
    Complete reception before channel filtering.

R:
    Select multiple Tx-Rx pairs.
    Each Rxy receives 6400 uint16 = 12800 Bytes.
    Wait for R_DONE before sending the next Rxy.

Temperature:
    Single-shot command Y only.
    Periodic temperature is repeated Y commands from PC.
    No legacy MCU 1 Hz T mode.

Power:
    v : Enable UGW power.
    0 : Stop UGW and disable power.

Serial arbitration:
    UGW acquisition has priority.
    Temperature requests are skipped and marked when UGW is busy.
    No concurrent UART transactions.

Temperature save:
    Optional CSV, disabled by default.

UGW save:
    MATLAB-compatible UGW_result MAT structure.

ToF:
    Folder-based M-mode Tx3 -> Rx4 analysis.
    XCorr_Wave_W1_Lag2 processing.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import threading
import queue
import time
import csv

from pathlib import Path
from datetime import datetime

import numpy as np
import serial
from serial.tools import list_ports

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from scipy.io import savemat, loadmat

from scipy.signal import (
    butter,
    lfilter,
    detrend,
    correlate
)


# ============================================================
# 1. Fixed parameters
# ============================================================

DEFAULT_BAUDRATE = 1382400

# Actual STM32 ADC sampling rate
ADC_FS_HZ = 80e6 / (6.5 + 12.5)

DEFAULT_ADC_BITS = 12
DEFAULT_VREF = 3.3

DEFAULT_TIMEOUT = 10.0

DEFAULT_UGW_INTERVAL = 15.0
DEFAULT_TEMP_INTERVAL = 10.0

# Independent data directories
DEFAULT_BASE_FOLDER = (
    Path.home() / "Documents" / "STM32_UGW_data"
)

DEFAULT_UGW_FOLDER = DEFAULT_BASE_FOLDER / "UGW"
DEFAULT_TEMP_FOLDER = DEFAULT_BASE_FOLDER / "Temperature"

# Compatibility
DEFAULT_SAVE_FOLDER = DEFAULT_UGW_FOLDER

# M mode
M_CHANNELS = (1, 2, 3)
M_RX = 4
M_SAMPLES_PER_PATH = 800
M_TOTAL_BYTES = 3 * M_SAMPLES_PER_PATH * 2

# R mode
R_SAMPLES_PER_PATH = 6400
R_BYTES_PER_PATH = R_SAMPLES_PER_PATH * 2

# Temperature
TEMP_COMMAND = b"Y"
TEMP_RESPONSE_PREFIX = "T,"

TEMP_RESPONSE_TIMEOUT = 2.5

# Power commands
POWER_ON_COMMAND = b"v"
POWER_OFF_COMMAND = b"0"

# ToF parameters
TOF_HP_HZ = 170e3
TOF_LP_HZ = 300e3
TOF_INTERP_FACTOR = 10

TOF_SEARCH_US = (5.0, 100.0)
TOF_MAX_LAG_US = 2.0
TOF_MIN_CORR = 0.35


# ============================================================
# 2. Main application
# ============================================================


PANEL_BG = "#f8fafc"
PANEL_BORDER = "#d8e2ed"


def configure_panel_styles(parent, px):
    style = ttk.Style(parent)
    style.configure("Monitor.TFrame", background=PANEL_BG)
    for kind in ("TLabel", "Muted.TLabel", "Section.TLabel", "TCheckbutton", "TRadiobutton"):
        style.configure(f"Monitor.{kind}", background=PANEL_BG)
    for kind in ("TCheckbutton", "TRadiobutton"):
        style.map(f"Monitor.{kind}", background=[("disabled", PANEL_BG), ("!disabled", PANEL_BG)])
    style.configure("Monitor.TLabelframe", background=PANEL_BG, bordercolor=PANEL_BORDER,
                    lightcolor=PANEL_BORDER, darkcolor=PANEL_BORDER, relief="solid", borderwidth=1)
    style.configure("Monitor.TLabelframe.Label", background=PANEL_BG, foreground="#172b4d",
                    font=("Microsoft YaHei UI", 10, "bold"))
    for kind in ("TButton", "Primary.TButton"):
        style.configure(f"Monitor.{kind}", padding=(px(7), px(4)))


class UGWHost:

    def __init__(self, parent, px):
        self.parent = parent
        self.px = px

        # ----------------------------------------------------
        # Serial
        # ----------------------------------------------------

        self.ser = None

        self.worker = None
        self.stop_event = None

        self.temp_auto_thread = None
        self.temp_auto_stop = threading.Event()

        self.event_queue = queue.Queue()

        # Serial mutual exclusion
        self.serial_lock = threading.Lock()
        self.state_lock = threading.Lock()

        self.ugw_requested = False
        self.ugw_active = False

        self.temp_active = False
        self.temp_auto_running = False

        # Power state:
        # None  = unknown
        # True  = v command sent
        # False = 0 command sent
        #
        # The MCU does not return an explicit power-state ACK.
        # Therefore this represents the last command sent,
        # not independently verified hardware power state.
        self.power_enabled = None
        self.power_busy = False

        # ----------------------------------------------------
        # UGW data
        # ----------------------------------------------------

        self.latest_result = None

        self.acquisition_count = 0
        self.plot_cleared = False

        # ----------------------------------------------------
        # Temperature data
        # ----------------------------------------------------

        self.temp_samples = []

        self.temp_data_lock = threading.Lock()

        self.temp_auto_start_mono = None
        self.temp_auto_index = 0

        # Separate CSV paths
        self.temp_single_csv_path = None
        self.temp_auto_csv_path = None

        self.temp_auto_save_enabled = False
        self.temp_auto_save_dir = str(DEFAULT_TEMP_FOLDER)

        self.temp_single_save_enabled = False
        self.temp_single_save_dir = str(DEFAULT_TEMP_FOLDER)

        self.temp_window = None
        self.temp_value_vars = {}

        self.temp_fig = None
        self.temp_ax = None
        self.temp_canvas = None

        self.temp_line_cvd = None
        self.temp_line_table = None

        # ----------------------------------------------------
        # GUI variables
        # ----------------------------------------------------

        self.port_var = tk.StringVar()

        self.baud_var = tk.StringVar(
            value=str(DEFAULT_BAUDRATE)
        )

        self.mode_var = tk.StringVar(
            value="M"
        )

        # M channel selection
        self.channel_vars = {
            ch: tk.BooleanVar(value=True)
            for ch in M_CHANNELS
        }

        self.channel_checkbuttons = {}

        # R path selection
        default_r_pairs = {
            (1, 2),
            (3, 4)
        }

        self.r_pair_vars = {
            (tx, rx): tk.BooleanVar(
                value=(tx, rx) in default_r_pairs
            )
            for tx in range(1, 5)
            for rx in range(1, 5)
            if tx != rx
        }

        self.r_pair_checkbuttons = {}

        # Display
        self.plot_voltage_var = tk.BooleanVar(
            value=False
        )

        self.time_axis_var = tk.BooleanVar(
            value=False
        )

        self.display_fs_var = tk.StringVar(
            value=f"{ADC_FS_HZ:.0f}"
        )

        # Fixed hardware parameters
        self.bits_var = tk.StringVar(
            value=str(DEFAULT_ADC_BITS)
        )

        self.vref_var = tk.StringVar(
            value=str(DEFAULT_VREF)
        )

        self.timeout_var = tk.StringVar(
            value=str(DEFAULT_TIMEOUT)
        )

        # UGW saving
        self.save_var = tk.BooleanVar(
            value=True
        )

        self.save_dir_var = tk.StringVar(
            value=str(DEFAULT_UGW_FOLDER)
        )

        # UGW automatic acquisition
        self.auto_var = tk.BooleanVar(
            value=False
        )

        self.interval_var = tk.StringVar(
            value=str(DEFAULT_UGW_INTERVAL)
        )

        # Temperature
        self.temp_interval_var = tk.StringVar(
            value=str(DEFAULT_TEMP_INTERVAL)
        )

        self.temp_save_var = tk.BooleanVar(
            value=False
        )

        self.temp_save_dir_var = tk.StringVar(
            value=str(DEFAULT_TEMP_FOLDER)
        )
        self.tof_dir = ""

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        self.connection_var = tk.StringVar(
            value="未连接"
        )

        self.status_var = tk.StringVar(
            value="等待连接 STM32"
        )

        self.count_var = tk.StringVar(
            value="UGW 采集次数：0"
        )

        self.temp_status_var = tk.StringVar(
            value="温度：尚未采集"
        )

        self.power_status_var = tk.StringVar(
            value="UGW电源：状态未知"
        )

        self.build_ui()

        self.refresh_ports()

        self.update_mode()

        self.update_auto()

        self.update_buttons()

        self.event_job = self.parent.after(50, self.process_events)


    # ========================================================
    # 3. GUI
    # ========================================================

    def _preference_variables(self):
        return {
            "port": self.port_var,
            "baud": self.baud_var,
            "mode": self.mode_var,
            "ugw_interval": self.interval_var,
            "temperature_interval": self.temp_interval_var,
            "auto_ugw": self.auto_var,
            "save_mat": self.save_var,
            "save_temperature_csv": self.temp_save_var,
            "plot_voltage": self.plot_voltage_var,
            "use_time_axis": self.time_axis_var,
            "display_sample_rate": self.display_fs_var,
            "receive_timeout": self.timeout_var,
            **{f"m_tx{channel}": variable for channel, variable in self.channel_vars.items()},
            **{f"r_tx{tx}_rx{rx}": variable for (tx, rx), variable in self.r_pair_vars.items()},
        }

    def get_preferences(self):
        return {name: variable.get() for name, variable in self._preference_variables().items()}

    def restore_preferences(self, preferences):
        if not isinstance(preferences, dict):
            return

        for name, variable in self._preference_variables().items():
            value = preferences.get(name)
            if type(value) is not type(variable.get()) or value == "":
                continue
            if name == "port" and value not in self.port_combo["values"]:
                continue
            if name == "mode" and value not in ("M", "R"):
                continue
            variable.set(value)

        self.update_mode()
        self.update_auto()

    def build_ui(self):
        p = self.px
        configure_panel_styles(self.parent, p)
        self.layout = ttk.Frame(self.parent, padding=p(10))
        self.layout.pack(fill=tk.BOTH, expand=True)

        serial_panel = ttk.LabelFrame(self.layout, text="串口连接", style="Monitor.TLabelframe", padding=p(8))
        serial_panel.pack(fill=tk.X, pady=(0, p(7)))
        self._build_serial_bar(serial_panel)

        self.control_grid = ttk.Frame(self.layout)
        self.control_grid.pack(fill=tk.X)
        self.control_grid.columnconfigure(0, weight=5)
        self.control_grid.columnconfigure(1, weight=6)

        self.settings_panel = ttk.LabelFrame(
            self.control_grid, text="超声设置", style="Monitor.TLabelframe", padding=p(8)
        )
        self._build_ugw_settings(self.settings_panel)
        self.actions_panel = ttk.Frame(self.control_grid)

        temperature = ttk.LabelFrame(
            self.actions_panel, text="温度采集", style="Monitor.TLabelframe", padding=p(8)
        )
        temperature.pack(fill=tk.X, pady=(0, p(7)))
        self._build_temperature_controls(temperature)

        operations = ttk.LabelFrame(
            self.actions_panel, text="超声采集", style="Monitor.TLabelframe", padding=p(8)
        )
        operations.pack(fill=tk.X, expand=True, anchor=tk.N)
        self._build_ugw_controls(operations)
        self.controls_stacked = None
        self.reflow_controls(self.parent.winfo_screenwidth())

        self.waveform_panel = ttk.LabelFrame(
            self.layout, text="超声实时波形", style="Monitor.TLabelframe", padding=p(5)
        )
        self.waveform_panel.pack(fill=tk.BOTH, expand=True, pady=p(7))
        self.fig = Figure(figsize=(8, 3.5), dpi=95, facecolor=PANEL_BG)
        self._empty_waveform()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.waveform_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        log_panel = ttk.LabelFrame(self.layout, text="运行日志", style="Monitor.TLabelframe", padding=p(5))
        log_panel.pack(fill=tk.X)
        log_scroll = ttk.Scrollbar(log_panel, orient=tk.VERTICAL)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text = tk.Text(
            log_panel, height=4, width=1, state=tk.DISABLED, font=("Consolas", 9),
            background=PANEL_BG, foreground="#334155", relief=tk.FLAT, borderwidth=0,
            highlightthickness=0, padx=p(4), pady=p(2), yscrollcommand=log_scroll.set,
        )
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.configure(command=self.log_text.yview)
        self.log("请先连接 STM32。采集超声前，请打开 UGW 电源。")

    def _build_serial_bar(self, panel):
        p = self.px
        ttk.Label(panel, text="串口", style="Monitor.TLabel").pack(side=tk.LEFT)
        self.port_combo = ttk.Combobox(panel, textvariable=self.port_var, width=9, state="readonly")
        self.port_combo.pack(side=tk.LEFT, padx=p(4))
        self.refresh_button = self._button(panel, "刷新", self.refresh_ports)
        self.refresh_button.pack(side=tk.LEFT)
        ttk.Label(panel, text="波特率", style="Monitor.TLabel").pack(side=tk.LEFT, padx=(p(10), p(4)))
        self.baud_combo = ttk.Combobox(
            panel, textvariable=self.baud_var, width=10,
            values=("115200", "230400", "460800", "921600", "1382400", "2000000"),
        )
        self.baud_combo.pack(side=tk.LEFT)
        self.connect_button = self._button(panel, "连接", self.connect_serial, primary=True)
        self.connect_button.pack(side=tk.LEFT, padx=p(4))
        self.disconnect_button = self._button(panel, "断开", self.disconnect_serial)
        self.disconnect_button.pack(side=tk.LEFT)
        ttk.Label(panel, textvariable=self.connection_var, style="Monitor.Muted.TLabel").pack(
            side=tk.RIGHT, padx=(p(8), 0)
        )

    def _build_ugw_settings(self, panel):
        p = self.px
        mode = ttk.Frame(panel, style="Monitor.TFrame", padding=p(4))
        mode.pack(fill=tk.X)
        ttk.Label(mode, text="采集模式", style="Monitor.Section.TLabel").pack(
            anchor=tk.W, pady=(0, p(5))
        )

        choices = ttk.Frame(mode, style="Monitor.TFrame")
        choices.pack(fill=tk.X, pady=(0, p(6)))
        for value, title in (("M", "M · 三路依次采集"), ("R", "R · 多路径单发单收")):
            ttk.Radiobutton(
                choices, text=title, variable=self.mode_var, value=value,
                command=self.update_mode, style="Monitor.TRadiobutton",
            ).pack(side=tk.LEFT, padx=(0, p(12)))

        self.m_options = ttk.Frame(mode, style="Monitor.TFrame")
        self.r_options = ttk.Frame(mode, style="Monitor.TFrame")
        m_channels = ttk.Frame(self.m_options, style="Monitor.TFrame")
        m_channels.pack(fill=tk.X)
        for ch in M_CHANNELS:
            check = ttk.Checkbutton(
                m_channels, text=f"Tx{ch} → Rx4", variable=self.channel_vars[ch],
                command=self.on_channel_selection_change, style="Monitor.TCheckbutton",
            )
            check.pack(side=tk.LEFT, padx=(0, p(10)))
            self.channel_checkbuttons[ch] = check
        ttk.Label(
            self.m_options, text="每轮接收三路，勾选项用于绘图和保存。", style="Monitor.Muted.TLabel"
        ).pack(anchor=tk.W, pady=(p(5), 0))

        pair_grid = ttk.Frame(self.r_options, style="Monitor.TFrame")
        pair_grid.pack(anchor=tk.W)
        for row, tx in enumerate(range(1, 5)):
            ttk.Label(pair_grid, text=f"Tx{tx}", style="Monitor.TLabel").grid(
                row=row, column=0, sticky=tk.W, padx=(0, p(9))
            )
            for rx in range(1, 5):
                if tx == rx:
                    ttk.Label(pair_grid, text="—", style="Monitor.Muted.TLabel").grid(
                        row=row, column=rx, padx=p(5)
                    )
                else:
                    check = ttk.Checkbutton(
                        pair_grid, text=f"R{tx}{rx}", variable=self.r_pair_vars[tx, rx],
                        style="Monitor.TCheckbutton",
                    )
                    check.grid(row=row, column=rx, sticky=tk.W, padx=p(5))
                    self.r_pair_checkbuttons[tx, rx] = check
        pair_buttons = ttk.Frame(self.r_options, style="Monitor.TFrame")
        pair_buttons.pack(anchor=tk.W, pady=(p(4), 0))
        self._button(pair_buttons, "全选", self.select_all_r_pairs).pack(side=tk.LEFT, padx=(0, p(5)))
        self._button(pair_buttons, "清空", self.clear_r_pairs).pack(side=tk.LEFT)

        ttk.Separator(panel).pack(fill=tk.X, pady=p(8))
        display = ttk.Frame(panel, style="Monitor.TFrame", padding=p(4))
        display.pack(fill=tk.X)
        ttk.Label(display, text="波形设置", style="Monitor.Section.TLabel").grid(
            row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, p(5))
        )
        for column, (label, variable) in enumerate((
            ("显示电压", self.plot_voltage_var), ("使用时间轴 / μs", self.time_axis_var),
        )):
            ttk.Checkbutton(
                display, text=label, variable=variable, command=self.plot_latest, style="Monitor.TCheckbutton",
            ).grid(row=1, column=column * 2, columnspan=2, sticky=tk.W)
        for index, (label, variable, state) in enumerate((
            ("显示采样率 / Hz", self.display_fs_var, "normal"),
            ("接收超时 / s", self.timeout_var, "normal"),
            ("ADC 位数", self.bits_var, "readonly"),
            ("参考电压 / V", self.vref_var, "readonly"),
        )):
            row, column = 2 + index // 2, 2 * (index % 2)
            ttk.Label(display, text=label, style="Monitor.TLabel").grid(
                row=row, column=column, sticky=tk.W, padx=(0, p(4)), pady=p(3)
            )
            ttk.Entry(display, textvariable=variable, width=8, state=state).grid(
                row=row, column=column + 1, sticky=tk.W, padx=(0, p(8))
            )
        ttk.Label(
            display, text="显示采样率只影响绘图时间轴。", style="Monitor.Muted.TLabel",
        ).grid(row=4, column=0, columnspan=4, sticky=tk.W)

    def _build_temperature_controls(self, panel):
        p = self.px
        timing = ttk.Frame(panel, style="Monitor.TFrame")
        timing.pack(fill=tk.X)
        self.temp_once_button = self._button(timing, "单次测温", self.temperature_once)
        self.temp_once_button.pack(side=tk.LEFT, padx=(0, p(5)))
        self._button(timing, "温度趋势", self.show_temperature_window).pack(side=tk.LEFT)
        ttk.Label(timing, text="周期 / s", style="Monitor.TLabel").pack(side=tk.LEFT, padx=(p(10), p(4)))
        ttk.Entry(timing, textvariable=self.temp_interval_var, width=6).pack(side=tk.LEFT)
        self.temp_start_button = self._button(timing, "开始定时", self.start_temperature_auto, primary=True)
        self.temp_start_button.pack(side=tk.LEFT, padx=p(5))
        self.temp_stop_button = self._button(timing, "停止定时", self.stop_temperature_auto)
        self.temp_stop_button.pack(side=tk.LEFT)

        storage = ttk.Frame(panel, style="Monitor.TFrame")
        storage.pack(fill=tk.X, pady=(p(6), p(3)))
        self.temp_save_check = ttk.Checkbutton(
            storage, text="保存温度 CSV", variable=self.temp_save_var, style="Monitor.TCheckbutton"
        )
        self.temp_save_check.pack(side=tk.LEFT)
        ttk.Entry(storage, textvariable=self.temp_save_dir_var, width=12).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=p(5)
        )
        self._button(storage, "选择目录", self.choose_temp_folder).pack(side=tk.LEFT)
        ttk.Label(panel, textvariable=self.temp_status_var, style="Monitor.Muted.TLabel").pack(anchor=tk.W)

    def _build_ugw_controls(self, panel):
        p = self.px
        power = ttk.Frame(panel, style="Monitor.TFrame")
        power.pack(fill=tk.X)
        self.power_on_button = self._button(power, "打开 UGW 电源", self.turn_on_ugw_power)
        self.power_on_button.pack(side=tk.LEFT, padx=(0, p(5)))
        self.power_off_button = self._button(power, "停止并关电源", self.turn_off_ugw_power)
        self.power_off_button.pack(side=tk.LEFT)
        ttk.Label(power, textvariable=self.power_status_var, style="Monitor.Muted.TLabel").pack(
            side=tk.LEFT, padx=(p(7), 0)
        )
        ttk.Separator(panel).pack(fill=tk.X, pady=(p(6), p(3)))
        storage = ttk.Frame(panel, style="Monitor.TFrame", padding=(0, p(2)))
        storage.pack(fill=tk.X)
        storage.columnconfigure(1, weight=1)
        self.save_check = ttk.Checkbutton(
            storage, text="保存超声 MAT", variable=self.save_var, style="Monitor.TCheckbutton"
        )
        self.save_check.grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(storage, textvariable=self.save_dir_var, width=13).grid(
            row=0, column=1, sticky="ew", padx=p(5)
        )
        self._button(storage, "选择目录", self.choose_folder).grid(row=0, column=2)
        auto = ttk.Frame(storage, style="Monitor.TFrame")
        auto.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(p(6), 0))
        self.auto_check = ttk.Checkbutton(
            auto, text="启用自动采集", variable=self.auto_var,
            command=self.update_auto, style="Monitor.TCheckbutton",
        )
        self.auto_check.pack(side=tk.LEFT)
        ttk.Label(auto, text="周期 / s", style="Monitor.TLabel").pack(side=tk.LEFT, padx=(p(10), p(4)))
        ttk.Entry(auto, textvariable=self.interval_var, width=7).pack(side=tk.LEFT)
        ttk.Label(
            storage, text="自动采集保存 MAT，R 模式按路径分别保存。", style="Monitor.Muted.TLabel",
        ).grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(p(4), 0))

        actions = ttk.Frame(panel, style="Monitor.TFrame")
        actions.pack(fill=tk.X, pady=(p(6), p(3)))
        self.once_button = self._button(actions, "单轮采集", self.acquire_once, primary=True)
        self.auto_start_button = self._button(actions, "开始自动 M / R", self.start_auto, primary=True)
        self.stop_button = self._button(actions, "停止 UGW", self.stop_acquisition)
        self.clear_button = self._button(actions, "清空波形", self.clear_plot)
        self.tof_button = self._button(actions, "文件夹 ToF 分析", self.analyze_tof_folder)
        for button in (self.once_button, self.auto_start_button, self.stop_button, self.clear_button, self.tof_button):
            button.pack(side=tk.LEFT, padx=(0, p(5)))
        status = ttk.Frame(panel, style="Monitor.TFrame")
        status.pack(fill=tk.X)
        ttk.Label(status, textvariable=self.count_var, style="Monitor.Muted.TLabel").pack(side=tk.LEFT)
        self._button(status, "次数清零", self.reset_acquisition_count).pack(side=tk.LEFT, padx=(p(6), 0))
        ttk.Label(status, textvariable=self.status_var, style="Monitor.Muted.TLabel",
                  wraplength=p(270)).pack(side=tk.LEFT, padx=(p(8), 0))

    @staticmethod
    def _button(parent, text, command, *, primary=False):
        style = "Monitor.Primary.TButton" if primary else "Monitor.TButton"
        return ttk.Button(parent, text=text, command=command, style=style, width=0)

    def reflow_controls(self, available_width):
        gap = self.px(7)
        required_width = self.settings_panel.winfo_reqwidth() + self.actions_panel.winfo_reqwidth()
        stacked = available_width < required_width + self.px(24) + gap
        if stacked == self.controls_stacked:
            return False
        self.controls_stacked = stacked
        if stacked:
            self.settings_panel.grid(row=0, column=0, columnspan=2, sticky="ew", padx=0)
            self.actions_panel.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(gap, 0))
        else:
            self.settings_panel.grid(row=0, column=0, columnspan=1, sticky="nsew", padx=(0, gap))
            self.actions_panel.grid(row=0, column=1, columnspan=1, sticky="nsew", pady=0)
        return True

    def minimum_size(self):
        controls = [panel for panel in self.layout.winfo_children() if panel is not self.waveform_panel]
        width = max(panel.winfo_reqwidth() for panel in controls)
        height = sum(panel.winfo_reqheight() for panel in controls)
        return width + self.px(20), height + self.px(300)

    def _empty_waveform(self):
        self.fig.clear()
        axis = self.fig.add_subplot(111)
        axis.set_axis_off()
        axis.text(.5, .5, "等待超声采集", transform=axis.transAxes,
                  ha="center", va="center", color="#64748b", fontsize=11)


    # ========================================================
    # 4. Mode / channel selection
    # ========================================================

    def update_mode(self):
        if self.mode_var.get() == "M":
            self.r_options.pack_forget()
            self.m_options.pack(fill=tk.X)
        else:
            self.m_options.pack_forget()
            self.r_options.pack(fill=tk.X)
        self.parent.event_generate("<<MonitorLayoutChanged>>", when="tail")

    def on_channel_selection_change(self):

        selected = [
            ch for ch in M_CHANNELS
            if self.channel_vars[ch].get()
        ]

        self.log(
            f"M 模式显示/保存通道：{selected}"
        )


    def select_all_r_pairs(self):

        for var in self.r_pair_vars.values():
            var.set(True)


    def clear_r_pairs(self):

        for var in self.r_pair_vars.values():
            var.set(False)


    def selected_r_pairs(self):

        return [
            pair
            for pair, var in self.r_pair_vars.items()
            if var.get()
        ]


    def update_auto(self):

        if self.auto_var.get():

            self.save_var.set(True)

            self.save_check.state(
                ["disabled"]
            )

        else:

            self.save_check.state(
                ["!disabled"]
            )

        self.update_buttons()


    # ========================================================
    # 5. Serial management
    # ========================================================

    def refresh_ports(self):

        ports = [
            p.device
            for p in list_ports.comports()
        ]

        self.port_combo["values"] = ports

        if self.port_var.get() not in ports:

            self.port_var.set(
                ports[0] if ports else ""
            )

        self.log(
            f"可用串口：{ports}"
        )


    def connect_serial(self):

        if self.ser is not None and self.ser.is_open:
            return

        port = self.port_var.get().strip()

        if not port:

            messagebox.showwarning(
                "串口错误",
                "请先选择串口。"
            )

            return

        try:

            baud = int(
                self.baud_var.get()
            )

            self.ser = serial.Serial(

                port=port,

                baudrate=baud,

                bytesize=serial.EIGHTBITS,

                parity=serial.PARITY_NONE,

                stopbits=serial.STOPBITS_ONE,

                timeout=0.05,

                write_timeout=1.0

            )

            self.ser.reset_input_buffer()

            self.power_enabled = None

            self.power_status_var.set(
                "UGW电源：状态未知，请发送 v"
            )

            self.connection_var.set(
                f"已连接：{port} @ {baud}"
            )

            self.status_var.set(
                "串口已连接"
            )

            self.log(
                f"串口连接成功：{port} @ {baud}"
            )

        except Exception as exc:

            self.ser = None

            messagebox.showerror(
                "连接失败",
                str(exc)
            )

        self.update_buttons()


    def disconnect_serial(self):

        if self.worker is not None and self.worker.is_alive():

            messagebox.showwarning(
                "正在采集",
                "请先停止 UGW 自动采集。"
            )

            return

        with self.state_lock:

            if self.temp_active or self.temp_auto_running:

                messagebox.showwarning(
                    "温度采集中",
                    "请先停止温度定时采集并等待单次温度读取结束。"
                )

                return

        if self.power_busy:

            messagebox.showwarning(
                "电源操作中",
                "请等待电源操作完成。"
            )

            return

        if self.ser is not None:

            try:
                self.ser.close()
            except Exception:
                pass

        self.ser = None

        self.power_enabled = None

        self.power_status_var.set(
            "UGW电源：状态未知"
        )

        self.connection_var.set(
            "未连接"
        )

        self.status_var.set(
            "串口已断开"
        )

        self.log(
            "串口已断开。"
        )

        self.update_buttons()


    def choose_folder(self):

        folder = filedialog.askdirectory(
            parent=self.parent,
            title="选择 UGW MAT 数据保存目录",
            initialdir=self.save_dir_var.get() or str(DEFAULT_UGW_FOLDER)
        )

        if folder:

            self.save_dir_var.set(folder)

            self.log(
                f"UGW 保存目录：{folder}"
            )


    def choose_temp_folder(self):

        folder = filedialog.askdirectory(
            parent=self.parent,
            title="选择温度 CSV 保存目录",
            initialdir=self.temp_save_dir_var.get() or str(DEFAULT_TEMP_FOLDER)
        )

        if folder:

            self.temp_save_dir_var.set(folder)

            self.log(
                f"温度 CSV 保存目录：{folder}"
            )


    # ========================================================
    # 6. Power control
    # ========================================================

    def turn_on_ugw_power(self):

        if self.ser is None or not self.ser.is_open:

            messagebox.showwarning(
                "串口未连接",
                "请先连接 STM32。"
            )

            return

        if self.power_enabled is True:

            messagebox.showinfo(
                "UGW 电源",
                "已经发送过开启指令 v。"
            )

            return

        self.start_power_command(
            POWER_ON_COMMAND,
            True
        )


    def turn_off_ugw_power(self):

        if self.ser is None or not self.ser.is_open:

            messagebox.showwarning(
                "串口未连接",
                "请先连接 STM32。"
            )

            return

        if not messagebox.askyesno(
            "确认关闭",
            "将发送 0 指令。\n\n"
            "该指令会停止 UGW、ADC、DAC 等相关工作，"
            "并关闭两路供电使能。\n\n"
            "确定继续吗？"
        ):

            return

        self.start_power_command(
            POWER_OFF_COMMAND,
            False
        )


    def start_power_command(
        self,
        command,
        new_state
    ):

        if self.power_busy:
            return

        worker_busy = (
            self.worker is not None
            and self.worker.is_alive()
        )

        with self.state_lock:

            serial_busy = (
                self.ugw_requested
                or self.ugw_active
                or self.temp_active
            )

        if (
            worker_busy
            or serial_busy
            or self.temp_auto_running
        ):

            messagebox.showwarning(
                "设备忙",
                "请先停止 UGW 和温度定时采集，"
                "等待当前串口事务完成后再控制电源。"
            )

            return

        self.power_busy = True

        self.update_buttons()

        threading.Thread(
            target=self.power_command_worker,
            args=(
                self.ser,
                command,
                new_state
            ),
            daemon=True
        ).start()


    def power_command_worker(
        self,
        ser,
        command,
        new_state
    ):

        acquired = False

        try:

            acquired = self.serial_lock.acquire(
                timeout=2.0
            )

            if not acquired:

                raise RuntimeError(
                    "串口正被占用，电源指令未发送。"
                )

            if ser is not self.ser or not ser.is_open:

                raise RuntimeError(
                    "串口连接已经失效。"
                )

            ser.write(command)

            ser.flush()

            # v -> hv_enable_request
            # HV_Run performs two GPIO operations with delays.
            if command == POWER_ON_COMMAND:

                time.sleep(0.20)

            else:

                time.sleep(0.05)

            self.event_queue.put(
                (
                    "power_result",
                    new_state,
                    command.decode("ascii")
                )
            )

        except Exception as exc:

            self.event_queue.put(
                ("power_error", str(exc))
            )

        finally:

            if acquired:
                self.serial_lock.release()

            self.event_queue.put(
                ("power_finished",)
            )


    # ========================================================
    # 7. Build UGW configuration
    # ========================================================

    def build_config(self, auto):

        mode = self.mode_var.get()

        if mode == "M":

            selected = [
                ch for ch in M_CHANNELS
                if self.channel_vars[ch].get()
            ]

            if not selected:

                raise ValueError(
                    "M 模式至少勾选一个显示/保存通道。"
                )

            selected_indices = [
                ch - 1 for ch in selected
            ]

            selected_labels = [
                f"Tx{ch} -> Rx4"
                for ch in selected
            ]

            pairs = []

            samples = M_SAMPLES_PER_PATH

            total_channels = 3

            total_bytes = M_TOTAL_BYTES

        else:

            pairs = self.selected_r_pairs()

            if not pairs:

                raise ValueError(
                    "R 模式至少勾选一组 Tx-Rx 路径。"
                )

            selected = list(
                range(1, len(pairs) + 1)
            )

            selected_indices = list(
                range(len(pairs))
            )

            selected_labels = [
                f"Tx{tx} -> Rx{rx}"
                for tx, rx in pairs
            ]

            samples = R_SAMPLES_PER_PATH

            total_channels = len(pairs)

            total_bytes = R_BYTES_PER_PATH * len(pairs)

        display_fs = float(
            self.display_fs_var.get()
        )

        # Hardware parameters are fixed.
        # GUI cannot modify the actual ADC interpretation.
        bits = DEFAULT_ADC_BITS
        vref = DEFAULT_VREF

        timeout = float(
            self.timeout_var.get()
        )

        baud = int(
            self.baud_var.get()
        )

        if display_fs <= 0:

            raise ValueError(
                "显示采样率必须大于零。"
            )

        if timeout <= 0:

            raise ValueError(
                "接收超时时间必须大于零。"
            )

        interval = DEFAULT_UGW_INTERVAL

        if auto:

            interval = float(
                self.interval_var.get()
            )

            if interval <= 0:

                raise ValueError(
                    "UGW 自动采集周期必须大于零。"
                )

        save_enabled = bool(
            self.save_var.get() or auto
        )

        save_folder = Path(
            self.save_dir_var.get()
        ).expanduser()

        if save_enabled:

            save_folder.mkdir(
                parents=True,
                exist_ok=True
            )

        return {

            "mode": mode,

            "samples": samples,

            "total_channels": total_channels,

            "total_bytes_expected": total_bytes,

            "selected_channels": selected,

            "selected_indices": selected_indices,

            "selected_labels": selected_labels,

            "r_pairs": pairs,

            "display_fs": display_fs,

            "actual_fs": ADC_FS_HZ,

            "bits": bits,

            "vref": vref,

            "adc_max": 2 ** bits - 1,

            "timeout": timeout,

            "baud": baud,

            "interval": interval,

            "auto": auto,

            "save_enabled": save_enabled,

            "save_folder": save_folder,

            "plot_voltage": self.plot_voltage_var.get(),

            "time_axis": self.time_axis_var.get(),

            "port": self.port_var.get(),

        }


    # ========================================================
    # 8. UGW acquisition control
    # ========================================================

    def acquire_once(self):

        if self.auto_var.get():

            messagebox.showinfo(
                "自动采集模式",
                "请使用‘开始自动 M / R’按钮。"
            )

            return

        self.start_worker(
            auto=False
        )


    def start_auto(self):

        if not self.auto_var.get():

            messagebox.showwarning(
                "自动采集未启用",
                "请先勾选启用自动采集。"
            )

            return

        self.save_var.set(True)

        self.start_worker(
            auto=True
        )


    def start_worker(self, auto):

        if self.ser is None or not self.ser.is_open:

            messagebox.showwarning(
                "串口未连接",
                "请先连接 STM32。"
            )

            return

        if self.power_busy:

            messagebox.showwarning(
                "电源操作中",
                "请等待电源操作完成。"
            )

            return

        if self.power_enabled is not True:

            messagebox.showwarning(
                "UGW 电源未开启",
                "请先点击「打开 UGW 电源 (v)」。"
            )

            return

        if self.worker is not None and self.worker.is_alive():
            return

        try:

            cfg = self.build_config(auto)

        except Exception as exc:

            messagebox.showerror(
                "参数错误",
                str(exc)
            )

            return

        self.stop_event = threading.Event()

        self.worker = threading.Thread(

            target=self.worker_main,

            args=(
                self.ser,
                cfg,
                self.stop_event
            ),

            daemon=True

        )

        self.worker.start()

        if cfg["mode"] == "M":

            desc = "M（Tx1/Tx2/Tx3 → Rx4）"

        else:

            desc = "R（" + ", ".join(
                f"R{tx}{rx}"
                for tx, rx in cfg["r_pairs"]
            ) + "）"

        self.log(
            f"启动 {desc}{' 自动采集' if auto else ' 单轮采集'}。"
        )

        self.status_var.set(
            "UGW 采集任务启动"
        )

        self.update_buttons()


    # ========================================================
    # 9. Exact serial reception
    # ========================================================

    @staticmethod
    def read_exactly(
        ser,
        byte_count,
        timeout,
        stop_event
    ):

        received = bytearray()

        start = time.monotonic()

        while len(received) < byte_count:

            if stop_event.is_set():

                raise InterruptedError(
                    "用户停止 UGW 采集"
                )

            if time.monotonic() - start > timeout:

                raise TimeoutError(
                    f"串口接收超时："
                    f"{len(received)}/{byte_count} Bytes"
                )

            chunk = ser.read(
                min(4096, byte_count - len(received))
            )

            if chunk:
                received.extend(chunk)

        return bytes(received)


    @staticmethod
    def read_done_marker(
        ser,
        expected,
        timeout=2.0
    ):

        buf = bytearray()

        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:

            chunk = ser.read(1)

            if chunk:

                buf.extend(chunk)

                if expected.encode("ascii") in buf:
                    break

                if len(buf) > 512:
                    del buf[:-256]

        decoded = buf.decode(
            "ascii",
            errors="replace"
        ).strip()

        if expected not in decoded:

            raise TimeoutError(
                f"接收完二进制数据后未收到 {expected} 完成标志；"
                f"为避免下一条指令与 MCU 冲突，已停止本轮任务。"
                f"收到内容：{decoded!r}"
            )

        return decoded


    # ========================================================
    # 10. Result construction
    # ========================================================

    def _base_result(
        self,
        cfg,
        command,
        raw,
        labels,
        tx_channels,
        rx_channel,
        transmitted_channels,
        status,
        bytes_received
    ):

        raw = np.asarray(
            raw,
            dtype=np.uint16
        )

        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)

        voltage = (
            raw.astype(np.float64)
            / cfg["adc_max"]
            * cfg["vref"]
        )

        samples = raw.shape[0]

        # Actual time metadata is always based on hardware Fs.
        sample_index = np.arange(
            samples,
            dtype=np.int64
        )

        time_us = (
            sample_index.astype(np.float64)
            / cfg["actual_fs"]
            * 1e6
        )

        invalid = int(
            np.sum(raw > cfg["adc_max"])
        )

        return {

            "Command": command,

            "Mode": cfg["mode"],

            "FixedRxChannel": int(rx_channel),

            "TxChannels": list(tx_channels),

            "ChannelNumbers": list(
                range(1, raw.shape[1] + 1)
            ),

            "ChannelLabels": list(labels),

            "TransmittedChannels": list(
                transmitted_channels
            ),

            "TransmittedChannelCount": int(
                len(transmitted_channels)
            ),

            "SelectedChannelCount": int(
                raw.shape[1]
            ),

            "BytesReceived": int(
                bytes_received
            ),

            "SamplesPerPacket": int(
                samples
            ),

            "ADC_Bits": int(
                cfg["bits"]
            ),

            "ADC_Vref_V": float(
                cfg["vref"]
            ),

            "ADC_raw": raw,

            "Voltage_V": voltage,

            "UseTimeAxis": True,

            "Fs_Hz": float(
                ADC_FS_HZ
            ),

            "DisplayFs_Hz": float(
                ADC_FS_HZ
            ),

            "XData": time_us,

            "SampleIndex": sample_index,

            "Time_us": time_us,

            "STM32_Message": status,

            "SerialPort": cfg["port"],

            "BaudRate": int(
                cfg["baud"]
            ),

            "AcquisitionTime": datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            ),

            "InvalidCount": invalid,

            "InvalidCountAllChannels": invalid,

        }


    # ========================================================
    # 11. M mode
    # ========================================================

    def acquire_m_data(
        self,
        ser,
        cfg,
        stop_event
    ):

        ser.reset_input_buffer()

        ser.write(b"M")

        ser.flush()

        self.event_queue.put(
            (
                "log",
                "发送 M；等待 3×800 个 uint16（4800 Bytes）"
            )
        )

        raw_bytes = self.read_exactly(
            ser,
            M_TOTAL_BYTES,
            cfg["timeout"],
            stop_event
        )

        expected_tail = b"M_DONE\r\n"
        try:
            tail = self.read_exactly(ser, len(expected_tail), 2.0, stop_event)
        except TimeoutError as exc:
            raise TimeoutError(f"M 帧尾缺失或不完整，已停止本轮采集。{exc}") from exc
        if tail != expected_tail:
            raise RuntimeError(
                f"M 帧尾错误，已停止本轮采集。"
                f"预期 {expected_tail.hex(' ').upper()}，"
                f"实际收到 {tail.hex(' ').upper()}。"
            )

        # C array:
        # uint16_t ugw_m_frame.samples[3][800]
        #
        # Memory layout:
        # channel1, channel2, channel3

        raw_all = (
            np.frombuffer(
                raw_bytes,
                dtype="<u2"
            )
            .copy()
            .reshape(3, M_SAMPLES_PER_PATH)
            .T
        )

        # Filter only after receiving all three channels.
        selected = cfg["selected_indices"]

        raw = raw_all[:, selected].copy()

        labels = [
            f"Tx{ch} -> Rx4"
            for ch in cfg["selected_channels"]
        ]

        result = self._base_result(

            cfg,

            "M",

            raw,

            labels,

            cfg["selected_channels"],

            4,

            [1, 2, 3],

            tail.decode("ascii").strip(),

            M_TOTAL_BYTES

        )

        result["ChannelNumbers"] = list(
            cfg["selected_channels"]
        )

        result["TransmittedChannelCount"] = 3

        result["SelectedChannelCount"] = len(
            cfg["selected_channels"]
        )

        result["InvalidCountAllChannels"] = int(
            np.sum(raw_all > cfg["adc_max"])
        )

        return result


    # ========================================================
    # 12. R mode: one Rxy command
    # ========================================================

    def acquire_r_data(
        self,
        ser,
        cfg,
        stop_event,
        tx,
        rx
    ):

        command = f"R{tx}{rx}"

        ser.reset_input_buffer()

        ser.write(
            command.encode("ascii")
        )

        ser.flush()

        self.event_queue.put(
            (
                "log",
                f"发送 {command}；等待 6400 个 uint16（12800 Bytes）"
            )
        )

        raw_bytes = self.read_exactly(
            ser,
            R_BYTES_PER_PATH,
            cfg["timeout"],
            stop_event
        )

        raw = (
            np.frombuffer(
                raw_bytes,
                dtype="<u2"
            )
            .copy()
            .reshape(-1, 1)
        )

        status = self.read_done_marker(
            ser,
            "R_DONE",
            timeout=2.0
        )

        if (
            "UGW_SINGLE_RX_ERROR" in status
            or "R error" in status
        ):

            raise RuntimeError(
                f"MCU 返回错误：{status}"
            )

        result = self._base_result(

            cfg,

            command,

            raw,

            [f"Tx{tx} -> Rx{rx}"],

            [tx],

            rx,

            [tx],

            status,

            R_BYTES_PER_PATH

        )

        result["ChannelNumbers"] = [1]

        result["TransmittedChannelCount"] = 1

        result["SelectedChannelCount"] = 1

        result["FixedRxChannel"] = rx

        result["TxChannels"] = [tx]

        return result


    # ========================================================
    # 13. Combine R results for visualization
    # ========================================================

    @staticmethod
    def combine_r_results(results, cfg):

        if len(results) == 1:
            return results[0]

        raw = np.column_stack([
            r["ADC_raw"][:, 0]
            for r in results
        ]).astype(np.uint16)

        voltage = np.column_stack([
            r["Voltage_V"][:, 0]
            for r in results
        ])

        first = results[0]

        labels = [
            r["ChannelLabels"][0]
            for r in results
        ]

        txs = [
            r["TxChannels"][0]
            for r in results
        ]

        rxs = [
            r["FixedRxChannel"]
            for r in results
        ]

        return {

            **first,

            "Command": "R_BATCH",

            "Mode": "R",

            "FixedRxChannel": -1,

            "TxChannels": txs,

            "RxChannels": rxs,

            "ChannelNumbers": list(
                range(1, len(results) + 1)
            ),

            "ChannelLabels": labels,

            "TransmittedChannels": txs,

            "TransmittedChannelCount": len(results),

            "SelectedChannelCount": len(results),

            "BytesReceived": sum(
                r["BytesReceived"]
                for r in results
            ),

            "ADC_raw": raw,

            "Voltage_V": voltage,

            "InvalidCount": int(
                sum(r["InvalidCount"] for r in results)
            ),

            "InvalidCountAllChannels": int(
                sum(r["InvalidCountAllChannels"] for r in results)
            ),

            "STM32_Message": " | ".join(
                r["STM32_Message"]
                for r in results
            ),

        }


    # ========================================================
    # 14. UGW worker
    # ========================================================

    def worker_main(
        self,
        ser,
        cfg,
        stop_event
    ):

        cycle = 0

        try:

            while not stop_event.is_set():

                cycle += 1

                cycle_start = time.monotonic()

                # Set request before obtaining the serial lock.
                # This gives UGW priority over temperature.
                with self.state_lock:
                    self.ugw_requested = True

                # One whole M transaction or one whole R batch
                # is protected by the serial lock.
                with self.serial_lock:

                    if stop_event.is_set():
                        break

                    with self.state_lock:
                        self.ugw_active = True

                    try:

                        if cfg["mode"] == "M":

                            results = [
                                self.acquire_m_data(
                                    ser,
                                    cfg,
                                    stop_event
                                )
                            ]

                        else:

                            results = []

                            for tx, rx in cfg["r_pairs"]:

                                if stop_event.is_set():

                                    raise InterruptedError(
                                        "用户停止 R 多路径采集"
                                    )

                                results.append(
                                    self.acquire_r_data(
                                        ser,
                                        cfg,
                                        stop_event,
                                        tx,
                                        rx
                                    )
                                )

                        save_paths = []

                        if cfg["save_enabled"]:

                            for result in results:

                                path = self.save_mat_file(
                                    result,
                                    cfg["save_folder"]
                                )

                                save_paths.append(
                                    str(path)
                                )

                                self.event_queue.put(
                                    (
                                        "log",
                                        f"MAT 已保存：{path}"
                                    )
                                )

                        if cfg["mode"] == "M":

                            combined = results[0]

                        else:

                            combined = self.combine_r_results(
                                results,
                                cfg
                            )

                        self.event_queue.put(
                            (
                                "result",
                                combined,
                                cycle,
                                save_paths
                            )
                        )

                    finally:

                        with self.state_lock:

                            self.ugw_active = False

                            self.ugw_requested = False

                if not cfg["auto"]:
                    break

                # Start-to-start period
                next_start = (
                    cycle_start + cfg["interval"]
                )

                wait_time = max(
                    0.0,
                    next_start - time.monotonic()
                )

                if wait_time > 0:

                    if stop_event.wait(wait_time):
                        break

        except InterruptedError as exc:

            self.event_queue.put(
                ("log", str(exc))
            )

        except Exception as exc:

            self.event_queue.put(
                ("error", str(exc))
            )

        finally:

            with self.state_lock:

                self.ugw_requested = False

                self.ugw_active = False

            self.event_queue.put(
                ("finished",)
            )


    # ========================================================
    # 15. MAT saving
    # ========================================================

    @staticmethod
    def save_mat_file(result, folder):

        folder = Path(folder)

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        stem = (
            f"UGW_{result['Command']}_{timestamp}"
        )

        mat_path = folder / f"{stem}.mat"

        UGW_result = {

            "Command": result["Command"],

            "Mode": result["Mode"],

            "FixedRxChannel": result["FixedRxChannel"],

            "TxChannels": np.asarray(
                result["TxChannels"],
                dtype=np.uint8
            ).reshape(1, -1),

            "ChannelNumbers": np.asarray(
                result["ChannelNumbers"],
                dtype=np.uint8
            ).reshape(1, -1),

            "ChannelLabels": np.asarray(
                result["ChannelLabels"],
                dtype=object
            ).reshape(1, -1),

            "TransmittedChannels": np.asarray(
                result["TransmittedChannels"],
                dtype=np.uint8
            ).reshape(1, -1),

            "TransmittedChannelCount": result[
                "TransmittedChannelCount"
            ],

            "SelectedChannelCount": result[
                "SelectedChannelCount"
            ],

            "BytesReceived": result[
                "BytesReceived"
            ],

            "SamplesPerPacket": result[
                "SamplesPerPacket"
            ],

            "ADC_Bits": result[
                "ADC_Bits"
            ],

            "ADC_Vref_V": result[
                "ADC_Vref_V"
            ],

            # Raw ADC data is preserved.
            "ADC_raw": result[
                "ADC_raw"
            ],

            # Voltage is derived using fixed hardware parameters.
            "Voltage_V": result[
                "Voltage_V"
            ],

            "UseTimeAxis": True,

            # Actual hardware sampling rate
            "Fs_Hz": ADC_FS_HZ,

            "DisplayFs_Hz": ADC_FS_HZ,

            "XData": np.asarray(
                result["XData"]
            ).reshape(-1, 1),

            "SampleIndex": np.asarray(
                result["SampleIndex"]
            ).reshape(-1, 1),

            "Time_us": np.asarray(
                result["Time_us"]
            ).reshape(-1, 1),

            "STM32_Message": result[
                "STM32_Message"
            ],

            "SerialPort": result[
                "SerialPort"
            ],

            "BaudRate": result[
                "BaudRate"
            ],

            "AcquisitionTime": result[
                "AcquisitionTime"
            ],

            "InvalidCount": result[
                "InvalidCount"
            ],

            "InvalidCountAllChannels": result[
                "InvalidCountAllChannels"
            ],

        }

        if "RxChannels" in result:

            UGW_result["RxChannels"] = np.asarray(
                result["RxChannels"],
                dtype=np.uint8
            ).reshape(1, -1)

        savemat(
            str(mat_path),
            {"UGW_result": UGW_result},
            do_compression=True,
            long_field_names=True
        )

        return mat_path


    # ========================================================
    # 16. Temperature acquisition
    # ========================================================

    def _ugw_busy_snapshot(self):

        with self.state_lock:

            return bool(
                self.ugw_requested
                or self.ugw_active
            )


    def _temperature_transaction(self, ser):

        # Called only while holding serial_lock.
        ser.reset_input_buffer()

        ser.write(TEMP_COMMAND)

        ser.flush()

        deadline = (
            time.monotonic()
            + TEMP_RESPONSE_TIMEOUT
        )

        line_buf = bytearray()

        while time.monotonic() < deadline:

            chunk = ser.read(1)

            if not chunk:
                continue

            if chunk in (b"\n", b"\r"):

                if not line_buf:
                    continue

                line = line_buf.decode(
                    "ascii",
                    errors="replace"
                ).strip()

                line_buf.clear()

                if line.startswith(TEMP_RESPONSE_PREFIX):

                    fields = line.split(",")

                    if len(fields) < 5:
                        continue

                    return {

                        "PC_Time": datetime.now().astimezone().isoformat(
                            timespec="milliseconds"
                        ),

                        "MCU_Tick_ms": int(
                            float(fields[1])
                        ),

                        "Resistance_Ohm": float(
                            fields[2]
                        ),

                        "Temperature_CVD_C": float(
                            fields[3]
                        ),

                        "Temperature_Table_C": float(
                            fields[4]
                        ),

                        "Status": "OK",

                    }

            else:

                line_buf.extend(chunk)

                if len(line_buf) > 256:
                    line_buf.clear()

        raise TimeoutError(
            "发送 Y 后 2.5 s 内没有收到 T, 开头的温度响应。"
        )


    # --------------------------------------------------------
    # Temperature CSV
    # --------------------------------------------------------

    def _append_temp_csv(self, row, path):

        if not path:
            return

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        new_file = not path.exists()

        fieldnames = [
            "Index",
            "ScheduledPC_Time",
            "PC_Time",
            "MCU_Tick_ms",
            "Resistance_Ohm",
            "Temperature_CVD_C",
            "Temperature_Table_C",
            "Status",
            "Elapsed_s"
        ]

        with open(
            path,
            "a",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            if new_file:
                writer.writeheader()

            writer.writerow(row)


    def _ensure_temp_single_csv_path(self):

        if self.temp_single_csv_path is None:

            folder = Path(
                self.temp_single_save_dir
            ).expanduser()

            folder.mkdir(
                parents=True,
                exist_ok=True
            )

            stamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )

            self.temp_single_csv_path = (
                folder / f"Temperature_Single_{stamp}.csv"
            )

        return self.temp_single_csv_path


    def _record_temp_auto(
        self,
        index,
        scheduled_time,
        result=None,
        status="SKIPPED"
    ):

        with self.temp_data_lock:

            if self.temp_auto_start_mono is None:

                elapsed = 0.0

            else:

                elapsed = max(
                    0.0,
                    time.monotonic() - self.temp_auto_start_mono
                )

            row = {

                "Index": index,

                "ScheduledPC_Time": scheduled_time,

                "PC_Time": (
                    result["PC_Time"]
                    if result else ""
                ),

                "MCU_Tick_ms": (
                    result["MCU_Tick_ms"]
                    if result else ""
                ),

                "Resistance_Ohm": (
                    result["Resistance_Ohm"]
                    if result else np.nan
                ),

                "Temperature_CVD_C": (
                    result["Temperature_CVD_C"]
                    if result else np.nan
                ),

                "Temperature_Table_C": (
                    result["Temperature_Table_C"]
                    if result else np.nan
                ),

                "Status": (
                    result["Status"]
                    if result else status
                ),

                "Elapsed_s": elapsed,

            }

            self.temp_samples.append(row)

        if (
            self.temp_auto_save_enabled
            and self.temp_auto_csv_path
        ):

            self._append_temp_csv(
                row,
                self.temp_auto_csv_path
            )

        self.event_queue.put(
            ("temp_auto_sample", row)
        )


    # --------------------------------------------------------
    # Single temperature acquisition
    # --------------------------------------------------------

    def temperature_once(self):

        if self.ser is None or not self.ser.is_open:

            messagebox.showwarning(
                "串口未连接",
                "请先连接 STM32。"
            )

            return

        if self.power_busy:
            return

        with self.state_lock:

            if self.ugw_requested or self.ugw_active:

                self.event_queue.put(
                    (
                        "temp_skipped",
                        "单次温度请求",
                        "UGW 正在采集，未发送 Y"
                    )
                )

                return

            if self.temp_active or self.temp_auto_running:

                self.event_queue.put(
                    (
                        "temp_skipped",
                        "单次温度请求",
                        "另一项温度采集正在进行，未重复发送 Y"
                    )
                )

                return

            self.temp_active = True

        self.temp_single_save_enabled = bool(
            self.temp_save_var.get()
        )

        self.temp_single_save_dir = (
            self.temp_save_dir_var.get()
        )

        self.temp_single_csv_path = None

        threading.Thread(
            target=self._temperature_once_worker,
            daemon=True
        ).start()

        self.update_buttons()


    def _temperature_once_worker(self):

        acquired = self.serial_lock.acquire(
            blocking=False
        )

        if not acquired:

            with self.state_lock:
                self.temp_active = False

            self.event_queue.put(
                (
                    "temp_skipped",
                    "单次温度请求",
                    "串口正被其他任务占用，未发送 Y"
                )
            )

            self.event_queue.put(
                ("temp_once_finished",)
            )

            return

        try:

            with self.state_lock:

                if self.ugw_requested or self.ugw_active:

                    self.temp_active = False

                    self.event_queue.put(
                        (
                            "temp_skipped",
                            "单次温度请求",
                            "UGW 已开始，未发送 Y"
                        )
                    )

                    return

            if self.power_busy:

                return

            result = self._temperature_transaction(
                self.ser
            )

            self.event_queue.put(
                ("temp_once_result", result)
            )

            if self.temp_single_save_enabled:

                path = self._ensure_temp_single_csv_path()

                row = {

                    "Index": "single",

                    "ScheduledPC_Time": result["PC_Time"],

                    **result,

                    "Elapsed_s": "",

                }

                self._append_temp_csv(
                    row,
                    path
                )

                self.event_queue.put(
                    (
                        "log",
                        f"单次温度 CSV 已保存：{path}"
                    )
                )

        except Exception as exc:

            self.event_queue.put(
                ("temp_error", str(exc))
            )

        finally:

            with self.state_lock:
                self.temp_active = False

            self.serial_lock.release()

            self.event_queue.put(
                ("temp_once_finished",)
            )


    # --------------------------------------------------------
    # Automatic temperature acquisition
    # --------------------------------------------------------

    def start_temperature_auto(self):

        if self.ser is None or not self.ser.is_open:

            messagebox.showwarning(
                "串口未连接",
                "请先连接 STM32。"
            )

            return

        if self.power_busy:
            return

        if self.temp_auto_running:
            return

        try:

            interval = float(
                self.temp_interval_var.get()
            )

            if interval < 0.2:

                raise ValueError(
                    "温度周期建议不小于 0.2 s；"
                    "MAX31865 单次转换本身约需 60 ms。"
                )

        except Exception as exc:

            messagebox.showerror(
                "温度周期错误",
                str(exc)
            )

            return

        self.temp_auto_stop.clear()

        self.temp_auto_running = True

        self.temp_auto_save_enabled = bool(
            self.temp_save_var.get()
        )

        self.temp_auto_save_dir = (
            self.temp_save_dir_var.get()
        )

        self.temp_save_check.state(
            ["disabled"]
        )

        self.temp_auto_index = 0

        self.temp_auto_start_mono = time.monotonic()

        with self.temp_data_lock:
            self.temp_samples.clear()

        self.temp_auto_csv_path = None

        if self.temp_auto_save_enabled:

            folder = Path(
                self.temp_auto_save_dir
            ).expanduser()

            folder.mkdir(
                parents=True,
                exist_ok=True
            )

            stamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )

            self.temp_auto_csv_path = (
                folder / f"Temperature_Auto_{stamp}.csv"
            )

            self.log(
                f"温度 CSV 文件：{self.temp_auto_csv_path}"
            )

        self.temp_auto_thread = threading.Thread(

            target=self.temperature_auto_worker,

            args=(
                interval,
                self.temp_auto_stop
            ),

            daemon=True

        )

        self.temp_auto_thread.start()

        self.log(
            f"温度定时采集启动：每 {interval:g} s 发送一次 Y；"
            f"温度保存={'开启' if self.temp_auto_save_enabled else '关闭'}。"
        )

        self.temp_status_var.set(
            "温度定时采集中"
        )

        self.update_buttons()


    def temperature_auto_worker(
        self,
        interval,
        stop_event
    ):

        next_tick = time.monotonic()

        index = 0

        while not stop_event.is_set():

            wait_time = max(
                0.0,
                next_tick - time.monotonic()
            )

            if stop_event.wait(wait_time):
                break

            index += 1

            scheduled = datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            )

            # Check UGW state before trying to access UART.
            with self.state_lock:

                busy = bool(
                    self.ugw_requested
                    or self.ugw_active
                    or self.temp_active
                )

            if busy:

                reason = (
                    "SKIPPED_UGW_BUSY"
                    if self._ugw_busy_snapshot()
                    else "SKIPPED_SERIAL_BUSY"
                )

                self._record_temp_auto(
                    index,
                    scheduled,
                    None,
                    reason
                )

            else:

                acquired = self.serial_lock.acquire(
                    blocking=False
                )

                if not acquired:

                    self._record_temp_auto(
                        index,
                        scheduled,
                        None,
                        "SKIPPED_SERIAL_BUSY"
                    )

                else:

                    should_read = False

                    with self.state_lock:

                        if (
                            not self.ugw_requested
                            and not self.ugw_active
                            and not self.temp_active
                        ):

                            self.temp_active = True

                            should_read = True

                    if not should_read:

                        self.serial_lock.release()

                        self._record_temp_auto(
                            index,
                            scheduled,
                            None,
                            "SKIPPED_UGW_BUSY"
                        )

                    else:

                        try:

                            result = self._temperature_transaction(
                                self.ser
                            )

                            self._record_temp_auto(
                                index,
                                scheduled,
                                result,
                                "OK"
                            )

                        except Exception as exc:

                            self._record_temp_auto(
                                index,
                                scheduled,
                                None,
                                f"ERROR: {exc}"
                            )

                        finally:

                            with self.state_lock:
                                self.temp_active = False

                            self.serial_lock.release()

            # Avoid rapid catch-up requests if one transaction
            # takes longer than the configured period.
            next_tick += interval

            if next_tick <= time.monotonic():
                next_tick = time.monotonic() + interval

        self.event_queue.put(
            ("temp_auto_finished",)
        )


    def stop_temperature_auto(self):

        if self.temp_auto_running:

            self.temp_auto_stop.set()

            self.log(
                "已请求停止温度定时采集。"
            )


    # ========================================================
    # 17. Temperature visualization
    # ========================================================

    def show_temperature_window(self):

        if (
            self.temp_window is not None
            and self.temp_window.winfo_exists()
        ):

            self.temp_window.lift()

            return

        win = tk.Toplevel(
            self.parent
        )

        win.title(
            "温度实时读数与定时采集波形"
        )

        win.geometry(
            "1050x680"
        )

        win.configure(background=PANEL_BG)
        self.temp_window = win

        # ----------------------------------------------------
        # Readout panel
        # ----------------------------------------------------

        readout = ttk.LabelFrame(
            win,
            style="Monitor.TLabelframe",
            text="最新温度读数",
            padding=10
        )

        readout.pack(
            fill=tk.X,
            padx=10,
            pady=8
        )

        self.temp_value_vars = {

            "cvd": tk.StringVar(value="-- ℃"),

            "table": tk.StringVar(value="-- ℃"),

            "resistance": tk.StringVar(value="-- Ω"),

            "tick": tk.StringVar(value="-- ms"),

            "time": tk.StringVar(value="尚未采集"),

        }

        fields = [

            ("CVD 温度", "cvd"),

            ("查表温度", "table"),

            ("RTD 电阻", "resistance"),

            ("MCU 时间戳", "tick"),

        ]

        for label, key in fields:

            cell = ttk.Frame(readout, style="Monitor.TFrame")

            cell.pack(
                side=tk.LEFT,
                expand=True,
                fill=tk.X,
                padx=8
            )

            ttk.Label(
                cell,
                style="Monitor.TLabel",
                text=label
            ).pack(anchor="center")

            ttk.Label(
                cell,
                style="Monitor.TLabel",
                textvariable=self.temp_value_vars[key],
                font=("Microsoft YaHei", 18, "bold")
            ).pack(
                anchor="center",
                pady=3
            )

        ttk.Label(
            readout,
            style="Monitor.TLabel",
            textvariable=self.temp_value_vars["time"]
        ).pack(
            anchor=tk.W,
            pady=(8, 0)
        )

        # ----------------------------------------------------
        # Temperature trend plot
        # ----------------------------------------------------

        graph_box = ttk.LabelFrame(
            win,
            style="Monitor.TLabelframe",
            text="温度定时采集趋势（跳过的采样点会留空）",
            padding=5
        )

        graph_box.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=8
        )

        self.temp_fig = Figure(
            figsize=(10, 5),
            dpi=95
        )

        self.temp_ax = self.temp_fig.add_subplot(111)

        self.temp_ax.set_xlabel(
            "Elapsed time / s"
        )

        self.temp_ax.set_ylabel(
            "Temperature / °C"
        )

        self.temp_ax.grid(
            True,
            linestyle=":",
            alpha=0.5
        )

        self.temp_line_cvd, = self.temp_ax.plot(
            [],
            [],
            "o-",
            label="CVD"
        )

        self.temp_line_table, = self.temp_ax.plot(
            [],
            [],
            "s-",
            label="Lookup table"
        )

        self.temp_ax.legend(
            loc="best"
        )

        self.temp_canvas = FigureCanvasTkAgg(
            self.temp_fig,
            master=graph_box
        )

        self.temp_canvas.draw()

        self.temp_canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True
        )

        ttk.Label(
            win,
            style="Monitor.TLabel",
            text=(
                "说明：定时温度采集只重复发送 Y；"
                "若与 UGW 采集重叠，该时刻不发送 Y，并在曲线上标记缺测。"
            ),
            foreground="#526071"
        ).pack(
            anchor=tk.W,
            padx=12,
            pady=(0, 8)
        )

        self.refresh_temperature_plot()


    def update_temperature_readout(self, result):

        if (
            self.temp_window is None
            or not self.temp_window.winfo_exists()
        ):
            return

        self.temp_value_vars["cvd"].set(
            f"{result['Temperature_CVD_C']:.2f} ℃"
        )

        self.temp_value_vars["table"].set(
            f"{result['Temperature_Table_C']:.2f} ℃"
        )

        self.temp_value_vars["resistance"].set(
            f"{result['Resistance_Ohm']:.3f} Ω"
        )

        self.temp_value_vars["tick"].set(
            f"{result['MCU_Tick_ms']} ms"
        )

        self.temp_value_vars["time"].set(
            f"PC 时间：{result['PC_Time']}"
        )


    def refresh_temperature_plot(self):

        if (
            self.temp_window is None
            or not self.temp_window.winfo_exists()
            or self.temp_ax is None
        ):
            return

        with self.temp_data_lock:

            rows = list(
                self.temp_samples
            )

        if rows:

            x = np.asarray(
                [r["Elapsed_s"] for r in rows],
                dtype=float
            )

            y1 = np.asarray(
                [r["Temperature_CVD_C"] for r in rows],
                dtype=float
            )

            y2 = np.asarray(
                [r["Temperature_Table_C"] for r in rows],
                dtype=float
            )

            # NaN points remain gaps in the curves.
            self.temp_line_cvd.set_data(
                x,
                y1
            )

            self.temp_line_table.set_data(
                x,
                y2
            )

            self.temp_ax.relim()

            self.temp_ax.autoscale_view()

        else:

            self.temp_line_cvd.set_data([], [])

            self.temp_line_table.set_data([], [])

        self.temp_canvas.draw_idle()


    # ========================================================
    # 18. UGW plotting
    # ========================================================

    def plot_latest(self):

        if (
            self.latest_result is None
            or self.plot_cleared
        ):
            return

        result = self.latest_result

        raw = result["ADC_raw"]

        voltage = result["Voltage_V"]

        n_samples, n_channels = raw.shape

        try:

            display_fs = float(
                self.display_fs_var.get()
            )

            if display_fs <= 0:
                raise ValueError

        except Exception:

            display_fs = ADC_FS_HZ

        # Display time axis only.
        # This setting does not modify saved data.
        if self.time_axis_var.get():

            x = (
                np.arange(n_samples)
                / display_fs
                * 1e6
            )

            xlabel = "Time / us (display rate)"

        else:

            x = np.arange(n_samples)

            xlabel = "Sample Index"

        if self.plot_voltage_var.get():

            data = voltage

            ylabel = "Voltage / V"

        else:

            data = raw.astype(float)

            ylabel = "ADC Raw"

        self.fig.clear()

        if n_channels <= 4:

            axs = self.fig.subplots(
                n_channels,
                1,
                sharex=True,
                squeeze=False
            ).reshape(-1)

        else:

            rows = 4

            cols = int(
                np.ceil(n_channels / rows)
            )

            axs = self.fig.subplots(
                rows,
                cols,
                sharex=True,
                squeeze=False
            ).reshape(-1)

        axes = list(axs)

        labels = result["ChannelLabels"]

        for i, ax in enumerate(axes):

            if i >= n_channels:

                ax.set_visible(False)

                continue

            y = data[:, i]

            ax.plot(
                x,
                y,
                linewidth=0.85
            )

            ax.set_ylabel(ylabel)

            ax.grid(
                True,
                linestyle=":",
                alpha=0.5
            )

            ax.set_title(
                f"{labels[i]} | "
                f"Min={np.min(y):.4g} | "
                f"Max={np.max(y):.4g}",
                loc="left",
                fontsize=9
            )

            ax.set_xlabel(xlabel)

        self.fig.suptitle(
            f"STM32 UGW | {result['Command']} | "
            f"{n_channels} path(s) displayed",
            fontsize=12
        )

        self.fig.tight_layout()

        self.canvas.draw_idle()


    def reset_acquisition_count(self):
        self.acquisition_count = 0
        self.count_var.set("UGW 采集次数：0")
        self.log("UGW 采集次数已清零。")

    def clear_plot(self):
        self.plot_cleared = True
        self._empty_waveform()
        self.canvas.draw_idle()
        self.status_var.set("UGW 波形已清空")
        self.log("UGW 波形已清空。")


    # ========================================================
    # 19. Folder ToF analysis
    # ========================================================

    @staticmethod
    def _mat_field(obj, name, default=None):

        if isinstance(obj, dict):
            return obj.get(name, default)

        return getattr(
            obj,
            name,
            default
        )


    @staticmethod
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


    @staticmethod
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


    @classmethod
    def _extract_tx3_rx4(cls, ugw):

        mode = cls._mat_string(
            cls._mat_field(ugw, "Mode", "")
        ).upper()

        command = cls._mat_string(
            cls._mat_field(ugw, "Command", "")
        ).upper()

        if mode == "R" or command.startswith("R"):

            raise ValueError(
                "R 模式数据不是本次 M 模式 Tx3->Rx4 ToF 分析对象"
            )

        raw_value = cls._mat_field(
            ugw,
            "ADC_raw",
            None
        )

        if raw_value is None:

            raise ValueError(
                "缺少 ADC_raw 字段"
            )

        raw = np.asarray(
            raw_value,
            dtype=np.float64
        )

        if raw.ndim == 1:

            raw = raw.reshape(-1, 1)

        elif raw.ndim != 2:

            raise ValueError(
                "ADC_raw 不是二维矩阵"
            )

        if raw.shape[0] <= 4 and raw.shape[1] > 4:
            raw = raw.T

        channel_numbers = cls._mat_ints(
            cls._mat_field(ugw, "ChannelNumbers", None)
        )

        tx_channels = cls._mat_ints(
            cls._mat_field(ugw, "TxChannels", None)
        )

        channel_ids = []

        if len(channel_numbers) == raw.shape[1]:

            channel_ids = channel_numbers

        elif len(tx_channels) == raw.shape[1]:

            channel_ids = tx_channels

        if 3 in channel_ids:

            col = channel_ids.index(3)

        elif not channel_ids and raw.shape[1] >= 3:

            col = 2

        else:

            raise ValueError(
                "此 MAT 没有保存 Tx3；采集 M 模式时需要勾选 Tx3"
            )

        waveform = raw[:, col].copy()

        if waveform.size < 100 or not np.all(np.isfinite(waveform)):

            raise ValueError(
                "Tx3 波形长度不足或含 NaN/Inf"
            )

        return waveform


    @staticmethod
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


    @staticmethod
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


    def analyze_tof_folder(self):

        folder = filedialog.askdirectory(
            parent=self.parent,

            title="选择包含 M 模式 MAT 文件的文件夹",

            initialdir=self.tof_dir or self.save_dir_var.get() or str(DEFAULT_UGW_FOLDER)

        )

        if not folder:
            return

        self.tof_dir = folder
        folder = Path(folder)

        files = sorted(
            folder.glob("*.mat"),
            key=lambda p: p.name.lower()
        )

        if not files:

            messagebox.showwarning(
                "没有数据",
                "所选文件夹中没有 MAT 文件。"
            )

            return

        self.log(
            f"开始文件夹 ToF 分析：{folder}；发现 {len(files)} 个 MAT"
        )

        waves = []

        skipped = 0

        for path in files:

            try:

                mat = loadmat(
                    str(path),
                    squeeze_me=True,
                    struct_as_record=False
                )

                if "UGW_result" in mat:

                    ugw = mat["UGW_result"]

                elif "UGW_Struct" in mat:

                    ugw = mat["UGW_Struct"]

                else:

                    raise ValueError(
                        "缺少 UGW_result / UGW_Struct"
                    )

                raw = self._extract_tx3_rx4(
                    ugw
                )

                processed, fs_interp = (
                    self._preprocess_tof_wave(raw)
                )

                waves.append({

                    "name": path.name,

                    "wave": processed,

                    "fs": fs_interp

                })

            except Exception as exc:

                skipped += 1

                self.log(
                    f"ToF 跳过 {path.name}：{exc}"
                )

        if not waves:

            messagebox.showwarning(
                "ToF 分析失败",
                "没有有效的 M 模式 Tx3 数据。"
                "请确认采集时勾选了 Tx3，并且文件是 M 模式。"
            )

            return

        reference = waves[0]["wave"]

        reference_name = waves[0]["name"]

        fs_interp = waves[0]["fs"]

        rows = []

        for index, item in enumerate(
            waves,
            start=1
        ):

            if (
                not np.isclose(item["fs"], fs_interp)
                or item["wave"].size != reference.size
            ):

                delta = np.nan
                quality = np.nan
                status = "长度/采样率不一致"

            else:

                delta, quality = self._relative_tof_xcorr(
                    item["wave"],
                    reference,
                    fs_interp
                )

                if not np.isfinite(quality):

                    status = "相关计算失败"

                elif quality < TOF_MIN_CORR:

                    status = "相关质量不足"

                elif not np.isfinite(delta):

                    status = "ToF 无效"

                else:

                    status = "有效"

            rows.append({

                "ContinuousIndex": index,

                "FileName": item["name"],

                "DeltaToF_us": delta,

                "XCorrCoefficient": quality,

                "Status": status,

            })

        x = np.asarray(
            [r["ContinuousIndex"] for r in rows],
            dtype=float
        )

        y = np.asarray(
            [r["DeltaToF_us"] for r in rows],
            dtype=float
        )

        valid = np.isfinite(y)

        xv = x[valid]

        yv = y[valid]

        slope = intercept = r2 = rmse = np.nan

        if yv.size >= 2:

            p = np.polyfit(
                xv,
                yv,
                1
            )

            slope = float(p[0])

            intercept = float(p[1])

            fit = np.polyval(
                p,
                xv
            )

            ss_res = float(
                np.sum((yv - fit) ** 2)
            )

            ss_tot = float(
                np.sum((yv - np.mean(yv)) ** 2)
            )

            if ss_tot > 1e-15:

                r2 = 1.0 - ss_res / ss_tot

            rmse = float(
                np.sqrt(np.mean((yv - fit) ** 2))
            )

        csv_path = folder / (
            "ToF_XCorr_Wave_W1_Lag2_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + ".csv"
        )

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "ContinuousIndex",
                    "FileName",
                    "DeltaToF_us",
                    "XCorrCoefficient",
                    "Status"
                ]
            )

            writer.writeheader()

            writer.writerows(rows)

        # ----------------------------------------------------
        # ToF visualization window
        # ----------------------------------------------------

        win = tk.Toplevel(
            self.parent
        )

        win.title(
            "STM32 Tx3 -> Rx4 | ToF 分析"
        )

        win.geometry(
            "1120x700"
        )

        fig = Figure(
            figsize=(10.5, 5.5),
            dpi=100
        )

        ax = fig.add_subplot(111)

        ax.plot(
            x,
            y,
            "o-",
            linewidth=1.2,
            markersize=4,
            label="Delta ToF"
        )

        if np.isfinite(slope) and xv.size:

            fit_x = np.linspace(
                float(np.min(xv)),
                float(np.max(xv)),
                200
            )

            ax.plot(
                fit_x,
                slope * fit_x + intercept,
                "--",
                linewidth=1.5,
                label="Linear fit"
            )

        ax.axhline(
            0,
            color="gray",
            linestyle=":",
            linewidth=0.8
        )

        ax.set_xlabel(
            "Acquisition index"
        )

        ax.set_ylabel(
            "Delta ToF / us"
        )

        ax.set_title(
            "STM32 Tx3 -> Rx4 | XCorr_Wave_W1_Lag2"
        )

        ax.grid(
            True,
            alpha=0.35
        )

        ax.legend(
            loc="best"
        )

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(
            fig,
            master=win
        )

        canvas.draw()

        canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True
        )

        ttk.Label(
            win,
            text=(
                f"MAT 文件：{len(files)}    "
                f"有效波形：{len(waves)}    "
                f"有效 ToF：{int(yv.size)}    "
                f"跳过：{skipped}\n"

                f"统一参考：{reference_name}\n"

                f"R²={r2:.5f}    "
                f"RMSE={rmse:.5f} us    "
                f"Slope={slope:.6f} us/index\n"

                f"HP=170 kHz, LP=300 kHz, "
                f"插值=10×, 搜索=5~100 us, "
                f"最大时移=±2 us, 最低相关系数=0.35"
            ),
            justify=tk.LEFT
        ).pack(
            fill=tk.X,
            padx=10,
            pady=8
        )

        self.log(
            f"ToF 分析完成：R²={r2:.5f}, "
            f"RMSE={rmse:.5f} us, "
            f"slope={slope:.6f} us/index"
        )

        self.log(
            f"ToF CSV：{csv_path}"
        )


    # ========================================================
    # 20. GUI event processing
    # ========================================================

    def process_events(self):

        try:

            while True:

                event = self.event_queue.get_nowait()

                kind = event[0]

                # --------------------------------------------
                # Log
                # --------------------------------------------

                if kind == "log":

                    self.log(event[1])

                # --------------------------------------------
                # UGW result
                # --------------------------------------------

                elif kind == "result":

                    result = event[1]

                    cycle = event[2]

                    save_paths = event[3]

                    self.latest_result = result

                    self.plot_cleared = False

                    self.acquisition_count += 1

                    self.count_var.set(
                        f"UGW 采集次数：{self.acquisition_count}"
                    )

                    self.plot_latest()

                    self.status_var.set(
                        f"UGW 第 {cycle} 轮完成"
                    )

                    self.log(
                        f"{result['Command']} 完成："
                        f"显示/保存 {result['SelectedChannelCount']} 路，"
                        f"本轮接收 {result['BytesReceived']} Bytes。"
                    )

                    if result["InvalidCount"] > 0:

                        self.log(
                            f"警告：ADC 超量程计数 "
                            f"{result['InvalidCount']}，请检查数据包对齐。"
                        )

                    if result["STM32_Message"]:

                        self.log(
                            f"STM32 完成标志：{result['STM32_Message']}"
                        )

                # --------------------------------------------
                # UGW error
                # --------------------------------------------

                elif kind == "error":

                    self.status_var.set(
                        "UGW 采集错误"
                    )

                    self.log(
                        "UGW 错误：" + event[1]
                    )

                    messagebox.showerror(
                        "UGW 采集错误",
                        event[1]
                    )

                # --------------------------------------------
                # UGW finished
                # --------------------------------------------

                elif kind == "finished":

                    self.worker = None

                    self.stop_event = None

                    self.status_var.set(
                        "UGW 任务结束"
                    )

                    self.update_buttons()

                # --------------------------------------------
                # Power result
                # --------------------------------------------

                elif kind == "power_result":

                    self.power_enabled = event[1]

                    command = event[2]

                    if command == "v":

                        self.power_status_var.set(
                            "已发送 v（硬件状态未确认）"
                        )

                        self.log(
                            "已发送 UGW 电源开启指令 v。"
                        )

                    elif command == "0":

                        self.power_status_var.set(
                            "已发送 0（停止并关闭）"
                        )

                        self.log(
                            "已发送全局停止及电源关闭指令 0。"
                        )

                elif kind == "power_error":

                    self.power_enabled = None

                    self.power_status_var.set(
                        "UGW电源：状态未知"
                    )

                    self.log(
                        "电源控制错误：" + event[1]
                    )

                    messagebox.showerror(
                        "电源控制错误",
                        event[1]
                    )

                elif kind == "power_finished":

                    self.power_busy = False

                    self.update_buttons()

                # --------------------------------------------
                # Single temperature result
                # --------------------------------------------

                elif kind == "temp_once_result":

                    result = event[1]

                    self.update_temperature_readout(
                        result
                    )

                    self.temp_status_var.set(
                        f"温度：CVD {result['Temperature_CVD_C']:.2f} ℃ / "
                        f"查表 {result['Temperature_Table_C']:.2f} ℃"
                    )

                    self.log(
                        f"温度单次采集："
                        f"R={result['Resistance_Ohm']:.3f} Ω，"
                        f"CVD={result['Temperature_CVD_C']:.2f} ℃，"
                        f"查表={result['Temperature_Table_C']:.2f} ℃。"
                    )

                elif kind == "temp_once_finished":

                    self.update_buttons()

                # --------------------------------------------
                # Automatic temperature sample
                # --------------------------------------------

                elif kind == "temp_auto_sample":

                    row = event[1]

                    if row["Status"] == "OK":

                        self.temp_status_var.set(
                            f"温度：CVD {row['Temperature_CVD_C']:.2f} ℃ / "
                            f"查表 {row['Temperature_Table_C']:.2f} ℃"
                        )

                        self.update_temperature_readout(
                            row
                        )

                        self.log(
                            f"温度定时 #{row['Index']}："
                            f"CVD={row['Temperature_CVD_C']:.2f} ℃，"
                            f"查表={row['Temperature_Table_C']:.2f} ℃。"
                        )

                    else:

                        self.temp_status_var.set(
                            f"温度定时 #{row['Index']}：缺测（{row['Status']}）"
                        )

                        self.log(
                            f"温度定时 #{row['Index']} 缺测："
                            f"{row['Status']}；未完成有效温度采集。"
                        )

                    self.refresh_temperature_plot()

                # --------------------------------------------
                # Temperature skipped
                # --------------------------------------------

                elif kind == "temp_skipped":

                    self.temp_status_var.set(
                        "温度：本次未采（UGW/串口忙）"
                    )

                    self.log(
                        f"温度采集跳过：{event[1]}；原因：{event[2]}。"
                    )

                # --------------------------------------------
                # Temperature error
                # --------------------------------------------

                elif kind == "temp_error":

                    self.temp_status_var.set(
                        "温度采集错误"
                    )

                    self.log(
                        "温度采集错误：" + event[1]
                    )

                # --------------------------------------------
                # Automatic temperature finished
                # --------------------------------------------

                elif kind == "temp_auto_finished":

                    self.temp_auto_running = False

                    self.temp_save_check.state(
                        ["!disabled"]
                    )

                    self.temp_status_var.set(
                        "温度定时采集已停止"
                    )

                    self.log(
                        "温度定时采集结束。"
                    )

                    self.update_buttons()

        except queue.Empty:

            pass

        self.event_job = self.parent.after(50, self.process_events)


    # ========================================================
    # 21. Stop / button states / log
    # ========================================================

    def stop_acquisition(self):

        if self.stop_event is not None:

            self.stop_event.set()

            self.status_var.set(
                "正在停止 UGW……"
            )

            self.log(
                "已请求停止 UGW 采集。"
            )


    def update_buttons(self):

        connected = (
            self.ser is not None
            and self.ser.is_open
        )

        busy = (
            self.worker is not None
            and self.worker.is_alive()
        )

        temp_busy = (
            self.temp_active
            or self.temp_auto_running
        )

        powered = (
            self.power_enabled is True
        )

        idle = (
            connected
            and not busy
            and not self.power_busy
        )

        power_idle = (
            idle
            and not temp_busy
        )

        self.connect_button.configure(
            state=(
                "disabled"
                if connected or busy or self.power_busy
                else "normal"
            )
        )

        self.disconnect_button.configure(
            state=(
                "normal"
                if connected
                and not busy
                and not self.power_busy
                and not temp_busy
                else "disabled"
            )
        )

        self.refresh_button.configure(
            state=(
                "disabled"
                if connected or busy or self.power_busy
                else "normal"
            )
        )

        self.once_button.configure(
            state=(
                "normal"
                if idle and powered and not self.auto_var.get()
                else "disabled"
            )
        )

        self.auto_start_button.configure(
            state=(
                "normal"
                if idle and powered and self.auto_var.get()
                else "disabled"
            )
        )

        self.stop_button.configure(
            state="normal" if busy else "disabled"
        )

        self.auto_check.configure(
            state=(
                "disabled"
                if busy or self.power_busy
                else "normal"
            )
        )

        self.power_on_button.configure(
            state=(
                "normal"
                if power_idle and not powered
                else "disabled"
            )
        )

        self.power_off_button.configure(
            state=(
                "normal"
                if power_idle and self.power_enabled is not False
                else "disabled"
            )
        )

        self.temp_once_button.configure(
            state=(
                "normal"
                if connected
                and not self.power_busy
                and not busy
                and not self.temp_auto_running
                and not self.temp_active
                else "disabled"
            )
        )

        self.temp_start_button.configure(
            state=(
                "normal"
                if connected
                and not self.power_busy
                and not self.temp_auto_running
                else "disabled"
            )
        )

        self.temp_stop_button.configure(
            state=(
                "normal"
                if self.temp_auto_running
                else "disabled"
            )
        )

        if self.temp_auto_running:

            self.temp_save_check.state(
                ["disabled"]
            )

        else:

            self.temp_save_check.state(
                ["!disabled"]
            )


    def log(self, message):

        if not hasattr(self, "log_text"):
            return

        timestamp = datetime.now().strftime(
            "%H:%M:%S"
        )

        self.log_text.configure(
            state=tk.NORMAL
        )

        self.log_text.insert(
            tk.END,
            f"[{timestamp}] {message}\n"
        )

        self.log_text.see(
            tk.END
        )

        self.log_text.configure(
            state=tk.DISABLED
        )


    # ========================================================
    # 22. Close
    # ========================================================

    def shutdown(self):
        self.parent.after_cancel(self.event_job)
        if self.stop_event is not None:
            self.stop_event.set()
        self.temp_auto_stop.set()

        for worker in (self.worker, self.temp_auto_thread):
            if worker is not None:
                worker.join(timeout=1.0)

        if self.ser is not None:
            try:
                self.ser.close()
            except OSError:
                pass
