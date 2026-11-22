"""界面样式、可滚动设置区与 Windows 高 DPI 初始化。"""
import ctypes
import os
import tkinter as tk
from tkinter import ttk


def enable_high_dpi():
    # 必须在创建首个窗口之前调用；仅影响本进程，不改变系统显示设置。
    if os.name != "nt":
        return
    try:
        api = ctypes.windll.user32.SetProcessDpiAwarenessContext
        api.argtypes = [ctypes.c_void_p]
        api.restype = ctypes.c_bool
        if api(ctypes.c_void_p(-4)):
            return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        pass


def apply_theme(root, scale):
    px = lambda x: max(1, round(x * scale))
    style = ttk.Style(root)
    style.theme_use("clam")
    font = ("Microsoft YaHei UI", 10)
    # 通用 *Font 会覆盖 ttk 各样式的标题字号，保持样式自身的字体层级。
    root.option_add("*TCombobox*Listbox.font", font)
    root.configure(bg="#edf2f7")
    style.configure(".", font=font, background="#edf2f7", foreground="#263449")
    style.configure("TFrame", background="#edf2f7")
    style.configure("Card.TFrame", background="white")
    style.configure("TLabel", background="white", foreground="#334155")
    style.configure("Muted.TLabel", foreground="#718096", font=("Microsoft YaHei UI", 9))
    style.configure("Section.TLabel", foreground="#172b4d", font=("Microsoft YaHei UI", 11, "bold"))
    style.layout("TButton", [("Button.border", {"sticky": "nswe", "border": "1", "children": [
        ("Button.padding", {"sticky": "nswe", "children": [("Button.label", {"sticky": "nswe"})]})]})])
    style.configure("TButton", background="#f1f5fa", foreground="#334155", borderwidth=0,
                    bordercolor="#f1f5fa", lightcolor="#f1f5fa", darkcolor="#f1f5fa", padding=(px(11), px(7)))
    style.map("TButton", background=[("pressed", "#d5e0ef"), ("active", "#e2eaf5"), ("disabled", "#f1f5f9")],
              bordercolor=[("active", "#e2eaf5")], lightcolor=[("active", "#e2eaf5")], darkcolor=[("active", "#e2eaf5")],
              foreground=[("disabled", "#94a3b8")])
    style.configure("Primary.TButton", background="#2563eb", foreground="white", bordercolor="#2563eb",
                    lightcolor="#2563eb", darkcolor="#2563eb", font=("Microsoft YaHei UI", 10, "bold"))
    style.map("Primary.TButton", background=[("pressed", "#1e40af"), ("active", "#1d4ed8"), ("disabled", "#a9bfe8")],
              bordercolor=[("active", "#1d4ed8")], lightcolor=[("active", "#1d4ed8")], darkcolor=[("active", "#1d4ed8")],
              foreground=[("disabled", "white")])
    style.configure("TEntry", fieldbackground="#f8fafc", bordercolor="#d9e2ef", lightcolor="#d9e2ef", darkcolor="#d9e2ef", padding=px(6))
    style.configure("TCombobox", fieldbackground="#f8fafc", background="#e9eff7", bordercolor="#d9e2ef",
                    lightcolor="#d9e2ef", darkcolor="#d9e2ef", arrowsize=px(13), padding=px(5))
    style.map("TCombobox", fieldbackground=[("readonly", "#f8fafc"), ("disabled", "#f1f5f9")],
              foreground=[("disabled", "#a0aec0")], selectbackground=[("readonly", "#e6eefc")], selectforeground=[("readonly", "#1e293b")])
    style.configure("TCheckbutton", background="white", padding=(0, px(4)), focusthickness=0,
                    indicatorbackground="white", indicatorforeground="white", indicatorsize=px(14),
                    upperbordercolor="#b8c6d9", lowerbordercolor="#b8c6d9")
    style.map("TCheckbutton", background=[("active", "white")],
              indicatorbackground=[("selected", "#2563eb"), ("active", "#eef3fb")],
              upperbordercolor=[("selected", "#2563eb")], lowerbordercolor=[("selected", "#2563eb")])
    style.configure("TNotebook", background="white", borderwidth=0, tabmargins=0)
    style.configure("TNotebook.Tab", padding=(px(12), px(8)), background="#edf2f7", foreground="#64748b")
    style.map("TNotebook.Tab", background=[("selected", "white")], foreground=[("selected", "#2563eb")],
              padding=[("selected", (px(12), px(8)))], expand=[("selected", (0, 0, 0, 0))])
    for orient in ("Vertical", "Horizontal"):
        style.configure(f"{orient}.TScrollbar", background="#d5dfed", troughcolor="#f8fafc", borderwidth=0, arrowsize=px(10),
                        bordercolor="#f8fafc", lightcolor="#d5dfed", darkcolor="#d5dfed", arrowcolor="#64748b")
        style.map(f"{orient}.TScrollbar", background=[("active", "#b8c6d9")])
    return px


class ScrollArea(ttk.Frame):
    """窗口小于内容最小尺寸时出现上下、左右滚动条；足够大时内容铺满窗口。"""

    def __init__(self, parent, min_size):
        super().__init__(parent)
        self.min_size = min_size
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0, bg="#edf2f7", width=1, height=1)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.hbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.vbar.set, xscrollcommand=self.hbar.set)
        self.body = ttk.Frame(self.canvas)
        self.item = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.pending = None
        self.bind("<Configure>", lambda e: self.refresh())

    def watch(self, *widgets):
        for widget in widgets:
            widget.bind("<Configure>", lambda e: self.schedule_refresh(), add="+")

    def schedule_refresh(self):
        if self.pending is None:
            self.pending = self.after_idle(self.refresh)

    def refresh(self):
        if self.pending is not None:
            self.after_cancel(self.pending)
            self.pending = None
        width, height = self.winfo_width(), self.winfo_height()
        if width <= 1 or height <= 1:
            return
        min_w, min_h = self.min_size()
        bar = self.vbar.winfo_reqwidth()
        need_h = width < min_w
        need_v = height - (bar if need_h else 0) < min_h
        need_h = width - (bar if need_v else 0) < min_w
        view_w, view_h = width - (bar if need_v else 0), height - (bar if need_h else 0)
        self.canvas.place(x=0, y=0, width=view_w, height=view_h)
        self.vbar.place(x=view_w, y=0, width=bar, height=view_h) if need_v else self.vbar.place_forget()
        self.hbar.place(x=0, y=view_h, width=view_w, height=bar) if need_h else self.hbar.place_forget()
        body_w = max(view_w, min_w)
        body_h = max(view_h, min_h)
        self.canvas.itemconfigure(self.item, width=body_w, height=body_h)
        self.canvas.configure(scrollregion=(0, 0, body_w, body_h))

    def bind_scrolling(self):
        def wheel(event, axis):
            if (isinstance(event.widget, str) or not self.winfo_ismapped()
                    or not str(event.widget).startswith(str(self) + ".")
                    or event.widget.winfo_class() in ("Text", "Listbox")):
                return
            view = self.canvas.xview if axis == "x" else self.canvas.yview
            if view() != (0.0, 1.0):
                view("scroll", -int(event.delta / 120) if abs(event.delta) >= 120 else (-1 if event.delta > 0 else 1), "units")
        self.bind_all("<MouseWheel>", lambda e: wheel(e, "y"), add="+")
        self.bind_all("<Shift-MouseWheel>", lambda e: wheel(e, "x"), add="+")


class ScrollPanel(ttk.Frame):
    def __init__(self, parent, px):
        super().__init__(parent, style="Card.TFrame")
        self.canvas = tk.Canvas(self, highlightthickness=0, bg="white", width=1, height=1)
        scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.body = ttk.Frame(self.canvas, style="Card.TFrame", padding=px(16))
        item = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(item, width=e.width))
        self.body.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

    def bind_scrolling(self):
        def wheel(event):
            self.canvas.yview_scroll(-int(event.delta / 120) if abs(event.delta) >= 120 else (-1 if event.delta > 0 else 1), "units")
            return "break"
        def visit(widget):
            widget.bind("<MouseWheel>", wheel)
            for child in widget.winfo_children():
                visit(child)
        visit(self)
