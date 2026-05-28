#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Crypy Premium 2D Crystal & CDW Interactive Builder
Author: Antigravity Team
Date: 2026-05-28
"""

import sys
import os
import ast
import math
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib
# Set Matplotlib backend to TkAgg (must be set before importing pyplot)
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Try to import crypy, fallback to local path
try:
    import crypy as cp
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import crypy as cp

# Set Windows High-DPI awareness
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

# PRESETS DATA
PRESETS = {
    "2H-TaSe2 (3x3 CDW)": {
        "l1": 1.0,
        "l2": 1.0,
        "angle": 120.0,
        "rot": 0.0,
        "ox": 0.0,
        "oy": 0.0,
        "sub_n1": 3,
        "sub_n2": 3,
        "xmin": -5.0,
        "xmax": 5.0,
        "ymin": -5.0,
        "ymax": 5.0,
        "view_xmin": -1.2,
        "view_xmax": 1.2,
        "view_ymin": -1.2,
        "view_ymax": 1.2,
        "basis": (
            "# 2H-TaSe2 CDW superstructure (n1=3, n2=3)\n"
            "# kind,label,ij_points,color,marker_or_linewidth,size\n"
            "point,Ta,\"(2, 1)\",#ff4d4d,o,9\n"
            "point,Se,\"(1, 2)\",#2ecc71,o,9\n"
            "line,bond1,\"[((2, 1), (1, -1))]\",#3498db,-,1.8\n"
            "line,bond2,\"[((1, 2), (-1, 1))]\",#3498db,-,1.8\n"
            "line,bond3,\"[((2, 1), (1, 2))]\",#3498db,-,1.8\n"
        ),
        "show_cdw": True,
        "cdw_kx": 0.0,
        "cdw_ky": 2.0944, # 2*pi/3 for 3x3 superstructure along y
        "cdw_color": "#a29bfe",
        "cdw_lw": 1.5,
        "cdw_cmin": -4,
        "cdw_cmax": 4
    },
    "Graphene / Honeycomb": {
        "l1": 1.0,
        "l2": 1.0,
        "angle": 120.0,
        "rot": 0.0,
        "ox": 0.0,
        "oy": 0.0,
        "sub_n1": 1,
        "sub_n2": 1,
        "xmin": -4.0,
        "xmax": 4.0,
        "ymin": -4.0,
        "ymax": 4.0,
        "view_xmin": -1.5,
        "view_xmax": 1.5,
        "view_ymin": -1.5,
        "view_ymax": 1.5,
        "basis": (
            "# Graphene Honeycomb Lattice (n1=1, n2=1)\n"
            "# kind,label,ij_points,color,marker_or_linewidth,size\n"
            "point,C1,\"(1/3, 2/3)\",#e056fd,o,8\n"
            "point,C2,\"(2/3, 1/3)\",#ff7675,o,8\n"
            "line,bond1,\"[((1/3, 2/3), (2/3, 1/3))]\",#74b9ff,-,2.0\n"
            "line,bond2,\"[((1/3, 2/3), (-1/3, 1/3))]\",#74b9ff,-,2.0\n"
            "line,bond3,\"[((1/3, 2/3), (2/3, -2/3))]\",#74b9ff,-,2.0\n"
        ),
        "show_cdw": False,
        "cdw_kx": 0.0,
        "cdw_ky": 1.0,
        "cdw_color": "#a29bfe",
        "cdw_lw": 1.0,
        "cdw_cmin": -5,
        "cdw_cmax": 5
    },
    "Simple Triangular": {
        "l1": 1.0,
        "l2": 1.0,
        "angle": 120.0,
        "rot": 0.0,
        "ox": 0.0,
        "oy": 0.0,
        "sub_n1": 1,
        "sub_n2": 1,
        "xmin": -5.0,
        "xmax": 5.0,
        "ymin": -5.0,
        "ymax": 5.0,
        "view_xmin": -2.0,
        "view_xmax": 2.0,
        "view_ymin": -2.0,
        "view_ymax": 2.0,
        "basis": (
            "# Simple Triangular Lattice (n1=1, n2=1)\n"
            "# kind,label,ij_points,color,marker_or_linewidth,size\n"
            "point,Atom,\"(0, 0)\",#fdcb6e,o,10\n"
            "line,bond1,\"[((0, 0), (1, 0))]\",#b2bec3,-,1.5\n"
            "line,bond2,\"[((0, 0), (0, 1))]\",#b2bec3,-,1.5\n"
            "line,bond3,\"[((0, 0), (-1, 1))]\",#b2bec3,-,1.5\n"
        ),
        "show_cdw": False,
        "cdw_kx": 0.0,
        "cdw_ky": 1.0,
        "cdw_color": "#a29bfe",
        "cdw_lw": 1.0,
        "cdw_cmin": -5,
        "cdw_cmax": 5
    },
    "Simple Square": {
        "l1": 1.0,
        "l2": 1.0,
        "angle": 90.0,
        "rot": 0.0,
        "ox": 0.0,
        "oy": 0.0,
        "sub_n1": 1,
        "sub_n2": 1,
        "xmin": -5.0,
        "xmax": 5.0,
        "ymin": -5.0,
        "ymax": 5.0,
        "view_xmin": -2.0,
        "view_xmax": 2.0,
        "view_ymin": -2.0,
        "view_ymax": 2.0,
        "basis": (
            "# Simple Square Lattice (n1=1, n2=1)\n"
            "# kind,label,ij_points,color,marker_or_linewidth,size\n"
            "point,Atom,\"(0, 0)\",#00cec9,o,10\n"
            "line,bond1,\"[((0, 0), (1, 0))]\",#b2bec3,-,1.5\n"
            "line,bond2,\"[((0, 0), (0, 1))]\",#b2bec3,-,1.5\n"
        ),
        "show_cdw": False,
        "cdw_kx": 0.0,
        "cdw_ky": 1.0,
        "cdw_color": "#a29bfe",
        "cdw_lw": 1.0,
        "cdw_cmin": -5,
        "cdw_cmax": 5
    }
}


class ScrollableFrame(ttk.Frame):
    """A standard scrollable container using Canvas and Scrollbar"""
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, bg="#1e1e24")
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas, style="TFrame")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Configure canvas to resize the scrollable frame to fill width
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.canvas.bind('<Enter>', self._bind_mousewheel)
        self.canvas.bind('<Leave>', self._unbind_mousewheel)

    def _on_canvas_configure(self, event):
        # Resize inner frame to match canvas width
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _bind_mousewheel(self, event):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, event):
        self.canvas.unbind_all("<MouseWheel>")


class CrypyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Crypy 2D Crystal & CDW Premium Builder")
        self.geometry("1360x860")
        self.minsize(1024, 700)
        
        self.auto_render_active = True
        self.theme_setup()
        self.variables_setup()
        self.ui_setup()
        
        # Load default preset
        self.load_preset_data("2H-TaSe2 (3x3 CDW)")
        
        # Render the initial plot
        self.render()

    def theme_setup(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Modern Dark Color Palette
        self.bg = "#1e1e24"
        self.card = "#2a2a35"
        self.border = "#3f3f4e"
        self.accent = "#7055ff"
        self.accent_hover = "#856dff"
        self.text_color = "#f3f3f6"
        self.text_muted = "#a0a0b2"
        self.entry_bg = "#323241"

        self.configure(bg=self.bg)

        # Style configurations
        self.style.configure('.',
            background=self.bg,
            foreground=self.text_color,
            font=('Segoe UI', 10),
            bordercolor=self.border,
            lightcolor=self.border,
            darkcolor=self.border
        )

        self.style.configure('TFrame', background=self.bg)
        self.style.configure('Card.TFrame', background=self.card, relief='flat', borderwidth=0)
        self.style.configure('TLabel', background=self.bg, foreground=self.text_color)
        self.style.configure('Card.TLabel', background=self.card, foreground=self.text_color)
        self.style.configure('Title.TLabel', background=self.bg, foreground=self.text_color, font=('Segoe UI', 12, 'bold'))
        self.style.configure('Header.TLabel', background=self.bg, foreground=self.accent, font=('Segoe UI', 11, 'bold'))
        self.style.configure('Section.TLabel', background=self.bg, foreground=self.text_muted, font=('Segoe UI', 9, 'italic'))

        # Buttons
        self.style.configure('TButton',
            background=self.accent,
            foreground=self.text_color,
            bordercolor=self.accent,
            borderwidth=1,
            focusthickness=0,
            focuscolor=self.accent,
            padding=(10, 5)
        )
        self.style.map('TButton',
            background=[('active', self.accent_hover), ('disabled', self.border)],
            foreground=[('active', self.text_color), ('disabled', self.text_muted)],
            bordercolor=[('active', self.accent_hover)]
        )

        # Entry fields
        self.style.configure('TEntry',
            fieldbackground=self.entry_bg,
            foreground=self.text_color,
            bordercolor=self.border,
            lightcolor=self.border,
            darkcolor=self.border,
            insertcolor=self.text_color,
            padding=4
        )

        # Checkbuttons
        self.style.configure('TCheckbutton', background=self.bg, foreground=self.text_color, focuscolor=self.bg)
        self.style.map('TCheckbutton',
            background=[('active', self.bg)],
            foreground=[('active', self.text_color)]
        )

        # Labelframes
        self.style.configure('TLabelframe',
            background=self.bg,
            bordercolor=self.border,
            borderwidth=1,
            relief='solid'
        )
        self.style.configure('TLabelframe.Label',
            background=self.bg,
            foreground=self.accent,
            font=('Segoe UI', 10, 'bold'),
            padding=(6, 0)
        )

        # Scales / Sliders
        self.style.configure('Horizontal.TScale',
            background=self.bg,
            troughcolor=self.entry_bg,
            slidercolor=self.accent,
            lightcolor=self.accent,
            darkcolor=self.accent
        )

        # Comboboxes
        self.style.configure('TCombobox',
            fieldbackground=self.entry_bg,
            background=self.bg,
            foreground=self.text_color,
            bordercolor=self.border,
            arrowcolor=self.text_color
        )
        self.style.map('TCombobox',
            fieldbackground=[('readonly', self.entry_bg)],
            foreground=[('readonly', self.text_color)]
        )

    def variables_setup(self):
        # Numeric / Control Tk variables
        self.var_l1 = tk.DoubleVar(value=1.0)
        self.var_l2 = tk.DoubleVar(value=1.0)
        self.var_angle = tk.DoubleVar(value=120.0)
        self.var_rot = tk.DoubleVar(value=0.0)
        
        self.var_ox = tk.StringVar(value="0.0")
        self.var_oy = tk.StringVar(value="0.0")
        
        self.var_sub_n1 = tk.StringVar(value="3")
        self.var_sub_n2 = tk.StringVar(value="3")
        
        self.var_xmin = tk.StringVar(value="-5")
        self.var_xmax = tk.StringVar(value="5")
        self.var_ymin = tk.StringVar(value="-5")
        self.var_ymax = tk.StringVar(value="5")
        
        self.var_view_xmin = tk.StringVar(value="-1.2")
        self.var_view_xmax = tk.StringVar(value="1.2")
        self.var_view_ymin = tk.StringVar(value="-1.2")
        self.var_view_ymax = tk.StringVar(value="1.2")
        
        # Display Option Booleans
        self.show_crystal = tk.BooleanVar(value=True)
        self.show_lattice = tk.BooleanVar(value=False)
        self.show_basis_labels = tk.BooleanVar(value=True)
        self.show_primitive = tk.BooleanVar(value=True)
        self.show_wigner = tk.BooleanVar(value=True)
        self.show_parallelogram = tk.BooleanVar(value=True)
        self.show_grid = tk.BooleanVar(value=True)
        
        # CDW parameters
        self.var_show_cdw = tk.BooleanVar(value=False)
        self.var_cdw_kx = tk.StringVar(value="0.0")
        self.var_cdw_ky = tk.StringVar(value="2.0944")
        self.var_cdw_color = tk.StringVar(value="#a29bfe")
        self.var_cdw_lw = tk.DoubleVar(value=1.5)
        self.var_cdw_cmin = tk.StringVar(value="-4")
        self.var_cdw_cmax = tk.StringVar(value="4")

        # Set traces for auto-rendering
        for var in [self.var_l1, self.var_l2, self.var_angle, self.var_rot, self.var_cdw_lw]:
            var.trace_add("write", self.on_slider_change)
        
        for var in [self.show_crystal, self.show_lattice, self.show_basis_labels, 
                    self.show_primitive, self.show_wigner, self.show_parallelogram, 
                    self.show_grid, self.var_show_cdw]:
            var.trace_add("write", lambda *args: self.render())

    def ui_setup(self):
        # Outer Layout structure: Left sidebar (scrollable), Right Plot area
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # Left control column
        left_panel = ScrollableFrame(self)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.scroll_inner = left_panel.scrollable_frame
        
        # Right Plot Column
        right_panel = ttk.Frame(self, style="TFrame")
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=1)

        # Configure widgets in Left Panel
        self.setup_presets_frame()
        self.setup_lattice_geometry_frame()
        self.setup_ranges_frame()
        self.setup_basis_editor_frame()
        self.setup_cdw_frame()
        self.setup_display_options_frame()
        self.setup_action_buttons_frame()

        # Matplotlib Figure Embed (Right Panel)
        self.figure = plt.figure(figsize=(7, 7), facecolor='#1e1e24')
        original_manager = self.figure.canvas.manager
        
        self.ax = self.figure.add_subplot(111, facecolor='#2a2a35')
        
        # Apply custom dark theme styling to matplotlib elements
        self.ax.tick_params(colors='#f3f3f6', which='both')
        self.ax.xaxis.label.set_color('#f3f3f6')
        self.ax.yaxis.label.set_color('#f3f3f6')
        self.ax.spines['bottom'].set_color('#3f3f4e')
        self.ax.spines['top'].set_color('#3f3f4e')
        self.ax.spines['left'].set_color('#3f3f4e')
        self.ax.spines['right'].set_color('#3f3f4e')

        self.canvas = FigureCanvasTkAgg(self.figure, master=right_panel)
        if original_manager is not None:
            self.canvas.manager = original_manager
            original_manager.canvas = self.canvas
            
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        
        # Style embedded toolbar
        toolbar_frame = ttk.Frame(right_panel, style="TFrame")
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="left", fill="x", padx=5)

    def setup_presets_frame(self):
        frame = ttk.LabelFrame(self.scroll_inner, text="격자 프리셋 (Preset)", padding=8)
        frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(frame, text="원하는 물리 모델 프리셋 선택:").pack(anchor="w", pady=(0, 4))
        self.preset_combo = ttk.Combobox(
            frame, 
            values=list(PRESETS.keys()), 
            state="readonly", 
            width=30
        )
        self.preset_combo.set("2H-TaSe2 (3x3 CDW)")
        self.preset_combo.pack(fill="x", ipady=2)
        self.preset_combo.bind("<<ComboboxSelected>>", self.on_preset_selected)

    def setup_lattice_geometry_frame(self):
        frame = ttk.LabelFrame(self.scroll_inner, text="격자 벡터 및 기하 구조 (Geometry)", padding=8)
        frame.pack(fill="x", pady=(0, 10))
        
        # 1. Vector a1 Length slider
        ttk.Label(frame, text="a1 벡터 길이 (L1):").grid(row=0, column=0, sticky="w", pady=2)
        s_l1 = ttk.Scale(frame, from_=0.1, to=5.0, variable=self.var_l1, style="Horizontal.TScale")
        s_l1.grid(row=0, column=1, sticky="ew", padx=5)
        lbl_l1 = ttk.Label(frame, text="1.00")
        lbl_l1.grid(row=0, column=2, sticky="e")
        self.var_l1.trace_add("write", lambda *args: lbl_l1.configure(text=f"{self.var_l1.get():.2f}"))

        # 2. Vector a2 Length slider
        ttk.Label(frame, text="a2 벡터 길이 (L2):").grid(row=1, column=0, sticky="w", pady=2)
        s_l2 = ttk.Scale(frame, from_=0.1, to=5.0, variable=self.var_l2, style="Horizontal.TScale")
        s_l2.grid(row=1, column=1, sticky="ew", padx=5)
        lbl_l2 = ttk.Label(frame, text="1.00")
        lbl_l2.grid(row=1, column=2, sticky="e")
        self.var_l2.trace_add("write", lambda *args: lbl_l2.configure(text=f"{self.var_l2.get():.2f}"))

        # 3. Angle Slider
        ttk.Label(frame, text="두 기저 벡터 사이 각도 (θ):").grid(row=2, column=0, sticky="w", pady=2)
        s_ang = ttk.Scale(frame, from_=10.0, to=170.0, variable=self.var_angle, style="Horizontal.TScale")
        s_ang.grid(row=2, column=1, sticky="ew", padx=5)
        lbl_ang = ttk.Label(frame, text="120.0°")
        lbl_ang.grid(row=2, column=2, sticky="e")
        self.var_angle.trace_add("write", lambda *args: lbl_ang.configure(text=f"{self.var_angle.get():.1f}°"))

        # 4. Lattice Rotation Slider
        ttk.Label(frame, text="전체 격자 회전각 (φ):").grid(row=3, column=0, sticky="w", pady=2)
        s_rot = ttk.Scale(frame, from_=-180.0, to=180.0, variable=self.var_rot, style="Horizontal.TScale")
        s_rot.grid(row=3, column=1, sticky="ew", padx=5)
        lbl_rot = ttk.Label(frame, text="0.0°")
        lbl_rot.grid(row=3, column=2, sticky="e")
        self.var_rot.trace_add("write", lambda *args: lbl_rot.configure(text=f"{self.var_rot.get():.1f}°"))

        frame.columnconfigure(1, weight=1)

        # Divider line
        ttk.Separator(frame, orient="horizontal").grid(row=4, column=0, columnspan=3, sticky="ew", pady=8)

        # 5. Lattice Origin entry
        subframe = ttk.Frame(frame, style="TFrame")
        subframe.grid(row=5, column=0, columnspan=3, sticky="ew")
        ttk.Label(subframe, text="격자 원점 (Origin Ox, Oy):").pack(side="left", padx=(0, 5))
        e_ox = ttk.Entry(subframe, textvariable=self.var_ox, width=6, style="TEntry")
        e_ox.pack(side="left", padx=2)
        e_ox.bind("<FocusOut>", lambda e: self.render())
        e_ox.bind("<Return>", lambda e: self.render())
        
        e_oy = ttk.Entry(subframe, textvariable=self.var_oy, width=6, style="TEntry")
        e_oy.pack(side="left", padx=2)
        e_oy.bind("<FocusOut>", lambda e: self.render())
        e_oy.bind("<Return>", lambda e: self.render())

        # 6. Read-only Calculated vectors (extremely useful for users!)
        self.lbl_calc_vectors = ttk.Label(
            frame, 
            text="a1 = [1.000, 0.000]\na2 = [-0.500, 0.866]", 
            justify="left", 
            style="Section.TLabel"
        )
        self.lbl_calc_vectors.grid(row=6, column=0, columnspan=3, sticky="w", pady=(8, 0))

    def setup_ranges_frame(self):
        frame = ttk.LabelFrame(self.scroll_inner, text="초격자 및 시각화 범위 (Bounds & Substructure)", padding=8)
        frame.pack(fill="x", pady=(0, 10))

        # Substructure factors n1, n2
        sub1 = ttk.Frame(frame, style="TFrame")
        sub1.pack(fill="x", pady=2)
        ttk.Label(sub1, text="하부격자 배율 (sub n1, n2):").pack(side="left")
        e_n1 = ttk.Entry(sub1, textvariable=self.var_sub_n1, width=5, style="TEntry")
        e_n1.pack(side="left", padx=(5, 2))
        e_n1.bind("<FocusOut>", lambda e: self.render())
        e_n1.bind("<Return>", lambda e: self.render())
        
        e_n2 = ttk.Entry(sub1, textvariable=self.var_sub_n2, width=5, style="TEntry")
        e_n2.pack(side="left", padx=2)
        e_n2.bind("<FocusOut>", lambda e: self.render())
        e_n2.bind("<Return>", lambda e: self.render())

        # Generate Bounds (xmin, xmax, ymin, ymax)
        sub2 = ttk.Frame(frame, style="TFrame")
        sub2.pack(fill="x", pady=4)
        ttk.Label(sub2, text="격자 생성 X 범위:").pack(side="left")
        e_xmin = ttk.Entry(sub2, textvariable=self.var_xmin, width=5, style="TEntry")
        e_xmin.pack(side="left", padx=(5, 2))
        e_xmin.bind("<FocusOut>", lambda e: self.render())
        e_xmin.bind("<Return>", lambda e: self.render())
        ttk.Label(sub2, text="to").pack(side="left", padx=2)
        e_xmax = ttk.Entry(sub2, textvariable=self.var_xmax, width=5, style="TEntry")
        e_xmax.pack(side="left", padx=2)
        e_xmax.bind("<FocusOut>", lambda e: self.render())
        e_xmax.bind("<Return>", lambda e: self.render())

        sub3 = ttk.Frame(frame, style="TFrame")
        sub3.pack(fill="x", pady=2)
        ttk.Label(sub3, text="격자 생성 Y 범위:").pack(side="left")
        e_ymin = ttk.Entry(sub3, textvariable=self.var_ymin, width=5, style="TEntry")
        e_ymin.pack(side="left", padx=(5, 2))
        e_ymin.bind("<FocusOut>", lambda e: self.render())
        e_ymin.bind("<Return>", lambda e: self.render())
        ttk.Label(sub3, text="to").pack(side="left", padx=2)
        e_ymax = ttk.Entry(sub3, textvariable=self.var_ymax, width=5, style="TEntry")
        e_ymax.pack(side="left", padx=2)
        e_ymax.bind("<FocusOut>", lambda e: self.render())
        e_ymax.bind("<Return>", lambda e: self.render())

        # View boundaries
        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=6)

        sub4 = ttk.Frame(frame, style="TFrame")
        sub4.pack(fill="x", pady=2)
        ttk.Label(sub4, text="그래프 뷰포트 X:").pack(side="left")
        e_vx1 = ttk.Entry(sub4, textvariable=self.var_view_xmin, width=6, style="TEntry")
        e_vx1.pack(side="left", padx=(10, 2))
        e_vx1.bind("<FocusOut>", lambda e: self.render())
        e_vx1.bind("<Return>", lambda e: self.render())
        ttk.Label(sub4, text="to").pack(side="left", padx=2)
        e_vx2 = ttk.Entry(sub4, textvariable=self.var_view_xmax, width=6, style="TEntry")
        e_vx2.pack(side="left", padx=2)
        e_vx2.bind("<FocusOut>", lambda e: self.render())
        e_vx2.bind("<Return>", lambda e: self.render())

        sub5 = ttk.Frame(frame, style="TFrame")
        sub5.pack(fill="x", pady=2)
        ttk.Label(sub5, text="그래프 뷰포트 Y:").pack(side="left")
        e_vy1 = ttk.Entry(sub5, textvariable=self.var_view_ymin, width=6, style="TEntry")
        e_vy1.pack(side="left", padx=(10, 2))
        e_vy1.bind("<FocusOut>", lambda e: self.render())
        e_vy1.bind("<Return>", lambda e: self.render())
        ttk.Label(sub5, text="to").pack(side="left", padx=2)
        e_vy2 = ttk.Entry(sub5, textvariable=self.var_view_ymax, width=6, style="TEntry")
        e_vy2.pack(side="left", padx=2)
        e_vy2.bind("<FocusOut>", lambda e: self.render())
        e_vy2.bind("<Return>", lambda e: self.render())

    def setup_basis_editor_frame(self):
        frame = ttk.LabelFrame(self.scroll_inner, text="단위 격자 기저 원자/결합 (Basis Artists)", padding=8)
        frame.pack(fill="both", expand=True, pady=(0, 10))

        # Guide line
        ttk.Label(
            frame, 
            text="CSV형식: kind,label,ij_points,color,style,size\n(ij_points는 하부 격자단위 분수/정수 인덱스)",
            justify="left",
            style="Section.TLabel"
        ).pack(anchor="w", pady=(0, 4))

        # Text editor
        editor_frame = ttk.Frame(frame, style="TFrame")
        editor_frame.pack(fill="both", expand=True)
        
        self.basis_editor = tk.Text(
            editor_frame, 
            height=10, 
            bg=self.entry_bg, 
            fg=self.text_color,
            insertbackground=self.text_color,
            selectbackground=self.accent,
            selectforeground=self.text_color,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightcolor=self.accent,
            highlightbackground=self.border,
            font=("Consolas", 10),
            undo=True
        )
        self.basis_editor.pack(side="left", fill="both", expand=True)
        
        scroll = ttk.Scrollbar(editor_frame, orient="vertical", command=self.basis_editor.yview)
        scroll.pack(side="right", fill="y")
        self.basis_editor.configure(yscrollcommand=scroll.set)

        # Trigger rendering when focus leaves editor or user presses Ctrl+Return
        self.basis_editor.bind("<FocusOut>", lambda e: self.render())
        self.basis_editor.bind("<Control-Return>", lambda e: self.render())

    def setup_cdw_frame(self):
        frame = ttk.LabelFrame(self.scroll_inner, text="전하밀도파 평면파 변조 (CDW Plane Wave Overlay)", padding=8)
        frame.pack(fill="x", pady=(0, 10))

        # Toggle to show plane waves
        ttk.Checkbutton(frame, text="CDW 평면파 위상선 표시", variable=self.var_show_cdw).pack(anchor="w", pady=2)

        # kx, ky inputs
        inputs1 = ttk.Frame(frame, style="TFrame")
        inputs1.pack(fill="x", pady=2)
        ttk.Label(inputs1, text="파동 벡터 kx, ky:").pack(side="left")
        e_kx = ttk.Entry(inputs1, textvariable=self.var_cdw_kx, width=8, style="TEntry")
        e_kx.pack(side="left", padx=(5, 2))
        e_kx.bind("<FocusOut>", lambda e: self.render())
        e_kx.bind("<Return>", lambda e: self.render())
        
        e_ky = ttk.Entry(inputs1, textvariable=self.var_cdw_ky, width=8, style="TEntry")
        e_ky.pack(side="left", padx=2)
        e_ky.bind("<FocusOut>", lambda e: self.render())
        e_ky.bind("<Return>", lambda e: self.render())

        # Density index range cmin, cmax
        inputs2 = ttk.Frame(frame, style="TFrame")
        inputs2.pack(fill="x", pady=2)
        ttk.Label(inputs2, text="파동 인덱스 범위:").pack(side="left")
        e_cmin = ttk.Entry(inputs2, textvariable=self.var_cdw_cmin, width=5, style="TEntry")
        e_cmin.pack(side="left", padx=(5, 2))
        e_cmin.bind("<FocusOut>", lambda e: self.render())
        e_cmin.bind("<Return>", lambda e: self.render())
        ttk.Label(inputs2, text="to").pack(side="left", padx=2)
        e_cmax = ttk.Entry(inputs2, textvariable=self.var_cdw_cmax, width=5, style="TEntry")
        e_cmax.pack(side="left", padx=2)
        e_cmax.bind("<FocusOut>", lambda e: self.render())
        e_cmax.bind("<Return>", lambda e: self.render())

        # Width and Color controls
        inputs3 = ttk.Frame(frame, style="TFrame")
        inputs3.pack(fill="x", pady=4)
        ttk.Label(inputs3, text="위상선 두께:").grid(row=0, column=0, sticky="w")
        s_cdw_w = ttk.Scale(inputs3, from_=0.2, to=4.0, variable=self.var_cdw_lw, style="Horizontal.TScale")
        s_cdw_w.grid(row=0, column=1, sticky="ew", padx=5)
        lbl_cdw_w = ttk.Label(inputs3, text="1.5")
        lbl_cdw_w.grid(row=0, column=2, sticky="e")
        self.var_cdw_lw.trace_add("write", lambda *args: lbl_cdw_w.configure(text=f"{self.var_cdw_lw.get():.1f}"))

        ttk.Label(inputs3, text="위상선 색상:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        e_cdw_clr = ttk.Entry(inputs3, textvariable=self.var_cdw_color, width=10, style="TEntry")
        e_cdw_clr.grid(row=1, column=1, sticky="w", padx=5, pady=(4, 0))
        e_cdw_clr.bind("<FocusOut>", lambda e: self.render())
        e_cdw_clr.bind("<Return>", lambda e: self.render())
        
        inputs3.columnconfigure(1, weight=1)

    def setup_display_options_frame(self):
        frame = ttk.LabelFrame(self.scroll_inner, text="시각화 옵션 (Display Options)", padding=8)
        frame.pack(fill="x", pady=(0, 10))

        # Checkboxes for various plotting components
        ttk.Checkbutton(frame, text="결정 기저 구조 그리기 (Crystal)", variable=self.show_crystal).grid(row=0, column=0, sticky="w", pady=2)
        ttk.Checkbutton(frame, text="격자점들 표시 (Lattice Points)", variable=self.show_lattice).grid(row=0, column=1, sticky="w", pady=2)
        
        ttk.Checkbutton(frame, text="기저 원소 라벨링 (Basis Labels)", variable=self.show_basis_labels).grid(row=1, column=0, sticky="w", pady=2)
        ttk.Checkbutton(frame, text="기저 벡터 화살표 (Gizmo Vectors)", variable=self.show_primitive).grid(row=1, column=1, sticky="w", pady=2)
        
        ttk.Checkbutton(frame, text="비그너-자이츠 셀 (Wigner-Seitz)", variable=self.show_wigner).grid(row=2, column=0, sticky="w", pady=2)
        ttk.Checkbutton(frame, text="평행사변형 단위 격자 (Unit Cell)", variable=self.show_parallelogram).grid(row=2, column=1, sticky="w", pady=2)
        
        ttk.Checkbutton(frame, text="배경 그리드선 표시 (Grid)", variable=self.show_grid).grid(row=3, column=0, sticky="w", pady=2)
        
        self.auto_render_toggle = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="슬라이더 값 실시간 반영 (Live Update)", variable=self.auto_render_toggle, command=self.on_auto_render_toggled).grid(row=3, column=1, sticky="w", pady=2)

        frame.columnconfigure((0, 1), weight=1)

    def setup_action_buttons_frame(self):
        frame = ttk.Frame(self.scroll_inner, style="TFrame")
        frame.pack(fill="x", pady=(5, 15))
        frame.columnconfigure((0, 1, 2), weight=1)

        ttk.Button(frame, text="수동 렌더링", command=self.manual_render).grid(row=0, column=0, sticky="ew", padx=2)
        ttk.Button(frame, text="PNG 이미지 저장", command=self.save_image).grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Button(frame, text="기본값 초기화", command=self.reset_to_preset_default).grid(row=0, column=2, sticky="ew", padx=2)

    def on_auto_render_toggled(self):
        self.auto_render_active = self.auto_render_toggle.get()
        if self.auto_render_active:
            self.render()

    def on_slider_change(self, *args):
        if self.auto_render_active:
            self.render()

    def manual_render(self):
        self.render()

    def on_preset_selected(self, event):
        preset_name = self.preset_combo.get()
        self.load_preset_data(preset_name)
        self.render()

    def load_preset_data(self, name):
        if name not in PRESETS:
            return
        
        # Temporarily disable auto render to prevent multi-render during value changes
        prev_auto = self.auto_render_active
        self.auto_render_active = False

        data = PRESETS[name]
        self.var_l1.set(data["l1"])
        self.var_l2.set(data["l2"])
        self.var_angle.set(data["angle"])
        self.var_rot.set(data["rot"])
        
        self.var_ox.set(str(data["ox"]))
        self.var_oy.set(str(data["oy"]))
        self.var_sub_n1.set(str(data["sub_n1"]))
        self.var_sub_n2.set(str(data["sub_n2"]))
        
        self.var_xmin.set(str(data["xmin"]))
        self.var_xmax.set(str(data["xmax"]))
        self.var_ymin.set(str(data["ymin"]))
        self.var_ymax.set(str(data["ymax"]))
        
        self.var_view_xmin.set(str(data["view_xmin"]))
        self.var_view_xmax.set(str(data["view_xmax"]))
        self.var_view_ymin.set(str(data["view_ymin"]))
        self.var_view_ymax.set(str(data["view_ymax"]))
        
        self.basis_editor.delete("1.0", "end")
        self.basis_editor.insert("1.0", data["basis"])
        
        self.var_show_cdw.set(data["show_cdw"])
        self.var_cdw_kx.set(str(data["cdw_kx"]))
        self.var_cdw_ky.set(str(data["cdw_ky"]))
        self.var_cdw_color.set(data["cdw_color"])
        self.var_cdw_lw.set(data["cdw_lw"])
        self.var_cdw_cmin.set(str(data["cdw_cmin"]))
        self.var_cdw_cmax.set(str(data["cdw_cmax"]))

        self.auto_render_active = prev_auto

    def reset_to_preset_default(self):
        name = self.preset_combo.get()
        self.load_preset_data(name)
        self.render()

    def get_parsed_float(self, var, name_for_err):
        try:
            return float(var.get())
        except ValueError:
            raise ValueError(f"'{name_for_err}' 입력값이 유효한 실수가 아닙니다.")

    def get_parsed_int_or_float(self, var, name_for_err):
        try:
            val = float(var.get())
            return int(val) if val.is_integer() else val
        except ValueError:
            raise ValueError(f"'{name_for_err}' 입력값이 올바르지 않습니다.")

    def parse_csv_line(self, line):
        parts = []
        current = []
        quote = None
        bracket_depth = 0
        for char in line:
            if quote:
                if char == quote:
                    quote = None
                else:
                    current.append(char)
                continue
            if char in {"'", '"'}:
                quote = char
                continue
            if char in "([{":
                bracket_depth += 1
            elif char in ")]}":
                bracket_depth -= 1
            if char == "," and bracket_depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(char)
        parts.append("".join(current))
        return parts

    def populate_basis_from_text(self, basis):
        raw_text = self.basis_editor.get("1.0", "end")
        for line_no, raw_line in enumerate(raw_text.splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = self.parse_csv_line(line)
            if len(parts) != 6:
                raise ValueError(f"기저 정의 라인 {line_no}: 6개의 열이 채워져야 합니다 (현재 {len(parts)}개)")

            kind, label, points_expr, color, marker_or_linewidth, size_text = [p.strip() for p in parts]
            
            try:
                # Use safe eval instead of ast.literal_eval to support fractions like 1/3, 2/3
                points = eval(points_expr, {"__builtins__": None, "math": math, "np": np})
            except Exception:
                raise ValueError(f"기저 정의 라인 {line_no}: '{points_expr}' 올바른 수학적 좌표 표현식이 아닙니다. (예: (1/3, 2/3) 또는 [((0,0), (1,0))])")
                
            try:
                size = float(size_text)
            except ValueError:
                raise ValueError(f"기저 정의 라인 {line_no}: 크기 값 '{size_text}'은(는) 숫자여야 합니다.")

            if kind == "point":
                marker = marker_or_linewidth or "o"
                # Keep closure binding clean
                def point_generator(x, y, c=color, m=marker, ms=size):
                    return plt.plot(x, y, color=c, marker=m, linestyle="None", markersize=ms)
                basis.add_artist(point_generator, points, label=label)

            elif kind == "line":
                linestyle = marker_or_linewidth or "-"
                def line_generator(x, y, c=color, ls=linestyle, lw=size):
                    return plt.plot(x, y, color=c, linestyle=ls, linewidth=lw)
                basis.add_artist(line_generator, points, label=label)
            else:
                raise ValueError(f"기저 정의 라인 {line_no}: kind 필드는 'point' 또는 'line'이어야 합니다 (현재 '{kind}')")

    def build_crystal_objects(self):
        # 1. Clear previous tracked instances in crypy to prevent memory leaks
        cp.TrackedInstance.clear_all_project_instances()

        # 2. Get parameter values
        l1 = self.var_l1.get()
        l2 = self.var_l2.get()
        angle_deg = self.var_angle.get()
        rot_deg = self.var_rot.get()
        
        ox = self.get_parsed_float(self.var_ox, "Origin Ox")
        oy = self.get_parsed_float(self.var_oy, "Origin Oy")
        
        # 3. Calculate 2D vectors a1 and a2 from lengths, angle and rotation
        rad_rot = math.radians(rot_deg)
        rad_angle = math.radians(angle_deg)
        
        a1 = np.array([
            l1 * math.cos(rad_rot),
            l1 * math.sin(rad_rot)
        ])
        a2 = np.array([
            l2 * math.cos(rad_rot + rad_angle),
            l2 * math.sin(rad_rot + rad_angle)
        ])

        # Show calculated vectors to user
        self.lbl_calc_vectors.configure(
            text=f"계산된 격자 벡터:\n"
                 f"a1 = [{a1[0]:.4f}, {a1[1]:.4f}]\n"
                 f"a2 = [{a2[0]:.4f}, {a2[1]:.4f}]"
        )

        pv = cp.PrimitiveVector2D(a1, a2, O=[ox, oy], gizmowidth=2)
        
        sub_n1 = self.get_parsed_int_or_float(self.var_sub_n1, "sub n1")
        sub_n2 = self.get_parsed_int_or_float(self.var_sub_n2, "sub n2")
        
        sub_pv = pv.get_sub_structure(sub_n1, sub_n2)
        basis = cp.Basis2D(sub_pv)
        self.populate_basis_from_text(basis)

        lattice = cp.LatticePoints2D(pv)
        xmin = self.get_parsed_float(self.var_xmin, "Lattice Xmin")
        xmax = self.get_parsed_float(self.var_xmax, "Lattice Xmax")
        ymin = self.get_parsed_float(self.var_ymin, "Lattice Ymin")
        ymax = self.get_parsed_float(self.var_ymax, "Lattice Ymax")
        lattice.generate_points_by_xylim((xmin, xmax), (ymin, ymax))

        crystal = cp.Crystal2D(basis, lattice)
        return pv, sub_pv, lattice, basis, crystal

    def render(self):
        try:
            # Build and generate crystal parameters
            pv, sub_pv, lattice, basis, crystal = self.build_crystal_objects()
            
            # Prepare matplotlib embedded drawing
            # Ensure the figure is managed by pyplot to avoid Gcf desynchronization ("figure not managed by pyplot")
            from matplotlib import _pylab_helpers
            if not _pylab_helpers.Gcf.has_fignum(self.figure.number):
                manager = self.figure.canvas.manager
                if manager is not None:
                    _pylab_helpers.Gcf.figs[self.figure.number] = manager
                else:
                    from matplotlib.backend_bases import FigureManagerBase
                    manager = FigureManagerBase(self.figure.canvas, self.figure.number)
                    _pylab_helpers.Gcf.figs[self.figure.number] = manager

            plt.figure(self.figure.number)
            self.ax.clear()
            plt.sca(self.ax)
            
            # Drawing elements depending on UI toggles
            if self.show_crystal.get():
                crystal.plot_crystal()
            if self.show_lattice.get():
                # Grey background dots for general lattice coordinates
                lattice.plot_scatter(c="#57606f", s=15, marker="+", zorder=1)
            if self.show_basis_labels.get():
                basis.plot_basis()
            if self.show_primitive.get():
                pv.plot_gizmo()
            if self.show_wigner.get():
                pv.plot_wigner_seitz_2d()
            if self.show_parallelogram.get():
                pv.plot_paral_2d()
            
            # Plane Wave Overlay (CDW Phase Lines)
            if self.var_show_cdw.get():
                kx = self.get_parsed_float(self.var_cdw_kx, "CDW kx")
                ky = self.get_parsed_float(self.var_cdw_ky, "CDW ky")
                cmin = int(self.get_parsed_float(self.var_cdw_cmin, "CDW Index Min"))
                cmax = int(self.get_parsed_float(self.var_cdw_cmax, "CDW Index Max"))
                
                # Check for zero wavevector to avoid zero-division errors
                if kx != 0.0 or ky != 0.0:
                    ox = self.get_parsed_float(self.var_ox, "Origin Ox")
                    oy = self.get_parsed_float(self.var_oy, "Origin Oy")
                    color = self.var_cdw_color.get().strip() or "#a29bfe"
                    lw = self.var_cdw_lw.get()
                    
                    cp.Collection.plot_plane_wave_lines(
                        k=[kx, ky],
                        length=40,
                        index_range=(cmin, cmax),
                        origin=(ox, oy),
                        color=color,
                        ax=self.ax,
                        lw=lw
                    )

            # Boundaries
            vx_min = self.get_parsed_float(self.var_view_xmin, "Viewport Xmin")
            vx_max = self.get_parsed_float(self.var_view_xmax, "Viewport Xmax")
            vy_min = self.get_parsed_float(self.var_view_ymin, "Viewport Ymin")
            vy_max = self.get_parsed_float(self.var_view_ymax, "Viewport Ymax")
            
            self.ax.set_xlim(vx_min, vx_max)
            self.ax.set_ylim(vy_min, vy_max)
            
            # Set beautiful grid lines inside the dark canvas
            if self.show_grid.get():
                self.ax.grid(True, color="#3f3f4e", linestyle=":", linewidth=0.8, zorder=0)
            else:
                self.ax.grid(False)

            self.ax.set_aspect("equal", adjustable="box")
            
            # Refresh Matplotlib layout
            self.figure.tight_layout()
            self.canvas.draw_idle()
            
        except Exception as err:
            # Output rendering errors directly to title or console/message without popping an annoying modal in slider loop
            # Only pop warning box on manual click or serious structure errors
            if not self.auto_render_active:
                messagebox.showerror("렌더링 실패", f"오류: {str(err)}")
            else:
                # Show subtle visual warning by styling grid or vector label
                self.lbl_calc_vectors.configure(text=f"오류: {str(err)}", foreground="#ff4d4d")

    def save_image(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG 고해상도 이미지", "*.png"), ("SVG 벡터 그래픽스", "*.svg"), ("모든 파일", "*.*")],
            initialfile="2H_TaSe2_crystal_model.png"
        )
        if not path:
            return
        try:
            self.figure.savefig(path, dpi=300, bbox_inches="tight", facecolor='#1e1e24')
            messagebox.showinfo("저장 완료", f"이미지가 성공적으로 저장되었습니다:\n{path}")
        except Exception as err:
            messagebox.showerror("저장 실패", f"이미지를 저장하지 못했습니다.\n오류: {str(err)}")


def main():
    app = CrypyApp()
    app.mainloop()


if __name__ == "__main__":
    main()
