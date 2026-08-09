import base64
import io
import math
import numpy as np
import flet as ft
from PIL import Image, ImageDraw, ImageFont

# 標稱電壓列表[cite: 2]
VOLTAGE_OPTIONS = [
    ("161", 161000.0),
    ("22.8", 22800.0),
    ("11.4", 11400.0),
    ("6.6", 6600.0),
    ("4.16", 4160.0),
    ("3.3", 3300.0),
    ("0.46", 460.0),
    ("0.38", 380.0),
    ("0.22", 220.0),
]

# Schneider P3 曲線家族與型態對應字典[cite: 2]
CURVE_FAMILY_MAP = {
    "IEC": ["NI", "VI", "EI", "LTI"],
    "IEEE": ["MI", "VI", "EI", "LTI", "LTVI", "LTEI", "STI", "STEI"],
    "IEEE2": ["MI", "NI", "VI", "EI"]
}

# 繪圖尺寸與邊界參數 (對齊原 Matplotlib 圖表比例)[cite: 2]
FIG_W_PX, FIG_H_PX = 500, 320
LEFT_MARGIN, RIGHT_MARGIN = 0.12, 0.95
TOP_MARGIN, BOTTOM_MARGIN = 0.88, 0.14

# -----------------------------------------------------------------------------
# 跳脫時間計算邏輯 (50/51)[cite: 2]
# -----------------------------------------------------------------------------
def calc_trip_time(standard, curve_type, I_base, Ip_base, TMS_TD, enable_51, enable_50, inst_ip_base, inst_time):
    t = np.full_like(I_base, np.nan, dtype=float)

    # 1. 51 反時保護[cite: 2]
    if enable_51 and Ip_base > 0:
        M = I_base / Ip_base
        valid_51 = M >= 1.001

        if standard == "IEC":
            iec_params = {
                "NI": (0.14, 0.02),
                "VI": (13.5, 1.0),
                "EI": (80.0, 2.0),
                "LTI": (120.0, 1.0)
            }
            if curve_type in iec_params:
                A, B = iec_params[curve_type]
                t[valid_51] = TMS_TD * (A / (np.power(M[valid_51], B) - 1.0))

        elif standard in ["IEEE", "ANSI"]:
            ieee_params = {
                "MI": (0.0515, 0.1140, 0.02),
                "VI": (19.61, 0.4910, 2.0),
                "EI": (28.2, 0.1217, 2.0),
                "LTI": (0.086, 0.185, 0.02),
                "LTVI": (28.55, 0.712, 2.0),
                "LTEI": (64.07, 0.250, 2.0),
                "STI": (0.16758, 0.11858, 0.02),
                "STEI": (1.281, 0.005, 2.0)
            }
            if curve_type in ieee_params:
                A, B, C = ieee_params[curve_type]
                t[valid_51] = TMS_TD * ((A / (np.power(M[valid_51], C) - 1.0)) + B)

        elif standard == "IEEE2":
            ieee2_params = {
                "MI": (0.1735, 0.6791, 0.8, -0.08, 0.1271),
                "NI": (0.0274, 2.2614, 0.3, -4.1899, 9.1272),
                "VI": (0.0615, 0.7989, 0.34, -0.284, 4.0505),
                "EI": (0.0399, 0.2294, 0.5, 3.0094, 0.7222)
            }
            if curve_type in ieee2_params:
                A, B, C, D, E = ieee2_params[curve_type]
                m_val = M[valid_51]
                valid_m = m_val > C
                
                m_sub = m_val[valid_m] - C
                t_calc = TMS_TD * (A + (B / m_sub) + (D / (m_sub ** 2)) + (E / (m_sub ** 3)))
                
                idx_51 = np.where(valid_51)[0]
                t[idx_51[valid_m]] = t_calc

    # 2. 50 瞬跳保護[cite: 2]
    if enable_50 and inst_ip_base > 0:
        inst_mask = I_base >= inst_ip_base
        if enable_51:
            t[inst_mask] = np.nanmin([t[inst_mask], np.full(np.sum(inst_mask), inst_time)], axis=0)
        else:
            t[inst_mask] = inst_time

    t[t > 10000] = np.nan
    return t

# -----------------------------------------------------------------------------
# 純 PIL 高效對數圖表繪製引擎[cite: 2]
# -----------------------------------------------------------------------------
def render_trip_curve_pil(stage_configs, default_colors):
    img = Image.new("RGB", (FIG_W_PX, FIG_H_PX), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    x_min, x_max = 10.0, 100000.0
    y_min, y_max = 0.001, 360.0

    log_x_min, log_x_max = math.log10(x_min), math.log10(x_max)
    log_y_min, log_y_max = math.log10(y_min), math.log10(y_max)

    plot_x0 = int(FIG_W_PX * LEFT_MARGIN)
    plot_x1 = int(FIG_W_PX * RIGHT_MARGIN)
    plot_y0 = int(FIG_H_PX * (1 - TOP_MARGIN))
    plot_y1 = int(FIG_H_PX * (1 - BOTTOM_MARGIN))

    def val_to_px(val_x, val_y):
        lx = math.log10(max(val_x, x_min))
        ly = math.log10(max(val_y, y_min))
        px = plot_x0 + (lx - log_x_min) / (log_x_max - log_x_min) * (plot_x1 - plot_x0)
        py = plot_y1 - (ly - log_y_min) / (log_y_max - log_y_min) * (plot_y1 - plot_y0)
        return px, py

    # 畫背景外框[cite: 2]
    draw.rectangle([plot_x0, plot_y0, plot_x1, plot_y1], outline="#333333", width=1)

    # 畫 X 軸對數網格[cite: 2]
    for dec in range(1, 6):
        base_val = 10**dec
        for sub in range(1, 10):
            v = base_val * sub
            if v > x_max: break
            px, _ = val_to_px(v, y_min)
            is_major = (sub == 1)
            draw.line([(px, plot_y0), (px, plot_y1)], fill="#E0E0E0" if not is_major else "#B0BEC5", width=1)
            if is_major and px <= plot_x1:
                label = f"{int(v)}" if v < 1000 else f"{int(v//1000)}k"
                draw.text((px - 8, plot_y1 + 4), label, fill="#333333", font=font)

    # 畫 Y 軸對數網格[cite: 2]
    y_ticks = [0.001, 0.01, 0.1, 1, 10, 100]
    for y_val in y_ticks:
        _, py = val_to_px(x_min, y_val)
        draw.line([(plot_x0, py), (plot_x1, py)], fill="#B0BEC5", width=1)
        draw.text((plot_x0 - 32, py - 6), f"{y_val:g}", fill="#333333", font=font)

    # 繪製各迴路 Trip Curve[cite: 2]
    active_voltages = [cfg["voltage"] for cfg in stage_configs if cfg["enable_51"] or cfg["enable_50"]]
    v_base = max(active_voltages) if active_voltages else 161000.0
    v_base_kv = f"{v_base/1000:.2f}kV" if v_base >= 1000 else f"{int(v_base)}V"

    I_base_range = np.logspace(np.log10(x_min), np.log10(x_max), 600)
    legend_items = []

    for i, config in enumerate(stage_configs):
        if not config["enable_51"] and not config["enable_50"]:
            continue

        ratio = config["voltage"] / v_base
        ip_base = config["ip"] * ratio
        inst_ip_base = config["inst_ip"] * ratio

        if not config["enable_51"] and config["enable_50"]:
            I_curve = np.array([inst_ip_base, inst_ip_base, x_max])
            t_curve = np.array([y_max, config["inst_time"], config["inst_time"]])
        else:
            I_curve = I_base_range.copy()
            t_curve = calc_trip_time(
                standard=config["std"], curve_type=config["type"], I_base=I_curve, 
                Ip_base=ip_base, TMS_TD=config["tms"], enable_51=config["enable_51"],
                enable_50=config["enable_50"], inst_ip_base=inst_ip_base, inst_time=config["inst_time"]
            )

            if config["enable_51"] and config["enable_50"] and inst_ip_base > 0:
                t_51_at_inst = calc_trip_time(
                    standard=config["std"], curve_type=config["type"], I_base=np.array([inst_ip_base]),
                    Ip_base=ip_base, TMS_TD=config["tms"], enable_51=True, enable_50=False,
                    inst_ip_base=0, inst_time=0
                )[0]
                if not np.isnan(t_51_at_inst) and t_51_at_inst > config["inst_time"]:
                    idx = np.searchsorted(I_curve, inst_ip_base)
                    I_curve = np.insert(I_curve, idx, inst_ip_base)
                    t_curve = np.insert(t_curve, idx, t_51_at_inst)

        # 將座標轉為點陣繪圖點[cite: 2]
        pts = []
        for ix, tx in zip(I_curve, t_curve):
            if not np.isnan(tx) and y_min <= tx <= y_max:
                pts.append(val_to_px(ix, tx))

        if len(pts) > 1:
            draw.line(pts, fill=default_colors[i], width=2)

        v_str = f"{config['voltage']/1000:.1f}kV" if config['voltage'] >= 1000 else f"{int(config['voltage'])}V"
        legend_items.append((f"{config['name']} ({v_str})", default_colors[i]))

    # 標題與標籤文字[cite: 2]
    draw.text((plot_x0 + 80, 8), "Time-Current Characteristic", fill="#111111", font=font)
    draw.text((plot_x0 + 60, plot_y1 + 22), f"Current (A) [Reflected to {v_base_kv}]", fill="#333333", font=font)

    # 畫圖例 (Legend)[cite: 2]
    leg_x = plot_x1 - 105
    leg_y = plot_y0 + 10
    for name, color in legend_items:
        draw.rectangle([leg_x, leg_y + 2, leg_x + 10, leg_y + 8], fill=color)
        draw.text((leg_x + 14, leg_y - 1), name, fill="#222222", font=font)
        leg_y += 12

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_b64}"

# -----------------------------------------------------------------------------
# Flet 主應用程式
# -----------------------------------------------------------------------------
def main(page: ft.Page):
    page.title = "保護協調曲線 (Schneider Electric)"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 6
    page.spacing = 6

    # 調整為直式 UI 尺寸[cite: 2]
    page.window.width = 480
    page.window.height = 840
    page.window.resizable = True

    default_colors = ["#E63946", "#F4A261", "#2A9D8F", "#457B9D", "#1D3557", "#8D99AE"]
    
    stage_configs = [
        {"name": "IED_1", "voltage": 161000.0, "enable_51": True,  "std": "IEC",   "type": "NI", "ip": 200,  "tms": 0.4, "enable_50": True,  "inst_ip": 1200, "inst_time": 0.03},
        {"name": "IED_2", "voltage": 22800.0,  "enable_51": True,  "std": "IEC",   "type": "NI", "ip": 800,  "tms": 0.3, "enable_50": True,  "inst_ip": 7000, "inst_time": 0.02},
        {"name": "IED_3", "voltage": 11400.0,  "enable_51": True,  "std": "IEC",   "type": "NI", "ip": 1500, "tms": 0.2, "enable_50": True,  "inst_ip": 12000, "inst_time": 0.01},
        {"name": "IED_4", "voltage": 380.0,    "enable_51": False, "std": "IEEE2", "type": "EI", "ip": 800,  "tms": 0.4, "enable_50": False, "inst_ip": 3000, "inst_time": 0.03},
        {"name": "IED_5", "voltage": 4160.0,   "enable_51": False, "std": "IEEE",  "type": "VI", "ip": 1200, "tms": 0.5, "enable_50": False, "inst_ip": 10000,"inst_time": 0.03},
        {"name": "IED_6", "voltage": 220.0,    "enable_51": False, "std": "IEEE",  "type": "VI", "ip": 2000, "tms": 0.6, "enable_50": False, "inst_ip": 15000,"inst_time": 0.03},
    ]

    current_selected_index = [0]
    IMG_W, IMG_H = FIG_W_PX, FIG_H_PX

    chart_image = ft.Image(src="", fit="fill", width=IMG_W, height=IMG_H)

    cursor_line = ft.Container(
        content=ft.Column(
            controls=[ft.Container(height=4, width=1.5, bgcolor="#E63946") for _ in range(40)],
            spacing=3,
            alignment=ft.MainAxisAlignment.START,
        ),
        width=2,
        visible=False,
        top=IMG_H * (1.0 - TOP_MARGIN),
        height=IMG_H * (TOP_MARGIN - BOTTOM_MARGIN),
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
    )

    hover_I_val_text = ft.Text("", size=9, weight=ft.FontWeight.BOLD, color="#1D3557")
    hover_details_column = ft.Column(spacing=1)

    hover_card = ft.Container(
        content=ft.Column(
            controls=[
                ft.Row(
                    controls=[
                        ft.Container(width=4),
                        ft.Text("📍 電流:", size=9, weight=ft.FontWeight.BOLD, color="#1D3557", expand=True),
                        hover_I_val_text,
                    ],
                    spacing=1,
                ),
                ft.Divider(height=1, color="#E0E0E0"),
                hover_details_column,
            ],
            spacing=2,
        ),
        padding=5,
        bgcolor="#FFFFFF",
        border=ft.Border.all(1, "#B0BEC5"),
        border_radius=5,
        shadow=ft.BoxShadow(spread_radius=1, blur_radius=4, color="black12"),
        visible=False,
        top=35,
        right=25,
        width=120,
    )

    def generate_static_chart_src():
        return render_trip_curve_pil(stage_configs, default_colors)

    def on_chart_hover(e):
        px = getattr(e.local_position, "x", None) if hasattr(e, "local_position") and e.local_position else getattr(e, "x", None)
        if px is None:
            return

        x_min_px = IMG_W * LEFT_MARGIN
        x_max_px = IMG_W * RIGHT_MARGIN

        if px < x_min_px or px > x_max_px:
            if cursor_line.visible or hover_card.visible:
                cursor_line.visible = False
                hover_card.visible = False
                cursor_line.update()
                hover_card.update()
            return

        norm_x = (px - x_min_px) / (x_max_px - x_min_px)
        calc_I = 10 ** (1.0 + norm_x * 4.0)

        cursor_line.left = px
        cursor_line.visible = True

        active_voltages = [cfg["voltage"] for cfg in stage_configs if cfg["enable_51"] or cfg["enable_50"]]
        v_base = max(active_voltages) if active_voltages else 161000.0

        hover_I_val_text.value = f"{calc_I:,.1f} A"
        hover_details_column.controls.clear()

        for i, config in enumerate(stage_configs):
            if not config["enable_51"] and not config["enable_50"]:
                continue

            ratio = config["voltage"] / v_base
            ip_base = config["ip"] * ratio
            inst_ip_base = config["inst_ip"] * ratio

            t_val = calc_trip_time(
                standard=config["std"], curve_type=config["type"], I_base=np.array([calc_I]),
                Ip_base=ip_base, TMS_TD=config["tms"], enable_51=config["enable_51"],
                enable_50=config["enable_50"], inst_ip_base=inst_ip_base, inst_time=config["inst_time"]
            )[0]

            t_str = "不動作" if np.isnan(t_val) else f"{t_val:.3f} s"
            v_str = f"{config['voltage']/1000:.1f}kV" if config['voltage'] >= 1000 else f"{int(config['voltage'])}V"
            
            hover_details_column.controls.append(
                ft.Row(
                    controls=[
                        ft.Container(width=5, height=5, bgcolor=default_colors[i], border_radius=2.5),
                        ft.Text(f"{config['name']} ({v_str}):", size=8.5, weight=ft.FontWeight.W_500, expand=True),
                        ft.Text(t_str, size=8.5, weight=ft.FontWeight.BOLD, color="#2B2D42"),
                    ],
                    spacing=1,
                )
            )

        hover_card.visible = True
        cursor_line.update()
        hover_card.update()

    def on_chart_exit(e):
        if cursor_line.visible or hover_card.visible:
            cursor_line.visible = False
            hover_card.visible = False
            cursor_line.update()
            hover_card.update()

    chart_stack = ft.Stack(
        controls=[chart_image, cursor_line, hover_card],
        width=IMG_W,
        height=IMG_H,
    )

    chart_container = ft.Container(
        content=ft.GestureDetector(
            content=chart_stack,
            on_hover=on_chart_hover,
            on_exit=on_chart_exit,
        ),
        alignment=ft.Alignment(0, 0),
    )

    # 調整輸入框高度與內距提升視覺美觀[cite: 2]
    INPUT_HEIGHT = 48
    CHK_SLOT_WIDTH = 30
    style_text_10 = ft.TextStyle(size=12)
    pad_box = ft.Padding(6, 6, 6, 6)

    dd_loop_select = ft.Dropdown(
        label="迴路",
        options=[ft.dropdown.Option(key=str(i), text=cfg["name"]) for i, cfg in enumerate(stage_configs)],
        value="0", dense=True, text_size=11, label_style=style_text_10,
        content_padding=pad_box, height=INPUT_HEIGHT, expand=True
    )
    dd_voltage = ft.Dropdown(
        label="電壓(kV)",
        options=[ft.dropdown.Option(key=str(val), text=name) for name, val in VOLTAGE_OPTIONS],
        dense=True, text_size=11, label_style=style_text_10,
        content_padding=pad_box, height=INPUT_HEIGHT, expand=True
    )

    chk_enable_51 = ft.Checkbox(value=True)
    
    dd_std = ft.Dropdown(
        label="標準",
        options=[ft.dropdown.Option("IEC"), ft.dropdown.Option("IEEE"), ft.dropdown.Option("IEEE2")],
        dense=True, expand=True, text_size=11, label_style=style_text_10,
        content_padding=pad_box, height=INPUT_HEIGHT
    )
    dd_type = ft.Dropdown(
        label="型態", options=[], dense=True, expand=True,
        text_size=11, label_style=style_text_10, content_padding=pad_box, height=INPUT_HEIGHT
    )
    
    tf_ip = ft.TextField(label="51 Ip (A)", dense=True, expand=True, text_size=11, label_style=style_text_10, keyboard_type=ft.KeyboardType.NUMBER, content_padding=pad_box, height=INPUT_HEIGHT)
    tf_tms = ft.TextField(label="TMS/TD", dense=True, expand=True, text_size=11, label_style=style_text_10, keyboard_type=ft.KeyboardType.NUMBER, content_padding=pad_box, height=INPUT_HEIGHT)

    chk_enable_50 = ft.Checkbox(value=False)
    tf_inst_ip = ft.TextField(label="50 Ip (A)", dense=True, expand=True, text_size=11, label_style=style_text_10, keyboard_type=ft.KeyboardType.NUMBER, content_padding=pad_box, height=INPUT_HEIGHT)
    tf_inst_time = ft.TextField(label="時間 (s)", dense=True, expand=True, text_size=11, label_style=style_text_10, keyboard_type=ft.KeyboardType.NUMBER, content_padding=pad_box, height=INPUT_HEIGHT)

    tf_test_I = ft.TextField(label="電流(A)", dense=True, expand=True, text_size=11, label_style=style_text_10, keyboard_type=ft.KeyboardType.NUMBER, content_padding=pad_box, height=INPUT_HEIGHT)
    tf_test_result = ft.TextField(
        label="跳脫時間 (s)", value="-", read_only=True, dense=True, expand=True,
        text_size=11, label_style=style_text_10, content_padding=pad_box, height=INPUT_HEIGHT,
        text_style=ft.TextStyle(color="#E63946", weight=ft.FontWeight.BOLD)
    )

    def update_type_options(std_val: str, current_type_val: str = None):
        available_types = CURVE_FAMILY_MAP.get(std_val, ["NI", "VI", "EI"])
        dd_type.options = [ft.dropdown.Option(t) for t in available_types]
        if current_type_val in available_types:
            dd_type.value = current_type_val
        else:
            dd_type.value = available_types[0]

    def update_test_current_result():
        idx = current_selected_index[0]
        cfg = stage_configs[idx]
        active_voltages = [c["voltage"] for c in stage_configs if c["enable_51"] or c["enable_50"]]
        v_base = max(active_voltages) if active_voltages else 161000.0

        val_str = tf_test_I.value.strip() if tf_test_I.value else ""
        if not val_str:
            tf_test_result.value = "-"
            return

        try:
            test_i = float(val_str)
            if test_i <= 0:
                tf_test_result.value = "-"
            else:
                ratio = cfg["voltage"] / v_base
                ip_base = cfg["ip"] * ratio
                inst_ip_base = cfg["inst_ip"] * ratio

                t_val = calc_trip_time(
                    standard=cfg["std"], curve_type=cfg["type"],
                    I_base=np.array([test_i * ratio]),
                    Ip_base=ip_base, TMS_TD=cfg["tms"],
                    enable_51=cfg["enable_51"], enable_50=cfg["enable_50"],
                    inst_ip_base=inst_ip_base, inst_time=cfg["inst_time"]
                )[0]

                tf_test_result.value = "不動作" if np.isnan(t_val) else f"{t_val:.3f}"
        except ValueError:
            tf_test_result.value = "格式錯誤"

    # 原封不動的資料寫入邏輯[cite: 2]
    def load_loop_data_to_ui(idx: int):
        cfg = stage_configs[idx]
        dd_voltage.value = str(cfg["voltage"])
        chk_enable_51.value = cfg["enable_51"]
        dd_std.value = cfg["std"]
        
        update_type_options(cfg["std"], cfg["type"])

        tf_ip.value = str(cfg["ip"])
        tf_tms.value = str(cfg["tms"])
        chk_enable_50.value = cfg["enable_50"]
        tf_inst_ip.value = str(cfg["inst_ip"])
        tf_inst_time.value = str(cfg["inst_time"])
        update_test_current_result()

    def on_std_changed(e):
        update_type_options(dd_std.value)
        update_and_redraw(e)

    # 原封不動的選擇迴路處理[cite: 2]
    def on_loop_selected(e):
        if e.control.value is not None:
            idx = int(e.control.value)
            current_selected_index[0] = idx
            load_loop_data_to_ui(idx)
            chart_image.src = generate_static_chart_src()
            page.update()

    def update_and_redraw(e=None):
        idx = current_selected_index[0]
        cfg = stage_configs[idx]

        try: cfg["voltage"] = float(dd_voltage.value)
        except (ValueError, TypeError): cfg["voltage"] = 161000.0

        cfg["enable_51"] = chk_enable_51.value
        cfg["std"] = dd_std.value
        cfg["type"] = dd_type.value
        try: cfg["ip"] = float(tf_ip.value)
        except ValueError: cfg["ip"] = 100.0
        try: cfg["tms"] = float(tf_tms.value)
        except ValueError: cfg["tms"] = 0.1

        cfg["enable_50"] = chk_enable_50.value
        try: cfg["inst_ip"] = float(tf_inst_ip.value)
        except ValueError: cfg["inst_ip"] = 1000.0
        try: cfg["inst_time"] = float(tf_inst_time.value)
        except ValueError: cfg["inst_time"] = 0.03

        update_test_current_result()
        chart_image.src = generate_static_chart_src()
        page.update()

    def on_test_I_changed(e):
        update_test_current_result()
        tf_test_result.update()

    # 原封不動的事件綁定機制[cite: 2]
    dd_loop_select.on_select = on_loop_selected
    dd_voltage.on_select = update_and_redraw
    dd_std.on_select = on_std_changed
    dd_type.on_select = update_and_redraw

    chk_enable_51.on_change = update_and_redraw
    tf_ip.on_change = update_and_redraw
    tf_tms.on_change = update_and_redraw

    chk_enable_50.on_change = update_and_redraw
    tf_inst_ip.on_change = update_and_redraw
    tf_inst_time.on_change = update_and_redraw

    tf_test_I.on_change = on_test_I_changed

    # 初始載入 IED_1[cite: 2]
    load_loop_data_to_ui(0)
    chart_image.src = generate_static_chart_src()

    # --- 調整版面結構為直向 Stack/Column 上下分割 ---[cite: 2]
    top_panel = ft.Container(
        content=chart_container,
        alignment=ft.Alignment(0, 0),
        padding=0
    )

    bottom_panel = ft.Container(
        content=ft.Column(
            controls=[
                ft.Text("⚡ 保護參數設定", size=12, weight=ft.FontWeight.BOLD),
                ft.Row([ft.Container(width=CHK_SLOT_WIDTH), dd_loop_select, dd_voltage], spacing=4, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Divider(height=1),
                ft.Row([ft.Container(width=CHK_SLOT_WIDTH, content=chk_enable_51, alignment=ft.Alignment(0, 0)), dd_std, dd_type], spacing=4, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Row([ft.Container(width=CHK_SLOT_WIDTH), tf_ip, tf_tms], spacing=4, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Divider(height=1),
                ft.Row([ft.Container(width=CHK_SLOT_WIDTH, content=chk_enable_50, alignment=ft.Alignment(0, 0)), tf_inst_ip, tf_inst_time], spacing=4, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Divider(height=1),
                ft.Text("🎯 電流測試試算", size=11, weight=ft.FontWeight.BOLD, color="#1D3557"),
                ft.Row([ft.Container(width=CHK_SLOT_WIDTH), tf_test_I, tf_test_result], spacing=4, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ],
            spacing=4,
        ),
        padding=8,
        border=ft.Border.all(1, "#DDDDDD"),
        border_radius=8,
        bgcolor="#FAFAFA",
        expand=True,
    )

    page.add(
        ft.Column(
            controls=[
                top_panel,
                bottom_panel,
            ],
            expand=True,
            spacing=6,
            scroll=ft.ScrollMode.AUTO,
        )
    )

if __name__ == "__main__":
    ft.run(main)