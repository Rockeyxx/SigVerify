"""
styles.py — Centralized visual theme for the Signature Verification System

Applied once at the QApplication level via app.setStyleSheet(STYLE).
This keeps the theme maintainable from a single file, and means no
other file (app.py, enrollment.py, verification.py) needs inline
styling for the base look — only functionally-meaningful colors
(result green/red, degraded-mode orange) stay inline since they
carry meaning, not just decoration.
"""

# ---- Palette ----
PRIMARY_DARK   = "#1a2332"   # deep slate — headers, main window bg
PRIMARY        = "#24304a"   # panel backgrounds
ACCENT         = "#3d7dfc"   # accent blue — primary buttons
ACCENT_HOVER   = "#5c93ff"
TEXT_LIGHT     = "#eef1f7"
TEXT_MUTED     = "#9aa5b8"
BORDER         = "#3a4761"
SUCCESS        = "#2e7d32"
SUCCESS_BG     = "#e8f5e9"
DANGER         = "#c62828"
DANGER_BG      = "#ffebee"
WARNING        = "#e65100"

STYLE = f"""
QWidget {{
    background-color: {PRIMARY_DARK};
    color: {TEXT_LIGHT};
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}}

QLabel {{
    background: transparent;
}}

QLabel[role="title"] {{
    font-size: 22px;
    font-weight: 700;
    color: {TEXT_LIGHT};
}}

QLabel[role="subtitle"] {{
    font-size: 12px;
    color: {TEXT_MUTED};
}}

QLabel[role="section"] {{
    font-size: 13px;
    font-weight: 600;
    color: {TEXT_MUTED};
    margin-top: 6px;
}}

QLineEdit, QComboBox {{
    background-color: {PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 8px 10px;
    color: {TEXT_LIGHT};
}}

QLineEdit:focus, QComboBox:focus {{
    border: 1px solid {ACCENT};
}}

QComboBox::drop-down {{
    border: none;
}}

QComboBox QAbstractItemView {{
    background-color: {PRIMARY};
    color: {TEXT_LIGHT};
    selection-background-color: {ACCENT};
}}

QPushButton {{
    background-color: {PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 10px 14px;
    color: {TEXT_LIGHT};
    font-weight: 600;
}}

QPushButton:hover {{
    background-color: {BORDER};
}}

QPushButton:disabled {{
    color: {TEXT_MUTED};
    background-color: {PRIMARY_DARK};
}}

QPushButton[role="primary"] {{
    background-color: {ACCENT};
    border: none;
    color: white;
    font-size: 14px;
    padding: 12px 14px;
}}

QPushButton[role="primary"]:hover {{
    background-color: {ACCENT_HOVER};
}}

QPushButton[role="ghost"] {{
    background-color: transparent;
    border: none;
    color: {TEXT_MUTED};
    text-align: left;
    font-weight: 500;
}}

QPushButton[role="ghost"]:hover {{
    color: {ACCENT};
}}

QListWidget {{
    background-color: {PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    color: {TEXT_LIGHT};
}}

QLabel[role="result-idle"] {{
    font-size: 16px;
    font-weight: 600;
    padding: 16px;
    border: 1px solid {BORDER};
    border-radius: 8px;
    background-color: {PRIMARY};
    color: {TEXT_MUTED};
}}

QLabel[role="result-genuine"] {{
    font-size: 20px;
    font-weight: 700;
    padding: 16px;
    border: 1px solid {SUCCESS};
    border-radius: 8px;
    background-color: {SUCCESS_BG};
    color: {SUCCESS};
}}

QLabel[role="result-forged"] {{
    font-size: 20px;
    font-weight: 700;
    padding: 16px;
    border: 1px solid {DANGER};
    border-radius: 8px;
    background-color: {DANGER_BG};
    color: {DANGER};
}}

QLabel[role="warning"] {{
    color: {WARNING};
    font-weight: 700;
    font-size: 12px;
}}

QLabel[role="muted"] {{
    color: {TEXT_MUTED};
    font-size: 12px;
}}

QLabel[role="status-success"] {{
    color: {SUCCESS};
    font-weight: 700;
}}

QLabel[role="status-error"] {{
    color: {DANGER};
    font-weight: 700;
}}
QLabel[role="hero-title"] {{
    font-size: 30px;
    font-weight: 800;
    color: {TEXT_LIGHT};
    letter-spacing: 0.5px;
}}

QLabel[role="hero-subtitle"] {{
    font-size: 13px;
    color: {TEXT_MUTED};
    letter-spacing: 1px;
}}
QWidget[role="sidebar"] {{
    background-color: {PRIMARY};
    border-right: 1px solid {BORDER};
}}

QLabel[role="sidebar-name"] {{
    font-size: 16px;
    font-weight: 700;
    color: {TEXT_LIGHT};
}}

QPushButton[role="sidebar-nav"] {{
    background-color: transparent;
    border: none;
    text-align: left;
    padding: 10px 8px;
    color: {TEXT_LIGHT};
    font-weight: 600;
}}

QPushButton[role="sidebar-nav"]:hover {{
    background-color: {BORDER};
    border-radius: 6px;
}}

QPushButton[role="sidebar-nav-danger"] {{
    background-color: transparent;
    border: none;
    text-align: left;
    padding: 10px 8px;
    color: {DANGER};
    font-weight: 600;
}}

QPushButton[role="danger"] {{
    background-color: transparent;
    border: 1px solid {DANGER};
    color: {DANGER};
    border-radius: 6px;
    padding: 10px 14px;
    font-weight: 600;
}}

QPushButton[role="danger"]:hover {{
    background-color: {DANGER_BG};
}}
"""