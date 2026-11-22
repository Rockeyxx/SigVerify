"""
styles.py — Centralized visual theme: Teal + Ink palette

Card-focused layout: sidebar and content live in distinct panels with
subtle background contrast instead of heavy borders. Applied once at
the QApplication level via app.setStyleSheet(STYLE).
"""

# ---- Palette: Teal + Ink ----
PRIMARY_DARK   = "#0B1A17"   # deep ink-teal — page/window background
PRIMARY        = "#122822"   # panel/card background (one step lighter)
PRIMARY_LIGHT  = "#1A362E"   # nested card / hover background
ACCENT         = "#1D9E75"   # teal accent — primary buttons, active nav
ACCENT_HOVER   = "#28B98A"
ACCENT_DARK    = "#085041"   # dark teal for text-on-light-teal
TEXT_LIGHT     = "#EAF7F2"
TEXT_MUTED     = "#8FAFA6"
BORDER         = "#1E3D34"
SUCCESS        = "#1D9E75"
SUCCESS_BG     = "#DCF3EA"
SUCCESS_TEXT   = "#085041"
DANGER         = "#D8544E"
DANGER_BG      = "#FBE4E2"
DANGER_TEXT    = "#7A241F"
WARNING        = "#D89A3D"

STYLE = f"""
QWidget {{
    background-color: {PRIMARY_DARK};
    color: {TEXT_LIGHT};
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 15px;
}}

QLabel {{
    background: transparent;
}}

/* ---- Typography roles ---- */
QLabel[role="hero-title"] {{
    font-size: 28px;
    font-weight: 800;
    color: {TEXT_LIGHT};
    letter-spacing: 0.3px;
}}

QLabel[role="hero-subtitle"] {{
    font-size: 12px;
    color: {TEXT_MUTED};
    letter-spacing: 1px;
}}

QLabel[role="title"] {{
    font-size: 20px;
    font-weight: 700;
    color: {TEXT_LIGHT};
}}

QLabel[role="section"] {{
    font-size: 11px;
    font-weight: 700;
    color: {TEXT_MUTED};
    letter-spacing: 0.5px;
    margin-top: 4px;
}}

QLabel[role="muted"] {{
    color: {TEXT_MUTED};
    font-size: 12px;
}}

QLabel[role="warning"] {{
    color: {WARNING};
    font-weight: 700;
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

/* ---- Inputs ---- */
QLineEdit, QComboBox {{
    background-color: {PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 9px 12px;
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

/* ---- Buttons ---- */
QPushButton {{
    background-color: {PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 10px 14px;
    color: {TEXT_LIGHT};
    font-weight: 600;
}}

QPushButton:hover {{
    background-color: {PRIMARY_LIGHT};
}}

QPushButton:disabled {{
    color: {TEXT_MUTED};
    background-color: {PRIMARY_DARK};
}}

QPushButton[role="primary"] {{
    background-color: {ACCENT};
    border: none;
    color: #06231C;
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

QPushButton[role="danger"] {{
    background-color: transparent;
    border: 1px solid {DANGER};
    color: {DANGER};
    border-radius: 8px;
    padding: 10px 14px;
    font-weight: 600;
}}

QPushButton[role="danger"]:hover {{
    background-color: {DANGER_BG};
    color: {DANGER_TEXT};
}}

/* ---- Sidebar (card-focused: distinct panel, no heavy border) ---- */
QWidget[role="sidebar"] {{
    background-color: {PRIMARY};
    border-right: 1px solid {BORDER};
}}

QLabel[role="sidebar-name"] {{
    font-size: 15px;
    font-weight: 700;
    color: {TEXT_LIGHT};
}}

QPushButton[role="sidebar-nav"] {{
    background-color: transparent;
    border: none;
    text-align: left;
    padding: 10px 12px;
    color: {TEXT_LIGHT};
    font-weight: 600;
    border-radius: 8px;
}}

QPushButton[role="sidebar-nav"]:hover {{
    background-color: {PRIMARY_LIGHT};
}}

QPushButton[role="sidebar-nav-danger"] {{
    background-color: transparent;
    border: none;
    text-align: left;
    padding: 10px 12px;
    color: {DANGER};
    font-weight: 600;
}}

QPushButton[role="sidebar-toggle"] {{
    background-color: transparent;
    border: none;
    color: {TEXT_LIGHT};
    font-size: 16px;
    padding: 4px;
    border-radius: 6px;
}}

QPushButton[role="sidebar-toggle"]:hover {{
    background-color: {PRIMARY_LIGHT};
}}

/* ---- Cards (bento-style stat tiles) ---- */
QFrame[role="stat-card"] {{
    background-color: {PRIMARY};
    border: none;
    border-radius: 14px;
}}

QLabel[role="stat-value"] {{
    font-size: 26px;
    font-weight: 800;
    color: {TEXT_LIGHT};
}}

QLabel[role="stat-value-accent"] {{
    font-size: 26px;
    font-weight: 800;
    color: {ACCENT};
}}

QLabel[role="stat-title"] {{
    font-size: 10px;
    font-weight: 700;
    color: {TEXT_MUTED};
    letter-spacing: 0.5px;
}}

/* ---- Result display ---- */
QLabel[role="result-idle"] {{
    font-size: 15px;
    font-weight: 600;
    padding: 16px;
    border-radius: 14px;
    background-color: {PRIMARY};
    color: {TEXT_MUTED};
}}

QLabel[role="result-genuine"] {{
    font-size: 20px;
    font-weight: 800;
    padding: 16px;
    border-radius: 14px;
    background-color: {SUCCESS_BG};
    color: {SUCCESS_TEXT};
}}

QLabel[role="result-forged"] {{
    font-size: 20px;
    font-weight: 800;
    padding: 16px;
    border-radius: 14px;
    background-color: {DANGER_BG};
    color: {DANGER_TEXT};
}}

/* ---- Lists / Tables ---- */
QListWidget {{
    background-color: {PRIMARY};
    border: none;
    border-radius: 10px;
    color: {TEXT_LIGHT};
}}

QTableWidget {{
    background-color: {PRIMARY};
    border: none;
    border-radius: 10px;
    gridline-color: {BORDER};
    color: {TEXT_LIGHT};
}}

QHeaderView::section {{
    background-color: {PRIMARY_DARK};
    color: {TEXT_MUTED};
    padding: 8px;
    border: none;
    font-weight: 700;
}}

/* ---- Scroll bars (the pages scroll on small screens) ---- */
QScrollBar:vertical {{
    background: {PRIMARY_DARK};
    width: 10px;
    margin: 0;
}}

QScrollBar:horizontal {{
    background: {PRIMARY_DARK};
    height: 10px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background: {TEXT_MUTED};
    border-radius: 5px;
    min-height: 30px;
}}

QScrollBar::handle:horizontal {{
    background: {TEXT_MUTED};
    border-radius: 5px;
    min-width: 30px;
}}

QScrollBar::handle:hover {{
    background: {ACCENT};
}}

QScrollBar::add-line, QScrollBar::sub-line {{
    width: 0;
    height: 0;
    border: none;
    background: none;
}}

QScrollBar::add-page, QScrollBar::sub-page {{
    background: none;
}}

QAbstractScrollArea::corner {{
    background: {PRIMARY_DARK};
}}
"""