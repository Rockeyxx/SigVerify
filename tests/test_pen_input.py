"""Which pen input Qt uses on Windows: Wintab (the tablet driver, full pressure range) by
default, Windows Ink on request. Chosen before QApplication starts (ENG_REPORT G10)."""
from src.capture.pen_input import PEN_PLATFORM, choose_pen_input


def test_windows_uses_wintab_by_default():
    env = {}
    assert choose_pen_input(None, system="Windows", env=env, wintab=True) == "windows:nowmpointer"
    assert env["QT_QPA_PLATFORM"] == "windows:nowmpointer"


def test_windows_ink_on_request():
    env = {}
    assert choose_pen_input("ink", system="Windows", env=env) == PEN_PLATFORM["ink"] == "windows"
    assert env["QT_QPA_PLATFORM"] == "windows"


def test_the_flag_beats_a_platform_already_set():
    env = {"QT_QPA_PLATFORM": "windows"}
    assert choose_pen_input("wintab", system="Windows", env=env) == "windows:nowmpointer"


def test_a_platform_the_operator_set_is_kept_without_the_flag():
    env = {"QT_QPA_PLATFORM": "windows:nowmpointer,dpiawareness=1"}
    assert choose_pen_input(None, system="Windows", env=env) == "windows:nowmpointer,dpiawareness=1"


def test_linux_is_left_alone():
    env = {}
    assert choose_pen_input("wintab", system="Linux", env=env) is None and env == {}


def test_without_the_tablet_driver_windows_falls_back_to_ink():
    """No wintab32.dll (the XPPen driver is not installed): Wintab would give no pen at all."""
    env = {}
    assert choose_pen_input(None, system="Windows", env=env, wintab=False) == "windows"


def test_with_the_tablet_driver_windows_uses_wintab():
    env = {}
    assert choose_pen_input(None, system="Windows", env=env, wintab=True) == "windows:nowmpointer"


def test_an_explicit_wintab_is_kept_even_without_the_driver():
    env = {}
    assert choose_pen_input("wintab", system="Windows", env=env, wintab=False) == "windows:nowmpointer"
