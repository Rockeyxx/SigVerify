"""Which pen input Qt reads on Windows, chosen before QApplication starts (ENG_REPORT G10).

Windows Ink, Qt 5.15's default, reports pressure in 1024 steps. Wintab, the tablet driver's own
input, passes the device's full range (1/16383 measured with XPPen driver 4.0.21). Wintab needs
the driver's wintab32.dll: without it the pen would draw nothing, so the default falls back to
Windows Ink. Qt picks the input once at startup.
"""
import ctypes
import os
import platform

from config.settings import WINDOWS_PEN_API

PEN_PLATFORM = {"wintab": "windows:nowmpointer", "ink": "windows"}


def wintab_installed():
    """True when a tablet driver's wintab32.dll can be loaded (Windows only)."""
    try:
        ctypes.WinDLL("wintab32.dll")
        return True
    except (AttributeError, OSError):
        return False


def choose_pen_input(pen_api=None, system=None, env=None, wintab=None):
    """Set QT_QPA_PLATFORM for the Windows pen input and return it. pen_api "wintab" or "ink"
    is used as given. Without it: a value the operator already set is kept; else
    WINDOWS_PEN_API, falling back to Windows Ink when no Wintab driver is installed (wintab:
    whether it is, detected when None). Other systems are left alone (None)."""
    system = system or platform.system()
    env = os.environ if env is None else env
    if system != "Windows":
        return None
    if pen_api is None and env.get("QT_QPA_PLATFORM"):
        return env["QT_QPA_PLATFORM"]
    if pen_api is None:
        pen_api = WINDOWS_PEN_API
        if pen_api == "wintab" and not (wintab_installed() if wintab is None else wintab):
            pen_api = "ink"
    env["QT_QPA_PLATFORM"] = PEN_PLATFORM[pen_api]
    return env["QT_QPA_PLATFORM"]
