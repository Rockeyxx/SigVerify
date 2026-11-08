"""Which pen input Qt reads on Windows, chosen before QApplication starts (ENG_REPORT G10).

Windows Ink, Qt 5.15's default, reports pressure in 1024 steps and read the same signer at
about 0.6 of Linux. Wintab, the tablet driver's own input, passes the device's range. Qt picks
the input once at startup, so a wrong choice means restarting with the other one.
"""
import os
import platform

from config.settings import WINDOWS_PEN_API

PEN_PLATFORM = {"wintab": "windows:nowmpointer", "ink": "windows"}


def choose_pen_input(pen_api=None, system=None, env=None):
    """Set QT_QPA_PLATFORM for the chosen Windows pen input and return it. pen_api is "wintab"
    or "ink"; without it a value the operator already set is kept, else WINDOWS_PEN_API.
    Other systems are left alone (None)."""
    system = system or platform.system()
    env = os.environ if env is None else env
    if system != "Windows":
        return None
    if pen_api is None and env.get("QT_QPA_PLATFORM"):
        return env["QT_QPA_PLATFORM"]
    env["QT_QPA_PLATFORM"] = PEN_PLATFORM[pen_api or WINDOWS_PEN_API]
    return env["QT_QPA_PLATFORM"]
