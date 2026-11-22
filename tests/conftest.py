import os
import pytest

# Forced, not setdefault: a desktop session sets QT_QPA_PLATFORM (Hyprland: "wayland;xcb"), and a
# tiling window manager then resizes the test windows, so the size checks in test_gui.py fail.
os.environ["QT_QPA_PLATFORM"] = "offscreen"


@pytest.fixture(scope="session")
def app():
    """One QApplication for the whole run. A module-scoped one was destroyed when its module
    ended, which deletes every started worker thread still listed in src/gui/workers.py, so the
    next module's MainWindow failed with "wrapped C/C++ object ... has been deleted"."""
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
