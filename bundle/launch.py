"""Entry point of the packaged app (issue #42).

    SigVerify            starts the app
    SigVerify --debug    also shows every error (a console window on Windows) and saves it to
                         debug.log in the data folder, so a failed start can be seen and sent

The Windows package has no console, so without --debug an error has nowhere to go. The CI
start check runs the shipped app with --debug and reads debug.log.
"""

import os
import sys
import traceback


class _Tee:
    """Writes to every stream it was given, so errors reach the screen and the log."""

    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]

    def write(self, text):
        for s in self.streams:
            s.write(text)
            s.flush()
        return len(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def _start_app():
    from src.gui.app import main
    main()


def run(argv, start=_start_app):
    if "--debug" not in argv:
        start()
        return
    sys.argv = [a for a in argv if a != "--debug"]       # Qt must not see the flag
    # Same folder rule as src.gui.storage._data_dir; not imported, so an import error is logged too.
    home = os.path.abspath(os.path.expanduser(os.environ.get("SIGVER_HOME") or os.path.join("~", ".sigverify")))
    os.makedirs(home, exist_ok=True)
    log = open(os.path.join(home, "debug.log"), "w", encoding="utf-8")
    screen = sys.__stderr__
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.AllocConsole()
        screen = open("CONOUT$", "w", encoding="utf-8")
    sys.stdout = sys.stderr = _Tee(log, screen)
    try:
        start()
    except SystemExit:
        raise
    except BaseException:     # caught here, or the Windows package shows a blocking crash dialog
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run(sys.argv)
