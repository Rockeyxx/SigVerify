"""The app version shown to the user: APP_VERSION from config/settings.py (major.minor, changed by
hand for 1.1 or 2.0) plus the number of commits since that line last changed. The installer build
writes the result to VERSION, because the package has no git."""

import subprocess
from pathlib import Path

from config.settings import APP_VERSION, PROJECT_ROOT


def app_version(root=PROJECT_ROOT, base=APP_VERSION) -> str:
    root = Path(root)
    baked = root / "VERSION"
    if baked.is_file():
        return baked.read_text().strip()  # skylos: ignore[SKY-D215] the app's own folder and a fixed file name
    try:
        def git(*args):
            return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                                  text=True).stdout.strip()
        # the commit that introduced the current APP_VERSION line; the count starts there
        since = git("log", "-1", "--format=%H", "-S", f'APP_VERSION = "{base}"', "--", "config/settings.py")
        if not since:
            return base
        return f"{base}.{git('rev-list', '--count', f'{since}..HEAD')}"
    except (OSError, subprocess.CalledProcessError):
        return base
