"""Resolve app version from baked VERSION file or git commit count since APP_VERSION changed."""

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
