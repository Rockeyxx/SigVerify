"""The app version: APP_VERSION (major.minor, set by hand) plus the commits since that line changed."""
import subprocess

from src.app_version import app_version


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _commit(repo, settings_text=None):
    settings = repo / "config" / "settings.py"
    if settings_text is not None:
        settings.write_text(settings_text)
    else:
        with open(repo / "notes.txt", "a") as f:
            f.write("x\n")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c")


def test_each_commit_after_the_version_line_adds_one_and_a_new_line_starts_again(tmp_path):
    (tmp_path / "config").mkdir()
    _git(tmp_path, "init", "-q")
    _commit(tmp_path, 'APP_VERSION = "1.0"\n')
    assert app_version(tmp_path, "1.0") == "1.0.0"
    _commit(tmp_path)
    _commit(tmp_path)
    assert app_version(tmp_path, "1.0") == "1.0.2"
    _commit(tmp_path, 'APP_VERSION = "1.1"\n')
    assert app_version(tmp_path, "1.1") == "1.1.0"
    _commit(tmp_path)
    assert app_version(tmp_path, "1.1") == "1.1.1"


def test_a_baked_version_file_wins_and_no_git_falls_back(tmp_path):
    assert app_version(tmp_path, "1.0") == "1.0"               # no git, no file: the hand-set part only
    (tmp_path / "VERSION").write_text("1.0.7\n")
    assert app_version(tmp_path, "1.0") == "1.0.7"
