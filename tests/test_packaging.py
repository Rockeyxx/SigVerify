"""The installer build (issue #42): weight check, runtime dependency list, --debug launcher."""

import ast
import hashlib
import os
import re
import subprocess
import sys
from importlib.metadata import packages_distributions
from pathlib import Path

import pytest

from bundle import fetch_weights

REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ weight check

def test_check_accepts_matching_hash(tmp_path):
    f = tmp_path / "w.pth"
    f.write_bytes(b"weights")
    fetch_weights.check(f, hashlib.sha256(b"weights").hexdigest())


def test_check_rejects_wrong_hash(tmp_path):
    f = tmp_path / "w.pth"
    f.write_bytes(b"weights")
    with pytest.raises(ValueError, match="SHA-256"):
        fetch_weights.check(f, hashlib.sha256(b"other").hexdigest())


def test_install_leaves_destination_untouched_on_mismatch(tmp_path):
    downloaded, dest = tmp_path / "dl", tmp_path / "weights"
    downloaded.mkdir()
    dest.mkdir()
    (downloaded / "spatial_cnn.pth").write_bytes(b"tampered")
    (dest / "spatial_cnn.pth").write_bytes(b"good")
    with pytest.raises(ValueError):
        fetch_weights.install(downloaded, dest, {"spatial_cnn.pth": hashlib.sha256(b"good").hexdigest()})
    assert (dest / "spatial_cnn.pth").read_bytes() == b"good"


def test_install_moves_checked_files(tmp_path):
    downloaded, dest = tmp_path / "dl", tmp_path / "weights"
    downloaded.mkdir()
    dest.mkdir()
    (downloaded / "spatial_cnn.pth").write_bytes(b"good")
    fetch_weights.install(downloaded, dest, {"spatial_cnn.pth": hashlib.sha256(b"good").hexdigest()})
    assert (dest / "spatial_cnn.pth").read_bytes() == b"good"


# ------------------------------------------------------------------ requirements-app.txt

def _dist_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _listed(path: Path) -> set[str]:
    names = set()
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            names.add(_dist_name(re.split(r"[<>=!~;\[ ]", line)[0]))
    return names


def _imported_top_level(*folders: Path) -> set[str]:
    """Every module named in an import statement anywhere in these folders, including imports
    inside functions (PyInstaller follows those too, and scipy is only imported that way)."""
    names = set()
    for folder in folders:
        for path in folder.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    names |= {a.name.split(".")[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names.add(node.module.split(".")[0])
    return names


def test_requirements_app_covers_every_library_the_app_imports():
    modules = _imported_top_level(REPO / "src", REPO / "config") - set(sys.stdlib_module_names) - {"src", "config"}
    owners = packages_distributions()
    unknown = {m for m in modules if m not in owners}
    assert unknown == set(), "imported but not installed in this environment"
    listed = _listed(REPO / "requirements-app.txt")
    # One import can belong to several distributions (PyQt5 -> PyQt5, PyQt5-Qt5, PyQt5-sip); one listed is enough.
    missing = {m for m in modules if not {_dist_name(d) for d in owners[m]} & listed}
    assert missing == set(), "add the distributions of these modules to requirements-app.txt"


def test_requirements_app_leaves_out_research_libraries():
    listed = _listed(REPO / "requirements-app.txt")
    assert listed.isdisjoint({"matplotlib", "scikit-learn", "tqdm", "pytest", "torchvision"})


# ------------------------------------------------------------------ --debug launcher

def _run_launch(tmp_path, code):
    env = {**os.environ, "SIGVER_HOME": str(tmp_path), "QT_QPA_PLATFORM": "offscreen"}
    return subprocess.run([sys.executable, "-c", code], cwd=REPO / "bundle", env=env,
                          capture_output=True, text=True, timeout=60)


def test_debug_logs_a_startup_crash_and_exits_1(tmp_path):
    code = ("import launch\n"
            "def boom(): raise ModuleNotFoundError('No module named torch')\n"
            "launch.run(['SigVerify', '--debug'], boom)\n")
    result = _run_launch(tmp_path, code)
    assert result.returncode == 1
    log = (tmp_path / "debug.log").read_text()
    assert "Traceback" in log and "No module named torch" in log


def test_without_debug_no_log_is_written(tmp_path):
    result = _run_launch(tmp_path, "import launch\nlaunch.run(['SigVerify'], lambda: None)\n")
    assert result.returncode == 0
    assert not (tmp_path / "debug.log").exists()


def test_debug_flag_is_removed_before_qt_sees_it(tmp_path):
    code = ("import sys, launch\n"
            "launch.run(['SigVerify', '--debug', '-x'], lambda: print('ARGV', sys.argv))\n")
    result = _run_launch(tmp_path, code)
    assert result.returncode == 0
    assert "ARGV ['SigVerify', '-x']" in (tmp_path / "debug.log").read_text()


# ------------------------------------------------------------------ checks on the built package

from bundle import check_bundle  # noqa: E402


def test_layout_names_a_missing_weight(tmp_path):
    internal = tmp_path / "_internal"
    for rel in check_bundle.REQUIRED[1:]:
        (internal / rel).parent.mkdir(parents=True, exist_ok=True)
        (internal / rel).write_bytes(b"x")
    with pytest.raises(SystemExit, match="spatial_cnn.pth"):
        check_bundle.layout(tmp_path)


def _fake_app(tmp_path, body):
    exe = tmp_path / "fake_app"
    exe.write_text(f"#!{sys.executable}\nimport os, sys, time\n{body}\n")
    exe.chmod(0o755)
    return str(exe)


@pytest.mark.skipif(sys.platform == "win32", reason="the fake app is a shebang script")
def test_start_fails_when_the_app_exits(tmp_path):
    with pytest.raises(SystemExit, match="exited"):
        check_bundle.start(_fake_app(tmp_path, "sys.exit(3)"), seconds=5)


@pytest.mark.skipif(sys.platform == "win32", reason="the fake app is a shebang script")
def test_start_fails_on_a_logged_traceback(tmp_path):
    body = ("open(os.path.join(os.environ['SIGVER_HOME'], 'debug.log'), 'w').write('Traceback (most recent call last)')\n"
            "time.sleep(60)")
    with pytest.raises(SystemExit, match="Traceback"):
        check_bundle.start(_fake_app(tmp_path, body), seconds=2)


@pytest.mark.skipif(sys.platform == "win32", reason="the fake app is a shebang script")
def test_start_passes_an_app_that_keeps_running(tmp_path):
    check_bundle.start(_fake_app(tmp_path, "assert '--debug' in sys.argv\ntime.sleep(60)"), seconds=2)
