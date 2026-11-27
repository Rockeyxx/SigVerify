"""Installer build checks: weight inclusion, runtime dependency list, and --debug launcher."""

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


def test_the_installer_ships_the_two_shipped_weights_and_no_ensemble():
    """Installer bundles only the two shipped weights without the 6-seed temporal ensemble."""
    import re
    spec = (REPO / "bundle" / "sigverify.spec").read_text()
    assert set(re.findall(r'"([\w.-]+\.pth)"', spec)) == {"spatial_cnn.pth", "temporal_lstm.pth"}


@pytest.mark.parametrize("code_from_test", [0, 1])
def test_self_test_passes_the_kit_and_exits_with_its_result_without_starting_the_app(tmp_path, code_from_test):
    code = ("import launch\n"
            "def start(): print('APP STARTED')\n"
            f"def self_test(args): print('KIT', args); return {code_from_test}\n"
            "launch.run(['SigVerify', '--self-test', 'C:/kit'], start, self_test)\n")
    result = _run_launch(tmp_path, code)
    assert result.returncode == code_from_test
    assert "KIT ['C:/kit']" in result.stdout and "APP STARTED" not in result.stdout


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


def test_the_common_password_list_ships_with_the_app():
    from config.settings import COMMON_PASSWORDS_FILE, PROJECT_ROOT
    rel = Path(COMMON_PASSWORDS_FILE).relative_to(PROJECT_ROOT).as_posix()
    assert rel in check_bundle.REQUIRED
    assert '"common_passwords.txt"' in (REPO / "bundle" / "sigverify.spec").read_text()


def test_the_build_bakes_the_version_from_full_git_history():
    assert "VERSION" in check_bundle.REQUIRED
    assert "app_version()" in (REPO / "bundle" / "sigverify.spec").read_text()
    assert "fetch-depth: 0" in (REPO / ".github" / "workflows" / "release.yml").read_text()   # the count needs history


def test_every_merge_to_main_builds_a_prerelease_named_after_the_app_version():
    text = (REPO / ".github" / "workflows" / "release.yml").read_text()
    assert re.search(r"push:\s*\n\s*branches:\s*\[main\]", text)
    assert "app_version()" in text and "gh release create" in text and "--prerelease" in text
    assert "needs.tag.outputs.tag" in text            # build and attach use the tag that job made
    # One platform failing (the Windows runner refused for billing) still attaches the other's file.
    attach_if = re.search(r"\n  attach:\n(?:.*\n)*?    if: (.*)\n", text).group(1)
    assert "needs.build.result == 'failure'" in attach_if
