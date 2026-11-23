"""Issue #43: the whole app as one system, on the shipped weights and a real vault.

The checks live in src/self_test.py, so a team member runs the same ones on the installed app
(`SigVerify --self-test KIT`, kit from tools/make_test_kit.py). Here they run from source on a
kit built in a temporary folder. Real signings: in-house participant P01 (XPPen tablet and
blue-pen paper of the same signing, the one participant cleared for this use; the kit holds
copies, so the originals are never touched). Without that folder (CI) the kit's made-up signer
is used. NFI cannot stand in: its scans are black ink, which the app's crop refuses (A6).
"""
import os

import pytest

from config.settings import INHOUSE_DIR, SPATIAL_WEIGHTS_PATH, TEMPORAL_WEIGHTS_PATH

pytestmark = pytest.mark.skipif(
    not (os.path.exists(SPATIAL_WEIGHTS_PATH) and os.path.exists(TEMPORAL_WEIGHTS_PATH)),
    reason="the shipped weights are not downloaded (README: Model Weights)")

HAVE_P01 = os.path.isdir(os.path.join(INHOUSE_DIR, "P01"))


@pytest.fixture(scope="module")
def kit(tmp_path_factory):
    from tools import make_test_kit
    root = tmp_path_factory.mktemp("kit")
    signer = make_test_kit.p01_signer() if HAVE_P01 else make_test_kit.synthetic_signer(str(root / "drawn"))
    return make_test_kit.build(str(root / "kit"), signer)


def test_the_five_tests_pass_on_the_app_from_source(app, kit):
    from src import self_test
    report = self_test.run(kit)
    failed = [f"test {c['test']}: {c['name']} ({c['detail']})" for c in report.checks if not c["ok"]]
    print("\n" + "\n".join(f"{k}: {v}" for k, v in report.measured.items()))
    assert len(report.checks) >= 18 and not failed, failed


def test_a_lock_file_windows_will_not_let_us_read_does_not_stop_the_test(app, kit, monkeypatch):
    """Windows enforces the app's lock on signatures.db.lock, so reading it raises PermissionError
    (the first Windows run stopped there); Linux locks are advisory."""
    import builtins
    from src import self_test
    real_open = builtins.open

    def windows_open(path, mode="r", *args, **kwargs):
        if str(path).endswith(".lock") and mode == "rb":
            raise PermissionError(13, "Permission denied")
        return real_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", windows_open)
    report = self_test.run(kit)
    leak_check = next(c for c in report.checks if "plain form" in c["name"])
    assert report.passed and "signatures.db.lock (locked by the app)" in leak_check["detail"]


def test_a_ram_reading_of_zero_fails_the_check_instead_of_passing(app, kit, monkeypatch):
    """On Windows the reading failed silently and 0.00 GB passed "at most 4 GB"."""
    from src import self_test
    monkeypatch.setattr(self_test, "_peak_ram_gb", lambda: 0.0)
    ram = next(c for c in self_test.run(kit).checks if "peak RAM" in c["name"])
    assert not ram["ok"] and ram["detail"] == "could not be measured"


def test_the_results_file_says_pass_and_names_no_user_or_path(app, kit):
    from src import self_test
    assert self_test.main([kit]) == 0
    with open(os.path.join(kit, self_test.RESULTS_NAME), encoding="utf-8") as f:
        text = f.read()
    assert "Result:  PASS" in text and "[FAIL]" not in text
    assert os.path.expanduser("~") not in text and kit not in text and os.sep + "tmp" not in text
