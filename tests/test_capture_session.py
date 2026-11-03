"""The research logger: helpers (names, raw CSV, manifest, sheet grid) and the session flow."""
import csv
import os

import cv2
import numpy as np
import pytest

from config.settings import (CAMPAIGN_BREAK_AFTER, CAMPAIGN_GENUINE_PER_MODALITY,
                             CAMPAIGN_PRACTICE_SIGNINGS, SHEET_GRID)
from tools import capture_session as cs


def test_file_names_follow_the_owners_default(tmp_path):
    assert cs.signature_path(tmp_path, "P07", "tablet", "genuine", index=3) == \
        tmp_path / "P07" / "P07_tablet_genuine_03.csv"
    assert cs.signature_path(tmp_path, "P07", "paper", "forgercopy").name == "P07_paper_forgercopy.png"
    assert cs.signature_path(tmp_path, "P07", "tablet", "forgery", forger="P02", attempt=1).name == \
        "P07_tablet_forgery_byP02_a1.csv"


def test_a_bad_participant_code_is_refused(tmp_path):
    with pytest.raises(ValueError):
        cs.signature_path(tmp_path, "omar", "tablet", "genuine", index=1)


def test_raw_events_are_never_overwritten(tmp_path):
    path = tmp_path / "P07" / "x.csv"
    cs.write_raw_events(path, [(1.0, 87, 2.0, 3.0, 0.5, 10.0, -4.0, 1)], root=tmp_path)
    rows = list(csv.reader(open(path)))
    assert rows[0] == cs.RAW_FIELDS and len(rows) == 2
    with pytest.raises(FileExistsError):
        cs.write_raw_events(path, [], root=tmp_path)


def test_manifest_has_one_header_and_one_row_per_file(tmp_path):
    cs.append_manifest(tmp_path, {"file": "a.csv", "participant": "P07"})
    cs.append_manifest(tmp_path, {"file": "b.csv", "participant": "P07"})
    rows = list(csv.DictReader(open(tmp_path / "manifest.csv")))
    assert [r["file"] for r in rows] == ["a.csv", "b.csv"]


def test_next_index_resumes_after_the_last_saved_genuine(tmp_path):
    for i in (1, 2, 3):
        cs.write_raw_events(cs.signature_path(tmp_path, "P07", "tablet", "genuine", index=i), [],
                            root=tmp_path)
    assert cs.next_index(tmp_path, "P07", "tablet") == 4


def test_the_sheet_splits_into_its_boxes_without_borders(tmp_path):
    sheet = tmp_path / "sheet.png"
    cs.make_sheet(sheet)
    img = cv2.imread(str(sheet))
    t, b, l, r = cs.sheet_boxes(*img.shape[:2])[2]
    cv2.circle(img, ((l + r) // 2, (t + b) // 2), 40, (255, 0, 0), -1)      # blue ink in box 3
    crops = cs.split_sheet(img)
    assert len(crops) == SHEET_GRID[0] * SHEET_GRID[1]
    inked = [i for i, c in enumerate(crops) if (c < 200).any()]
    assert inked == [2]                                                    # borders and numbers stay out


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
from PyQt5.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PyQt5.QtGui import QTabletEvent  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _event(kind, x, y, pressure, ts):
    ev = QTabletEvent(kind, QPointF(x, y), QPointF(x, y), QTabletEvent.Stylus, QTabletEvent.Pen,
                      pressure, 0, 0, 0.0, 0.0, 0, Qt.NoModifier, 1, Qt.LeftButton, Qt.LeftButton)
    ev.setTimestamp(int(ts))
    return ev


def _sign_and_approve(session):
    """Three strokes at 4 ms inside the box, then 'I approve'."""
    canvas = session.capture.canvas
    for k, t0 in enumerate((1000, 1400, 1800)):
        for i in range(60):
            kind = QEvent.TabletPress if i == 0 else QEvent.TabletMove
            canvas.tabletEvent(_event(kind, 100.0 + 40 * k + 0.6 * i, 200.0 + 30 * np.sin(i / 9),
                                      0.45, t0 + 4 * i))
        canvas.tabletEvent(_event(QEvent.TabletRelease, 300, 200, 0.0, t0 + 240))
    session.capture.confirm_signature()


def test_a_genuine_session_saves_twelve_signatures_after_practice(app, tmp_path):
    session = cs.CaptureSession(root=tmp_path)
    session.start("P07", operator="op")
    for _ in range(CAMPAIGN_PRACTICE_SIGNINGS):
        _sign_and_approve(session)
    assert not (tmp_path / "P07").exists()                       # practice is never saved
    for _ in range(CAMPAIGN_BREAK_AFTER):
        _sign_and_approve(session)
    assert session.stage == "break"
    session.continue_after_break()
    for _ in range(CAMPAIGN_GENUINE_PER_MODALITY - CAMPAIGN_BREAK_AFTER):
        _sign_and_approve(session)
    assert session.stage == "paper"
    saved = sorted((tmp_path / "P07").glob("P07_tablet_genuine_*.csv"))
    assert len(saved) == CAMPAIGN_GENUINE_PER_MODALITY
    rows = list(csv.DictReader(open(tmp_path / "manifest.csv")))
    assert len(rows) == CAMPAIGN_GENUINE_PER_MODALITY
    assert rows[0]["os"] and rows[0]["device_model"] and rows[0]["index"] == "1"


def test_paper_sheets_become_genuines_then_the_forger_copy(app, tmp_path):
    sheet = tmp_path / "sheet.png"
    cs.make_sheet(sheet)
    session = cs.CaptureSession(root=tmp_path)
    session.start("P07", operator="op")
    session.stage = "paper"
    names = [p.name for p in session.save_paper([sheet, sheet])]
    assert names[:CAMPAIGN_GENUINE_PER_MODALITY] == [
        f"P07_paper_genuine_{i:02d}.png" for i in range(1, CAMPAIGN_GENUINE_PER_MODALITY + 1)]
    assert names[CAMPAIGN_GENUINE_PER_MODALITY] == "P07_paper_forgercopy.png"
    assert session.stage == "done"


def test_a_forgery_session_saves_two_attempts(app, tmp_path):
    session = cs.CaptureSession(root=tmp_path)
    session.start("P07", operator="op", forger="P02")
    _sign_and_approve(session)
    assert "again" in session.prompt.text().lower()
    _sign_and_approve(session)
    assert sorted(p.name for p in (tmp_path / "P07").glob("*forgery*")) == [
        "P07_tablet_forgery_byP02_a1.csv", "P07_tablet_forgery_byP02_a2.csv"]
    rows = list(csv.DictReader(open(tmp_path / "manifest.csv")))
    assert [(r["forger"], r["attempt"]) for r in rows] == [("P02", "1"), ("P02", "2")]
    assert session.stage == "paper"


def _imported(tmp_path):
    """A genuine session whose paper sheets were just imported."""
    sheet = tmp_path / "sheet.png"
    cs.make_sheet(sheet)
    session = cs.CaptureSession(root=tmp_path)
    session.start("P07", operator="op")
    session.stage = "paper"
    return session, session.save_paper([sheet, sheet])


def _manifest_files(tmp_path):
    return [r["file"] for r in csv.DictReader(open(tmp_path / "manifest.csv"))]


def test_ending_the_session_readies_the_window_for_the_next_participant(app, tmp_path):
    session, _ = _imported(tmp_path)
    session.end_session()
    assert session.stage == "idle" and session.participant is None
    assert session.code.text() == "" and not session.capture.isEnabled()


def test_deleting_one_paper_box_removes_its_file_and_manifest_row(app, tmp_path):
    session, saved = _imported(tmp_path)
    session.delete_paper(saved[2])
    assert not saved[2].exists() and saved[3].exists()
    files = _manifest_files(tmp_path)
    assert "P07/P07_paper_genuine_03.png" not in files and len(files) == len(saved) - 1


def test_replacing_one_paper_box_keeps_its_name_and_one_manifest_row(app, tmp_path):
    session, saved = _imported(tmp_path)
    new = tmp_path / "new.png"
    cv2.imwrite(str(new), np.zeros((50, 80, 3), np.uint8))
    session.replace_paper(saved[0], new)
    assert cv2.imread(str(saved[0])).shape[:2] == (50, 80)
    files = _manifest_files(tmp_path)
    assert files.count("P07/P07_paper_genuine_01.png") == 1 and len(files) == len(saved)


def test_undoing_the_paper_import_removes_it_and_allows_a_new_one(app, tmp_path):
    session, saved = _imported(tmp_path)
    session.undo_paper()
    assert not any(p.exists() for p in saved)
    assert not (tmp_path / "manifest.csv").exists() or _manifest_files(tmp_path) == []
    assert session.stage == "paper"
    assert session.save_paper([tmp_path / "sheet.png"] * 2) == saved


def _inked_sheet(tmp_path, blank=()):
    """A sheet with a blue signature-like mark in every box except those in blank."""
    sheet = tmp_path / "inked.png"
    cs.make_sheet(sheet)
    img = cv2.imread(str(sheet))
    for n, (t, b, l, r) in enumerate(cs.sheet_boxes(*img.shape[:2])):
        if n in blank:
            continue
        cx, cy = (l + r) // 2, (t + b) // 2
        cv2.ellipse(img, (cx, cy), (160, 70), -20, 0, 360, (190, 40, 20), 5)
        cv2.line(img, (cx - 150, cy + 60), (cx + 120, cy - 50), (190, 40, 20), 5)
    cv2.imwrite(str(sheet), img)
    return sheet


def test_paper_import_saves_the_ink_crop_the_model_sees_beside_each_box(app, tmp_path):
    session = cs.CaptureSession(root=tmp_path)
    session.start("P07", operator="op")
    session.stage = "paper"
    saved = session.save_paper([_inked_sheet(tmp_path)] * 2)
    for path in saved:
        ink = cv2.imread(str(cs.ink_path(path)), cv2.IMREAD_GRAYSCALE)
        assert ink is not None and set(np.unique(ink)) <= {0, 255}        # binarized, as the app does
        assert ink.shape[0] < cv2.imread(str(path)).shape[0]              # cropped to the ink
    assert len(_manifest_files(tmp_path)) == len(saved)                    # one row per signature


def test_a_box_without_a_signature_is_named_and_its_edits_follow_the_ink_file(app, tmp_path):
    session = cs.CaptureSession(root=tmp_path)
    session.start("P07", operator="op")
    session.stage = "paper"
    saved = session.save_paper([_inked_sheet(tmp_path, blank=(2,))] * 2)
    assert "P07_paper_genuine_03" in session.prompt.text()
    assert not cs.ink_path(saved[2]).exists() and cs.ink_path(saved[3]).exists()
    session.replace_paper(saved[2], saved[3])
    assert cs.ink_path(saved[2]).exists()
    session.delete_paper(saved[3])
    assert not cs.ink_path(saved[3]).exists()
    session.undo_paper()
    assert not any(cs.ink_path(p).exists() for p in saved)


def _phone_box(signature=True):
    """A box as a phone scan gives it: grey paper, a soft shadow, the dark page edge on the right,
    and a faint ballpoint signature broken into short pieces (each far below crop_to_ink's
    800 px piece size)."""
    box = np.full((742, 1148, 3), 235, np.uint8)
    cv2.circle(box, (150, 80), 140, (185, 190, 185), -1)                     # shadow, grey-green
    box = cv2.GaussianBlur(box, (0, 0), 25)
    cv2.line(box, (1100, 0), (1100, 741), (60, 60, 60), 6)                   # page edge
    if signature:
        for a in range(0, 360, 12):                                          # dashed faint oval
            cv2.ellipse(box, (500, 400), (200, 110), -30, a, a + 7, (200, 150, 140), 2)
        cv2.line(box, (380, 560), (560, 230), (200, 150, 140), 2)
    return box


def test_the_ink_crop_keeps_a_faint_broken_signature_and_drops_shadow_and_page_edge():
    top, bottom, left, right = cs.ink_box(_phone_box())
    assert left < 340 and right > 660 and top < 300 and bottom > 500         # the whole oval
    assert right < 1000                                                      # not the page edge
    crop = cs.ink_crop(_phone_box())
    assert set(np.unique(crop)) <= {0, 255} and (crop == 0).any()            # ink black on white


def test_a_box_with_only_shadow_and_page_edge_has_no_signature():
    assert cs.ink_box(_phone_box(signature=False)) is None
    assert cs.ink_crop(_phone_box(signature=False)) is None


def test_a_pdf_sheet_is_two_full_a4_pages_for_double_sided_printing(tmp_path):
    import re
    pdf = tmp_path / "sheet.pdf"
    cs.make_sheet(pdf)
    data = pdf.read_bytes()
    assert data.startswith(b"%PDF")
    boxes = [tuple(float(v) for v in m) for m in
             re.findall(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", data)]
    assert len(boxes) == 2                                                   # front and back
    for w, h in boxes:                                                       # A4 is 595 x 842 pt
        assert abs(w - 595.3) < 1.5 and abs(h - 841.9) < 1.5                # the grid fills the page


def test_choosing_a_box_shows_that_box_large_and_follows_replace_and_delete(app, tmp_path):
    session = cs.CaptureSession(root=tmp_path)
    session.start("P07", operator="op")
    session.stage = "paper"
    saved = session.save_paper([_inked_sheet(tmp_path)] * 2)
    session.paper_list.setCurrentIndex(4)
    assert session.shown == saved[4] and not session.selected.pixmap().isNull()
    before = session.selected.pixmap().toImage()
    new = tmp_path / "new.png"
    cv2.imwrite(str(new), np.zeros((50, 80, 3), np.uint8))
    session.replace_paper(saved[4], new)
    assert session.shown == saved[4]                                        # selection kept
    assert session.selected.pixmap().toImage() != before                    # and redrawn
    session.delete_paper(saved[4])
    assert session.paper_list.count() == len(saved) - 1 and session.shown == saved[5]


def test_the_language_toggle_switches_the_prompt_and_buttons_to_arabic_and_back(app, tmp_path):
    session = cs.CaptureSession(root=tmp_path)
    session.start("P07", operator="op")
    assert "Practice 1 of" in session.prompt.text()
    session.btn_lang.click()
    assert "تجربة 1 من" in session.prompt.text()                          # the shown prompt too
    assert session.capture.btn_confirm.text() == "أوافق" and session.btn_start.text() == "ابدأ"
    assert session.btn_lang.text() == "English"
    session.btn_lang.click()
    assert "Practice 1 of" in session.prompt.text() and session.capture.btn_confirm.text() == "I approve"


def test_prompts_stay_arabic_as_the_session_moves_on(app, tmp_path):
    session = cs.CaptureSession(root=tmp_path)
    session.set_language("ar")
    session.start("P07", operator="op")
    for _ in range(CAMPAIGN_PRACTICE_SIGNINGS):
        _sign_and_approve(session)
    assert f"التوقيع 1 من {CAMPAIGN_GENUINE_PER_MODALITY}" in session.prompt.text()


def test_forgery_mode_starts_from_the_form_in_arabic(app, tmp_path):
    session = cs.CaptureSession(root=tmp_path)
    session.set_language("ar")
    session.code.setText("P07")
    session.mode.setCurrentIndex(1)                                          # the forgery item
    session.forger_code.setText("P02")
    session.btn_start.click()
    assert session.stage == "forgery" and session.forger == "P02"


def test_an_arabic_prompt_sits_on_the_right(app, tmp_path):
    from PyQt5.QtWidgets import QStyle
    session = cs.CaptureSession(root=tmp_path)
    shown = lambda: QStyle.visualAlignment(session.prompt.layoutDirection(), session.prompt.alignment())
    assert shown() & Qt.AlignLeft
    session.set_language("ar")
    assert shown() & Qt.AlignRight


def test_files_outside_the_data_folder_or_behind_a_symlink_are_refused(tmp_path):
    root = tmp_path / "inhouse"
    (root / "P07").mkdir(parents=True)
    outside = tmp_path / "elsewhere.csv"
    outside.write_text("keep")
    link = root / "P07" / "P07_tablet_genuine_01.csv"
    link.symlink_to(outside)
    with pytest.raises(ValueError):                                        # a symlink in the folder
        cs.write_raw_events(link, [], root=root)
    with pytest.raises(ValueError):                                        # a path that escapes
        cs.write_raw_events(root / ".." / "escape.csv", [], root=root)
    assert outside.read_text() == "keep"
    (root / "manifest.csv").symlink_to(outside)
    with pytest.raises(ValueError):
        cs.append_manifest(root, {"file": "x"})
    assert outside.read_text() == "keep"
