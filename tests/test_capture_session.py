"""The research logger: helpers (names, raw CSV, manifest, sheet grid) and the session flow."""
import csv
import os
import shutil

import cv2
import numpy as np
import pytest
from PIL import Image

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


def _black_pixels_on_and_off_rows(crop, top, y):
    line = slice(y - top - 3, y - top + 4)
    off = np.ones(crop.shape[0], bool)
    off[line] = False
    return (crop[line] == 0).sum(), (crop[off] == 0).sum()


def test_the_ink_crop_drops_a_printed_black_line_inside_the_signature_box():
    # A blue signature over a printed "Signature: ____" line: the line falls inside the box, but it
    # is straight and not blue, so the crop the model sees must not carry it, and its darkness must
    # not push the faint blue strokes below the binarization threshold either.
    top, _, _, _ = cs.ink_box(_phone_box())
    clean_on, clean_off = _black_pixels_on_and_off_rows(cs.ink_crop(_phone_box()), top, 530)
    on_line = _phone_box()
    cv2.line(on_line, (300, 530), (700, 530), (40, 40, 40), 3)              # printed black line
    assert cs.ink_box(on_line) == cs.ink_box(_phone_box())                   # same box either way
    line_on, line_off = _black_pixels_on_and_off_rows(cs.ink_crop(on_line), top, 530)
    assert line_on <= clean_on + 20                                          # the line is not ink
    assert 0.95 <= line_off / clean_off <= 1.05                              # the signature survives


def test_the_ink_crop_keeps_a_straight_blue_flourish():
    # A long straight stroke in the signature's own blue ink is part of the signature.
    top, _, _, _ = cs.ink_box(_phone_box())
    flourish = _phone_box()
    cv2.line(flourish, (330, 520), (680, 520), (200, 150, 140), 3)          # blue underline
    assert cs.ink_box(flourish)[0] == top
    on, _ = _black_pixels_on_and_off_rows(cs.ink_crop(flourish), top, 520)
    assert on >= 0.8 * 3 * 350                                               # most of it is kept


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


def test_redo_keeps_the_box_hint_in_arabic(app, tmp_path):
    """The widget's Redo writes its own English line; the session must put the Arabic hint back."""
    session = cs.CaptureSession(root=tmp_path)
    session.set_language("ar")
    session.start("P07", operator="op")
    session.capture.btn_reset.click()
    assert session.capture.label.text() == cs.TEXT["ar"]["canvas_hint"]


# --- operator menu: a code (or forger + victim) that already has saved files ---------------

def _saved_tablet_files(root, participant, numbers, forger=None):
    """Tablet files with manifest rows, as an earlier session would have left them."""
    for n in numbers:
        kind = dict(role="forgery", forger=forger, attempt=n) if forger else \
            dict(role="genuine", index=n)
        path = cs.signature_path(root, participant, "tablet", **kind)
        cs.write_raw_events(path, [], root=root)
        cs.append_manifest(root, {"file": str(path.relative_to(root)), "participant": participant,
                                  "modality": "tablet", "role": kind["role"]})


def _all_genuine():
    return range(1, CAMPAIGN_GENUINE_PER_MODALITY + 1)


def _session(tmp_path, participant="P07", forger=None, confirm=True, typed=""):
    session = cs.CaptureSession(root=tmp_path)
    session._confirm = lambda _text: confirm
    session._typed = lambda _text: typed
    session.start(participant, operator="op", forger=forger)
    return session


def _shown(widget):
    return not widget.isHidden()


def _sheet(tmp_path):
    sheet = tmp_path / "sheet.png"
    cs.make_sheet(sheet)
    return sheet


def test_next_index_fills_a_deleted_number_first(tmp_path):
    _saved_tablet_files(tmp_path, "P07", [1, 2, 4])
    assert cs.next_index(tmp_path, "P07", "tablet") == 3


def test_saved_tablet_lists_extras_and_keeps_forgers_apart(tmp_path):
    _saved_tablet_files(tmp_path, "P07", [1, 13, 14])
    _saved_tablet_files(tmp_path, "P07", [1], forger="P02")
    _saved_tablet_files(tmp_path, "P07", [1, 2], forger="P03")
    assert [t[2] for t in cs.saved_tablet(tmp_path, "P07")] == [1, 13, 14]
    assert [t[3] for t in cs.saved_tablet(tmp_path, "P07", "P03")] == [1, 2]


def test_replacing_raw_events_swaps_the_file_and_leaves_no_temp(tmp_path):
    path = cs.signature_path(tmp_path, "P07", "tablet", "genuine", index=1)
    cs.write_raw_events(path, [], root=tmp_path)
    cs.replace_raw_events(path, [(1.0, 87, 2.0, 3.0, 0.5, 10.0, -4.0, 1)], root=tmp_path)
    assert len(list(csv.reader(open(path)))) == 2
    assert [p.name for p in path.parent.iterdir()] == [path.name]


def test_a_new_code_goes_straight_to_practice_without_a_menu(app, tmp_path):
    assert _session(tmp_path).stage == "practice"


def test_a_code_with_twelve_signatures_opens_the_menu_and_signs_nothing(app, tmp_path):
    _saved_tablet_files(tmp_path, "P07", _all_genuine())
    session = _session(tmp_path)
    assert session.stage == "menu" and not session.capture.isEnabled()
    assert f"{CAMPAIGN_GENUINE_PER_MODALITY}/{CAMPAIGN_GENUINE_PER_MODALITY}" in session.prompt.text()
    assert _shown(session.btn_paper) and _shown(session.btn_redo_tablet)
    assert not _shown(session.btn_tablet_continue) and not _shown(session.btn_review)


def test_extra_signatures_show_in_the_count_and_can_be_deleted(app, tmp_path):
    _saved_tablet_files(tmp_path, "P07", range(1, 15))
    session = _session(tmp_path)
    assert f"14/{CAMPAIGN_GENUINE_PER_MODALITY}" in session.prompt.text()
    session.tablet_pick.setCurrentIndex(14)                     # 0 is "all"; 14 is the 14th file
    session.btn_delete_tablet.click()
    assert [t[2] for t in cs.saved_tablet(tmp_path, "P07")] == list(range(1, 14))
    assert "P07/P07_tablet_genuine_14.csv" not in _manifest_files(tmp_path)
    assert session.stage == "menu"


def test_declining_a_delete_keeps_the_signature(app, tmp_path):
    _saved_tablet_files(tmp_path, "P07", [1, 2])
    session = _session(tmp_path, confirm=False)
    session.btn_delete_tablet.click()
    assert len(cs.saved_tablet(tmp_path, "P07")) == 2


def test_import_from_the_menu_needs_no_signing_and_ends_on_the_check(app, tmp_path):
    _saved_tablet_files(tmp_path, "P07", _all_genuine())
    session = _session(tmp_path)
    saved = session.save_paper([_sheet(tmp_path)] * 2)
    assert all(p.exists() for p in saved) and session.stage == "done"
    assert len(cs.saved_tablet(tmp_path, "P07")) == CAMPAIGN_GENUINE_PER_MODALITY


def test_the_tablet_can_end_now_and_the_paper_come_later(app, tmp_path):
    session = cs.CaptureSession(root=tmp_path)
    session.start("P07", operator="op")
    session.stage = "paper"
    session._show_buttons()
    assert _shown(session.btn_end)
    session.btn_end.click()
    assert session.stage == "idle" and session.participant is None


def test_import_after_a_deleted_box_fills_only_that_box(app, tmp_path):
    session, saved = _imported(tmp_path)
    session.end_session()
    kept = saved[1].read_bytes()
    saved[0].unlink()
    cs.remove_manifest_rows(tmp_path, [saved[0].relative_to(tmp_path)])
    session = _session(tmp_path)
    assert _shown(session.btn_paper)
    assert session.save_paper([tmp_path / "sheet.png"] * 2) == [saved[0]]
    assert saved[0].exists() and saved[1].read_bytes() == kept
    assert len(_manifest_files(tmp_path)) == len(saved)


def test_a_complete_paper_set_refuses_a_second_import(app, tmp_path):
    session, _ = _imported(tmp_path)
    with pytest.raises(FileExistsError):
        session.save_paper([tmp_path / "sheet.png"] * 2)


def test_review_opens_the_saved_paper_for_editing(app, tmp_path):
    session, saved = _imported(tmp_path)
    session.end_session()
    session = _session(tmp_path)
    session.btn_review.click()
    assert session.stage == "done" and session.paper_list.count() == len(saved)
    assert _shown(session.btn_replace)
    session.paper_list.setCurrentIndex(1)
    session.btn_delete.click()
    assert not saved[1].exists()


def test_declining_keeps_the_box_and_the_whole_import(app, tmp_path):
    session, saved = _imported(tmp_path)
    session._confirm = lambda _text: False
    session.btn_delete.click()
    session.btn_undo.click()
    assert all(p.exists() for p in saved)


def test_continue_fills_the_gap_and_never_makes_a_thirteenth(app, tmp_path):
    numbers = [n for n in _all_genuine() if n != 5]
    _saved_tablet_files(tmp_path, "P07", numbers)
    session = _session(tmp_path)
    session.btn_tablet_continue.click()
    session.btn_skip.click()
    assert session.index == 5
    _sign_and_approve(session)
    assert [t[2] for t in cs.saved_tablet(tmp_path, "P07")] == list(_all_genuine())
    assert session.stage == "paper"


def test_finishing_the_tablet_with_paper_complete_returns_to_the_menu(app, tmp_path):
    session, saved = _imported(tmp_path)
    session.end_session()
    _saved_tablet_files(tmp_path, "P07", [n for n in _all_genuine() if n != 1])
    session = _session(tmp_path)
    session.btn_tablet_continue.click()
    session.btn_skip.click()
    _sign_and_approve(session)
    assert session.stage == "menu"


def test_redoing_one_signature_replaces_it_only_on_approval(app, tmp_path):
    _saved_tablet_files(tmp_path, "P07", _all_genuine())
    session = _session(tmp_path)
    path = cs.saved_tablet(tmp_path, "P07")[2][0]
    old = path.read_bytes()
    session.tablet_pick.setCurrentIndex(3)
    session.btn_redo_tablet.click()
    assert session.stage == "redo" and session.capture.isEnabled()
    assert path.read_bytes() == old                              # nothing lost before approval
    _sign_and_approve(session)
    assert path.read_bytes() != old and session.stage == "menu"
    assert len(cs.saved_tablet(tmp_path, "P07")) == CAMPAIGN_GENUINE_PER_MODALITY
    assert _manifest_files(tmp_path).count(str(path.relative_to(tmp_path))) == 1


def test_redoing_all_walks_every_signature_and_can_stop_midway(app, tmp_path):
    _saved_tablet_files(tmp_path, "P07", [1, 2, 3])
    session = _session(tmp_path)
    session.btn_redo_tablet.click()                              # 0 = all
    assert len(session.redo) == CAMPAIGN_GENUINE_PER_MODALITY
    _sign_and_approve(session)
    assert _shown(session.btn_end)
    session.btn_end.click()
    assert session.stage == "idle"
    assert len(cs.saved_tablet(tmp_path, "P07")) == 3            # the rest stayed as they were


def test_deleting_a_person_needs_the_code_typed_and_removes_all_their_data(app, tmp_path):
    session, saved = _imported(tmp_path)
    session.end_session()
    _saved_tablet_files(tmp_path, "P07", _all_genuine())
    _saved_tablet_files(tmp_path, "P07", [1], forger="P02")      # someone forged them
    _saved_tablet_files(tmp_path, "P09", [1, 2])                 # another volunteer
    _saved_tablet_files(tmp_path, "P09", [1], forger="P07")      # they forged someone
    wrong = _session(tmp_path, typed="P09")
    wrong.btn_delete_all.click()
    assert saved[0].exists()
    right = _session(tmp_path, typed="P07")
    right.btn_delete_all.click()
    assert not (tmp_path / "P07").exists() and right.stage == "idle"
    assert not list(tmp_path.glob("*/*byP07*"))
    assert sorted(_manifest_files(tmp_path)) == ["P09/P09_tablet_genuine_01.csv",
                                                 "P09/P09_tablet_genuine_02.csv"]


def test_a_victim_with_all_genuines_can_still_be_forged_without_a_menu(app, tmp_path):
    _saved_tablet_files(tmp_path, "P07", _all_genuine())
    assert _session(tmp_path, forger="P02").stage == "forgery"


def test_a_forger_with_saved_attempts_opens_the_same_menu(app, tmp_path):
    _saved_tablet_files(tmp_path, "P07", [1, 2], forger="P02")
    session = _session(tmp_path, forger="P02")
    assert session.stage == "menu" and "P02" in session.prompt.text()
    assert _shown(session.btn_paper) and not _shown(session.btn_tablet_continue)
    saved = session.save_paper([_sheet(tmp_path)])
    assert [p.name for p in saved] == ["P07_paper_forgery_byP02_a1.png",
                                       "P07_paper_forgery_byP02_a2.png"]


def test_another_forger_on_the_same_victim_starts_fresh(app, tmp_path):
    _saved_tablet_files(tmp_path, "P07", [1, 2], forger="P02")
    assert _session(tmp_path, forger="P03").stage == "forgery"


def test_a_forger_continues_at_the_missing_attempt(app, tmp_path):
    _saved_tablet_files(tmp_path, "P07", [1], forger="P02")
    session = _session(tmp_path, forger="P02")
    session.btn_tablet_continue.click()
    assert session.stage == "forgery" and session.attempt == 2
    _sign_and_approve(session)
    assert session.stage == "paper"
    assert len(cs.saved_tablet(tmp_path, "P07", "P02")) == 2


def test_redoing_a_forgery_attempt_replaces_it(app, tmp_path):
    _saved_tablet_files(tmp_path, "P07", [1, 2], forger="P02")
    session = _session(tmp_path, forger="P02")
    path = cs.saved_tablet(tmp_path, "P07", "P02")[1][0]
    session.tablet_pick.setCurrentIndex(2)
    session.btn_redo_tablet.click()
    _sign_and_approve(session)
    assert path.stat().st_size > 0 and len(list(csv.reader(open(path)))) > 1
    row = [r for r in csv.DictReader(open(tmp_path / "manifest.csv"))
           if r["file"] == str(path.relative_to(tmp_path))]
    assert len(row) == 1 and (row[0]["forger"], row[0]["attempt"]) == ("P02", "2")


def test_deleting_a_forger_keeps_the_victim_and_other_forgers(app, tmp_path):
    _saved_tablet_files(tmp_path, "P07", [1])
    _saved_tablet_files(tmp_path, "P07", [1, 2], forger="P02")
    _saved_tablet_files(tmp_path, "P07", [1], forger="P03")
    session = _session(tmp_path, forger="P02", typed="P02")
    session.btn_delete_all.click()
    assert session.stage == "idle"
    assert [p.name for p in sorted((tmp_path / "P07").iterdir())] == [
        "P07_tablet_forgery_byP03_a1.csv", "P07_tablet_genuine_01.csv"]


def test_the_menu_follows_the_language_toggle(app, tmp_path):
    _saved_tablet_files(tmp_path, "P07", _all_genuine())
    session = _session(tmp_path)
    session.toggle_language()
    assert session.prompt.text() == session._t(*[session._last[0]], **session._last[1])
    assert session.btn_redo_tablet.text() == cs.TEXT["ar"]["btn_redo_tablet"]
    assert session.btn_end.text() == cs.TEXT["ar"]["btn_cancel"]
    assert session.tablet_pick.itemText(0) == cs.TEXT["ar"]["pick_all"]


def test_english_and_arabic_have_the_same_texts():
    assert set(cs.TEXT["en"]) == set(cs.TEXT["ar"])


# --- scans in any order, and PDFs straight from the scanner --------------------------------

def _sheet_with_ink(tmp_path, name, boxes):
    """A scanned page with a blue mark in the first `boxes` boxes."""
    path = tmp_path / "sheet.png"
    cs.make_sheet(path)
    img = cv2.imread(str(path))
    for t, b, l, r in cs.sheet_boxes(*img.shape[:2])[:boxes]:
        cv2.ellipse(img, ((l + r) // 2, (t + b) // 2), (180, 60), 0, 0, 360, (200, 60, 20), 6)
    out = tmp_path / name
    cv2.imwrite(str(out), img)
    return out


def _no_empty_box(tmp_path, participant="P07"):
    return all(cs.ink_path(t[0]).exists() for t in cs.saved_paper(tmp_path, participant))


def test_the_back_page_given_first_still_imports_in_signing_order(app, tmp_path):
    front = _sheet_with_ink(tmp_path, "front.png", SHEET_GRID[0] * SHEET_GRID[1])
    back = _sheet_with_ink(tmp_path, "back.png", 5)
    session = cs.CaptureSession(root=tmp_path)
    session.start("P07", operator="op")
    session.save_paper([back, front])
    assert _no_empty_box(tmp_path)


def test_a_scanner_pdf_with_the_back_page_first_imports_directly(app, tmp_path):
    if not shutil.which("pdftoppm"):
        pytest.skip("pdftoppm not installed")
    pages = [Image.open(_sheet_with_ink(tmp_path, n, k)).convert("RGB")
             for n, k in (("back.png", 5), ("front.png", SHEET_GRID[0] * SHEET_GRID[1]))]
    pdf = tmp_path / "SCAN.PDF"
    pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=300)
    session = cs.CaptureSession(root=tmp_path)
    session.start("P07", operator="op")
    assert len(session.save_paper([pdf])) == CAMPAIGN_GENUINE_PER_MODALITY + 1
    assert _no_empty_box(tmp_path)


def test_a_pdf_without_pdftoppm_gives_a_clear_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(cs.shutil, "which", lambda _name: None)
    with pytest.raises(ValueError, match="pdftoppm"):
        cs.read_sheets([tmp_path / "scan.pdf"])


def test_an_unreadable_scan_is_refused_by_name(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_text("not an image")
    with pytest.raises(ValueError, match="bad.png"):
        cs.read_sheets([bad])


def test_a_pale_stretch_of_stroke_does_not_cut_the_signature_in_two():
    """A scan app's whitening leaves thin ink nearly grey (P03, saturation about 14): the pale
    stretch must still join the two strong ends, or only the bigger end is kept."""
    box = np.full((500, 1000, 3), 255, np.uint8)
    cv2.line(box, (100, 250), (300, 250), (200, 60, 20), 4)                  # strong blue
    cv2.line(box, (300, 250), (600, 250), (170, 160, 160), 3)                # pale, saturation 15
    cv2.line(box, (600, 250), (720, 250), (200, 60, 20), 4)                  # strong blue
    _, _, left, right = cs.ink_box(box)
    assert left < 100 and right > 720
