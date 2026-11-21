"""Research capture session for the XPPen campaign (spec section 6): one volunteer, one sitting.

Tablet: practice (never saved), then CAMPAIGN_GENUINE_PER_MODALITY genuine signatures with a
break after CAMPAIGN_BREAK_AFTER; each is saved only after the signer approves it and the input
gate passes. Paper: import the scanned sheets as images or a PDF (print them with --make-sheet);
each sheet is cut into its grid of boxes, the fuller page first, in signing order. Forgery mode records a team forger's attempts at one
volunteer. Files go to INHOUSE_DIR/<participant>/, one manifest row each. Raw streams are
research data: keep INHOUSE_DIR on an encrypted disk, never in git or the app database.

A code (or forger and victim) that already has files opens the operator menu instead: continue
the tablet, import or review the paper, redo tablet signatures, or delete. The paper can wait:
end after the tablet part and import it later from the menu.

Run:  python tools/capture_session.py
      python tools/capture_session.py --make-sheet sheet.png
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from config.settings import (CAMPAIGN_BREAK_AFTER, CAMPAIGN_FORGERY_ATTEMPTS,  # noqa: E402
                             CAMPAIGN_GENUINE_PER_MODALITY, CAMPAIGN_PRACTICE_SIGNINGS,
                             INHOUSE_DIR, SHEET_DPI, SHEET_GRID, SHEET_MARGIN)
from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtGui import QFont, QImage, QPixmap  # noqa: E402
from PyQt5.QtWidgets import (QApplication, QComboBox, QFileDialog, QHBoxLayout,  # noqa: E402
                             QInputDialog, QLabel, QLineEdit, QMessageBox, QPushButton,
                             QVBoxLayout, QWidget)

from src.capture.ink import ink_box, ink_crop  # noqa: E402,F401  (ink_box: tests)
from src.capture.pen_input import PEN_PLATFORM, choose_pen_input  # noqa: E402
from src.capture.tablet import SignatureCaptureWidget  # noqa: E402

def imread(path):
    """cv2.imread that also reads a path with non-ASCII characters (a Windows Desktop folder
    named in Arabic, for example): cv2.imread returns None there without an error. The file
    is read as raw uint8 bytes and decoded. None when it is missing or not an image."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite(path, image):
    """cv2.imwrite for any path (see imread): encode in memory, then write the bytes.
    The format follows the file extension, as with cv2.imwrite."""
    ok, encoded = cv2.imencode(Path(path).suffix or ".png", image)
    if ok:
        encoded.tofile(str(path))
    return bool(ok)


PARTICIPANT = re.compile(r"^P\d{2,3}$")
RAW_FIELDS = ["t_ms", "kind", "x", "y", "pressure", "xtilt", "ytilt", "inside"]
MANIFEST_FIELDS = ["file", "participant", "modality", "role", "index", "forger", "attempt",
                   "operator", "os", "qt_platform", "device_model", "saved_at"]
_EXT = {"tablet": "csv", "paper": "png"}

# Every instruction the session window shows, in English and Arabic (the toggle switches them;
# file names, participant codes and the captions drawn inside preview images stay English).
TEXT = {
    "en": {
        "start_hint": "Enter the participant code and press Start.",
        "bad_code": "Participant and forger codes look like P07.",
        "practice_first": "Practice 1 of {n}: sign as you normally do. Nothing is saved.",
        "practice_next": "Practice {i} of {n}. Nothing is saved.",
        "genuine_first": "Signature {i} of {n}. Sign, look at it, then press I approve (or Redo).",
        "genuine_next": "Signature {i} of {n}.",
        "break": "Short break. Press Continue when ready.",
        "forgery_first": ("Forger {forger} imitates {participant}: attempt 1 of {n}. "
                          "Study the printed copy, then sign."),
        "forgery_again": "Sign again: attempt {i} of {n}.",
        "tablet_done": "Tablet done. Now import the scanned paper sheets.",
        "tablet_done_forgery": ("Tablet done. Import the forger's paper sheet "
                                "(boxes 1-{n} = attempts in order)."),
        "paper_saved": ("Saved {n} paper signatures. Session done. Check the crops: delete or "
                        "replace a bad one, or undo the whole import."),
        "paper_saved_no_ink": ("Saved {n} paper signatures. Session done. No signature found in: "
                               "{names}. Check the crops: delete or replace a bad one, or undo "
                               "the whole import."),
        "deleted": "Deleted {name}.",
        "replaced": "Replaced {name}.",
        "replaced_no_ink": "Replaced {name}. No signature found in it.",
        "undone": "Paper import undone. Import the scanned sheets again.",
        "not_saved": "Not saved: {error}",
        "not_replaced": "Not replaced: {error}",
        "canvas_hint": "Sign inside the box, then press I approve or Redo.",
        "btn_start": "Start", "btn_approve": "I approve", "btn_redo": "Redo",
        "btn_skip": "Skip practice", "btn_continue": "Continue",
        "btn_paper": "Import scanned paper sheets...", "btn_end": "End / Next participant",
        "btn_delete": "Delete", "btn_replace": "Replace...", "btn_undo": "Undo paper import",
        "mode_genuine": "Genuine", "mode_forgery": "Forgery",
        "ph_code": "Participant, e.g. P07", "ph_operator": "Operator", "ph_forger": "Forger, e.g. P02",
        "dlg_sheets": "Scanned sheets, in order", "dlg_replace": "Image of this one signature",
        "menu_genuine": ("{participant} already has saved data: tablet {tablet}/{tablet_n}, "
                         "paper {paper}/{paper_n}. What do you want to do?"),
        "menu_forgery": ("Forger {forger} on {participant} already has saved data: tablet "
                         "{tablet}/{tablet_n}, paper {paper}/{paper_n}. What do you want to do?"),
        "redo_one": "Redo {name} ({i} of {n}). The old one stays until you press I approve.",
        "review": ("Saved paper signatures. Check the crops: delete or replace a bad one, or "
                   "delete them all with Undo paper import."),
        "deleted_all": "Deleted {name}: {n} files.",
        "confirm_delete": "Permanently delete {what}? This cannot be undone.",
        "what_tablet": "{n} tablet signatures", "what_paper": "{n} paper signatures",
        "type_code": "This permanently deletes {n} files of {name}. Type {code} to confirm.",
        "pick_all": "All tablet signatures",
        "btn_tablet_continue": "Continue tablet", "btn_review": "Review paper",
        "btn_redo_tablet": "Redo tablet", "btn_delete_tablet": "Delete tablet",
        "btn_delete_person": "Delete whole participant",
        "btn_delete_forger": "Delete this forger's attempts",
        "btn_cancel": "Cancel", "btn_later": "Import later / Next participant",
        "btn_yes": "Yes", "btn_no": "No",
    },
    "ar": {
        "start_hint": "أدخل رمز المشارك ثم اضغط «ابدأ».",
        "bad_code": "رمز المشارك أو المزوّر يكون مثل P07.",
        "practice_first": "تجربة 1 من {n}: وقّع كما توقّع عادةً. لن يُحفظ شيء.",
        "practice_next": "تجربة {i} من {n}. لن يُحفظ شيء.",
        "genuine_first": "التوقيع {i} من {n}. وقّع، ثم انظر إليه، ثم اضغط «أوافق» (أو «إعادة»).",
        "genuine_next": "التوقيع {i} من {n}.",
        "break": "استراحة قصيرة. اضغط «متابعة» عندما تكون جاهزًا.",
        "forgery_first": ("المزوّر {forger} يقلّد توقيع {participant}: المحاولة 1 من {n}. "
                          "ادرس النسخة المطبوعة، ثم وقّع."),
        "forgery_again": "وقّع مرة أخرى: المحاولة {i} من {n}.",
        "tablet_done": "انتهى التوقيع على الجهاز اللوحي. الآن أدخل الصور.",
        "tablet_done_forgery": ("انتهى التوقيع على الجهاز اللوحي. أدخل صورة ورقة المزوّر "
                                "(المربعات 1-{n} هي المحاولات بالترتيب)."),
        "paper_saved": ("عدد التوقيعات الورقية المحفوظة: {n}. انتهت الجلسة. راجع القصاصات: "
                        "احذف قصاصة سيئة أو استبدلها، أو تراجع عن الإدخال كله."),
        "paper_saved_no_ink": ("عدد التوقيعات الورقية المحفوظة: {n}. انتهت الجلسة. لم يُعثر على "
                               "توقيع في: {names}. راجع القصاصات: احذف قصاصة سيئة أو استبدلها، "
                               "أو تراجع عن الإدخال كله."),
        "deleted": "حُذف {name}.",
        "replaced": "استُبدل {name}.",
        "replaced_no_ink": "استُبدل {name}. لم يُعثر فيه على توقيع.",
        "undone": "أُلغي إدخال الصور. أدخل الصور مرة أخرى.",
        "not_saved": "لم يُحفظ: {error}",
        "not_replaced": "لم يُستبدل: {error}",
        "canvas_hint": "وقّع داخل المربع، ثم اضغط «أوافق» أو «إعادة».",
        "btn_start": "ابدأ", "btn_approve": "أوافق", "btn_redo": "إعادة",
        "btn_skip": "تخطّي التجربة", "btn_continue": "متابعة",
        "btn_paper": "إدخال الصور...", "btn_end": "إنهاء / المشارك التالي",
        "btn_delete": "حذف", "btn_replace": "استبدال...", "btn_undo": "التراجع عن إدخال الصور",
        "mode_genuine": "أصلي", "mode_forgery": "تزوير",
        "ph_code": "رمز المشارك، مثل P07", "ph_operator": "المشغّل", "ph_forger": "رمز المزوّر، مثل P02",
        "dlg_sheets": "الصور، بالترتيب", "dlg_replace": "صورة هذا التوقيع وحده",
        "menu_genuine": ("لدى {participant} بيانات محفوظة: الجهاز اللوحي {tablet}/{tablet_n}، "
                         "الورق {paper}/{paper_n}. ماذا تريد أن تفعل؟"),
        "menu_forgery": ("لدى المزوّر {forger} على {participant} بيانات محفوظة: الجهاز اللوحي "
                         "{tablet}/{tablet_n}، الورق {paper}/{paper_n}. ماذا تريد أن تفعل؟"),
        "redo_one": "إعادة {name} ({i} من {n}). يبقى القديم حتى تضغط «أوافق».",
        "review": ("التوقيعات الورقية المحفوظة. راجع القصاصات: احذف قصاصة سيئة أو استبدلها، "
                   "أو احذفها كلها بزر «التراجع عن إدخال الصور»."),
        "deleted_all": "حُذف {name}: {n} ملفات.",
        "confirm_delete": "حذف {what} نهائيًا؟ لا يمكن التراجع عن ذلك.",
        "what_tablet": "توقيعات الجهاز اللوحي ({n})", "what_paper": "التوقيعات الورقية ({n})",
        "type_code": "سيُحذف {n} ملفات لـ {name} نهائيًا. اكتب {code} للتأكيد.",
        "pick_all": "كل توقيعات الجهاز اللوحي",
        "btn_tablet_continue": "متابعة الجهاز اللوحي", "btn_review": "مراجعة الورق",
        "btn_redo_tablet": "إعادة توقيع الجهاز اللوحي", "btn_delete_tablet": "حذف من الجهاز اللوحي",
        "btn_delete_person": "حذف المشارك كله",
        "btn_delete_forger": "حذف محاولات هذا المزوّر",
        "btn_cancel": "إلغاء", "btn_later": "إدخال الصور لاحقًا / المشارك التالي",
        "btn_yes": "نعم", "btn_no": "لا",
    },
}


def signature_path(root, participant, modality, role, index=None, forger=None, attempt=None):
    """Where one signature is saved (the owner's default names, spec section 6)."""
    for code in ([participant, forger] if forger else [participant]):
        if not PARTICIPANT.match(str(code)):
            raise ValueError(f"participant codes look like P07, got {code!r}")
    if role == "genuine":
        name = f"{participant}_{modality}_genuine_{index:02d}"
    elif role == "forgercopy":
        name = f"{participant}_{modality}_forgercopy"
    elif role == "forgery":
        name = f"{participant}_{modality}_forgery_by{forger}_a{attempt}"
    else:
        raise ValueError(f"unknown role {role!r}")
    return Path(root) / participant / f"{name}.{_EXT[modality]}"


def next_index(root, participant, modality):
    """The lowest free genuine number, so a session that crashed resumes where it stopped and a
    deleted signature is filled before anything new."""
    folder = Path(root) / participant
    taken = {int(m.group(1)) for p in folder.glob(f"{participant}_{modality}_genuine_*")
             if (m := re.search(r"_genuine_(\d+)\.", p.name))}
    return min(set(range(1, len(taken) + 2)) - taken)


def expected_files(root, participant, modality, forger=None):
    """Every signature a full session saves, as (path, role, index, attempt), in signing order."""
    if forger:
        return [(signature_path(root, participant, modality, "forgery", forger=forger, attempt=a),
                 "forgery", "", a) for a in range(1, CAMPAIGN_FORGERY_ATTEMPTS + 1)]
    files = [(signature_path(root, participant, modality, "genuine", index=i), "genuine", i, "")
             for i in range(1, CAMPAIGN_GENUINE_PER_MODALITY + 1)]
    if modality == "paper":
        files.append((signature_path(root, participant, "paper", "forgercopy"), "forgercopy", "", ""))
    return files


def saved_tablet(root, participant, forger=None):
    """The tablet signatures on disk, in number order, as expected_files gives them. Includes
    any beyond the expected count, so the operator can see and delete them."""
    pattern = rf"_forgery_by{forger}_a(\d+)\.csv$" if forger else r"_genuine_(\d+)\.csv$"
    found = []
    for path in (Path(root) / participant).glob(f"{participant}_tablet_*.csv"):
        if m := re.search(pattern, path.name):
            n = int(m.group(1))
            found.append((path, "forgery", "", n) if forger else (path, "genuine", n, ""))
    return sorted(found, key=lambda t: t[3] if forger else t[2])


def saved_paper(root, participant, forger=None):
    """The expected paper signatures that are on disk."""
    return [t for t in expected_files(root, participant, "paper", forger) if t[0].exists()]


def person_files(root, participant, forger=None):
    """What a full delete removes. With forger: that forger's attempts at participant. Without:
    everything of the person, their own folder and the attempts they made at others."""
    root = Path(root)
    if forger:
        return sorted((root / participant).glob(f"{participant}_*_forgery_by{forger}_*"))
    return sorted({*(root / participant).glob(f"{participant}_*"),
                   *root.glob(f"P*/*_by{participant}_*")})


def _check_inside(root, path):
    """Every file this tool writes, reads back or deletes must be a plain path inside the data
    folder: not a symlink, and not resolving outside root. Raises ValueError otherwise."""
    path, base = Path(path), Path(root).resolve()
    if path.is_symlink() or not path.resolve().is_relative_to(base):
        raise ValueError(f"refusing {path}: not a plain file inside {base}")
    return path


def write_raw_events(path, events, root):
    """One tablet signature: every Qt tablet event the canvas saw (RAW_FIELDS: screen pixels,
    y down, pressure 0..1, Qt tilt). Mode "x" refuses to overwrite a saved signature."""
    path = _check_inside(root, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", newline="") as f:  # skylos: ignore[SKY-D215,SKY-D324] checked by _check_inside: inside the data folder, not a symlink
        csv.writer(f).writerows([RAW_FIELDS] + [list(e) for e in events])


def replace_raw_events(path, events, root):
    """Put a new signature in place of a saved one. The old file stays until the new one is
    fully written, so a failure cannot lose it."""
    path = Path(path)
    temp = _check_inside(root, path.with_suffix(".tmp"))
    temp.unlink(missing_ok=True)
    write_raw_events(temp, events, root)
    os.replace(temp, _check_inside(root, path))


def append_manifest(root, row):
    """One row per saved file in <root>/manifest.csv; missing fields are left empty."""
    path = _check_inside(root, Path(root) / "manifest.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:  # skylos: ignore[SKY-D215,SKY-D324] checked by _check_inside: inside the data folder, not a symlink
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})


def remove_manifest_rows(root, files):
    """Drop the manifest rows whose file is in files (paths relative to root)."""
    path = _check_inside(root, Path(root) / "manifest.csv")
    if not path.exists():
        return
    files = {str(f) for f in files}
    with open(path, newline="") as f:  # skylos: ignore[SKY-D215,SKY-D325] checked by _check_inside: inside the data folder, not a symlink
        rows = [r for r in csv.DictReader(f) if r["file"] not in files]
    with open(path, "w", newline="") as f:  # skylos: ignore[SKY-D215,SKY-D324] checked by _check_inside above
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def ink_path(path):
    """Beside each raw paper box: the ink crop the model sees."""
    path = Path(path)
    return path.with_name(f"{path.stem}_ink.png")


def save_ink(path, root):
    """Write ink_path(path) from the raw box. Returns False, and leaves no ink file, when no
    signature is found."""
    _check_inside(root, ink_path(path)).unlink(missing_ok=True)
    crop = ink_crop(imread(str(path)))
    if crop is None:
        return False
    imwrite(str(ink_path(path)), crop)
    return True


def sheet_boxes(height, width, margin=SHEET_MARGIN):
    """(top, bottom, left, right) of each grid box, row by row, shrunk by margin on every side
    so the printed border and number never enter the crop."""
    rows, cols = SHEET_GRID
    boxes = []
    for r in range(rows):
        for c in range(cols):
            top, bottom = height * r / rows, height * (r + 1) / rows
            left, right = width * c / cols, width * (c + 1) / cols
            dy, dx = (bottom - top) * margin, (right - left) * margin
            boxes.append((int(top + dy), int(bottom - dy), int(left + dx), int(right - dx)))
    return boxes


def split_sheet(image):
    """A scanned sheet (the whole A4 page) cut into its boxes, in signing order."""
    h, w = image.shape[:2]
    return [image[t:b, l:r] for t, b, l, r in sheet_boxes(h, w)]


def pdf_pages(path, dpi=SHEET_DPI):
    """Each page of a scanner's PDF as an image, through the pdftoppm program (poppler)."""
    tool = shutil.which("pdftoppm")
    if tool is None:
        raise ValueError("reading a PDF needs pdftoppm (poppler): install it, or scan to PNG")
    with tempfile.TemporaryDirectory() as folder:
        try:
            subprocess.run([tool, "-r", str(dpi), "-png", str(path), str(Path(folder) / "page")],
                           check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"cannot read {Path(path).name}") from exc
        return [imread(str(p)) for p in sorted(Path(folder).glob("page*.png"))]


def at_sheet_dpi(page, dpi=SHEET_DPI):
    """A page scanned finer than the sheet's DPI, shrunk to it (an A4 page's short side is
    8.27 in). A 1200 DPI scan has 16x the pixels: cutting and cropping it took most of a
    minute and froze the window, for detail the model resizes away anyway."""
    short = int(8.27 * dpi)
    h, w = page.shape[:2]
    if min(h, w) <= short:
        return page
    scale = short / min(h, w)
    return cv2.resize(page, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def read_sheets(paths):
    """The scanned pages of these files, as images: an image is one page, a PDF is all of its.
    Every page comes back at the sheet's DPI or coarser."""
    pages = []
    for path in paths:
        if Path(path).suffix.lower() == ".pdf":
            pages += pdf_pages(path)
        elif (image := imread(str(path))) is not None:
            pages.append(image)
        else:
            raise ValueError(f"cannot read {Path(path).name}")
    return [at_sheet_dpi(page) for page in pages]


def signing_order(pages):
    """The boxes of these pages in signing order. The front is filled first, so the page with
    the most signatures comes first, whatever order the scanner or file dialog gave."""
    sheets = [split_sheet(page) for page in pages]
    sheets.sort(key=lambda boxes: sum(ink_crop(box) is not None for box in boxes), reverse=True)
    return [box for boxes in sheets for box in boxes]


def make_sheet(path, dpi=SHEET_DPI):
    """A printable A4 sheet: light-grey numbered boxes on the grid split_sheet cuts.
    Sign with a blue pen, inside the boxes; scan the full page at the same DPI."""
    w, h = int(8.27 * dpi), int(11.69 * dpi)
    img = np.full((h, w, 3), 255, np.uint8)
    for n, (t, b, l, r) in enumerate(sheet_boxes(h, w, margin=0.0), start=1):
        cv2.rectangle(img, (l + 10, t + 10), (r - 10, b - 10), (210, 210, 210), 2)
        cv2.putText(img, str(n), (l + 18, t + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (210, 210, 210), 2)
    if str(path).lower().endswith(".pdf"):
        # Front and back of one paper, each page exactly A4 at this DPI, with no margins, so
        # the grid fills the page as split_sheet expects. Print two-sided, flip on long edge.
        from PIL import Image
        page = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        page.save(str(path), save_all=True, append_images=[page.copy()], resolution=dpi)
    else:
        imwrite(str(path), img)


class CaptureSession(QWidget):
    """One volunteer, one sitting (or one forger at one volunteer), step by step."""

    def __init__(self, root=INHOUSE_DIR):
        super().__init__()
        self.root = Path(root)
        self.setWindowTitle("XPPen capture session")
        self.stage, self.index, self.attempt, self.practised = "idle", 0, 0, 0
        self.participant = self.operator = self.forger = None
        self.redo, self.redo_total = [], 0        # [(path, role, index, attempt)] still to sign
        self.lang, self._last = "en", ("start_hint", {})

        self.code = QLineEdit()
        self.op = QLineEdit()
        self.mode = QComboBox()
        self.mode.addItems(["", ""])                  # genuine, forgery; texts set by _retranslate
        self.forger_code = QLineEdit()
        self.btn_start = QPushButton()
        self.btn_start.clicked.connect(self._start_from_form)
        self.btn_lang = QPushButton()                 # العربية / English
        self.btn_lang.clicked.connect(self.toggle_language)
        form = QHBoxLayout()
        for w in (self.code, self.op, self.mode, self.forger_code, self.btn_start, self.btn_lang):
            form.addWidget(w)

        self.prompt = QLabel()
        font = QFont()
        font.setPointSize(18)
        font.setBold(True)
        self.prompt.setFont(font)
        self.prompt.setWordWrap(True)

        self.capture = SignatureCaptureWidget()
        self.capture.signature_confirmed.connect(self._on_signature)
        # Redo writes the widget's own English line; ours runs after it (connection order).
        self.capture.btn_reset.clicked.connect(
            lambda: self.capture.label.setText(self._t("canvas_hint")))
        self.capture.setEnabled(False)

        self.btn_skip = QPushButton()
        self.btn_skip.clicked.connect(self._end_practice)
        self.btn_continue = QPushButton()
        self.btn_continue.clicked.connect(self.continue_after_break)
        self.btn_paper = QPushButton()
        self.btn_paper.clicked.connect(self._choose_sheets)
        self.btn_end = QPushButton()
        self.btn_end.clicked.connect(self.end_session)
        self.preview = QLabel()
        buttons = QHBoxLayout()
        for w in (self.btn_skip, self.btn_continue, self.btn_paper, self.btn_end):
            buttons.addWidget(w)

        # The operator menu, for a code that already has saved files.
        self.tablet_pick = QComboBox()                # all, then each saved tablet signature
        self.btn_tablet_continue = QPushButton()
        self.btn_tablet_continue.clicked.connect(self.begin_tablet)
        self.btn_redo_tablet = QPushButton()
        self.btn_redo_tablet.clicked.connect(self._redo_picked)
        self.btn_delete_tablet = QPushButton()
        self.btn_delete_tablet.clicked.connect(self._delete_picked)
        self.btn_review = QPushButton()
        self.btn_review.clicked.connect(self.review_paper)
        self.btn_delete_all = QPushButton()
        self.btn_delete_all.clicked.connect(self._delete_everything)
        menu = QHBoxLayout()
        for w in (self.btn_tablet_continue, self.btn_review, self.tablet_pick,
                  self.btn_redo_tablet, self.btn_delete_tablet, self.btn_delete_all):
            menu.addWidget(w)

        # Editing the paper import: one box (delete / replace) or the whole import (undo).
        self.paper_saved = []                     # [(path, role, index, attempt)] of the last import
        self.paper_list = QComboBox()
        self.btn_delete = QPushButton()
        self.btn_delete.clicked.connect(self._delete_selected)
        self.btn_replace = QPushButton()
        self.btn_replace.clicked.connect(self._choose_replacement)
        self.btn_undo = QPushButton()
        self.btn_undo.clicked.connect(self._undo_confirmed)
        edit = QHBoxLayout()
        for w in (self.paper_list, self.btn_delete, self.btn_replace, self.btn_undo):
            edit.addWidget(w)
        self.edit_widgets = (self.paper_list, self.btn_delete, self.btn_replace, self.btn_undo)
        # The box chosen in paper_list, shown large (scan | ink crop) beside the labelled grid.
        self.selected = QLabel()
        self.shown = None
        self.paper_list.currentIndexChanged.connect(self._show_selected)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.prompt)
        layout.addWidget(self.capture)
        layout.addLayout(buttons)
        layout.addLayout(menu)
        layout.addLayout(edit)
        views = QHBoxLayout()
        views.addWidget(self.preview)
        views.addWidget(self.selected)
        layout.addLayout(views)
        self._retranslate()
        self._show_buttons()

    # --- flow ---------------------------------------------------------------
    def start(self, participant, operator, forger=None):
        signature_path(self.root, participant, "tablet", "genuine", index=1)   # validates the code
        if forger:
            signature_path(self.root, participant, "tablet", "forgery", forger=forger, attempt=1)
        self.participant, self.operator, self.forger = participant, operator, forger
        self.redo, self.paper_saved = [], []
        self._refresh_paper()
        if saved_tablet(self.root, participant, forger) or saved_paper(self.root, participant, forger):
            self.open_menu()
        else:
            self.begin_tablet()

    def begin_tablet(self):
        """Sign what is missing: a forger's next attempt, or practice before the genuines."""
        if self.forger:
            self.stage, self.attempt = "forgery", self._next_attempt()
            if self.attempt == 1:
                self._say("forgery_first", forger=self.forger, participant=self.participant,
                          n=CAMPAIGN_FORGERY_ATTEMPTS)
            else:
                self._say("forgery_again", i=self.attempt, n=CAMPAIGN_FORGERY_ATTEMPTS)
        else:
            self.stage, self.practised = "practice", 0
            self._say("practice_first", n=CAMPAIGN_PRACTICE_SIGNINGS)
        self.capture.setEnabled(True)
        self._show_buttons()

    def open_menu(self):
        """Show what is saved for this code and let the operator choose what to do."""
        self.stage, self.redo, self.paper_saved = "menu", [], []
        self.capture.setEnabled(False)
        self._refresh_paper()
        tablet = saved_tablet(self.root, self.participant, self.forger)
        self.tablet_pick.clear()
        self.tablet_pick.addItems([self._t("pick_all")] + [t[0].name for t in tablet])
        self._say("menu_forgery" if self.forger else "menu_genuine",
                  participant=self.participant, forger=self.forger, tablet=len(tablet),
                  tablet_n=len(expected_files(self.root, self.participant, "tablet", self.forger)),
                  paper=len(saved_paper(self.root, self.participant, self.forger)),
                  paper_n=len(expected_files(self.root, self.participant, "paper", self.forger)))
        self._show_buttons()

    def review_paper(self):
        """Open the saved paper in the same grid an import ends on."""
        self.stage = "done"
        self.paper_saved = saved_paper(self.root, self.participant, self.forger)
        self._say("review")
        self._refresh_paper()
        self._show_buttons()

    def redo_tablet(self, files):
        """Sign these again, one by one; each replaces its file only once approved."""
        self.stage, self.redo, self.redo_total = "redo", list(files), len(files)
        self.capture.setEnabled(True)
        self._say_redo()
        self._show_buttons()

    def delete_files(self, paths):
        """Permanently remove these files, their manifest rows, and any folder left empty."""
        for path in paths:
            _check_inside(self.root, path).unlink(missing_ok=True)
        remove_manifest_rows(self.root, [Path(p).relative_to(self.root) for p in paths])
        for folder in {Path(p).parent for p in paths}:
            if folder.exists() and not any(folder.iterdir()):
                folder.rmdir()

    def continue_after_break(self):
        self.stage = "genuine"
        self.capture.setEnabled(True)
        self._say("genuine_next", i=self.index, n=CAMPAIGN_GENUINE_PER_MODALITY)
        self._show_buttons()

    def _end_practice(self):
        self.stage = "genuine"
        self.index = next_index(self.root, self.participant, "tablet")
        self._say("genuine_first", i=self.index, n=CAMPAIGN_GENUINE_PER_MODALITY)
        self._show_buttons()

    def _on_signature(self, _trajectory):
        events = list(self.capture.canvas.raw_events)   # read before the widget clears its canvas
        if self.stage == "practice":
            self.practised += 1
            if self.practised >= CAMPAIGN_PRACTICE_SIGNINGS:
                self._end_practice()
            else:
                self._say("practice_next", i=self.practised + 1, n=CAMPAIGN_PRACTICE_SIGNINGS)
        elif self.stage == "genuine":
            self._save_tablet(events, "genuine", index=self.index)
            self.index = next_index(self.root, self.participant, "tablet")
            if self.index > CAMPAIGN_GENUINE_PER_MODALITY:
                self._tablet_finished("tablet_done")
            elif self.index == CAMPAIGN_BREAK_AFTER + 1:
                self.stage = "break"
                self.capture.setEnabled(False)
                self._say("break")
            else:
                self._say("genuine_next", i=self.index, n=CAMPAIGN_GENUINE_PER_MODALITY)
        elif self.stage == "forgery":
            self._save_tablet(events, "forgery", forger=self.forger, attempt=self.attempt)
            self.attempt = self._next_attempt()
            if self.attempt > CAMPAIGN_FORGERY_ATTEMPTS:
                self._tablet_finished("tablet_done_forgery", n=CAMPAIGN_FORGERY_ATTEMPTS)
            else:
                self._say("forgery_again", i=self.attempt, n=CAMPAIGN_FORGERY_ATTEMPTS)
        elif self.stage == "redo":
            path, role, index, attempt = self.redo.pop(0)
            replace_raw_events(path, events, self.root)
            remove_manifest_rows(self.root, [path.relative_to(self.root)])
            self._manifest(path, "tablet", role, index=index, attempt=attempt)
            if self.redo:
                self._say_redo()
            else:
                self.open_menu()
        self.capture.label.setText(self._t("canvas_hint"))   # the widget wrote "Accepted!" there
        self._show_buttons()

    def save_paper(self, sheet_paths):
        """Cut each sheet into its boxes in order; save the genuines then the forger copy
        (genuine session) or the forger's attempts (forgery session). A box already saved is
        kept, so only the missing ones are filled; a complete set is refused."""
        crops = signing_order(read_sheets(sheet_paths))
        expected = expected_files(self.root, self.participant, "paper", self.forger)
        if len(crops) < len(expected):
            raise ValueError(f"{len(crops)} boxes scanned, {len(expected)} needed")
        missing = [(crop, t) for crop, t in zip(crops, expected) if not t[0].exists()]
        if not missing:
            raise FileExistsError(expected[0][0])
        targets = [t for _, t in missing]
        no_ink = []
        for crop, (path, role, index, attempt) in missing:
            _check_inside(self.root, path).parent.mkdir(parents=True, exist_ok=True)
            imwrite(str(path), crop)
            self._manifest(path, "paper", role, index=index, attempt=attempt)
            if not save_ink(path, self.root):
                no_ink.append(path.stem)
        self.stage = "done"
        self.paper_saved = list(targets)
        if no_ink:
            self._say("paper_saved_no_ink", n=len(targets), names=", ".join(no_ink))
        else:
            self._say("paper_saved", n=len(targets))
        self._refresh_paper()
        self._show_buttons()
        return [t[0] for t in targets]

    def delete_paper(self, path):
        """Remove one imported paper signature and its manifest row."""
        path = _check_inside(self.root, path)
        path.unlink(missing_ok=True)  # skylos: ignore[SKY-D215] checked by _check_inside: inside the data folder, not a symlink
        _check_inside(self.root, ink_path(path)).unlink(missing_ok=True)
        remove_manifest_rows(self.root, [path.relative_to(self.root)])
        self.paper_saved = [t for t in self.paper_saved if t[0] != path]
        self._say("deleted", name=path.name)
        self._refresh_paper()

    def replace_paper(self, path, image_path):
        """Put the image of one signature in place of an imported box, under the same name.
        The one deliberate overwrite in this tool: the operator chose it."""
        path = Path(path)
        image = imread(str(image_path))
        if image is None:
            raise ValueError(f"cannot read {image_path}")
        _, role, index, attempt = next(t for t in self.paper_saved if t[0] == path)
        imwrite(str(_check_inside(self.root, path)), image)
        remove_manifest_rows(self.root, [path.relative_to(self.root)])
        self._manifest(path, "paper", role, index=index, attempt=attempt)
        found = save_ink(path, self.root)
        self._say("replaced" if found else "replaced_no_ink", name=path.name)
        self._refresh_paper()

    def undo_paper(self):
        """Delete the whole last paper import, so the sheets can be imported again."""
        for path, *_ in self.paper_saved:
            _check_inside(self.root, path).unlink(missing_ok=True)
            _check_inside(self.root, ink_path(path)).unlink(missing_ok=True)
        remove_manifest_rows(self.root, [t[0].relative_to(self.root) for t in self.paper_saved])
        self.paper_saved = []
        self.stage = "paper"
        self._say("undone")
        self._refresh_paper()
        self._show_buttons()

    def end_session(self):
        """Clear the window for the next participant. Everything is already saved."""
        self.stage, self.index, self.attempt, self.practised = "idle", 0, 0, 0
        self.participant = self.operator = self.forger = None
        self.paper_saved, self.redo = [], []
        for field in (self.code, self.forger_code):
            field.clear()
        self.capture.reset_capture()
        self.capture.setEnabled(False)
        self._refresh_paper()
        self._say("start_hint")
        self._show_buttons()

    # --- helpers ------------------------------------------------------------
    def _start_from_form(self):
        forger = self.forger_code.text().strip() if self.mode.currentIndex() == 1 else ""
        try:
            self.start(self.code.text().strip(), self.op.text().strip(), forger or None)
        except ValueError:
            self._say("bad_code")

    def _next_attempt(self):
        missing = [t for t in expected_files(self.root, self.participant, "tablet", self.forger)
                   if not t[0].exists()]
        return missing[0][3] if missing else CAMPAIGN_FORGERY_ATTEMPTS + 1

    def _tablet_finished(self, key, **values):
        """On to the paper, unless it is already complete."""
        if len(saved_paper(self.root, self.participant, self.forger)) == len(
                expected_files(self.root, self.participant, "paper", self.forger)):
            self.open_menu()
        else:
            self._to_paper(key, **values)

    def _say_redo(self):
        self._say("redo_one", name=self.redo[0][0].name, n=self.redo_total,
                  i=self.redo_total - len(self.redo) + 1)

    def _picked(self):
        """The chosen saved tablet signature as a one-item list; None when "all" is chosen."""
        i = self.tablet_pick.currentIndex()
        return None if i <= 0 else [saved_tablet(self.root, self.participant, self.forger)[i - 1]]

    def _redo_picked(self):
        self.redo_tablet(self._picked()
                         or expected_files(self.root, self.participant, "tablet", self.forger))

    def _delete_picked(self):
        files = self._picked() or saved_tablet(self.root, self.participant, self.forger)
        what = files[0][0].name if len(files) == 1 else self._t("what_tablet", n=len(files))
        if self._confirm(self._t("confirm_delete", what=what)):
            self.delete_files([t[0] for t in files])
            if (saved_tablet(self.root, self.participant, self.forger)
                    or saved_paper(self.root, self.participant, self.forger)):
                self.open_menu()
            else:
                self.end_session()

    def _delete_everything(self):
        """Withdrawal: the operator must type the code, because this cannot be undone."""
        code = self.forger or self.participant
        name = f"{self.forger} / {self.participant}" if self.forger else self.participant
        files = person_files(self.root, self.participant, self.forger)
        if self._typed(self._t("type_code", n=len(files), name=name, code=code)) == code:
            self.delete_files(files)
            self.end_session()
            self._say("deleted_all", name=name, n=len(files))

    def _delete_selected(self):
        if self.paper_saved and self._confirm(
                self._t("confirm_delete", what=self.paper_list.currentText())):
            self._edit_selected(self.delete_paper)

    def _undo_confirmed(self):
        what = self._t("what_paper", n=len(self.paper_saved))
        if self._confirm(self._t("confirm_delete", what=what)):
            self.undo_paper()

    def _confirm(self, text):
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setText(text)
        yes = box.addButton(self._t("btn_yes"), QMessageBox.YesRole)
        box.setDefaultButton(box.addButton(self._t("btn_no"), QMessageBox.NoRole))
        box.exec_()
        return box.clickedButton() == yes

    def _typed(self, text):
        value, ok = QInputDialog.getText(self, self.windowTitle(), text)
        return value.strip() if ok else ""

    def _choose_sheets(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, self._t("dlg_sheets"), "", "Scans (*.pdf *.PDF *.png *.jpg *.jpeg *.tif *.tiff)")
        if paths:
            try:
                self.save_paper(paths)
            except (ValueError, FileExistsError) as exc:
                self._say("not_saved", error=exc)

    def _edit_selected(self, action, *args):
        if self.paper_list.currentIndex() >= 0 and self.paper_saved:
            action(self.paper_saved[self.paper_list.currentIndex()][0], *args)

    def _choose_replacement(self):
        image, _ = QFileDialog.getOpenFileName(
            self, self._t("dlg_replace"), "", "Images (*.png *.jpg *.jpeg *.tif *.tiff)")
        if image:
            try:
                self._edit_selected(self.replace_paper, image)
            except ValueError as exc:
                self._say("not_replaced", error=exc)

    def _refresh_paper(self):
        """List what the last paper import saved, keep the chosen box chosen, and redraw the
        labelled grid and the large view of that box."""
        keep = self.paper_list.currentIndex()
        self.paper_list.blockSignals(True)
        self.paper_list.clear()
        self.paper_list.addItems([t[0].name for t in self.paper_saved])
        if self.paper_saved:
            self.paper_list.setCurrentIndex(min(max(keep, 0), len(self.paper_saved) - 1))
        self.paper_list.blockSignals(False)
        self._show_selected(self.paper_list.currentIndex())

    def _show_selected(self, i):
        """Frame box i in the grid and show it large: the scan beside its ink crop."""
        self.preview.clear()
        self.selected.clear()
        self.shown = None
        if not 0 <= i < len(self.paper_saved):
            return
        path = self.paper_saved[i][0]
        self.shown = path
        labels = [("copy" if role == "forgercopy" else f"a{attempt}" if role == "forgery"
                   else f"{index:02d}") for _, role, index, attempt in self.paper_saved]
        tiles = []
        for p, *_ in self.paper_saved:
            img = imread(str(ink_path(p) if ink_path(p).exists() else p))
            tiles.append(img if img is not None else np.full((120, 240, 3), 255, np.uint8))
        self._preview(tiles, labels, highlight=i)
        raw = imread(str(path))
        ink = imread(str(ink_path(path))) if ink_path(path).exists() else None
        big = np.hstack([_tile(raw, 420, 260, "scan"),
                         _tile(ink, 420, 260, "ink crop (model sees)" if ink is not None
                               else "no signature found")])
        self.selected.setPixmap(_pixmap(big))

    def _save_tablet(self, events, role, index="", forger=None, attempt=""):
        path = signature_path(self.root, self.participant, "tablet", role,
                              index=index or None, forger=forger, attempt=attempt or None)
        write_raw_events(path, events, self.root)
        self._manifest(path, "tablet", role, index=index, attempt=attempt)

    def _manifest(self, path, modality, role, index="", attempt=""):
        append_manifest(self.root, {
            "file": str(Path(path).relative_to(self.root)), "participant": self.participant,
            "modality": modality, "role": role, "index": index, "forger": self.forger or "",
            "attempt": attempt, "operator": self.operator,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), **self.capture.capture_info()})

    def _to_paper(self, key, **values):
        self.stage = "paper"
        self.capture.setEnabled(False)
        self._say(key, **values)

    # --- language -----------------------------------------------------------
    def set_language(self, lang):
        """Show every instruction in lang ("en" or "ar"), including the prompt already shown."""
        self.lang = lang
        self._retranslate()
        self._show_buttons()

    def toggle_language(self):
        self.set_language("ar" if self.lang == "en" else "en")

    def _t(self, key, **values):
        return TEXT[self.lang][key].format(**values)

    def _say(self, key, **values):
        self._last = (key, values)
        self.prompt.setText(self._t(key, **values))

    def _retranslate(self):
        for widget, key in ((self.btn_start, "btn_start"), (self.capture.btn_confirm, "btn_approve"),
                            (self.capture.btn_reset, "btn_redo"), (self.btn_skip, "btn_skip"),
                            (self.btn_continue, "btn_continue"), (self.btn_paper, "btn_paper"),
                            (self.btn_delete, "btn_delete"), (self.btn_replace, "btn_replace"),
                            (self.btn_undo, "btn_undo"),
                            (self.btn_tablet_continue, "btn_tablet_continue"),
                            (self.btn_review, "btn_review"),
                            (self.btn_redo_tablet, "btn_redo_tablet"),
                            (self.btn_delete_tablet, "btn_delete_tablet")):
            widget.setText(self._t(key))
        for field, key in ((self.code, "ph_code"), (self.op, "ph_operator"),
                           (self.forger_code, "ph_forger")):
            field.setPlaceholderText(self._t(key))
        self.mode.setItemText(0, self._t("mode_genuine"))
        self.mode.setItemText(1, self._t("mode_forgery"))
        if self.tablet_pick.count():
            self.tablet_pick.setItemText(0, self._t("pick_all"))
        self.btn_lang.setText("English" if self.lang == "ar" else "العربية")
        rtl = self.lang == "ar"
        self.prompt.setLayoutDirection(Qt.RightToLeft if rtl else Qt.LeftToRight)
        self.prompt.setAlignment(Qt.AlignLeading | Qt.AlignVCenter)   # right in Arabic, left in English
        self.capture.label.setText(self._t("canvas_hint"))
        key, values = self._last
        self._say(key, **values)

    def _show_buttons(self):
        self.btn_skip.setVisible(self.stage == "practice")
        self.btn_continue.setVisible(self.stage == "break")
        menu = self.stage == "menu"
        tablet = menu and saved_tablet(self.root, self.participant, self.forger)
        paper = menu and saved_paper(self.root, self.participant, self.forger)
        expected = (lambda modality: expected_files(self.root, self.participant, modality,
                                                    self.forger))
        self.btn_paper.setVisible(self.stage == "paper"
                                  or (menu and len(paper) < len(expected("paper"))))
        self.btn_end.setVisible(self.stage in ("done", "paper", "menu", "redo"))
        self.btn_end.setText(self._t({"menu": "btn_cancel", "paper": "btn_later"}.get(
            self.stage, "btn_end")))
        self.btn_tablet_continue.setVisible(
            menu and any(not t[0].exists() for t in expected("tablet")))
        self.btn_review.setVisible(bool(paper))
        for w in (self.tablet_pick, self.btn_redo_tablet, self.btn_delete_tablet):
            w.setVisible(bool(tablet))
        self.btn_delete_all.setVisible(menu)
        self.btn_delete_all.setText(self._t("btn_delete_forger" if self.forger
                                            else "btn_delete_person"))
        for w in self.edit_widgets:
            w.setVisible(self.stage == "done" and bool(self.paper_saved))

    def _preview(self, crops, labels=None, highlight=None):
        """The grid of boxes, 4 per row, each labelled; the highlighted one framed in red."""
        if not crops:
            return
        labels = labels or [""] * len(crops)
        thumbs = [_tile(c, 240, 120, text, frame=(n == highlight))
                  for n, (c, text) in enumerate(zip(crops, labels))]
        while len(thumbs) % 4:
            thumbs.append(np.full_like(thumbs[0], 255))
        grid = np.vstack([np.hstack(thumbs[i:i + 4]) for i in range(0, len(thumbs), 4)])
        self.preview.setPixmap(_pixmap(grid))


def _tile(image, width, height, text="", frame=False):
    """image fitted into a white width x height tile (aspect kept), with a caption, and a red
    frame when chosen. A missing image gives an empty tile."""
    tile = np.full((height, width, 3), 255, np.uint8)
    if image is not None:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        scale = min((width - 8) / image.shape[1], (height - 28) / image.shape[0])
        w, h = max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))
        x, y = (width - w) // 2, 22 + (height - 22 - h) // 2
        tile[y:y + h, x:x + w] = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    cv2.putText(tile, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(tile, (0, 0), (width - 1, height - 1), (0, 0, 255) if frame else (200, 200, 200),
                  4 if frame else 1)
    return tile


def _pixmap(bgr):
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QImage.Format_RGB888)
    return QPixmap.fromImage(image.copy())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--make-sheet", metavar="PNG|PDF",
                    help="write the printable paper sheet and exit (.pdf: front and back, A4)")
    ap.add_argument("--root", default=INHOUSE_DIR, help="data folder (default: INHOUSE_DIR)")
    ap.add_argument("--pen-api", choices=sorted(PEN_PLATFORM), default=None,
                    help="Windows pen input (default: wintab when the tablet driver is installed, else ink); "
                         "use ink if the pen draws nothing")
    args = ap.parse_args()
    if args.make_sheet:
        make_sheet(args.make_sheet)
        two_sided = ", two-sided (flip on long edge)" if args.make_sheet.lower().endswith(".pdf") else ""
        print(f"sheet written to {args.make_sheet}: print at 100% (actual size) on A4{two_sided}; "
              "scan each full page, front first")
        return
    choose_pen_input(args.pen_api)
    app = QApplication(sys.argv)
    window = CaptureSession(root=args.root)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
