"""Research capture session for the XPPen campaign (spec section 6): one volunteer, one sitting.

Tablet: practice (never saved), then CAMPAIGN_GENUINE_PER_MODALITY genuine signatures with a
break after CAMPAIGN_BREAK_AFTER; each is saved only after the signer approves it and the input
gate passes. Paper: import the scanned sheets (print them with --make-sheet); each sheet is cut
into its grid of boxes in signing order. Forgery mode records a team forger's attempts at one
volunteer. Files go to INHOUSE_DIR/<participant>/, one manifest row each. Raw streams are
research data: keep INHOUSE_DIR on an encrypted disk, never in git or the app database.

Run:  python tools/capture_session.py
      python tools/capture_session.py --make-sheet sheet.png
"""
import argparse
import csv
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import (CAMPAIGN_BREAK_AFTER, CAMPAIGN_FORGERY_ATTEMPTS,  # noqa: E402
                             CAMPAIGN_GENUINE_PER_MODALITY, CAMPAIGN_PRACTICE_SIGNINGS,
                             INHOUSE_DIR, SHEET_GRID, SHEET_MARGIN)
from PyQt5.QtGui import QFont, QImage, QPixmap  # noqa: E402
from PyQt5.QtWidgets import (QApplication, QComboBox, QFileDialog, QHBoxLayout,  # noqa: E402
                             QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget)

from src.capture.ink import ink_box, ink_crop  # noqa: E402,F401  (ink_box: tests)
from src.capture.tablet import SignatureCaptureWidget  # noqa: E402

PARTICIPANT = re.compile(r"^P\d{2,3}$")
RAW_FIELDS = ["t_ms", "kind", "x", "y", "pressure", "xtilt", "ytilt", "inside"]
MANIFEST_FIELDS = ["file", "participant", "modality", "role", "index", "forger", "attempt",
                   "operator", "os", "qt_platform", "device_model", "saved_at"]
_EXT = {"tablet": "csv", "paper": "png"}


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
    """The next free genuine number, so a session that crashed resumes where it stopped."""
    folder = Path(root) / participant
    taken = [int(m.group(1)) for p in folder.glob(f"{participant}_{modality}_genuine_*")
             if (m := re.search(r"_genuine_(\d+)\.", p.name))]
    return max(taken, default=0) + 1


def write_raw_events(path, events):
    """One tablet signature: every Qt tablet event the canvas saw (RAW_FIELDS: screen pixels,
    y down, pressure 0..1, Qt tilt). Mode "x" refuses to overwrite a saved signature."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", newline="") as f:
        csv.writer(f).writerows([RAW_FIELDS] + [list(e) for e in events])


def append_manifest(root, row):
    """One row per saved file in <root>/manifest.csv; missing fields are left empty."""
    path = Path(root) / "manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})


def remove_manifest_rows(root, files):
    """Drop the manifest rows whose file is in files (paths relative to root)."""
    path = Path(root) / "manifest.csv"
    if not path.exists():
        return
    files = {str(f) for f in files}
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["file"] not in files]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def ink_path(path):
    """Beside each raw paper box: the ink crop the model sees."""
    path = Path(path)
    return path.with_name(f"{path.stem}_ink.png")


def save_ink(path):
    """Write ink_path(path) from the raw box. Returns False, and leaves no ink file, when no
    signature is found."""
    ink_path(path).unlink(missing_ok=True)
    crop = ink_crop(cv2.imread(str(path)))
    if crop is None:
        return False
    cv2.imwrite(str(ink_path(path)), crop)
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


def make_sheet(path, dpi=300):
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
        cv2.imwrite(str(path), img)


class CaptureSession(QWidget):
    """One volunteer, one sitting (or one forger at one volunteer), step by step."""

    def __init__(self, root=INHOUSE_DIR):
        super().__init__()
        self.root = Path(root)
        self.setWindowTitle("XPPen capture session")
        self.stage, self.index, self.attempt, self.practised = "idle", 0, 0, 0
        self.participant = self.operator = self.forger = None

        self.code = QLineEdit(placeholderText="Participant, e.g. P07")
        self.op = QLineEdit(placeholderText="Operator")
        self.mode = QComboBox()
        self.mode.addItems(["Genuine", "Forgery"])
        self.forger_code = QLineEdit(placeholderText="Forger, e.g. P02")
        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self._start_from_form)
        form = QHBoxLayout()
        for w in (self.code, self.op, self.mode, self.forger_code, self.btn_start):
            form.addWidget(w)

        self.prompt = QLabel("Enter the participant code and press Start.")
        font = QFont()
        font.setPointSize(18)
        font.setBold(True)
        self.prompt.setFont(font)
        self.prompt.setWordWrap(True)

        self.capture = SignatureCaptureWidget()
        self.capture.btn_confirm.setText("I approve")
        self.capture.btn_reset.setText("Redo")
        self.capture.signature_confirmed.connect(self._on_signature)
        self.capture.setEnabled(False)

        self.btn_skip = QPushButton("Skip practice")
        self.btn_skip.clicked.connect(self._end_practice)
        self.btn_continue = QPushButton("Continue")
        self.btn_continue.clicked.connect(self.continue_after_break)
        self.btn_paper = QPushButton("Import scanned paper sheets...")
        self.btn_paper.clicked.connect(self._choose_sheets)
        self.btn_end = QPushButton("End / Next participant")
        self.btn_end.clicked.connect(self.end_session)
        self.preview = QLabel()
        buttons = QHBoxLayout()
        for w in (self.btn_skip, self.btn_continue, self.btn_paper, self.btn_end):
            buttons.addWidget(w)

        # Editing the paper import: one box (delete / replace) or the whole import (undo).
        self.paper_saved = []                     # [(path, role, index, attempt)] of the last import
        self.paper_list = QComboBox()
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.clicked.connect(lambda: self._edit_selected(self.delete_paper))
        self.btn_replace = QPushButton("Replace...")
        self.btn_replace.clicked.connect(self._choose_replacement)
        self.btn_undo = QPushButton("Undo paper import")
        self.btn_undo.clicked.connect(self.undo_paper)
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
        layout.addLayout(edit)
        views = QHBoxLayout()
        views.addWidget(self.preview)
        views.addWidget(self.selected)
        layout.addLayout(views)
        self._show_buttons()

    # --- flow ---------------------------------------------------------------
    def start(self, participant, operator, forger=None):
        signature_path(self.root, participant, "tablet", "genuine", index=1)   # validates the code
        if forger:
            signature_path(self.root, participant, "tablet", "forgery", forger=forger, attempt=1)
        self.participant, self.operator, self.forger = participant, operator, forger
        if forger:
            self.stage, self.attempt = "forgery", 1
            self._say(f"Forger {forger} imitates {participant}: attempt 1 of "
                      f"{CAMPAIGN_FORGERY_ATTEMPTS}. Study the printed copy, then sign.")
        else:
            self.stage, self.practised = "practice", 0
            self._say(f"Practice 1 of {CAMPAIGN_PRACTICE_SIGNINGS}: sign as you normally do. "
                      "Nothing is saved.")
        self.capture.setEnabled(True)
        self._show_buttons()

    def continue_after_break(self):
        self.stage = "genuine"
        self.capture.setEnabled(True)
        self._say(f"Signature {self.index} of {CAMPAIGN_GENUINE_PER_MODALITY}.")
        self._show_buttons()

    def _end_practice(self):
        self.stage = "genuine"
        self.index = next_index(self.root, self.participant, "tablet")
        self._say(f"Signature {self.index} of {CAMPAIGN_GENUINE_PER_MODALITY}. "
                  "Sign, look at it, then press I approve (or Redo).")
        self._show_buttons()

    def _on_signature(self, _tensor):
        events = list(self.capture.canvas.raw_events)   # read before the widget clears its canvas
        if self.stage == "practice":
            self.practised += 1
            if self.practised >= CAMPAIGN_PRACTICE_SIGNINGS:
                self._end_practice()
            else:
                self._say(f"Practice {self.practised + 1} of {CAMPAIGN_PRACTICE_SIGNINGS}. "
                          "Nothing is saved.")
        elif self.stage == "genuine":
            self._save_tablet(events, "genuine", index=self.index)
            self.index += 1
            if self.index > CAMPAIGN_GENUINE_PER_MODALITY:
                self._to_paper("Tablet done. Now import the scanned paper sheets.")
            elif self.index == CAMPAIGN_BREAK_AFTER + 1:
                self.stage = "break"
                self.capture.setEnabled(False)
                self._say("Short break. Press Continue when ready.")
            else:
                self._say(f"Signature {self.index} of {CAMPAIGN_GENUINE_PER_MODALITY}.")
        elif self.stage == "forgery":
            self._save_tablet(events, "forgery", forger=self.forger, attempt=self.attempt)
            self.attempt += 1
            if self.attempt > CAMPAIGN_FORGERY_ATTEMPTS:
                self._to_paper(f"Tablet done. Import the forger's paper sheet "
                               f"(boxes 1-{CAMPAIGN_FORGERY_ATTEMPTS} = attempts in order).")
            else:
                self._say(f"Sign again: attempt {self.attempt} of {CAMPAIGN_FORGERY_ATTEMPTS}.")
        self._show_buttons()

    def save_paper(self, sheet_paths):
        """Cut each sheet into its boxes in order; save the genuines then the forger copy
        (genuine session) or the forger's attempts (forgery session)."""
        crops = [c for p in sheet_paths for c in split_sheet(cv2.imread(str(p)))]
        if self.forger:
            targets = [(signature_path(self.root, self.participant, "paper", "forgery",
                                       forger=self.forger, attempt=a), "forgery", "", a)
                       for a in range(1, CAMPAIGN_FORGERY_ATTEMPTS + 1)]
        else:
            targets = [(signature_path(self.root, self.participant, "paper", "genuine", index=i),
                        "genuine", i, "") for i in range(1, CAMPAIGN_GENUINE_PER_MODALITY + 1)]
            targets.append((signature_path(self.root, self.participant, "paper", "forgercopy"),
                            "forgercopy", "", ""))
        if len(crops) < len(targets):
            raise ValueError(f"{len(crops)} boxes scanned, {len(targets)} needed")
        for _, (path, *_rest) in zip(crops, targets):
            if path.exists():
                raise FileExistsError(path)
        no_ink = []
        for crop, (path, role, index, attempt) in zip(crops, targets):
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), crop)
            self._manifest(path, "paper", role, index=index, attempt=attempt)
            if not save_ink(path):
                no_ink.append(path.stem)
        self.stage = "done"
        self.paper_saved = list(targets)
        warning = f" No signature found in: {', '.join(no_ink)}." if no_ink else ""
        self._say(f"Saved {len(targets)} paper signatures. Session done.{warning} Check the "
                  "crops: delete or replace a bad one, or undo the whole import.")
        self._refresh_paper()
        self._show_buttons()
        return [t[0] for t in targets]

    def delete_paper(self, path):
        """Remove one imported paper signature and its manifest row."""
        path = Path(path)
        path.unlink(missing_ok=True)
        ink_path(path).unlink(missing_ok=True)
        remove_manifest_rows(self.root, [path.relative_to(self.root)])
        self.paper_saved = [t for t in self.paper_saved if t[0] != path]
        self._say(f"Deleted {path.name}.")
        self._refresh_paper()

    def replace_paper(self, path, image_path):
        """Put the image of one signature in place of an imported box, under the same name.
        The one deliberate overwrite in this tool: the operator chose it."""
        path = Path(path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"cannot read {image_path}")
        _, role, index, attempt = next(t for t in self.paper_saved if t[0] == path)
        cv2.imwrite(str(path), image)
        remove_manifest_rows(self.root, [path.relative_to(self.root)])
        self._manifest(path, "paper", role, index=index, attempt=attempt)
        found = save_ink(path)
        self._say(f"Replaced {path.name}." + ("" if found else " No signature found in it."))
        self._refresh_paper()

    def undo_paper(self):
        """Delete the whole last paper import, so the sheets can be imported again."""
        for path, *_ in self.paper_saved:
            path.unlink(missing_ok=True)
            ink_path(path).unlink(missing_ok=True)
        remove_manifest_rows(self.root, [t[0].relative_to(self.root) for t in self.paper_saved])
        self.paper_saved = []
        self.stage = "paper"
        self._say("Paper import undone. Import the scanned sheets again.")
        self._refresh_paper()
        self._show_buttons()

    def end_session(self):
        """Clear the window for the next participant. Everything is already saved."""
        self.stage, self.index, self.attempt, self.practised = "idle", 0, 0, 0
        self.participant = self.operator = self.forger = None
        self.paper_saved = []
        for field in (self.code, self.forger_code):
            field.clear()
        self.capture.reset_capture()
        self.capture.setEnabled(False)
        self._refresh_paper()
        self._say("Enter the participant code and press Start.")
        self._show_buttons()

    # --- helpers ------------------------------------------------------------
    def _start_from_form(self):
        forger = self.forger_code.text().strip() if self.mode.currentText() == "Forgery" else ""
        try:
            self.start(self.code.text().strip(), self.op.text().strip(), forger or None)
        except ValueError as exc:
            self._say(str(exc))

    def _choose_sheets(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Scanned sheets, in order", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff)")
        if paths:
            try:
                self.save_paper(paths)
            except (ValueError, FileExistsError) as exc:
                self._say(f"Not saved: {exc}")

    def _edit_selected(self, action, *args):
        if self.paper_list.currentIndex() >= 0 and self.paper_saved:
            action(self.paper_saved[self.paper_list.currentIndex()][0], *args)

    def _choose_replacement(self):
        image, _ = QFileDialog.getOpenFileName(
            self, "Image of this one signature", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff)")
        if image:
            try:
                self._edit_selected(self.replace_paper, image)
            except ValueError as exc:
                self._say(f"Not replaced: {exc}")

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
            img = cv2.imread(str(ink_path(p) if ink_path(p).exists() else p))
            tiles.append(img if img is not None else np.full((120, 240, 3), 255, np.uint8))
        self._preview(tiles, labels, highlight=i)
        raw = cv2.imread(str(path))
        ink = cv2.imread(str(ink_path(path))) if ink_path(path).exists() else None
        big = np.hstack([_tile(raw, 420, 260, "scan"),
                         _tile(ink, 420, 260, "ink crop (model sees)" if ink is not None
                               else "no signature found")])
        self.selected.setPixmap(_pixmap(big))

    def _save_tablet(self, events, role, index="", forger=None, attempt=""):
        path = signature_path(self.root, self.participant, "tablet", role,
                              index=index or None, forger=forger, attempt=attempt or None)
        write_raw_events(path, events)
        self._manifest(path, "tablet", role, index=index, attempt=attempt)

    def _manifest(self, path, modality, role, index="", attempt=""):
        append_manifest(self.root, {
            "file": str(Path(path).relative_to(self.root)), "participant": self.participant,
            "modality": modality, "role": role, "index": index, "forger": self.forger or "",
            "attempt": attempt, "operator": self.operator,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), **self.capture.capture_info()})

    def _to_paper(self, text):
        self.stage = "paper"
        self.capture.setEnabled(False)
        self._say(text)

    def _say(self, text):
        self.prompt.setText(text)

    def _show_buttons(self):
        self.btn_skip.setVisible(self.stage == "practice")
        self.btn_continue.setVisible(self.stage == "break")
        self.btn_paper.setVisible(self.stage == "paper")
        self.btn_end.setVisible(self.stage == "done")
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
    args = ap.parse_args()
    if args.make_sheet:
        make_sheet(args.make_sheet)
        two_sided = ", two-sided (flip on long edge)" if args.make_sheet.lower().endswith(".pdf") else ""
        print(f"sheet written to {args.make_sheet}: print at 100% (actual size) on A4{two_sided}; "
              "scan each full page, front first")
        return
    app = QApplication(sys.argv)
    window = CaptureSession(root=args.root)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
