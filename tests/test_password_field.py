"""The shared password field (#58): eye button, live rule hint, Generate, on every screen."""
import os

import pytest
import torch  # noqa: F401 — before PyQt5: on Windows the other order fails to load c10.dll

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt5.QtWidgets")

from src.database.db_manager import password_problem  # noqa: E402
from src.gui import password_field  # noqa: E402
from src.gui.password_field import PasswordField, new_password_helpers  # noqa: E402

Password, Normal = QtWidgets.QLineEdit.Password, QtWidgets.QLineEdit.Normal


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_the_eye_shows_and_hides_the_text(app):
    field = PasswordField("Password")
    assert field.echoMode() == Password and field.placeholderText() == "Password"
    field.eye.trigger()
    assert field.echoMode() == Normal
    field.eye.trigger()
    assert field.echoMode() == Password


def test_the_text_is_hidden_again_when_the_page_is_left(app):
    page = QtWidgets.QWidget()
    field = PasswordField("Password", page)
    page.show()
    field.eye.trigger()
    page.hide()
    assert field.echoMode() == Password


def test_the_hint_says_what_is_missing_while_typing(app):
    field, confirm = PasswordField("New"), PasswordField("Confirm")
    _, hint = new_password_helpers(field, confirm)
    field.setText("short")
    assert "at least 15" in hint.text()
    field.setText("passwordpassword")
    assert "too common" in hint.text()
    field.setText("correct horse battery staple")
    assert hint.text() == "Long enough"


def test_generate_fills_both_fields_and_shows_the_password_with_its_warning(app, monkeypatch):
    shown = []
    monkeypatch.setattr(password_field, "show_generated", lambda parent, pw: shown.append(pw))
    field, confirm = PasswordField("New"), PasswordField("Confirm")
    button, _ = new_password_helpers(field, confirm)
    button.click()
    assert field.text() == confirm.text() == shown[0]
    assert password_problem(shown[0]) is None


def test_the_generated_password_dialog_copies_and_warns(app):
    dialog = password_field.GeneratedPasswordDialog(None, "Abc123Abc123Abc123Ab")
    texts = " ".join(label.text() for label in dialog.findChildren(QtWidgets.QLabel))
    assert "cannot be recovered" in texts
    dialog.copyButton.click()
    assert QtWidgets.QApplication.clipboard().text() == "Abc123Abc123Abc123Ab"
