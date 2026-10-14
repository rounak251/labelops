"""Smoke-test the actual Streamlit workflow without a running browser."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_loads_and_records_review(tmp_path, monkeypatch):
    monkeypatch.setenv("LABELOPS_WORKSPACE", str(tmp_path / "ui"))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30)
    app.run()
    assert not app.exception
    assert app.title[0].value == "Better labels. Better models."
    sample_id = app.selectbox[0].value
    app.radio[0].set_value(5)
    save = next(button for button in app.button if button.label == "Save review")
    save.click().run()
    assert not app.exception
    from labelops.store import AnnotationStore

    rows = AnnotationStore(tmp_path / "ui" / "annotations.sqlite3").annotations()
    assert any(row["sample_id"] == sample_id and row["label"] == 5 for row in rows)
    corrected_digit = next(radio for radio in app.radio if radio.label == "Corrected digit")
    corrected_digit.set_value(6)
    next(button for button in app.button if button.label == "Save revision").click().run()
    assert not app.exception
    store = AnnotationStore(tmp_path / "ui" / "annotations.sqlite3")
    assert len(store.events()) == 2
    assert store.annotations()[0]["label"] == 6
