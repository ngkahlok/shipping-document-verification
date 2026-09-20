"""Headless Streamlit checks for app.py, via streamlit.testing.v1.AppTest.

    ./.venv/bin/pytest tests/test_app.py

No network/API key required for these -- they specifically exercise the
"Gemini backend selected but unavailable" path, which must degrade to the
rule-based result visibly rather than hang, crash, or silently mis-score
(see pipeline/gemini_classify.py's _call_with_retry: a missing API key must
fail immediately, not enter the retry/backoff loop -- a prior bug here made
a single email take ~7s and a full run ~an hour).

An optional live test (needs a real GEMINI_API_KEY) is included at the
bottom, skipped automatically when the key isn't set.
"""
import os
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).parent.parent


def _fresh_app_test():
    return AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=60)


def _backend_selectbox(at):
    return [s for s in at.sidebar.selectbox if s.key == "classifier_backend"][0]


def test_default_load_has_no_exceptions_and_defaults_to_rules():
    at = _fresh_app_test()
    at.run()
    assert not at.exception
    assert _backend_selectbox(at).value == "rules"


def test_gemini_backend_without_api_key_falls_back_visibly(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    at = _fresh_app_test()
    at.run()

    _backend_selectbox(at).set_value("gemini").run()
    assert not at.exception
    assert any("GEMINI_API_KEY not set" in w.value for w in at.sidebar.warning)

    at.sidebar.radio[0].set_value("🔍 Email Inspector").run()
    at.selectbox[0].set_value("email_001").run()
    assert not at.exception
    assert any("fell back to the rule-based result" in w.value for w in at.main.warning)


@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="needs a real GEMINI_API_KEY")
def test_gemini_backend_with_real_key_produces_gemini_decisions():
    at = _fresh_app_test()
    at.run()
    _backend_selectbox(at).set_value("gemini").run()
    assert not at.exception

    at.sidebar.radio[0].set_value("🔍 Email Inspector").run()
    at.selectbox[0].set_value("email_001").run()
    assert not at.exception
    # rendered as "decided by: **gemini**" (or a fallback, if the live call itself failed)
    assert any("decided by" in c.value for c in at.caption)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
