"""Headless Streamlit checks for app.py, via streamlit.testing.v1.AppTest.

    ./.venv/bin/pytest tests/test_app.py -v

No real network/API key required for the tests below -- they exercise the
"Gemini attempted, call fails" path by monkeypatching the actual API call
(pipeline/gemini_classify.py's _call_gemini_api_once), which is precise and
fast, rather than relying on a missing key (which would just mean Gemini is
never attempted at all under the current combined design, not what these
tests are checking) or hitting the real network (slow, costs quota, and
this session already hit a real daily-quota exhaustion once).

Three environment-specific gotchas worth knowing before touching this file:

1. gemini_classify.py's config constants (CACHE_DIR, MAX_RETRIES, etc.) are
   computed once at import time from env vars -- setting the env var after
   import has no effect. Patch the already-imported module's *attributes*
   directly (monkeypatch.setattr(gemini_classify, "CACHE_DIR", ...)), which
   AppTest's own `import gemini_classify` resolves to the same object via
   sys.modules.
2. app.py calls load_dotenv(".env") unconditionally on every script
   execution (AppTest re-execs the whole script per .run() call). If a real
   .env file exists on the machine running these tests (it does, for local
   dev), that would silently re-populate GEMINI_API_KEY etc. even after
   monkeypatch.delenv -- so tests that need precise env control disable
   dotenv itself for the duration (see _disable_dotenv below).
3. run_pipeline's @st.cache_data cache is process-global, not scoped per
   AppTest instance -- tests that need a fresh pipeline computation call
   st.cache_data.clear() first, or a later test can silently see an earlier
   test's stale cached rows (both share gemini_enabled=True as part of the
   cache key).
"""
import sys
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

import gemini_classify  # noqa: E402


def _fresh_app_test():
    return AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=60)


def _disable_dotenv(monkeypatch):
    """Stop app.py's own load_dotenv(".env") call from silently
    re-populating env vars a test just deleted, if a real .env exists."""
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)


def _isolate_gemini_state(monkeypatch, tmp_path):
    """Point the cache/pending stores at a scratch dir so tests never read
    or write this machine's real (possibly non-empty, from live testing)
    cache/pending files."""
    monkeypatch.setattr(gemini_classify, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(gemini_classify, "PENDING_STORE_PATH", tmp_path / "pending.json")


def test_no_api_key_is_pure_rules(monkeypatch, tmp_path):
    _disable_dotenv(monkeypatch)
    _isolate_gemini_state(monkeypatch, tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("SDOC_CLASSIFIER", raising=False)
    st.cache_data.clear()

    at = _fresh_app_test()
    at.run()
    assert not at.exception
    assert any("not configured" in c.value for c in at.sidebar.caption)
    assert gemini_classify.list_pending() == []


def test_gemini_failure_queues_as_pending(monkeypatch, tmp_path):
    _disable_dotenv(monkeypatch)
    _isolate_gemini_state(monkeypatch, tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
    monkeypatch.delenv("SDOC_CLASSIFIER", raising=False)
    monkeypatch.setattr(gemini_classify, "MAX_RETRIES", 1)  # don't sleep through real backoff

    def always_fails(email, rule_trace):
        raise gemini_classify.GeminiClassifyError("simulated outage")

    monkeypatch.setattr(gemini_classify, "_call_gemini_api_once", always_fails)
    st.cache_data.clear()

    at = _fresh_app_test()
    at.run()
    assert not at.exception

    pending_ids = {p["email_id"] for p in gemini_classify.list_pending()}
    assert "email_001" in pending_ids

    at.sidebar.radio[0].set_value("🔍 Email Inspector").run()
    at.selectbox[0].set_value("email_001").run()
    assert not at.exception
    assert any("queued for automatic retry" in i.value for i in at.main.info)


def test_pending_resolves_on_next_rerun(monkeypatch, tmp_path):
    _disable_dotenv(monkeypatch)
    _isolate_gemini_state(monkeypatch, tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
    monkeypatch.delenv("SDOC_CLASSIFIER", raising=False)
    monkeypatch.setattr(gemini_classify, "MAX_RETRIES", 1)
    # Long throttle for run 1: proves the failure stays queued rather than
    # being retried immediately within the same script execution. Run 2
    # drops it to 0 so the item is "due" and gets picked up.
    monkeypatch.setattr(gemini_classify, "POLL_INTERVAL_SECONDS", 3600)

    # warm_cache attempts all 520 emails concurrently, so the failure must be
    # scoped to email_001's own call count, not a counter shared across every
    # email's (unordered, multi-threaded) attempt.
    calls = {"email_001": 0}

    def flaky(email, rule_trace):
        eid = email["email_id"]
        if eid == "email_001":
            calls["email_001"] += 1
            if calls["email_001"] == 1:
                raise gemini_classify.GeminiClassifyError("simulated outage")
        return {"category": "GENERAL",
                 "scores": {c: 0.0 for c in gemini_classify.CATEGORIES},
                 "reasoning": "resolved on retry"}

    monkeypatch.setattr(gemini_classify, "_call_gemini_api_once", flaky)
    st.cache_data.clear()

    at = _fresh_app_test()
    at.run()
    assert not at.exception
    assert any(p["email_id"] == "email_001" for p in gemini_classify.list_pending())

    # Now it's due for a retry; the next script execution (what "next
    # refresh" means -- any page nav or widget interaction) should pick it
    # up automatically, with nothing clicked and no button involved.
    monkeypatch.setattr(gemini_classify, "POLL_INTERVAL_SECONDS", 0)
    at.run()
    assert not at.exception
    assert gemini_classify.list_pending() == []


@pytest.mark.skipif(not __import__("os").environ.get("GEMINI_API_KEY"),
                     reason="needs a real GEMINI_API_KEY")
def test_gemini_with_real_key_produces_gemini_decisions():
    st.cache_data.clear()
    at = _fresh_app_test()
    at.run()
    assert not at.exception

    at.sidebar.radio[0].set_value("🔍 Email Inspector").run()
    at.selectbox[0].set_value("email_001").run()
    assert not at.exception
    # rendered as "decided by: **gemini**" (or a queued/fallback message, if the live call itself failed)
    assert any("decided by" in c.value for c in at.caption)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
