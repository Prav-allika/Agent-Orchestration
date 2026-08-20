"""Import this before any `orchestration.*` import in a Streamlit page.

Streamlit runs each page as an independent script, so the normal
editable-install sys.path setup isn't guaranteed to apply; this puts
src/ on sys.path explicitly regardless of how `streamlit run` was invoked.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
