"""Canonical legal-document registry for the controlled experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple


_BACKEND_DIR = Path(__file__).resolve().parents[2]
_DATA_DIR = _BACKEND_DIR / "data"

# (file path, stable source key, display name)
DOCUMENTS: List[Tuple[str, str, str]] = [
    (
        str(_DATA_DIR / "muluki_ain.txt"),
        "muluki_ain",
        "मुलुकी देवानी (संहिता) ऐन, २०७४",
    ),
    (
        str(_DATA_DIR / "domestic_violence.txt"),
        "domestic_violence",
        "घरेलु हिंसा (कसूर र सजाय) ऐन, २०६६",
    ),
]

OFFICIAL_SOURCE_URLS: Dict[str, str] = {
    "muluki_ain": (
        "https://repository.lawcommission.gov.np/np/documents/"
        "%E0%A4%AE%E0%A5%81%E0%A4%B2%E0%A5%81%E0%A4%95%E0%A5%80-"
        "%E0%A4%A6%E0%A5%87%E0%A4%B5%E0%A4%BE%E0%A4%A8%E0%A5%80-"
        "%E0%A4%B8%E0%A4%82%E0%A4%B9%E0%A4%BF%E0%A4%A4%E0%A4%BE-"
        "%E0%A4%90%E0%A4%A8-%E0%A5%A8/"
    ),
    "domestic_violence": (
        "https://repository.lawcommission.gov.np/np/documents/"
        "%E0%A4%98%E0%A4%B0%E0%A5%87%E0%A4%B2%E0%A5%81-"
        "%E0%A4%B9%E0%A4%BF%E0%A4%82%E0%A4%B8%E0%A4%BE-"
        "%E0%A4%95%E0%A4%B8%E0%A5%82%E0%A4%B0-%E0%A4%B0-"
        "%E0%A4%B8%E0%A4%9C%E0%A4%BE%E0%A4%AF-%E0%A4%90%E0%A4%A8/"
    ),
}

# High-confidence conversion artifacts observed in the supplied Domestic
# Violence Act. These are not harmless spelling variants; many alter ordinary
# legal words (for example, section references and party labels).
_FORBIDDEN_ARTIFACTS: Dict[str, Tuple[str, ...]] = {
    "domestic_violence": (
        "ऐनकको",
        "र्दफा",
        "संशकोध",
        "कारबाी",
        "पीमडलिे",
        "तकोकीएकको",
    ),
}


def audit_registered_corpus() -> List[str]:
    """Return corpus-admission errors without modifying any source file."""
    errors: List[str] = []
    seen_sources = set()
    for path_text, source, _ in DOCUMENTS:
        path = Path(path_text)
        if source in seen_sources:
            errors.append(f"duplicate source key: {source}")
        seen_sources.add(source)
        if not path.is_file():
            errors.append(f"missing source file: {path}")
            continue
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            errors.append(f"empty source file: {path}")
            continue
        for artifact in _FORBIDDEN_ARTIFACTS.get(source, ()):
            count = text.count(artifact)
            if count:
                errors.append(
                    f"{source}: conversion artifact {artifact!r} occurs {count} time(s)"
                )
    return errors


def require_admitted_corpus() -> None:
    """Fail before embedding when registered legal text is not trustworthy."""
    errors = audit_registered_corpus()
    if errors:
        details = "\n  - ".join(errors)
        raise RuntimeError(
            "Corpus admission failed. Replace/correct the source against the "
            f"official Nepal Law Commission text before indexing:\n  - {details}"
        )


def infer_explicit_source_keys(query: str) -> List[str]:
    """Identify Acts explicitly named in a query; return no speculative match."""
    normalized = " ".join(query.lower().split())
    matches: List[str] = []
    if "मुलुकी देवानी" in normalized or "देवानी संहिता" in normalized:
        matches.append("muluki_ain")
    if "घरेलु हिंसा" in normalized and (
        "ऐन" in normalized or "कसूर र सजाय" in normalized
    ):
        matches.append("domestic_violence")
    return matches
