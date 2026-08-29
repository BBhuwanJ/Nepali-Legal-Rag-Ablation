"""Transparent, label-independent source routing for the two-Act corpus.

The router uses only the user's query and a frozen lexical profile. It never
sees evaluation labels, required sections, or retrieved chunks. Profiles are
conservative: diversification activates only when both Acts have strong
signals or the query explicitly asks about both Acts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

from shared.corpus import DOCUMENTS, infer_explicit_source_keys


ROUTER_VERSION = "lexical-v1"

SOURCE_PROFILES: Dict[str, Tuple[Tuple[str, int], ...]] = {
    "domestic_violence": (
        ("घरेलु हिंसा", 5),
        ("घरेलु सम्बन्ध", 4),
        ("पीडक", 3),
        ("संरक्षणात्मक आदेश", 4),
        ("आर्थिक यातना", 4),
        ("मानसिक यातना", 4),
        ("शारीरिक यातना", 4),
        ("यौनजन्य यातना", 4),
        ("दाइजो", 3),
        ("आर्थिक स्रोतबाट वञ्चित", 4),
        ("रासायनिक पदार्थ", 3),
    ),
    "muluki_ain": (
        ("सम्बन्ध विच्छेद", 5),
        ("अंशबण्डा", 5),
        ("अपुताली", 5),
        ("उत्तराधिकार", 4),
        ("करार", 4),
        ("संरक्षकत्व", 4),
        ("धर्मपुत्र", 4),
        ("धर्मपुत्री", 4),
        ("नाबालकलाई जिम्मा", 4),
        ("दीर्घकालीन जिम्मा", 4),
        ("मानाचामल", 4),
        ("खान-लाउन", 3),
        ("खान लाउन", 3),
        ("रोजगारी गर्न रोक", 3),
    ),
}

EXPLICIT_BOTH_CUES = (
    "दुवै ऐन", "दुबै ऐन", "दुवै कानून", "दुबै कानून",
    "दुवै कानुन", "दुबै कानुन",
)


@dataclass(frozen=True)
class SourceRoute:
    version: str
    route_type: str
    sources: Tuple[str, ...]
    scores: Dict[str, int]
    matched_phrases: Dict[str, Tuple[str, ...]]
    explicit_sources: Tuple[str, ...]
    reason: str

    @property
    def diversify(self) -> bool:
        return self.route_type == "multi_source" and len(self.sources) > 1

    def to_dict(self) -> dict:
        data = asdict(self)
        data["sources"] = list(self.sources)
        data["explicit_sources"] = list(self.explicit_sources)
        data["matched_phrases"] = {
            source: list(phrases) for source, phrases in self.matched_phrases.items()
        }
        data["diversify"] = self.diversify
        return data


def classify_source_route(query: str) -> SourceRoute:
    """Return a deterministic route using query text only."""
    normalized = " ".join(str(query).lower().split())
    registered = tuple(source for _, source, _ in DOCUMENTS)
    explicit = tuple(infer_explicit_source_keys(normalized))
    scores = {source: 0 for source in registered}
    matches: Dict[str, List[str]] = {source: [] for source in registered}

    for source in explicit:
        if source in scores:
            scores[source] += 100
            matches[source].append("<explicit-act-name>")

    for source, profile in SOURCE_PROFILES.items():
        if source not in scores:
            continue
        for phrase, weight in profile:
            if phrase in normalized:
                scores[source] += weight
                matches[source].append(phrase)

    signalled = tuple(source for source in registered if scores[source] >= 3)
    asks_both = any(cue in normalized for cue in EXPLICIT_BOTH_CUES)

    if len(explicit) > 1 or len(signalled) > 1:
        selected = set(explicit + signalled)
        sources = tuple(source for source in registered if source in selected)
        route_type = "multi_source"
        reason = "strong lexical or explicit signals identify more than one Act"
    elif asks_both and len(registered) == 2:
        sources = registered
        route_type = "multi_source"
        reason = "query explicitly asks about both Acts in the registered corpus"
    elif len(signalled) == 1:
        sources = signalled
        route_type = "single_source"
        reason = "one Act has a strong lexical signal"
    elif len(explicit) == 1:
        sources = explicit
        route_type = "single_source"
        reason = "one Act is explicitly named"
    else:
        sources = registered
        route_type = "global"
        reason = "no sufficiently strong source signal; use the global ranking"

    return SourceRoute(
        version=ROUTER_VERSION,
        route_type=route_type,
        sources=sources,
        scores=scores,
        matched_phrases={source: tuple(matches[source]) for source in registered},
        explicit_sources=explicit,
        reason=reason,
    )
