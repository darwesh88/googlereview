from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TOKEN_TEMPLATE = "<{concept_id}>"
TOKEN_PATTERN = re.compile(r"<([A-Za-z0-9_:-]+)>")


@dataclass(frozen=True)
class ConceptEntry:
    concept_id: str
    canonical: str
    aliases: tuple[str, ...]


class ConceptLexicon:
    def __init__(self, entries: Iterable[ConceptEntry]) -> None:
        self.entries = list(entries)
        if not self.entries:
            raise ValueError("Concept lexicon cannot be empty")

        self.id_to_entry = {entry.concept_id: entry for entry in self.entries}
        self.alias_to_id: dict[str, str] = {}

        aliases: list[str] = []
        for entry in self.entries:
            for alias in (entry.canonical, *entry.aliases):
                normalized = self._normalize_phrase(alias)
                existing = self.alias_to_id.get(normalized)
                if existing and existing != entry.concept_id:
                    raise ValueError(f"Alias '{alias}' maps to both {existing} and {entry.concept_id}")
                self.alias_to_id[normalized] = entry.concept_id
                aliases.append(normalized)

        unique_aliases = sorted(set(aliases), key=lambda value: (-len(value), value))
        pattern = "|".join(re.escape(alias) for alias in unique_aliases)
        self.alias_pattern = re.compile(rf"(?<!\w)({pattern})(?!\w)", re.IGNORECASE)

    @staticmethod
    def _normalize_phrase(text: str) -> str:
        return " ".join(text.lower().split())

    @classmethod
    def load(cls, path: str | Path) -> "ConceptLexicon":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        concepts = payload.get("concepts", [])
        entries = [
            ConceptEntry(
                concept_id=item["id"],
                canonical=item["canonical"],
                aliases=tuple(item.get("aliases", [])),
            )
            for item in concepts
        ]
        return cls(entries)

    def encode_text(self, text: str) -> tuple[str, Counter[str]]:
        counts: Counter[str] = Counter()

        def replace(match: re.Match[str]) -> str:
            surface = self._normalize_phrase(match.group(0))
            concept_id = self.alias_to_id[surface]
            counts[concept_id] += 1
            return TOKEN_TEMPLATE.format(concept_id=concept_id)

        rewritten = self.alias_pattern.sub(replace, text)
        return rewritten, counts

    def decode_text(self, text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            concept_id = match.group(1)
            entry = self.id_to_entry.get(concept_id)
            if entry is None:
                return match.group(0)
            return entry.canonical

        return TOKEN_PATTERN.sub(replace, text)

    def summarize_counts(self, counts: Counter[str]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for concept_id, count in counts.most_common():
            entry = self.id_to_entry[concept_id]
            rows.append({
                "id": concept_id,
                "canonical": entry.canonical,
                "count": count,
            })
        return rows
