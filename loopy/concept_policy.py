from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from loopy.concept_middleware import ConceptLexicon, TOKEN_TEMPLATE

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+")


@dataclass(frozen=True)
class AliasPolicyRule:
    alias: str
    concept_id: str
    require_any_before: tuple[str, ...] = ()
    require_any_after: tuple[str, ...] = ()
    require_any_window: tuple[str, ...] = ()
    forbid_any_before: tuple[str, ...] = ()
    forbid_any_after: tuple[str, ...] = ()
    forbid_any_window: tuple[str, ...] = ()
    window: int = 4


@dataclass(frozen=True)
class RewriteDecision:
    allow: bool
    reason: str
    before: tuple[str, ...]
    after: tuple[str, ...]


class ContextualRewritePolicy:
    def __init__(self, rules: Iterable[AliasPolicyRule]) -> None:
        self.rules: dict[tuple[str, str], AliasPolicyRule] = {}
        for rule in rules:
            key = (self._normalize(rule.alias), rule.concept_id)
            self.rules[key] = rule

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().split())

    @classmethod
    def load(cls, path: str | Path) -> "ContextualRewritePolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rules = [
            AliasPolicyRule(
                alias=item["alias"],
                concept_id=item["concept_id"],
                require_any_before=tuple(item.get("require_any_before", [])),
                require_any_after=tuple(item.get("require_any_after", [])),
                require_any_window=tuple(item.get("require_any_window", [])),
                forbid_any_before=tuple(item.get("forbid_any_before", [])),
                forbid_any_after=tuple(item.get("forbid_any_after", [])),
                forbid_any_window=tuple(item.get("forbid_any_window", [])),
                window=int(item.get("window", 4)),
            )
            for item in payload.get("rules", [])
        ]
        return cls(rules)

    def decide(self, text: str, start: int, end: int, alias: str, concept_id: str) -> RewriteDecision:
        key = (self._normalize(alias), concept_id)
        rule = self.rules.get(key)
        if rule is None:
            return RewriteDecision(True, "no_rule", (), ())

        before = tuple(word.lower() for word in WORD_RE.findall(text[:start])[-rule.window:])
        after = tuple(word.lower() for word in WORD_RE.findall(text[end:])[:rule.window])
        around = before + after

        if rule.require_any_before and not any(word in before for word in rule.require_any_before):
            return RewriteDecision(False, "missing_before", before, after)
        if rule.require_any_after and not any(word in after for word in rule.require_any_after):
            return RewriteDecision(False, "missing_after", before, after)
        if rule.require_any_window and not any(word in around for word in rule.require_any_window):
            return RewriteDecision(False, "missing_window", before, after)
        if rule.forbid_any_before and any(word in before for word in rule.forbid_any_before):
            return RewriteDecision(False, "forbidden_before", before, after)
        if rule.forbid_any_after and any(word in after for word in rule.forbid_any_after):
            return RewriteDecision(False, "forbidden_after", before, after)
        if rule.forbid_any_window and any(word in around for word in rule.forbid_any_window):
            return RewriteDecision(False, "forbidden_window", before, after)
        return RewriteDecision(True, "rule_pass", before, after)


class ContextualConceptEncoder:
    def __init__(self, lexicon: ConceptLexicon, policy: ContextualRewritePolicy) -> None:
        self.lexicon = lexicon
        self.policy = policy

    def encode_text(self, text: str) -> tuple[str, Counter[str], Counter[str], list[dict[str, object]]]:
        replacements: Counter[str] = Counter()
        skips: Counter[str] = Counter()
        trace: list[dict[str, object]] = []
        parts: list[str] = []
        last = 0

        for match in self.lexicon.alias_pattern.finditer(text):
            parts.append(text[last:match.start()])
            surface = match.group(0)
            normalized = self.lexicon._normalize_phrase(surface)
            concept_id = self.lexicon.alias_to_id[normalized]
            decision = self.policy.decide(text, match.start(), match.end(), normalized, concept_id)

            if decision.allow:
                parts.append(TOKEN_TEMPLATE.format(concept_id=concept_id))
                replacements[concept_id] += 1
            else:
                parts.append(surface)
                skips[f"{normalized}:{decision.reason}"] += 1
                trace.append(
                    {
                        "alias": normalized,
                        "concept_id": concept_id,
                        "reason": decision.reason,
                        "before": list(decision.before),
                        "after": list(decision.after),
                        "surface": surface,
                    }
                )
            last = match.end()

        parts.append(text[last:])
        return "".join(parts), replacements, skips, trace
