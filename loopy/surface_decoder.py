from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from loopy.concept_middleware import ConceptLexicon
from loopy.train_token_lm import detokenize, tokenize

CONCEPT_TOKEN_RE = re.compile(r"<([A-Za-z0-9_:-]+)>")


@dataclass(frozen=True)
class AliasExample:
    concept_id: str
    true_alias: str
    context_tokens: tuple[str, ...]
    prefix_text: str


@dataclass(frozen=True)
class AliasPrediction:
    concept_id: str
    predicted_alias: str
    score: float
    canonical_alias: str
    canonical_score: float
    used_copy: bool
    margin_over_canonical: float


class ContextualAliasDecoder:
    def __init__(
        self,
        lexicon: ConceptLexicon,
        window: int = 4,
        min_alias_count: int = 2,
        base_switch_margin: float = 0.6,
        dominance_penalty: float = 0.8,
        copy_bonus: float = 3.0,
    ) -> None:
        self.lexicon = lexicon
        self.window = window
        self.min_alias_count = min_alias_count
        self.base_switch_margin = base_switch_margin
        self.dominance_penalty = dominance_penalty
        self.copy_bonus = copy_bonus
        self.alias_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.context_counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.lower().split())

    @classmethod
    def _normalize_token(cls, token: str) -> str:
        token = token.strip()
        if not token:
            return ""
        if CONCEPT_TOKEN_RE.fullmatch(token):
            return token.lower()
        if re.fullmatch(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", token):
            return token.lower()
        return ""

    @classmethod
    def _normalized_word_tokens(cls, text: str) -> list[str]:
        return [token for token in (cls._normalize_token(value) for value in tokenize(text)) if token]

    @classmethod
    def _contains_alias(cls, alias: str, text: str) -> bool:
        alias_tokens = [token for token in cls._normalize_text(alias).split() if token]
        if not alias_tokens:
            return False
        text_tokens = cls._normalized_word_tokens(text)
        width = len(alias_tokens)
        for index in range(0, max(0, len(text_tokens) - width + 1)):
            if text_tokens[index:index + width] == alias_tokens:
                return True
        return False

    def _candidate_aliases(self, concept_id: str) -> list[str]:
        entry = self.lexicon.id_to_entry[concept_id]
        candidates: list[str] = []
        for alias in (entry.canonical, *entry.aliases):
            normalized = self._normalize_text(alias)
            if normalized not in candidates:
                candidates.append(normalized)
        return candidates

    def fit(self, texts: Iterable[str]) -> None:
        for text in texts:
            for example in extract_alias_examples(text, self.lexicon, self.window):
                self.alias_counts[example.concept_id][example.true_alias] += 1
                for token in example.context_tokens:
                    self.context_counts[example.concept_id][example.true_alias][token] += 1

    def predict_alias(
        self,
        concept_id: str,
        context_tokens: Iterable[str],
        memory_text: str | None = None,
        recent_output_text: str | None = None,
    ) -> AliasPrediction:
        candidates = self._candidate_aliases(concept_id)
        alias_prior = self.alias_counts.get(concept_id, Counter())
        canonical_alias = self._normalize_text(self.lexicon.id_to_entry[concept_id].canonical)
        default_alias = alias_prior.most_common(1)[0][0] if alias_prior else canonical_alias
        memory_text = memory_text or ""
        recent_output_text = recent_output_text or ""
        context = [token for token in context_tokens if token]

        best_alias = canonical_alias
        best_score = float("-inf")
        candidate_scores: dict[str, float] = {}
        copy_hits: dict[str, bool] = {}
        for candidate in candidates:
            score = math.log(alias_prior[candidate] + 1.0)
            ctx_counts = self.context_counts.get(concept_id, {}).get(candidate, Counter())
            for token in context:
                score += 0.35 * math.log(ctx_counts[token] + 1.0)
            has_copy = self._contains_alias(candidate, memory_text) or self._contains_alias(candidate, recent_output_text)
            if has_copy:
                score += self.copy_bonus
            copy_hits[candidate] = has_copy
            if candidate == default_alias:
                score += 0.05
            candidate_scores[candidate] = score
            if score > best_score:
                best_score = score
                best_alias = candidate

        canonical_score = candidate_scores.get(canonical_alias, float("-inf"))
        canonical_count = alias_prior[canonical_alias]
        total_count = sum(alias_prior.values())
        canonical_share = canonical_count / total_count if total_count else 1.0
        required_margin = self.base_switch_margin + self.dominance_penalty * canonical_share
        margin = best_score - canonical_score

        chosen_alias = best_alias
        used_copy = copy_hits.get(best_alias, False)
        if chosen_alias != canonical_alias:
            if used_copy:
                required_margin *= 0.5
            if alias_prior[chosen_alias] < self.min_alias_count or margin < required_margin:
                chosen_alias = canonical_alias
                used_copy = False

        chosen_score = candidate_scores.get(chosen_alias, canonical_score)
        return AliasPrediction(
            concept_id=concept_id,
            predicted_alias=chosen_alias,
            score=chosen_score,
            canonical_alias=canonical_alias,
            canonical_score=canonical_score,
            used_copy=used_copy,
            margin_over_canonical=chosen_score - canonical_score,
        )

    def decode_text(self, text: str, memory_text: str | None = None) -> str:
        tokens = tokenize(text)
        output_tokens: list[str] = []

        for index, token in enumerate(tokens):
            match = CONCEPT_TOKEN_RE.fullmatch(token)
            if match is None:
                output_tokens.append(token)
                continue

            concept_id = match.group(1)
            before = [
                self._normalize_token(value)
                for value in tokens[max(0, index - self.window):index]
            ]
            after = [
                self._normalize_token(value)
                for value in tokens[index + 1:index + 1 + self.window]
            ]
            context_tokens = [value for value in before + after if value]
            recent_text = detokenize(output_tokens[max(0, len(output_tokens) - self.window * 2):])
            prediction = self.predict_alias(
                concept_id,
                context_tokens,
                memory_text=memory_text,
                recent_output_text=recent_text,
            )
            output_tokens.append(prediction.predicted_alias)

        return detokenize(output_tokens)

    def to_dict(self) -> dict[str, object]:
        return {
            "window": self.window,
            "min_alias_count": self.min_alias_count,
            "base_switch_margin": self.base_switch_margin,
            "dominance_penalty": self.dominance_penalty,
            "copy_bonus": self.copy_bonus,
            "alias_counts": {concept: dict(counter) for concept, counter in self.alias_counts.items()},
            "context_counts": {
                concept: {alias: dict(counter) for alias, counter in alias_map.items()}
                for concept, alias_map in self.context_counts.items()
            },
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def extract_alias_examples(text: str, lexicon: ConceptLexicon, window: int = 4) -> list[AliasExample]:
    rewritten, _ = lexicon.encode_text(text)
    rewritten_tokens = tokenize(rewritten)
    concept_positions: list[tuple[int, str]] = []
    for index, token in enumerate(rewritten_tokens):
        match = CONCEPT_TOKEN_RE.fullmatch(token)
        if match is not None:
            concept_positions.append((index, match.group(1)))

    original_matches = list(lexicon.alias_pattern.finditer(text))
    examples: list[AliasExample] = []
    for (index, concept_id), original_match in zip(concept_positions, original_matches):
        true_alias = ContextualAliasDecoder._normalize_text(original_match.group(0))
        before = [
            ContextualAliasDecoder._normalize_token(value)
            for value in rewritten_tokens[max(0, index - window):index]
        ]
        after = [
            ContextualAliasDecoder._normalize_token(value)
            for value in rewritten_tokens[index + 1:index + 1 + window]
        ]
        context_tokens = tuple(value for value in before + after if value)
        examples.append(
            AliasExample(
                concept_id=concept_id,
                true_alias=true_alias,
                context_tokens=context_tokens,
                prefix_text=text[:original_match.start()],
            )
        )
    return examples
