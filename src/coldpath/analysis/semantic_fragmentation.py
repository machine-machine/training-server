"""Semantic contract alignment and liquidity fragmentation analysis.

Aligns economically equivalent contracts across venues and detects
friction-aware parity violations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class FragmentationConfig:
    """Configuration for semantic alignment and parity checks."""

    min_similarity: float = 0.55
    min_executable_edge: float = 0.01
    parity_tolerance: float = 0.03


@dataclass
class MarketContract:
    """Venue-specific contract metadata."""

    venue: str
    symbol: str
    description: str
    outcome: str = "yes"
    price: float = 0.5
    metadata: dict[str, Any] | None = None
    fee_bps: float = 10.0
    settlement_risk_bps: float = 5.0
    capital_lockup_bps: float = 5.0


@dataclass
class ContractAlignment:
    """A semantically aligned contract with canonical identity."""

    canonical_id: str
    contract: MarketContract
    similarity: float


@dataclass
class ParityOpportunity:
    """Detected parity violation after fragmentation frictions."""

    canonical_id: str
    opportunity_type: str
    side: str
    gross_edge: float
    net_edge: float
    legs: list[MarketContract]

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_id": self.canonical_id,
            "opportunity_type": self.opportunity_type,
            "side": self.side,
            "gross_edge": self.gross_edge,
            "net_edge": self.net_edge,
            "legs": [
                {
                    "venue": leg.venue,
                    "symbol": leg.symbol,
                    "outcome": leg.outcome,
                    "price": leg.price,
                }
                for leg in self.legs
            ],
        }


@dataclass
class AlignmentResult:
    """Output of semantic alignment."""

    alignments: list[ContractAlignment]

    def groups(self) -> dict[str, list[ContractAlignment]]:
        grouped: dict[str, list[ContractAlignment]] = {}
        for item in self.alignments:
            grouped.setdefault(item.canonical_id, []).append(item)
        return grouped


class SemanticFragmentationAnalyzer:
    """Resolve semantic identity and detect executable parity gaps."""

    STOPWORDS = {
        "the",
        "a",
        "an",
        "will",
        "be",
        "is",
        "to",
        "of",
        "on",
        "at",
        "by",
        "for",
        "in",
        "and",
        "or",
        "does",
        "do",
        "with",
        "market",
    }

    def __init__(self, config: FragmentationConfig | None = None):
        self.config = config or FragmentationConfig()

    def align_contracts(self, contracts: list[MarketContract]) -> AlignmentResult:
        """Map venue contracts to canonical semantic identities."""
        alignments: list[ContractAlignment] = []
        prototypes: dict[str, set[str]] = {}

        for contract in contracts:
            tokens = self._tokenize(contract.description, contract.metadata)
            canonical_id, similarity = self._match_or_create(tokens, prototypes)
            alignments.append(
                ContractAlignment(
                    canonical_id=canonical_id,
                    contract=contract,
                    similarity=similarity,
                )
            )

        return AlignmentResult(alignments=alignments)

    def detect_parity_opportunities(
        self,
        alignment: AlignmentResult,
    ) -> list[ParityOpportunity]:
        """Detect friction-aware parity violations within and across venues."""
        opportunities: list[ParityOpportunity] = []

        for canonical_id, grouped in alignment.groups().items():
            contracts = [item.contract for item in grouped]
            opportunities.extend(self._cross_venue_opportunities(canonical_id, contracts))
            opportunities.extend(self._within_market_parity(canonical_id, contracts))

        return sorted(opportunities, key=lambda x: x.net_edge, reverse=True)

    def _cross_venue_opportunities(
        self,
        canonical_id: str,
        contracts: list[MarketContract],
    ) -> list[ParityOpportunity]:
        opportunities: list[ParityOpportunity] = []

        # Same-outcome spread opportunities across venues.
        for i, left in enumerate(contracts):
            for right in contracts[i + 1 :]:
                if left.outcome.lower() != right.outcome.lower():
                    continue
                if left.venue == right.venue:
                    continue

                buy_leg, sell_leg = (left, right) if left.price <= right.price else (right, left)
                gross_edge = float(sell_leg.price - buy_leg.price)
                friction = self._combined_friction([buy_leg, sell_leg])
                net_edge = gross_edge - friction
                if net_edge >= self.config.min_executable_edge:
                    opportunities.append(
                        ParityOpportunity(
                            canonical_id=canonical_id,
                            opportunity_type="cross_venue_same_outcome",
                            side=f"buy:{buy_leg.venue} sell:{sell_leg.venue}",
                            gross_edge=gross_edge,
                            net_edge=net_edge,
                            legs=[buy_leg, sell_leg],
                        )
                    )

        return opportunities

    def _within_market_parity(
        self,
        canonical_id: str,
        contracts: list[MarketContract],
    ) -> list[ParityOpportunity]:
        opportunities: list[ParityOpportunity] = []

        by_venue: dict[str, dict[str, MarketContract]] = {}
        for contract in contracts:
            venue_book = by_venue.setdefault(contract.venue, {})
            venue_book[contract.outcome.lower()] = contract

        for venue, market in by_venue.items():
            yes = market.get("yes")
            no = market.get("no")
            if yes is None or no is None:
                continue

            parity = yes.price + no.price
            deviation = abs(parity - 1.0)
            if deviation <= self.config.parity_tolerance:
                continue

            friction = self._combined_friction([yes, no])
            net_edge = deviation - friction
            if net_edge >= self.config.min_executable_edge:
                opportunities.append(
                    ParityOpportunity(
                        canonical_id=canonical_id,
                        opportunity_type="within_venue_yes_no_parity",
                        side=f"{venue}",
                        gross_edge=deviation,
                        net_edge=net_edge,
                        legs=[yes, no],
                    )
                )

        return opportunities

    def _match_or_create(
        self,
        tokens: set[str],
        prototypes: dict[str, set[str]],
    ) -> tuple[str, float]:
        best_id: str | None = None
        best_similarity = 0.0

        for canonical_id, prototype in prototypes.items():
            similarity = self._jaccard(tokens, prototype)
            if similarity > best_similarity:
                best_similarity = similarity
                best_id = canonical_id

        if best_id is not None and best_similarity >= self.config.min_similarity:
            prototypes[best_id] = prototypes[best_id].union(tokens)
            return best_id, best_similarity

        new_id = f"event_{len(prototypes) + 1}"
        prototypes[new_id] = set(tokens)
        return new_id, 1.0

    def _tokenize(self, description: str, metadata: dict[str, Any] | None) -> set[str]:
        text = description.lower()
        if metadata:
            values = " ".join(str(v).lower() for v in metadata.values())
            text = f"{text} {values}"

        cleaned = re.sub(r"[^a-z0-9\s-]", " ", text)
        tokens = {token for token in cleaned.split() if token and token not in self.STOPWORDS}

        date_tokens = re.findall(r"\d{4}-\d{2}-\d{2}", text)
        tokens.update(date_tokens)
        return tokens

    def _jaccard(self, left: set[str], right: set[str]) -> float:
        if not left and not right:
            return 1.0
        union = left.union(right)
        if not union:
            return 0.0
        return len(left.intersection(right)) / len(union)

    def _combined_friction(self, legs: Iterable[MarketContract]) -> float:
        friction_bps = 0.0
        for leg in legs:
            friction_bps += leg.fee_bps + leg.settlement_risk_bps + leg.capital_lockup_bps
        return friction_bps / 10000.0
