"""
Graph-Based Features for network analysis and MEV detection.

Captures wallet network centrality, holder clustering, wash trading patterns,
correlated token signals, sector performance, cohort survival, and MEV activity.

Implements 8 features across 4 sub-categories:
- Wallet Network (2 features): deployer centrality, holder clustering coefficient
- Wash Trading (1 feature): circular transaction pattern detection
- Token Correlations (3 features): correlated momentum, sector strength, launch cohort survival
- MEV Activity (2 features): searcher attention, sandwich probability

These features complement the core 50-feature unified vector from
``coldpath.learning.feature_engineering`` by adding graph-theoretic signals
derived from wallet networks, token correlations, and MEV behavior.
"""

import logging
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GraphFeatures:
    """8 graph-based features capturing network and MEV dynamics."""

    # Wallet Network
    deployer_centrality: float = 0.0  # PageRank of deployer wallet in wallet graph
    holder_cluster_coefficient: float = 0.0  # Clustering coefficient in holder graph

    # Wash Trading Detection
    wash_trade_detection: float = 0.0  # Circular transaction pattern score (0-1)

    # Token Correlations
    correlated_tokens_momentum: float = 0.0  # Average momentum of correlated tokens
    sector_relative_strength: float = 0.0  # Performance vs similar tokens
    launch_cohort_survival: float = 0.0  # Survival rate of tokens launched same day

    # MEV Activity
    searcher_attention: float = 0.0  # MEV searcher activity level (0-1)
    sandwich_probability: float = 0.0  # Historical sandwich rate for similar tokens


class GraphFeatureExtractor:
    """Extract graph-based features from network and MEV data.

    Processes wallet graphs, holder networks, token correlation data, and MEV
    activity to compute 8 graph-theoretic features grouped into wallet network
    analysis, wash trading detection, token correlations, and MEV metrics.

    Args:
        min_centrality_nodes: Minimum nodes required for centrality computation.
        min_holders_for_clustering: Minimum holders required for clustering coefficient.
        correlation_threshold: Minimum correlation coefficient for token grouping.
    """

    def __init__(
        self,
        min_centrality_nodes: int = 10,
        min_holders_for_clustering: int = 5,
        correlation_threshold: float = 0.7,
    ):
        self.min_centrality_nodes = min_centrality_nodes
        self.min_holders_for_clustering = min_holders_for_clustering
        self.correlation_threshold = correlation_threshold

    def extract(
        self,
        wallet_graph: nx.DiGraph | None = None,
        holder_data: dict[str, list[str]] | None = None,
        token_data: pd.DataFrame | None = None,
        mev_data: pd.DataFrame | None = None,
        current_mint: str | None = None,
        deployer_address: str | None = None,
    ) -> GraphFeatures:
        """Extract all graph-based features from network and MEV data.

        Args:
            wallet_graph: NetworkX DiGraph where nodes are wallet addresses and
                edges represent transactions (with optional 'amount' weights).
            holder_data: Dict mapping mint_address to list of holder addresses.
            token_data: DataFrame with columns: mint_address, price_change_pct,
                launched_at, is_alive (bool).
            mev_data: DataFrame with columns: tx_hash, searcher_address,
                sandwich_count, token_mint.
            current_mint: The mint address of the token being analyzed.
            deployer_address: The deployer wallet address for centrality analysis.

        Returns:
            Populated ``GraphFeatures`` dataclass.
        """
        features = GraphFeatures()

        # --- Wallet Network ---
        if wallet_graph is not None and deployer_address is not None:
            features.deployer_centrality = self._compute_deployer_centrality(
                wallet_graph, deployer_address
            )

        if holder_data is not None and current_mint is not None:
            features.holder_cluster_coefficient = self._compute_holder_clustering(
                holder_data, current_mint
            )

        # --- Wash Trading Detection ---
        if wallet_graph is not None:
            features.wash_trade_detection = self._detect_wash_trading(wallet_graph)

        # --- Token Correlations ---
        if token_data is not None and current_mint is not None:
            features.correlated_tokens_momentum = self._compute_correlated_momentum(
                token_data, current_mint
            )
            features.sector_relative_strength = self._compute_sector_strength(
                token_data, current_mint
            )
            features.launch_cohort_survival = self._compute_cohort_survival(
                token_data, current_mint
            )

        # --- MEV Activity ---
        if mev_data is not None and current_mint is not None:
            features.searcher_attention = self._compute_searcher_attention(mev_data, current_mint)
            features.sandwich_probability = self._compute_sandwich_probability(
                mev_data, current_mint
            )

        return features

    # ------------------------------------------------------------------
    # Wallet Network
    # ------------------------------------------------------------------

    def _compute_deployer_centrality(
        self, wallet_graph: nx.DiGraph, deployer_address: str
    ) -> float:
        """Compute PageRank centrality of deployer wallet in the wallet graph.

        PageRank measures the importance of a node based on the structure of
        incoming links. A high score indicates the deployer is well-connected
        or receives transactions from influential wallets.

        Args:
            wallet_graph: NetworkX DiGraph of wallet transactions.
            deployer_address: The deployer wallet address.

        Returns:
            PageRank score (0-1), or 0.0 if graph too small or deployer not found.
        """
        if wallet_graph.number_of_nodes() < self.min_centrality_nodes:
            return 0.0

        if deployer_address not in wallet_graph.nodes:
            logger.debug(f"Deployer {deployer_address} not found in wallet graph")
            return 0.0

        try:
            pagerank = nx.pagerank(wallet_graph, max_iter=100)
            score = pagerank.get(deployer_address, 0.0)
            return float(min(max(score, 0.0), 1.0))
        except Exception:
            logger.debug("PageRank computation failed", exc_info=True)
            return 0.0

    def _compute_holder_clustering(
        self, holder_data: dict[str, list[str]], current_mint: str
    ) -> float:
        """Compute clustering coefficient in the holder graph.

        Builds an undirected graph where holders are nodes and edges exist
        if two holders share holdings in multiple tokens. The clustering
        coefficient measures the degree to which holders form tight groups.

        High clustering may indicate coordinated behavior or wash trading rings.

        Args:
            holder_data: Dict mapping mint_address -> list of holder addresses.
            current_mint: The token mint being analyzed.

        Returns:
            Average clustering coefficient (0-1), or 0.0 if insufficient data.
        """
        if current_mint not in holder_data:
            return 0.0

        current_holders = set(holder_data[current_mint])
        if len(current_holders) < self.min_holders_for_clustering:
            return 0.0

        # Build holder graph: edge if two holders share 2+ tokens
        holder_tokens: dict[str, list[str]] = {}
        for mint, holders in holder_data.items():
            for holder in holders:
                if holder not in holder_tokens:
                    holder_tokens[holder] = []
                holder_tokens[holder].append(mint)

        # Only consider holders of the current token
        G = nx.Graph()
        G.add_nodes_from(current_holders)

        for holder in current_holders:
            tokens_held = set(holder_tokens.get(holder, []))
            for other_holder in current_holders:
                if holder < other_holder:  # Avoid duplicate edges
                    other_tokens = set(holder_tokens.get(other_holder, []))
                    shared = tokens_held & other_tokens
                    if len(shared) >= 2:
                        G.add_edge(holder, other_holder)

        if G.number_of_nodes() < 3:
            return 0.0

        try:
            clustering = nx.average_clustering(G)
            return float(min(max(clustering, 0.0), 1.0))
        except Exception:
            logger.debug("Clustering coefficient computation failed", exc_info=True)
            return 0.0

    # ------------------------------------------------------------------
    # Wash Trading Detection
    # ------------------------------------------------------------------

    def _detect_wash_trading(self, wallet_graph: nx.DiGraph) -> float:
        """Detect circular transaction patterns (wash trading).

        Searches for simple cycles in the wallet graph where wallets send
        tokens in a loop (A -> B -> C -> A). The score is the fraction of
        nodes involved in at least one cycle.

        Args:
            wallet_graph: NetworkX DiGraph of wallet transactions.

        Returns:
            Wash trading score (0-1): fraction of nodes in cycles.
        """
        if wallet_graph.number_of_nodes() < 3:
            return 0.0

        try:
            # Find all simple cycles up to length 4 (performance limit)
            cycles = list(nx.simple_cycles(wallet_graph, length_bound=4))
            if not cycles:
                return 0.0

            # Count unique nodes involved in any cycle
            nodes_in_cycles = set()
            for cycle in cycles:
                nodes_in_cycles.update(cycle)

            score = len(nodes_in_cycles) / wallet_graph.number_of_nodes()
            return float(min(max(score, 0.0), 1.0))
        except Exception:
            logger.debug("Wash trading detection failed", exc_info=True)
            return 0.0

    # ------------------------------------------------------------------
    # Token Correlations
    # ------------------------------------------------------------------

    def _compute_correlated_momentum(self, token_data: pd.DataFrame, current_mint: str) -> float:
        """Compute average momentum of tokens correlated with current token.

        Identifies tokens with similar price movements (high correlation) and
        computes their average recent momentum. If correlated tokens are rising,
        the current token may follow.

        Args:
            token_data: DataFrame with mint_address and price_change_pct columns.
            current_mint: The token mint being analyzed.

        Returns:
            Average momentum of correlated tokens (centered around 0), or 0.0 if
            insufficient data.
        """
        if token_data.empty or "price_change_pct" not in token_data.columns:
            return 0.0

        current_row = token_data[token_data["mint_address"] == current_mint]
        if current_row.empty:
            return 0.0

        current_momentum = current_row["price_change_pct"].iloc[0]

        # Simple correlation: find tokens with similar price changes
        # In a real implementation, you'd compute rolling correlation
        # Here we use absolute difference as a proxy
        token_data = token_data.copy()
        token_data["momentum_diff"] = abs(token_data["price_change_pct"] - current_momentum)
        correlated = token_data[
            (token_data["momentum_diff"] < 5.0)  # Within 5% movement
            & (token_data["mint_address"] != current_mint)
        ]

        if len(correlated) < 3:
            return 0.0

        avg_momentum = correlated["price_change_pct"].mean()
        return float(np.clip(avg_momentum / 100.0, -1.0, 1.0))  # Normalize to [-1, 1]

    def _compute_sector_strength(self, token_data: pd.DataFrame, current_mint: str) -> float:
        """Compute performance relative to similar tokens (sector).

        Compares the current token's performance to the median of its sector.
        Positive values indicate outperformance; negative indicates underperformance.

        Args:
            token_data: DataFrame with mint_address and price_change_pct columns.
            current_mint: The token mint being analyzed.

        Returns:
            Relative strength (-1 to 1): current_pct - sector_median_pct, normalized.
        """
        if token_data.empty or "price_change_pct" not in token_data.columns:
            return 0.0

        current_row = token_data[token_data["mint_address"] == current_mint]
        if current_row.empty:
            return 0.0

        current_pct = current_row["price_change_pct"].iloc[0]
        sector_median = token_data["price_change_pct"].median()

        relative_strength = (current_pct - sector_median) / 100.0
        return float(np.clip(relative_strength, -1.0, 1.0))

    def _compute_cohort_survival(self, token_data: pd.DataFrame, current_mint: str) -> float:
        """Compute survival rate of tokens launched on the same day.

        Finds tokens launched on the same day as the current token and computes
        the fraction that are still alive (trading volume > 0, not rugged).

        Low survival rates may indicate a risky launch environment.

        Args:
            token_data: DataFrame with mint_address, launched_at, and is_alive columns.
            current_mint: The token mint being analyzed.

        Returns:
            Survival rate (0-1): fraction of cohort tokens still alive.
        """
        required_cols = {"mint_address", "launched_at", "is_alive"}
        if not required_cols.issubset(token_data.columns):
            return 0.0

        current_row = token_data[token_data["mint_address"] == current_mint]
        if current_row.empty or pd.isna(current_row["launched_at"].iloc[0]):
            return 0.0

        launch_date = pd.to_datetime(current_row["launched_at"].iloc[0]).date()

        # Find cohort: tokens launched on same day
        token_data = token_data.copy()
        token_data["launch_date"] = pd.to_datetime(token_data["launched_at"]).dt.date
        cohort = token_data[
            (token_data["launch_date"] == launch_date)
            & (token_data["mint_address"] != current_mint)
        ]

        if len(cohort) < 5:
            return 0.0

        survival_rate = cohort["is_alive"].mean()
        return float(min(max(survival_rate, 0.0), 1.0))

    # ------------------------------------------------------------------
    # MEV Activity
    # ------------------------------------------------------------------

    def _compute_searcher_attention(self, mev_data: pd.DataFrame, current_mint: str) -> float:
        """Compute MEV searcher activity level for the current token.

        Measures how many unique MEV searchers have interacted with this token
        relative to a baseline. High attention indicates the token is being
        actively arbitraged or sandwiched.

        Args:
            mev_data: DataFrame with searcher_address and token_mint columns.
            current_mint: The token mint being analyzed.

        Returns:
            Searcher attention score (0-1), or 0.0 if no MEV data.
        """
        required_cols = {"searcher_address", "token_mint"}
        if not required_cols.issubset(mev_data.columns) or mev_data.empty:
            return 0.0

        token_mev = mev_data[mev_data["token_mint"] == current_mint]
        if token_mev.empty:
            return 0.0

        unique_searchers = token_mev["searcher_address"].nunique()

        # Normalize: assume 10+ searchers is very high attention
        normalized = min(unique_searchers / 10.0, 1.0)
        return float(normalized)

    def _compute_sandwich_probability(self, mev_data: pd.DataFrame, current_mint: str) -> float:
        """Compute historical sandwich attack rate for the current token.

        Calculates the fraction of transactions that were sandwiched by MEV bots.
        High sandwich probability indicates the token has poor liquidity or
        is a frequent MEV target.

        Args:
            mev_data: DataFrame with sandwich_count, tx_hash, and token_mint columns.
            current_mint: The token mint being analyzed.

        Returns:
            Sandwich probability (0-1): fraction of txs that were sandwiched.
        """
        required_cols = {"tx_hash", "token_mint", "sandwich_count"}
        if not required_cols.issubset(mev_data.columns) or mev_data.empty:
            return 0.0

        token_mev = mev_data[mev_data["token_mint"] == current_mint]
        if token_mev.empty or len(token_mev) < 5:
            return 0.0

        total_txs = len(token_mev)
        sandwiched_txs = (token_mev["sandwich_count"] > 0).sum()

        probability = sandwiched_txs / total_txs
        return float(min(max(probability, 0.0), 1.0))

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_feature_dict(self, features: GraphFeatures) -> dict[str, float]:
        """Convert ``GraphFeatures`` to a flat dictionary for model input."""
        return {
            "deployer_centrality": features.deployer_centrality,
            "holder_cluster_coefficient": features.holder_cluster_coefficient,
            "wash_trade_detection": features.wash_trade_detection,
            "correlated_tokens_momentum": features.correlated_tokens_momentum,
            "sector_relative_strength": features.sector_relative_strength,
            "launch_cohort_survival": features.launch_cohort_survival,
            "searcher_attention": features.searcher_attention,
            "sandwich_probability": features.sandwich_probability,
        }

    @staticmethod
    def feature_names() -> list[str]:
        """Return the ordered list of 8 graph-based feature names."""
        return [
            "deployer_centrality",
            "holder_cluster_coefficient",
            "wash_trade_detection",
            "correlated_tokens_momentum",
            "sector_relative_strength",
            "launch_cohort_survival",
            "searcher_attention",
            "sandwich_probability",
        ]
