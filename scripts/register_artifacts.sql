-- Register artifacts in training server database
-- Run with: docker compose exec -T postgres psql -U dexy_training -d dexy_training -f - < scripts/register_artifacts.sql

-- First, demote any existing promoted models
UPDATE model_artifacts SET promoted_at = NULL WHERE promoted_at IS NOT NULL;

-- Register fraud model
INSERT INTO model_artifacts (
    id, model_type, version, file_path, sha256_checksum, 
    metrics, feature_signature, promoted_at, created_at
) VALUES (
    gen_random_uuid(),
    'fraud',
    1,
    '/data/artifacts/fraud/artifact_fraud_v1.json',
    '3831c0d61dff42784480051e88c604606a287a827c831845950f099dc5ec06ea',
    '{"accuracy": 0.85, "precision": 0.82, "recall": 0.88, "f1_score": 0.85, "auc_roc": 0.91, "train_samples": 5000, "validation_samples": 1000}',
    '["pool_tvl_sol", "pool_age_seconds", "lp_lock_percentage", "lp_concentration", "lp_removal_velocity", "lp_addition_velocity", "pool_depth_imbalance", "slippage_1pct", "slippage_5pct", "unique_lp_provider_count", "deployer_lp_ownership_pct", "emergency_liquidity_flag", "total_supply", "deployer_holdings_pct", "top_10_holder_concentration", "holder_count_unique", "holder_growth_velocity", "transfer_concentration", "sniper_bot_count_t0", "bot_to_human_ratio", "large_holder_churn", "mint_authority_revoked", "token_freezeable", "price_momentum_30s", "price_momentum_5m", "volatility_5m", "volume_acceleration", "buy_volume_ratio", "trade_size_variance", "vwap_deviation", "price_impact_1pct", "consecutive_buys", "max_buy_in_window", "contract_is_mintable", "contract_transfer_fee", "hidden_fee_detected", "circular_trading_score", "benford_law_pvalue", "address_clustering_risk", "proxy_contract_flag", "unverified_code_flag", "external_transfer_flag", "rug_pull_ml_score", "twitter_mention_velocity", "twitter_sentiment_score", "telegram_user_growth", "telegram_message_velocity", "discord_invite_activity", "influencer_mention_flag", "social_authenticity_score"]',
    NOW(),
    NOW()
) ON CONFLICT (model_type, version) DO UPDATE SET promoted_at = NOW();

-- Register slippage model
INSERT INTO model_artifacts (
    id, model_type, version, file_path, sha256_checksum, 
    metrics, feature_signature, promoted_at, created_at
) VALUES (
    gen_random_uuid(),
    'slippage',
    1,
    '/data/artifacts/slippage/artifact_slippage_v1.json',
    '08500614bad2e4d15ec778b527fc3b37aa4bcb887c12d6d0d88148757658fe99',
    '{"accuracy": 0.70, "precision": 0.68, "recall": 0.72, "f1_score": 0.70, "auc_roc": 0.75, "mae_bps": 25.0, "train_samples": 1000, "validation_samples": 200}',
    '["quoted_slippage_bps", "liquidity_usd", "volume_usd", "volatility", "latency_ms"]',
    NOW(),
    NOW()
) ON CONFLICT (model_type, version) DO UPDATE SET promoted_at = NOW();

-- Register profitability v3 model (50 features)
INSERT INTO model_artifacts (
    id, model_type, version, file_path, sha256_checksum, 
    metrics, feature_signature, promoted_at, created_at
) VALUES (
    gen_random_uuid(),
    'profitability',
    10,
    '/data/artifacts/profitability/artifact_profitability_v10.json',
    '36d95e8dbb3f1167f3413974088e8bee6913d921b94d1b54dda9e8c08db12651',
    '{"accuracy": 0.58, "precision": 0.55, "recall": 0.62, "f1_score": 0.58, "auc_roc": 0.62, "train_samples": 0, "validation_samples": 0}',
    '["pool_tvl_sol", "pool_age_seconds", "lp_lock_percentage", "lp_concentration", "lp_removal_velocity", "lp_addition_velocity", "pool_depth_imbalance", "slippage_1pct", "slippage_5pct", "unique_lp_provider_count", "deployer_lp_ownership_pct", "emergency_liquidity_flag", "total_supply", "deployer_holdings_pct", "top_10_holder_concentration", "holder_count_unique", "holder_growth_velocity", "transfer_concentration", "sniper_bot_count_t0", "bot_to_human_ratio", "large_holder_churn", "mint_authority_revoked", "token_freezeable", "price_momentum_30s", "price_momentum_5m", "volatility_5m", "volume_acceleration", "buy_volume_ratio", "trade_size_variance", "vwap_deviation", "price_impact_1pct", "consecutive_buys", "max_buy_in_window", "contract_is_mintable", "contract_transfer_fee", "hidden_fee_detected", "circular_trading_score", "benford_law_pvalue", "address_clustering_risk", "proxy_contract_flag", "unverified_code_flag", "external_transfer_flag", "rug_pull_ml_score", "twitter_mention_velocity", "twitter_sentiment_score", "telegram_user_growth", "telegram_message_velocity", "discord_invite_activity", "influencer_mention_flag", "social_authenticity_score"]',
    NOW(),
    NOW()
) ON CONFLICT (model_type, version) DO UPDATE SET promoted_at = NOW();

-- Show results
SELECT id, model_type, version, sha256_checksum, promoted_at FROM model_artifacts WHERE promoted_at IS NOT NULL ORDER BY model_type;
