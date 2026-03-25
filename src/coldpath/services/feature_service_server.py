"""
gRPC server implementation for FeatureService.

Provides real-time social and ML features to HotPath via gRPC.

Endpoints:
- GetSocialFeatures: Fetch social metrics for a token
- GetOnlineFeatures: Get features from the feature store
- StreamSocialSignals: Stream real-time social updates

Usage:
    from coldpath.services.feature_service_server import FeatureServiceServer

    server = FeatureServiceServer(feature_store, telegram_collector)
    await server.serve("0.0.0.0:50052")
"""

import asyncio
import logging
from typing import Optional

import grpc
from grpc import aio

logger = logging.getLogger(__name__)

try:
    from coldpath.services.feature_store import (
        FeatureStore,
        TokenFeatures,  # noqa: F401
    )
    from coldpath.services.telegram_signals import (
        TelegramMetrics,  # noqa: F401
        TelegramSignalCollector,
    )

    COLDPATH_SERVICES_AVAILABLE = True
except ImportError:
    COLDPATH_SERVICES_AVAILABLE = False
    logger.warning("ColdPath services not available, feature server will return stubs")


class FeatureServiceServer:
    """gRPC server for feature serving to HotPath."""

    def __init__(
        self,
        feature_store: Optional["FeatureStore"] = None,
        telegram_collector: Optional["TelegramSignalCollector"] = None,
        port: int = 50052,
    ):
        self.feature_store = feature_store
        self.telegram_collector = telegram_collector
        self.port = port
        self._server: aio.Server | None = None
        self._subscriptions: dict[str, asyncio.Task] = {}

    async def serve(self, bind_address: str = "0.0.0.0:50052"):
        """Start the gRPC server."""
        self._server = aio.server()

        try:
            from engine_hotpath.proto import (
                hotpath_pb2,  # noqa: F401
                hotpath_pb2_grpc,
            )

            hotpath_pb2_grpc.add_FeatureServiceServicer_to_server(
                _FeatureServiceServicer(self), self._server
            )
        except ImportError:
            logger.warning("Proto files not generated, using stub implementation")
            return

        self._server.add_insecure_port(bind_address)
        await self._server.start()
        logger.info(f"FeatureService server started on {bind_address}")

    async def stop(self, grace: float = 5.0):
        """Stop the gRPC server."""
        if self._server:
            await self._server.stop(grace)
            logger.info("FeatureService server stopped")

    async def get_social_features(
        self,
        token_mint: str,
        telegram_url: str | None = None,
        twitter_handle: str | None = None,
        include_sentiment: bool = True,
    ) -> dict[str, float]:
        """Get social features for a token.

        Returns dict with keys matching hotpath.proto SocialFeaturesResponse.
        """
        features = {
            "twitter_mention_velocity": 0.0,
            "twitter_sentiment_score": 0.0,
            "telegram_user_growth": 0.0,
            "telegram_message_velocity": 0.0,
            "discord_invite_activity": 0.0,
            "influencer_mention_flag": 0.0,
            "social_authenticity_score": 0.5,
            "found": False,
        }

        if not COLDPATH_SERVICES_AVAILABLE:
            return features

        try:
            if self.telegram_collector and telegram_url:
                metrics = await self.telegram_collector.get_group_metrics(
                    telegram_url,
                    include_messages=include_sentiment,
                )

                feature_dict = metrics.to_feature_dict()

                features["twitter_mention_velocity"] = feature_dict.get(
                    "twitter_mention_velocity", 0.0
                )
                features["twitter_sentiment_score"] = feature_dict.get(
                    "twitter_sentiment_score", 0.0
                )
                features["telegram_user_growth"] = feature_dict.get("telegram_user_growth", 0.0)
                features["telegram_message_velocity"] = feature_dict.get(
                    "telegram_message_velocity", 0.0
                )
                features["discord_invite_activity"] = feature_dict.get(
                    "discord_invite_activity", 0.0
                )
                features["influencer_mention_flag"] = feature_dict.get(
                    "influencer_mention_flag", -1.0
                )
                features["social_authenticity_score"] = feature_dict.get(
                    "social_authenticity_score", 0.5
                )
                features["found"] = True

                logger.debug(
                    f"Retrieved social features for {token_mint}: "
                    f"tg_growth={features['telegram_user_growth']:.3f}"
                )
            elif self.feature_store:
                token_features = await self.feature_store.get_token_features(token_mint)
                if token_features:
                    features["twitter_mention_velocity"] = (
                        token_features.twitter_mentions_24h / 24.0
                        if token_features.twitter_mentions_24h > 0
                        else 0.0
                    )
                    features["twitter_sentiment_score"] = (
                        token_features.sentiment_score - 0.5
                    ) * 2.0
                    features["telegram_user_growth"] = 0.0  # Not tracked in feature store yet
                    features["telegram_message_velocity"] = 0.0
                    features["discord_invite_activity"] = 0.0
                    features["influencer_mention_flag"] = -1.0
                    features["social_authenticity_score"] = 1.0 - (token_features.fraud_score * 0.5)
                    features["found"] = True

        except Exception as e:
            logger.debug(f"Failed to get social features for {token_mint}: {e}")

        return features

    async def get_online_features(
        self,
        entity_ids: list[str],
        feature_names: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Get online features from the feature store."""
        if not self.feature_store:
            return {eid: {} for eid in entity_ids}

        return await self.feature_store.get_online_features(entity_ids, feature_names)


class _FeatureServiceServicer:
    """Internal gRPC servicer implementation."""

    def __init__(self, server: FeatureServiceServer):
        self.server = server

    async def GetSocialFeatures(self, request, context):
        """Handle GetSocialFeatures RPC."""
        try:
            from engine_hotpath.proto import hotpath_pb2

            features = await self.server.get_social_features(
                token_mint=request.token_mint,
                telegram_url=request.telegram_url if request.HasField("telegram_url") else None,
                twitter_handle=request.twitter_handle
                if request.HasField("twitter_handle")
                else None,
                include_sentiment=request.include_sentiment,
            )

            return hotpath_pb2.SocialFeaturesResponse(
                token_mint=request.token_mint,
                found=features["found"],
                twitter_mention_velocity=features["twitter_mention_velocity"],
                twitter_sentiment_score=features["twitter_sentiment_score"],
                telegram_user_growth=features["telegram_user_growth"],
                telegram_message_velocity=features["telegram_message_velocity"],
                discord_invite_activity=features["discord_invite_activity"],
                influencer_mention_flag=features["influencer_mention_flag"],
                social_authenticity_score=features["social_authenticity_score"],
                timestamp_ms=int(asyncio.get_event_loop().time() * 1000),
            )
        except Exception as e:
            logger.error(f"GetSocialFeatures error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return hotpath_pb2.SocialFeaturesResponse(
                token_mint=request.token_mint,
                found=False,
                error=str(e),
            )

    async def GetOnlineFeatures(self, request, context):
        """Handle GetOnlineFeatures RPC."""
        try:
            from engine_hotpath.proto import hotpath_pb2

            features = await self.server.get_online_features(
                entity_ids=list(request.entity_ids),
                feature_names=list(request.feature_names) if request.feature_names else None,
            )

            entities = [
                hotpath_pb2.EntityFeatures(
                    entity_id=eid,
                    features=fmap,
                    found=bool(fmap),
                )
                for eid, fmap in features.items()
            ]

            return hotpath_pb2.OnlineFeaturesResponse(entities=entities)
        except Exception as e:
            logger.error(f"GetOnlineFeatures error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return hotpath_pb2.OnlineFeaturesResponse(entities=[])

    async def StreamSocialSignals(self, request_iterator, context):
        """Handle StreamSocialSignals bidirectional streaming RPC."""
        try:
            from engine_hotpath.proto import hotpath_pb2

            async for request in request_iterator:
                if request.subscribe:
                    features = await self.server.get_social_features(
                        token_mint=request.token_mint,
                    )

                    response = hotpath_pb2.SocialFeaturesResponse(
                        token_mint=request.token_mint,
                        found=features["found"],
                        twitter_mention_velocity=features["twitter_mention_velocity"],
                        twitter_sentiment_score=features["twitter_sentiment_score"],
                        telegram_user_growth=features["telegram_user_growth"],
                        telegram_message_velocity=features["telegram_message_velocity"],
                        discord_invite_activity=features["discord_invite_activity"],
                        influencer_mention_flag=features["influencer_mention_flag"],
                        social_authenticity_score=features["social_authenticity_score"],
                    )

                    yield hotpath_pb2.SocialSignalUpdate(
                        token_mint=request.token_mint,
                        features=response,
                    )

                    if request.update_interval_ms > 0:
                        await asyncio.sleep(request.update_interval_ms / 1000.0)

        except Exception as e:
            logger.error(f"StreamSocialSignals error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))


async def main():
    """Run the feature service server standalone."""
    logging.basicConfig(level=logging.INFO)

    feature_store = None
    telegram_collector = None

    if COLDPATH_SERVICES_AVAILABLE:
        feature_store = FeatureStore()
        telegram_collector = TelegramSignalCollector()

    server = FeatureServiceServer(
        feature_store=feature_store,
        telegram_collector=telegram_collector,
    )

    await server.serve("0.0.0.0:50052")

    try:
        await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
