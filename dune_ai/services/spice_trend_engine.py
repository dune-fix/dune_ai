import asyncio
import logging
from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
from textblob import TextBlob

from config.settings import (
    SCAN_INTERVAL_SECONDS,
    SENTIMENT_ANALYSIS_BATCH_SIZE,
    SENTIMENT_THRESHOLD_POSITIVE,
    SENTIMENT_THRESHOLD_NEGATIVE
)
from config.logging_config import get_logger
from dune_ai.blockchain.solana_client import SolanaClient
from dune_ai.analytics.sentiment_analyzer import SentimentAnalyzer
from dune_ai.analytics.pattern_recognition import PatternRecognition
from dune_ai.models.token import Token, TokenSentiment


class SpiceTrendEngine:
    """
    Spice Trend Engine - Analyzes sentiment and trends in cryptocurrency communities
    to predict future market movements, like the spice melange that grants future vision
    in the Dune universe.

    This component analyzes various data sources to identify sentiment patterns,
    correlations between tokens, and emerging market trends.
    """

    def __init__(self, solana_client: SolanaClient):
        self.logger = get_logger("spice_trend_engine")
        self.solana_client = solana_client
        self.sentiment_analyzer = SentimentAnalyzer()
        self.pattern_recognition = PatternRecognition()

        # Data storage
        self.token_registry: Dict[str, Token] = {}  # address -> Token
        self.trend_data: Dict[str, List[float]] = {}  # token address -> list of trend scores over time
        self.token_correlations: Dict[str, Dict[str, float]] = {}  # token address -> {other token -> correlation}
        self.global_sentiment_history: List[float] = []  # Overall market sentiment over time

        # Source trackers
        self.data_sources: Dict[str, datetime] = {
            "twitter": datetime.now() - timedelta(days=1),
            "telegram": datetime.now() - timedelta(days=1),
            "discord": datetime.now() - timedelta(days=1),
            "reddit": datetime.now() - timedelta(days=1),
        }

        self.sentiment_keywords: Set[str] = {
            "bullish", "bearish", "moon", "dump", "pump", "scam", "gem", "legit",
            "rugpull", "hodl", "profit", "loss", "buy", "sell", "launch", "presale",
            "airdrop", "stake", "mint", "solana", "sol", "memecoin", "meme", "hype"
        }

        self.is_running = False
        self.logger.info("Spice Trend Engine initialized")

    async def start_analysis(self):
        """Start the trend analysis process"""
        self.logger.info("Starting trend analysis")
        self.is_running = True

        while self.is_running:
            try:
                # Step 1: Gather data from multiple sources
                await self._gather_data()

                # Step 2: Process and analyze sentiment
                await self._analyze_sentiment()

                # Step 3: Detect patterns and correlations
                await self._detect_patterns()

                # Step 4: Update token trend scores
                await self._update_trend_scores()

                # Step 5: Generate predictions
                await self._generate_predictions()

                # Log results
                self._log_analysis_results()

                # Sleep until next analysis cycle
                await asyncio.sleep(SCAN_INTERVAL_SECONDS * 2)  # Longer interval for analysis

            except Exception as e:
                self.logger.error(f"Error in trend analysis: {e}", exc_info=True)
                await asyncio.sleep(SCAN_INTERVAL_SECONDS * 2)  # Sleep on error

    async def _gather_data(self):
        """Gather data from multiple sources for analysis"""
        self.logger.info("Gathering data for trend analysis")

        # Get token list to analyze
        tokens = await self._get_tokens_to_analyze()

        # Update token registry
        for token in tokens:
            self.token_registry[token.address] = token

        # Initialize trend data for new tokens
        for address in self.token_registry:
            if address not in self.trend_data:
                self.trend_data[address] = []

        # In a real implementation, would gather data from multiple sources:
        # - Twitter (via TwitterSentinel)
        # - Telegram groups
        # - Discord servers
        # - Reddit
        # - Price data from DEXes

        self.logger.info(f"Data gathered for {len(self.token_registry)} tokens")

    async def _get_tokens_to_analyze(self) -> List[Token]:
        """Get list of tokens to analyze from the Solana client"""
        # In a real implementation, would get top tokens by market cap, volume, etc.
        # For demo purposes, return some sample tokens
        tokens = await self.solana_client.get_top_tokens(limit=100)

        # Filter for tokens that are likely meme coins
        meme_tokens = []
        for token in tokens:
            # Apply meme token heuristics
            token.is_meme_coin = self._is_likely_meme_coin(token)
            if token.is_meme_coin:
                meme_tokens.append(token)

        return meme_tokens

    def _is_likely_meme_coin(self, token: Token) -> bool:
        """Determine if a token is likely a meme coin based on its properties"""
        # Check token name and symbol for meme-related terms
        meme_terms = ["dog", "cat", "pepe", "frog", "doge", "shib", "inu", "moon", "elon",
                      "safe", "baby", "rocket", "meme", "wojak", "chad", "pamp", "dump"]

        name_lower = token.name.lower()
        symbol_lower = token.symbol.lower()

        # Check if any meme terms appear in the name or symbol
        for term in meme_terms:
            if term in name_lower or term in symbol_lower:
                return True

        # Check other heuristics
        if token.metrics.holders_count < 1000 and token.metrics.market_cap > 0:
            return True

        # By default, not classified as a meme coin
        return False

    async def _analyze_sentiment(self):
        """Analyze sentiment for all tracked tokens"""
        self.logger.info("Analyzing sentiment")

        # Process tokens in batches
        addresses = list(self.token_registry.keys())
        batch_size = SENTIMENT_ANALYSIS_BATCH_SIZE

        for i in range(0, len(addresses), batch_size):
            batch_addresses = addresses[i:i + batch_size]
            batch_tokens = [self.token_registry[addr] for addr in batch_addresses]

            # Analyze sentiment for each token
            for token in batch_tokens:
                # In a real implementation, would analyze from multiple data sources
                # For demo purposes, generate some sample sentiment data
                await self._analyze_token_sentiment(token)

        # Calculate global market sentiment
        overall_sentiment = 0.0
        token_count = len(self.token_registry)

        if token_count > 0:
            sentiment_sum = sum(token.sentiment.overall_sentiment for token in self.token_registry.values())
            overall_sentiment = sentiment_sum / token_count

        # Add to global sentiment history
        self.global_sentiment_history.append(overall_sentiment)
        self.logger.info(f"Global market sentiment: {overall_sentiment:.2f}")

    async def _analyze_token_sentiment(self, token: Token):
        """Analyze sentiment for a specific token"""
        # In a real implementation, would process actual data
        # For demo purposes, simulate sentiment analysis

        # Simulated sentiment metrics
        token.sentiment.positive_score = np.random.randint(5, 100)
        token.sentiment.negative_score = np.random.randint(5, 50)
        token.sentiment.neutral_score = np.random.randint(10, 200)

        # Calculate overall sentiment
        total_mentions = token.sentiment.positive_score + token.sentiment.negative_score + token.sentiment.neutral_score

        if total_mentions > 0:
            weighted_score = (
                                     token.sentiment.positive_score -
                                     token.sentiment.negative_score
                             ) / total_mentions

            # Normalize to range -1.0 to 1.0
            token.sentiment.overall_sentiment = max(min(weighted_score, 1.0), -1.0)

        # Update trending data
        if token.address in self.trend_data:
            # Add current sentiment to trend data
            self.trend_data[token.address].append(token.sentiment.overall_sentiment)

            # Keep only recent history (last 24 data points)
            if len(self.trend_data[token.address]) > 24:
                self.trend_data[token.address] = self.trend_data[token.address][-24:]

            # Calculate sentiment trend (rate of change)
            trend_data = self.trend_data[token.address]
            if len(trend_data) >= 2:
                token.sentiment.sentiment_trend = trend_data[-1] - trend_data[0]

    async def _detect_patterns(self):
        """Detect patterns and correlations between tokens"""
        self.logger.info("Detecting patterns and correlations")

        # Get tokens with sufficient trend data
        tokens_with_data = []
        for address, trends in self.trend_data.items():
            if len(trends) >= 5:  # Need at least 5 data points
                tokens_with_data.append((address, trends))

        # Calculate correlations between tokens
        for i, (address1, trends1) in enumerate(tokens_with_data):
            if address1 not in self.token_correlations:
                self.token_correlations[address1] = {}

            for j, (address2, trends2) in enumerate(tokens_with_data):
                if i != j:  # Don't compare token with itself
                    # Ensure both trend lists are the same length
                    min_length = min(len(trends1), len(trends2))
                    if min_length >= 5:
                        # Calculate correlation coefficient
                        correlation = np.corrcoef(trends1[-min_length:], trends2[-min_length:])[0, 1]

                        # Store correlation if significant
                        if not np.isnan(correlation) and abs(correlation) > 0.5:
                            self.token_correlations[address1][address2] = correlation

        # Find similar tokens
        for address, correlations in self.token_correlations.items():
            if address in self.token_registry:
                token = self.token_registry[address]

                # Get addresses of positively correlated tokens (sorted by correlation strength)
                similar_tokens = sorted(
                    [(addr, corr) for addr, corr in correlations.items() if corr > 0.7],
                    key=lambda x: x[1],
                    reverse=True
                )

                # Store top 5 similar tokens
                token.similarity_tokens = [addr for addr, _ in similar_tokens[:5]]

    async def _update_trend_scores(self):
        """Update trend scores for all tokens"""
        self.logger.info("Updating token trend scores")

        for address, token in self.token_registry.items():
            # Skip tokens with insufficient data
            if address not in self.trend_data or len(self.trend_data[address]) < 3:
                continue

            # Calculate trend score based on multiple factors
            trend_score = 0.0

            # Factor 1: Current sentiment (30%)
            sentiment_factor = (token.sentiment.overall_sentiment + 1) / 2  # Convert to 0-1 range
            trend_score += sentiment_factor * 0.3

            # Factor 2: Sentiment momentum (30%)
            trend_data = self.trend_data[address]
            momentum = 0.0
            if len(trend_data) >= 3:
                # Calculate moving average over last 3 periods
                recent_avg = sum(trend_data[-3:]) / 3
                # Calculate moving average over periods 6-4
                if len(trend_data) >= 6:
                    older_avg = sum(trend_data[-6:-3]) / 3
                    momentum = recent_avg - older_avg

                # Normalize momentum to 0-1 range
                momentum = (momentum + 1) / 2

            trend_score += momentum * 0.3

            # Factor 3: Correlation with other trending tokens (20%)
            correlation_score = 0.0
            if address in self.token_correlations:
                # Get correlations with high-trend tokens
                high_trend_tokens = [addr for addr, tok in self.token_registry.items()
                                     if tok.trend_score > 0.7 and addr != address]

                if high_trend_tokens:
                    # Average correlation with high-trend tokens
                    correlations = [self.token_correlations[address].get(high_addr, 0)
                                    for high_addr in high_trend_tokens
                                    if high_addr in self.token_correlations[address]]

                    if correlations:
                        correlation_score = max(0, sum(correlations) / len(correlations))

            trend_score += correlation_score * 0.2

            # Factor 4: Trading activity (20%)
            # In real implementation, would use actual trading data
            # For demo, simulate based on other factors
            activity_score = min(1.0, (sentiment_factor + momentum) / 1.5)
            trend_score += activity_score * 0.2

            # Update token trend score
            token.trend_score = trend_score

    async def _generate_predictions(self):
        """Generate price movement predictions for trending tokens"""
        self.logger.info("Generating trend predictions")

        # Process tokens with high trend scores
        trending_tokens = [token for token in self.token_registry.values() if token.trend_score > 0.6]

        for token in trending_tokens:
            # Generate prediction based on trend data
            prediction = await self._predict_token_movement(token)

            # Log prediction
            direction = "upward" if prediction > 0 else "downward"
            strength = abs(prediction)

            if strength > 0.7:
                confidence = "strong"
            elif strength > 0.4:
                confidence = "moderate"
            else:
                confidence = "weak"

            self.logger.info(f"Prediction for {token.symbol}: {confidence} {direction} trend (score: {prediction:.2f})")

    async def _predict_token_movement(self, token: Token) -> float:
        """Predict price movement for a token (returns a value from -1.0 to 1.0)"""
        # In a real implementation, would use machine learning models
        # For demo purposes, use a simple heuristic model

        # Start with sentiment as base
        prediction = token.sentiment.overall_sentiment * 0.4

        # Add sentiment trend
        prediction += token.sentiment.sentiment_trend * 0.3

        # Add global market sentiment factor
        if self.global_sentiment_history:
            global_sentiment = self.global_sentiment_history[-1]
            prediction += global_sentiment * 0.1

        # Add token-specific momentum
        if token.address in self.trend_data and len(self.trend_data[token.address]) >= 2:
            trends = self.trend_data[token.address]
            momentum = trends[-1] - trends[0]
            prediction += momentum * 0.2

        # Ensure result is in the range [-1.0, 1.0]
        return max(min(prediction, 1.0), -1.0)

    def _log_analysis_results(self):
        """Log analysis results and trending tokens"""
        # Get top trending tokens
        top_trending = sorted(
            self.token_registry.values(),
            key=lambda token: token.trend_score,
            reverse=True
        )[:10]

        if top_trending:
            self.logger.info("=== Top Trending Tokens ===")
            for i, token in enumerate(top_trending, 1):
                self.logger.info(
                    f"{i}. {token.symbol}: Trend Score {token.trend_score:.2f}, Sentiment {token.sentiment.overall_sentiment:.2f}")

    async def stop(self):
        """Stop the trend analysis process"""
        self.logger.info("Stopping Spice Trend Engine")
        self.is_running = False