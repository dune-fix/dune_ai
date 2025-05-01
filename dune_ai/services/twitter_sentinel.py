import asyncio
import logging
import re
from typing import List, Dict, Set, Optional
from datetime import datetime, timedelta
import tweepy
from textblob import TextBlob

from config.settings import (
    TWITTER_API_KEY,
    TWITTER_API_SECRET,
    TWITTER_ACCESS_TOKEN,
    TWITTER_ACCESS_SECRET,
    SCAN_INTERVAL_SECONDS
)
from config.logging_config import get_logger
from dune_ai.blockchain.solana_client import SolanaClient
from dune_ai.analytics.sentiment_analyzer import SentimentAnalyzer
from dune_ai.models.token import Token, TokenSentiment


class TwitterSentinel:
    """
    Twitter Sentinel - Real-time analysis of Twitter trends to detect new Solana meme coins
    and monitor cryptocurrency sentiment.

    This component monitors Twitter for mentions of Solana tokens, tracks sentiment,
    and identifies emerging trends like the Fremen scouts from Dune who monitor the desert.
    """

    def __init__(self, solana_client: SolanaClient):
        self.logger = get_logger("twitter_sentinel")
        self.solana_client = solana_client
        self.sentiment_analyzer = SentimentAnalyzer()

        # Twitter API client
        self.api = self._init_twitter_api()

        # Tracking data
        self.tracked_tokens: Dict[str, Token] = {}  # address -> Token
        self.tracked_keywords: Set[str] = {
            "solana", "sol", "meme coin", "memecoin", "$sol", "solana token",
            "solana gems", "100x", "1000x", "solana season", "next bonk"
        }
        self.token_mentions: Dict[str, int] = {}  # token symbol -> count
        self.address_pattern = re.compile(r"\b[1-9A-HJ-NP-Za-km-z]{32,44}\b")

        self.last_scan_time = datetime.now() - timedelta(minutes=60)  # Start with old time to ensure initial scan
        self.is_running = False

        self.logger.info("Twitter Sentinel initialized")

    def _init_twitter_api(self) -> tweepy.API:
        """Initialize Twitter API client"""
        auth = tweepy.OAuth1UserHandler(
            TWITTER_API_KEY,
            TWITTER_API_SECRET,
            TWITTER_ACCESS_TOKEN,
            TWITTER_ACCESS_SECRET
        )
        api = tweepy.API(auth)
        self.logger.info("Twitter API client initialized")
        return api

    async def start_monitoring(self):
        """Start the Twitter monitoring process"""
        self.logger.info("Starting Twitter monitoring")
        self.is_running = True

        while self.is_running:
            try:
                await self._scan_twitter()

                # Update tracking keywords based on discovered tokens
                self._update_tracking_keywords()

                # Process and analyze results
                await self._process_results()

                # Sleep until next scan
                await asyncio.sleep(SCAN_INTERVAL_SECONDS)

            except Exception as e:
                self.logger.error(f"Error in Twitter monitoring: {e}", exc_info=True)
                await asyncio.sleep(SCAN_INTERVAL_SECONDS * 2)  # Longer sleep on error

    async def _scan_twitter(self):
        """Scan Twitter for relevant tweets"""
        self.logger.info("Scanning Twitter for Solana token mentions")

        # Get current time and calculate time since last scan
        now = datetime.now()
        minutes_since_last_scan = int((now - self.last_scan_time).total_seconds() / 60) + 1

        # Make sure we don't request too far back (Twitter API limitations)
        minutes_since_last_scan = min(minutes_since_last_scan, 60 * 24 * 7)  # Max 1 week

        # Build search query from tracked keywords
        search_query = " OR ".join([f'"{kw}"' for kw in self.tracked_keywords])
        search_query = f"{search_query} -filter:retweets"

        # The actual Twitter search would be done here
        # In a real implementation, this would use tweepy's cursor to iterate through results
        try:
            # Sample implementation (limited by API)
            tweets = self.api.search_tweets(
                q=search_query,
                count=100,
                result_type="recent",
                tweet_mode="extended"
            )

            self.logger.info(f"Found {len(tweets)} tweets matching search criteria")

            # Process tweets
            for tweet in tweets:
                await self._process_tweet(tweet)

            self.last_scan_time = now

        except tweepy.TweepyException as e:
            self.logger.error(f"Twitter API error: {e}")

    async def _process_tweet(self, tweet):
        """Process an individual tweet to extract tokens and sentiment"""
        text = tweet.full_text if hasattr(tweet, 'full_text') else tweet.text
        user_followers = tweet.user.followers_count

        # Extract potential Solana addresses
        addresses = self.address_pattern.findall(text)

        # Check for token symbols
        for token in self.tracked_tokens.values():
            symbol = token.symbol.lower()
            if f"${symbol}" in text.lower() or f" {symbol} " in f" {text.lower()} ":
                # Increment mention count
                if symbol not in self.token_mentions:
                    self.token_mentions[symbol] = 0
                self.token_mentions[symbol] += 1

                # Update token sentiment
                sentiment = self._analyze_tweet_sentiment(text)
                await self._update_token_sentiment(token, sentiment, user_followers)

        # Look up potential token addresses
        for address in addresses:
            token = await self._verify_solana_address(address)
            if token:
                # Add to tracked tokens if new
                if token.address not in self.tracked_tokens:
                    self.tracked_tokens[token.address] = token
                    self.logger.info(f"New token discovered: {token.symbol} ({token.address})")

                # Update sentiment
                sentiment = self._analyze_tweet_sentiment(text)
                await self._update_token_sentiment(token, sentiment, user_followers)

    def _analyze_tweet_sentiment(self, text: str) -> float:
        """Analyze sentiment of tweet text using TextBlob for simple sentiment"""
        # In a real implementation, would use the SentimentAnalyzer for more advanced analysis
        # This is a simple implementation using TextBlob
        blob = TextBlob(text)
        return blob.sentiment.polarity  # Range from -1.0 (negative) to 1.0 (positive)

    async def _update_token_sentiment(self, token: Token, sentiment_score: float, user_followers: int):
        """Update token sentiment metrics based on a tweet"""
        # Update sentiment metrics
        if sentiment_score > 0.3:
            token.sentiment.positive_score += 1
        elif sentiment_score < -0.3:
            token.sentiment.negative_score += 1
        else:
            token.sentiment.neutral_score += 1

        # Increase tweet count
        token.sentiment.tweet_count += 1

        # Consider influential mentions (users with many followers)
        if user_followers > 10000:
            token.sentiment.influential_mentions += 1

        # Calculate overall sentiment (weighted average)
        total_mentions = token.sentiment.positive_score + token.sentiment.negative_score + token.sentiment.neutral_score
        if total_mentions > 0:
            # Weight positive and negative scores more highly than neutral
            weighted_score = (
                                     token.sentiment.positive_score * 1.0 -
                                     token.sentiment.negative_score * 1.0
                             ) / total_mentions
            token.sentiment.overall_sentiment = weighted_score

        # Update timestamp
        token.sentiment.last_updated = datetime.now()

    async def _verify_solana_address(self, address: str) -> Optional[Token]:
        """Verify if an address is a valid Solana token and retrieve token info"""
        try:
            token_info = await self.solana_client.get_token_info(address)
            if token_info:
                return Token(
                    address=address,
                    name=token_info.get("name", "Unknown"),
                    symbol=token_info.get("symbol", "UNKNOWN"),
                    decimals=token_info.get("decimals", 9)
                )
            return None
        except Exception as e:
            self.logger.warning(f"Error verifying token address {address}: {e}")
            return None

    def _update_tracking_keywords(self):
        """Update the list of keywords to track based on discovered tokens"""
        # Add token symbols to tracked keywords
        for token in self.tracked_tokens.values():
            symbol = token.symbol.lower()
            self.tracked_keywords.add(symbol)
            self.tracked_keywords.add(f"${symbol}")

        # Limit to top mentioned tokens to avoid tracking too many
        if len(self.token_mentions) > 100:
            # Keep top 100 mentioned tokens
            top_symbols = sorted(self.token_mentions.items(), key=lambda x: x[1], reverse=True)[:100]
            for symbol, _ in top_symbols:
                self.tracked_keywords.add(symbol)
                self.tracked_keywords.add(f"${symbol}")

    async def _process_results(self):
        """Process scan results and update token trend scores"""
        now = datetime.now()

        # Update trend scores for all tracked tokens
        for token in self.tracked_tokens.values():
            symbol = token.symbol.lower()
            mention_count = self.token_mentions.get(symbol, 0)

            # Calculate trend score based on multiple factors:
            # - Recent mention count
            # - Sentiment
            # - Change in mentions over time
            # - Influential mentions

            # Simple trend score calculation (would be more complex in real implementation)
            trend_score = 0.0

            # Factor 1: Mention count (normalized)
            if mention_count > 0:
                trend_score += min(mention_count / 100, 1.0) * 0.4  # 40% weight

            # Factor 2: Positive sentiment
            if token.sentiment.overall_sentiment > 0:
                trend_score += min(token.sentiment.overall_sentiment, 1.0) * 0.3  # 30% weight

            # Factor 3: Influential mentions
            if token.sentiment.influential_mentions > 0:
                trend_score += min(token.sentiment.influential_mentions / 10, 1.0) * 0.3  # 30% weight

            # Update token trend score
            token.trend_score = trend_score

        # Log high trending tokens
        high_trending = [t for t in self.tracked_tokens.values() if t.trend_score > 0.7]
        if high_trending:
            self.logger.info(f"High trending tokens: {', '.join([t.symbol for t in high_trending])}")

    async def stop(self):
        """Stop the Twitter monitoring process"""
        self.logger.info("Stopping Twitter Sentinel")
        self.is_running = False