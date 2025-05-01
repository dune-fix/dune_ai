import unittest
import asyncio
from unittest.mock import MagicMock, patch
from datetime import datetime

from dune_ai.services.twitter_sentinel import TwitterSentinel
from dune_ai.blockchain.solana_client import SolanaClient
from dune_ai.models.token import Token


class TestTwitterSentinel(unittest.TestCase):
    """Test suite for TwitterSentinel class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock Solana client
        self.mock_solana_client = MagicMock(spec=SolanaClient)

        # Initialize TwitterSentinel with mock client
        self.twitter_sentinel = TwitterSentinel(self.mock_solana_client)

        # Mock the Twitter API client
        self.twitter_sentinel.api = MagicMock()

    def test_initialization(self):
        """Test that TwitterSentinel initializes correctly"""
        self.assertEqual(self.twitter_sentinel.solana_client, self.mock_solana_client)
        self.assertIsNotNone(self.twitter_sentinel.sentiment_analyzer)
        self.assertFalse(self.twitter_sentinel.is_running)
        self.assertIsInstance(self.twitter_sentinel.tracked_keywords, set)
        self.assertIsInstance(self.twitter_sentinel.tracked_tokens, dict)

    @patch('tweepy.API')
    def test_init_twitter_api(self, mock_api):
        """Test Twitter API initialization"""
        # Reset the API to test initialization
        self.twitter_sentinel.api = None

        # Call the method
        api = self.twitter_sentinel._init_twitter_api()

        # Check that API was created
        self.assertIsNotNone(api)

    def test_analyze_tweet_sentiment(self):
        """Test sentiment analysis of a tweet"""
        # Test positive tweet
        positive_tweet = "Just bought some $SOL! This is going to moon! 🚀 #bullish"
        positive_score = self.twitter_sentinel._analyze_tweet_sentiment(positive_tweet)
        self.assertGreater(positive_score, 0)

        # Test negative tweet
        negative_tweet = "Selling all my $SOL, this project is going nowhere 📉 #bearish"
        negative_score = self.twitter_sentinel._analyze_tweet_sentiment(negative_tweet)
        self.assertLess(negative_score, 0)

        # Test neutral tweet
        neutral_tweet = "Here's the latest update on $SOL development"
        neutral_score = self.twitter_sentinel._analyze_tweet_sentiment(neutral_tweet)
        self.assertAlmostEqual(neutral_score, 0, delta=0.3)

    async def test_update_token_sentiment(self):
        """Test updating token sentiment based on a tweet"""
        # Create test token
        token = Token(
            address="test_address",
            name="Test Token",
            symbol="TEST",
            decimals=9
        )

        # Update with positive sentiment
        await self.twitter_sentinel._update_token_sentiment(token, 0.8, 5000)
        self.assertEqual(token.sentiment.positive_score, 1)
        self.assertEqual(token.sentiment.negative_score, 0)
        self.assertEqual(token.sentiment.tweet_count, 1)

        # Update with negative sentiment
        await self.twitter_sentinel._update_token_sentiment(token, -0.8, 5000)
        self.assertEqual(token.sentiment.positive_score, 1)
        self.assertEqual(token.sentiment.negative_score, 1)
        self.assertEqual(token.sentiment.tweet_count, 2)

        # Update with neutral sentiment
        await self.twitter_sentinel._update_token_sentiment(token, 0.0, 5000)
        self.assertEqual(token.sentiment.positive_score, 1)
        self.assertEqual(token.sentiment.negative_score, 1)
        self.assertEqual(token.sentiment.neutral_score, 1)
        self.assertEqual(token.sentiment.tweet_count, 3)

        # Check influential mentions for users with many followers
        await self.twitter_sentinel._update_token_sentiment(token, 0.5, 15000)
        self.assertEqual(token.sentiment.influential_mentions, 1)

    async def test_verify_solana_address(self):
        """Test verification of Solana token addresses"""
        # Mock the response from get_token_info
        mock_token_info = {
            "name": "Mock Token",
            "symbol": "MOCK",
            "decimals": 9
        }
        self.mock_solana_client.get_token_info.return_value = mock_token_info

        # Call the method
        token = await self.twitter_sentinel._verify_solana_address("mock_address")

        # Check token was created correctly
        self.assertIsNotNone(token)
        self.assertEqual(token.name, "Mock Token")
        self.assertEqual(token.symbol, "MOCK")
        self.assertEqual(token.decimals, 9)

        # Test error handling
        self.mock_solana_client.get_token_info.side_effect = Exception("API error")
        token = await self.twitter_sentinel._verify_solana_address("error_address")
        self.assertIsNone(token)

    def test_update_tracking_keywords(self):
        """Test updating of tracking keywords based on discovered tokens"""
        # Add some tokens to track
        self.twitter_sentinel.tracked_tokens = {
            "addr1": Token(address="addr1", name="Token One", symbol="ONE", decimals=9),
            "addr2": Token(address="addr2", name="Token Two", symbol="TWO", decimals=9)
        }

        # Call the method
        self.twitter_sentinel._update_tracking_keywords()

        # Check that symbols were added to keywords
        self.assertIn("one", self.twitter_sentinel.tracked_keywords)
        self.assertIn("$one", self.twitter_sentinel.tracked_keywords)
        self.assertIn("two", self.twitter_sentinel.tracked_keywords)
        self.assertIn("$two", self.twitter_sentinel.tracked_keywords)

    async def test_process_results(self):
        """Test processing of scan results"""
        # Add some tokens with mentions
        token1 = Token(address="addr1", name="Token One", symbol="ONE", decimals=9)
        token2 = Token(address="addr2", name="Token Two", symbol="TWO", decimals=9)

        self.twitter_sentinel.tracked_tokens = {
            "addr1": token1,
            "addr2": token2
        }

        self.twitter_sentinel.token_mentions = {
            "one": 50,
            "two": 10
        }

        # Add some sentiment data
        token1.sentiment.overall_sentiment = 0.8
        token1.sentiment.influential_mentions = 5

        token2.sentiment.overall_sentiment = -0.3
        token2.sentiment.influential_mentions = 1

        # Call the method
        await self.twitter_sentinel._process_results()

        # Check that trend scores were updated
        self.assertGreater(token1.trend_score, token2.trend_score)
        self.assertGreater(token1.trend_score, 0.5)


if __name__ == '__main__':
    unittest.main()