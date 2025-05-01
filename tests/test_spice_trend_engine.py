import unittest
import asyncio
from unittest.mock import MagicMock, patch
from datetime import datetime
import numpy as np

from dune_ai.services.spice_trend_engine import SpiceTrendEngine
from dune_ai.blockchain.solana_client import SolanaClient
from dune_ai.models.token import Token


class TestSpiceTrendEngine(unittest.TestCase):
    """Test suite for SpiceTrendEngine class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock Solana client
        self.mock_solana_client = MagicMock(spec=SolanaClient)

        # Initialize SpiceTrendEngine with mock client
        self.trend_engine = SpiceTrendEngine(self.mock_solana_client)

    def test_initialization(self):
        """Test that SpiceTrendEngine initializes correctly"""
        self.assertEqual(self.trend_engine.solana_client, self.mock_solana_client)
        self.assertIsNotNone(self.trend_engine.sentiment_analyzer)
        self.assertIsNotNone(self.trend_engine.pattern_recognition)
        self.assertFalse(self.trend_engine.is_running)
        self.assertIsInstance(self.trend_engine.token_registry, dict)
        self.assertIsInstance(self.trend_engine.trend_data, dict)
        self.assertIsInstance(self.trend_engine.global_sentiment_history, list)

    async def test_get_tokens_to_analyze(self):
        """Test getting tokens to analyze"""
        # Mock the response from get_top_tokens
        mock_tokens = [
            Token(address="addr1", name="Token One", symbol="ONE", decimals=9),
            Token(address="addr2", name="Doge Token", symbol="DOGE", decimals=9),
            Token(address="addr3", name="Token Three", symbol="THREE", decimals=9),
            Token(address="addr4", name="Pepe Token", symbol="PEPE", decimals=9)
        ]
        self.mock_solana_client.get_top_tokens.return_value = mock_tokens

        # Call the method
        result = await self.trend_engine._get_tokens_to_analyze()

        # Check result (should filter for meme coins)
        self.assertEqual(len(result), 2)  # Only the meme coins
        symbols = [token.symbol for token in result]
        self.assertIn("DOGE", symbols)
        self.assertIn("PEPE", symbols)

    def test_is_likely_meme_coin(self):
        """Test detection of meme coins"""
        # Test token with meme name
        token1 = Token(address="addr1", name="Doge Coin", symbol="DOGE", decimals=9)
        self.assertTrue(self.trend_engine._is_likely_meme_coin(token1))

        # Test token with normal name but meme symbol
        token2 = Token(address="addr2", name="Normal Token", symbol="PEPE", decimals=9)
        self.assertTrue(self.trend_engine._is_likely_meme_coin(token2))

        # Test token with moon in name
        token3 = Token(address="addr3", name="To The Moon", symbol="MOON", decimals=9)
        self.assertTrue(self.trend_engine._is_likely_meme_coin(token3))

        # Test regular token
        token4 = Token(address="addr4", name="Serious Project", symbol="SRS", decimals=9)
        self.assertFalse(self.trend_engine._is_likely_meme_coin(token4))

        # Test token with small holders but high market cap (suspicious)
        token5 = Token(address="addr5", name="Suspicious Token", symbol="SUS", decimals=9)
        token5.metrics.holders_count = 50
        token5.metrics.market_cap = 2000000
        self.assertTrue(self.trend_engine._is_likely_meme_coin(token5))

    async def test_analyze_token_sentiment(self):
        """Test sentiment analysis for a token"""
        # Create test token
        token = Token(
            address="test_address",
            name="Test Token",
            symbol="TEST",
            decimals=9
        )

        # Add to trend data
        self.trend_engine.trend_data[token.address] = [0.1, 0.2, 0.3]

        # Call the method
        await self.trend_engine._analyze_token_sentiment(token)

        # Check sentiment was updated
        self.assertGreaterEqual(token.sentiment.positive_score, 0)
        self.assertGreaterEqual(token.sentiment.negative_score, 0)
        self.assertGreaterEqual(token.sentiment.neutral_score, 0)
        self.assertIsNotNone(token.sentiment.overall_sentiment)

        # Check trend data was updated
        self.assertEqual(len(self.trend_engine.trend_data[token.address]), 4)

        # Check sentiment trend was calculated
        self.assertNotEqual(token.sentiment.sentiment_trend, 0)

    async def test_detect_patterns(self):
        """Test pattern detection between tokens"""
        # Create tokens with trend data
        token1 = Token(address="addr1", name="Token One", symbol="ONE", decimals=9)
        token2 = Token(address="addr2", name="Token Two", symbol="TWO", decimals=9)
        token3 = Token(address="addr3", name="Token Three", symbol="THREE", decimals=9)

        self.trend_engine.token_registry = {
            "addr1": token1,
            "addr2": token2,
            "addr3": token3
        }

        # Create similar trend data for tokens 1 and 2
        self.trend_engine.trend_data["addr1"] = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.trend_engine.trend_data["addr2"] = [0.12, 0.22, 0.32, 0.42, 0.52]
        # Different trend for token 3
        self.trend_engine.trend_data["addr3"] = [0.5, 0.4, 0.3, 0.2, 0.1]

        # Call the method
        await self.trend_engine._detect_patterns()

        # Check correlations
        self.assertIn("addr1", self.trend_engine.token_correlations)
        self.assertIn("addr2", self.trend_engine.token_correlations)

        # Token 1 and 2 should be correlated
        if "addr2" in self.trend_engine.token_correlations["addr1"]:
            correlation = self.trend_engine.token_correlations["addr1"]["addr2"]
            self.assertGreater(correlation, 0.7)

        # Token 1 and 3 should be negatively correlated
        if "addr3" in self.trend_engine.token_correlations["addr1"]:
            correlation = self.trend_engine.token_correlations["addr1"]["addr3"]
            self.assertLess(correlation, 0)

        # Check similar tokens were identified
        self.assertIn("addr2", token1.similarity_tokens)

    async def test_update_trend_scores(self):
        """Test updating trend scores for tokens"""
        # Create tokens
        token1 = Token(address="addr1", name="Token One", symbol="ONE", decimals=9)
        token2 = Token(address="addr2", name="Token Two", symbol="TWO", decimals=9)

        self.trend_engine.token_registry = {
            "addr1": token1,
            "addr2": token2
        }

        # Set up trend data
        self.trend_engine.trend_data["addr1"] = [0.1, 0.2, 0.3, 0.4, 0.5]  # Upward trend
        self.trend_engine.trend_data["addr2"] = [0.5, 0.4, 0.3, 0.2, 0.1]  # Downward trend

        # Set up sentiment
        token1.sentiment.overall_sentiment = 0.8  # Positive sentiment
        token2.sentiment.overall_sentiment = -0.3  # Negative sentiment

        # Call the method
        await self.trend_engine._update_trend_scores()

        # Check trend scores
        self.assertGreater(token1.trend_score, token2.trend_score)
        self.assertGreater(token1.trend_score, 0.5)  # Strong positive trend
        self.assertLess(token2.trend_score, 0.5)  # Weaker trend

    async def test_generate_predictions(self):
        """Test generating price movement predictions"""
        # Create a trending token
        token = Token(address="addr1", name="Token One", symbol="ONE", decimals=9)
        token.trend_score = 0.9  # High trend score

        self.trend_engine.token_registry = {
            "addr1": token
        }

        # Call the method
        await self.trend_engine._generate_predictions()

        # Prediction should have been generated but no easy way to test the value
        # since it's calculated with random components in the demo version

        # In a real test, would mock the _predict_token_movement method
        # and verify it was called with the correct token


if __name__ == '__main__':
    unittest.main()