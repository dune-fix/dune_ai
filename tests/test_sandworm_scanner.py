import unittest
import asyncio
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from dune_ai.services.sandworm_scanner import SandwormScanner
from dune_ai.blockchain.solana_client import SolanaClient
from dune_ai.models.token import Token
from dune_ai.models.wallet import Wallet


class TestSandwormScanner(unittest.TestCase):
    """Test suite for SandwormScanner class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock Solana client
        self.mock_solana_client = MagicMock(spec=SolanaClient)

        # Initialize SandwormScanner with mock client
        self.scanner = SandwormScanner(self.mock_solana_client)

    def test_initialization(self):
        """Test that SandwormScanner initializes correctly"""
        self.assertEqual(self.scanner.solana_client, self.mock_solana_client)
        self.assertIsNotNone(self.scanner.token_operations)
        self.assertFalse(self.scanner.is_running)
        self.assertIsInstance(self.scanner.tracked_tokens, dict)
        self.assertIsInstance(self.scanner.new_token_queue, list)
        self.assertIsInstance(self.scanner.recently_analyzed, set)
        self.assertEqual(self.scanner.tokens_analyzed_count, 0)
        self.assertEqual(self.scanner.tokens_discovered_count, 0)

    async def test_scan_for_new_tokens(self):
        """Test scanning for new token launches"""
        # Mock the response from get_recent_token_creations
        mock_addresses = ["addr1", "addr2", "addr3"]
        self.mock_solana_client.get_recent_token_creations.return_value = mock_addresses

        # Call the method
        await self.scanner._scan_for_new_tokens()

        # Check that addresses were added to queue
        self.assertEqual(len(self.scanner.new_token_queue), 3)
        for addr in mock_addresses:
            self.assertIn(addr, self.scanner.new_token_queue)

        # Test filtering of known addresses
        # Reset queue
        self.scanner.new_token_queue = []

        # Add a token to tracked tokens
        self.scanner.tracked_tokens["addr1"] = Token(
            address="addr1", name="Token One", symbol="ONE", decimals=9
        )

        # Add an address to recently analyzed
        self.scanner.recently_analyzed.add("addr2")

        # Call method again
        await self.scanner._scan_for_new_tokens()

        # Only addr3 should be added
        self.assertEqual(len(self.scanner.new_token_queue), 1)
        self.assertEqual(self.scanner.new_token_queue[0], "addr3")

    async def test_analyze_token(self):
        """Test token analysis"""
        # Mock token info
        mock_token_info = {
            "name": "Test Token",
            "symbol": "TEST",
            "decimals": 9,
            "creator": "creator_address"
        }
        self.mock_solana_client.get_token_info.return_value = mock_token_info

        # Mock token metrics
        mock_supply = {"circulating": 1000000}
        self.mock_solana_client.get_token_supply.return_value = mock_supply

        mock_price = {"price_usd": 1.0, "price_sol": 0.01}
        self.mock_solana_client.get_token_price.return_value = mock_price

        # Set token to be detected as a meme coin
        mock_token_info["name"] = "Doge Test"

        # Call the method
        result = await self.scanner._analyze_token("test_address")

        # Should return True for meme coin
        self.assertTrue(result)

        # Check token was added to tracked tokens
        self.assertIn("test_address", self.scanner.tracked_tokens)
        token = self.scanner.tracked_tokens["test_address"]
        self.assertEqual(token.name, "Doge Test")
        self.assertEqual(token.symbol, "TEST")
        self.assertTrue(token.is_meme_coin)

        # Check creator wallet should be analyzed
        self.mock_solana_client.get_wallet_info.assert_called_once()

    async def test_get_token_metrics(self):
        """Test getting token metrics"""
        # Create test token
        token = Token(
            address="test_address",
            name="Test Token",
            symbol="TEST",
            decimals=9
        )

        # Mock responses
        mock_supply = {"circulating": 1000000}
        self.mock_solana_client.get_token_supply.return_value = mock_supply

        mock_price = {"price_usd": 1.0, "price_sol": 0.01}
        self.mock_solana_client.get_token_price.return_value = mock_price

        # Call the method
        await self.scanner._get_token_metrics(token)

        # Check metrics were updated
        self.assertEqual(token.metrics.market_cap, 1000000)  # 1M supply * $1
        self.assertGreater(token.metrics.volume_24h, 0)
        self.assertIsNotNone(token.metrics.price_change_24h)
        self.assertGreater(token.metrics.liquidity, 0)
        self.assertGreater(token.metrics.holders_count, 0)

        # Check price history was updated
        self.assertEqual(len(token.price_history), 1)
        self.assertEqual(token.price_history[0].price_usd, 1.0)
        self.assertEqual(token.price_history[0].price_sol, 0.01)

    def test_is_potential_meme_coin(self):
        """Test meme coin detection"""
        # Test token with meme keyword in name
        token1 = Token(address="addr1", name="Doge Coin", symbol="DOGE", decimals=9)
        self.assertTrue(self.scanner._is_potential_meme_coin(token1))

        # Test token with meme keyword in symbol
        token2 = Token(address="addr2", name="Normal Token", symbol="PEPE", decimals=9)
        self.assertTrue(self.scanner._is_potential_meme_coin(token2))

        # Test token with suspicious holder distribution
        token3 = Token(address="addr3", name="Suspicious", symbol="SUS", decimals=9)
        token3.metrics.holders_count = 50
        token3.metrics.market_cap = 2000000
        self.assertTrue(self.scanner._is_potential_meme_coin(token3))

        # Test token with unusual price action
        token4 = Token(address="addr4", name="Pumping", symbol="PUMP", decimals=9)
        token4.metrics.price_change_24h = 1.5  # 150% gain
        self.assertTrue(self.scanner._is_potential_meme_coin(token4))

        # Test regular token
        token5 = Token(address="addr5", name="Normal Project", symbol="NORM", decimals=9)
        self.assertFalse(self.scanner._is_potential_meme_coin(token5))

        # Test creator history
        token6 = Token(address="addr6", name="Creator Test", symbol="TEST", decimals=9)
        token6.creator_address = "creator1"

        # Add creator to known deployers
        creator_wallet = Wallet(address="creator1")
        creator_wallet.created_tokens = {"token1", "token2", "token3"}
        self.scanner.known_deployers["creator1"] = creator_wallet

        # Token from creator who made multiple tokens before is likely a meme coin
        self.assertTrue(self.scanner._is_potential_meme_coin(token6))

    async def test_calculate_risk_score(self):
        """Test risk score calculation"""
        # Create test token
        token = Token(
            address="test_address",
            name="Test Token",
            symbol="TEST",
            decimals=9
        )

        # Default risk should be low
        risk_score = await self.scanner._calculate_risk_score(token)
        self.assertLess(risk_score, 0.3)

        # Add risk factors

        # Risk factor 1: No social presence
        token.website = None
        token.twitter = None
        token.telegram = None

        # Risk factor 2: Very recent launch
        token.launch_date = datetime.now() - timedelta(hours=12)

        # Risk factor 3: Few holders
        token.metrics.holders_count = 20

        # Risk factor 4: Extreme price movement
        token.metrics.price_change_24h = 3.0  # 300% gain

        # Recalculate risk
        risk_score = await self.scanner._calculate_risk_score(token)

        # Should now be high risk
        self.assertGreater(risk_score, 0.7)

    async def test_analyze_creator_wallet(self):
        """Test creator wallet analysis"""
        # Mock wallet info
        mock_wallet_info = {
            "sol_balance": 50.0,
            "first_seen": datetime.now() - timedelta(days=10)
        }
        self.mock_solana_client.get_wallet_info.return_value = mock_wallet_info

        # Mock created tokens
        mock_created_tokens = ["token1", "token2", "token3"]
        self.mock_solana_client.get_tokens_created_by_wallet.return_value = mock_created_tokens

        # Call the method
        await self.scanner._analyze_creator_wallet("creator_address")

        # Check wallet was added to known deployers
        self.assertIn("creator_address", self.scanner.known_deployers)
        wallet = self.scanner.known_deployers["creator_address"]

        # Check wallet data
        self.assertEqual(wallet.address, "creator_address")
        self.assertEqual(wallet.sol_balance, 50.0)
        self.assertEqual(len(wallet.created_tokens), 3)
        self.assertTrue(wallet.profile.is_contract_deployer)
        self.assertTrue(wallet.profile.is_developer)

        # Should have a risk score
        self.assertGreater(wallet.profile.risk_score, 0)

    def test_cleanup_old_data(self):
        """Test data cleanup"""
        # Add some tokens to tracked_tokens
        for i in range(1, 11):
            token = Token(
                address=f"addr{i}",
                name=f"Token {i}",
                symbol=f"TK{i}",
                decimals=9
            )
            token.trend_score = i / 10.0  # 0.1 to 1.0
            self.scanner.tracked_tokens[f"addr{i}"] = token

        # Add some addresses to recently_analyzed
        for i in range(5, 15):
            self.scanner.recently_analyzed.add(f"addr{i}")

        # Set max tokens to 5
        self.scanner.MAX_TOKENS_TO_TRACK = 5

        # Call the method
        self.scanner._cleanup_old_data()

        # Should keep only top 5 tokens by trend score
        self.assertEqual(len(self.scanner.tracked_tokens), 5)
        for i in range(6, 11):
            self.assertIn(f"addr{i}", self.scanner.tracked_tokens)

        # Should clean up recently_analyzed for untracked tokens
        for addr in self.scanner.recently_analyzed:
            self.assertIn(addr, self.scanner.tracked_tokens)

    async def test_get_top_trending_tokens(self):
        """Test getting top trending tokens"""
        # Add some tokens to tracked_tokens with different trend scores
        for i in range(1, 11):
            token = Token(
                address=f"addr{i}",
                name=f"Token {i}",
                symbol=f"TK{i}",
                decimals=9
            )
            token.trend_score = i / 10.0  # 0.1 to 1.0
            self.scanner.tracked_tokens[f"addr{i}"] = token

        # Get top 5
        top_tokens = await self.scanner.get_top_trending_tokens(limit=5)

        # Should return 5 tokens in descending order of trend score
        self.assertEqual(len(top_tokens), 5)
        self.assertEqual(top_tokens[0].symbol, "TK10")  # Highest trend score
        self.assertEqual(top_tokens[1].symbol, "TK9")
        self.assertEqual(top_tokens[2].symbol, "TK8")
        self.assertEqual(top_tokens[3].symbol, "TK7")
        self.assertEqual(top_tokens[4].symbol, "TK6")

    if __name__ == '__main__':
        unittest.main()