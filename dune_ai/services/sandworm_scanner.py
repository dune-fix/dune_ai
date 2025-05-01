import asyncio
import logging
from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime, timedelta
import time
import random

from config.settings import (
    SOLANA_RPC_URL,
    SCAN_INTERVAL_SECONDS,
    MIN_MARKET_CAP_THRESHOLD,
    MAX_TOKENS_TO_TRACK
)
from config.logging_config import get_logger
from dune_ai.blockchain.solana_client import SolanaClient
from dune_ai.blockchain.token_operations import TokenOperations
from dune_ai.models.token import Token, TokenMetrics
from dune_ai.models.wallet import Wallet


class SandwormScanner:
    """
    Sandworm Scanner - Automatically detects and analyzes new Solana token launches,
    particularly focusing on potential high-growth meme coins.

    Like the massive sandworms of Dune that can detect movement across the desert,
    this scanner monitors the Solana blockchain for new token activity.
    """

    def __init__(self, solana_client: SolanaClient):
        self.logger = get_logger("sandworm_scanner")
        self.solana_client = solana_client
        self.token_operations = TokenOperations(solana_client)

        # Token tracking
        self.tracked_tokens: Dict[str, Token] = {}  # address -> Token
        self.new_token_queue: List[str] = []  # Queue of new token addresses to analyze
        self.recently_analyzed: Set[str] = set()  # Addresses recently analyzed
        self.known_deployers: Dict[str, Wallet] = {}  # address -> Wallet

        # Performance metrics
        self.tokens_analyzed_count = 0
        self.tokens_discovered_count = 0
        self.start_time = datetime.now()

        self.is_running = False
        self.logger.info("Sandworm Scanner initialized")

    async def start_scanning(self):
        """Start the token scanning process"""
        self.logger.info("Starting Sandworm Scanner")
        self.is_running = True
        self.start_time = datetime.now()

        # Start the main scan loop and the analysis loop
        scan_task = asyncio.create_task(self._scan_loop())
        analysis_task = asyncio.create_task(self._analysis_loop())

        try:
            await asyncio.gather(scan_task, analysis_task)
        except asyncio.CancelledError:
            self.logger.info("Sandworm Scanner tasks cancelled")
            scan_task.cancel()
            analysis_task.cancel()
            try:
                await asyncio.gather(scan_task, analysis_task, return_exceptions=True)
            except Exception:
                pass

    async def _scan_loop(self):
        """Main scanning loop to detect new token launches"""
        self.logger.info("Starting token scanning loop")

        while self.is_running:
            try:
                # Scan for recent program transactions
                await self._scan_for_new_tokens()

                # Clean up old data
                self._cleanup_old_data()

                # Log performance metrics
                elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600
                if elapsed_hours > 0:
                    tokens_per_hour = self.tokens_analyzed_count / elapsed_hours
                    self.logger.info(
                        f"Performance: Analyzed {self.tokens_analyzed_count} tokens, "
                        f"discovered {self.tokens_discovered_count} potential meme coins "
                        f"(~{tokens_per_hour:.1f} tokens/hour)"
                    )

                # Sleep until next scan
                await asyncio.sleep(SCAN_INTERVAL_SECONDS)

            except Exception as e:
                self.logger.error(f"Error in scan loop: {e}", exc_info=True)
                await asyncio.sleep(SCAN_INTERVAL_SECONDS * 2)  # Sleep longer on error

    async def _analysis_loop(self):
        """Analysis loop for processing newly discovered tokens"""
        self.logger.info("Starting token analysis loop")

        while self.is_running:
            try:
                # Process tokens in the analysis queue
                await self._process_analysis_queue()

                # Sleep briefly to avoid consuming too many resources
                await asyncio.sleep(10)  # Short sleep between analysis batches

            except Exception as e:
                self.logger.error(f"Error in analysis loop: {e}", exc_info=True)
                await asyncio.sleep(30)  # Sleep longer on error

    async def _scan_for_new_tokens(self):
        """Scan the blockchain for new token launches"""
        self.logger.info("Scanning for new token launches")

        try:
            # Get recent token program transactions
            # In a real implementation, would query Solana for recent TokenProgram transactions
            # For demo purposes, simulate with a request to get recent token program signatures

            new_token_addresses = await self.solana_client.get_recent_token_creations()

            # Filter out already known tokens
            filtered_addresses = [
                addr for addr in new_token_addresses
                if addr not in self.tracked_tokens
                   and addr not in self.recently_analyzed
                   and addr not in self.new_token_queue
            ]

            # Add to analysis queue
            self.new_token_queue.extend(filtered_addresses)
            self.logger.info(f"Added {len(filtered_addresses)} new tokens to analysis queue")

        except Exception as e:
            self.logger.error(f"Error scanning for new tokens: {e}", exc_info=True)

    async def _process_analysis_queue(self):
        """Process the queue of tokens awaiting analysis"""
        # Process up to 10 tokens per batch
        batch_size = min(10, len(self.new_token_queue))

        if batch_size == 0:
            return  # No tokens to process

        self.logger.info(f"Processing {batch_size} tokens from analysis queue")

        for _ in range(batch_size):
            if not self.new_token_queue:
                break

            # Get next token address from queue
            token_address = self.new_token_queue.pop(0)

            # Analyze the token
            success = await self._analyze_token(token_address)

            # Mark as recently analyzed
            self.recently_analyzed.add(token_address)
            self.tokens_analyzed_count += 1

            # Avoid rate limiting
            await asyncio.sleep(1)

    async def _analyze_token(self, token_address: str) -> bool:
        """Analyze a token to determine if it's a potential meme coin"""
        self.logger.info(f"Analyzing token {token_address}")

        try:
            # Get token info
            token_info = await self.solana_client.get_token_info(token_address)

            if not token_info:
                self.logger.warning(f"Could not retrieve info for token {token_address}")
                return False

            # Create token object
            token = Token(
                address=token_address,
                name=token_info.get("name", "Unknown"),
                symbol=token_info.get("symbol", "UNKNOWN"),
                decimals=token_info.get("decimals", 9),
                creator_address=token_info.get("creator")
            )

            # Get token metrics
            await self._get_token_metrics(token)

            # Skip tokens with very low market cap
            if token.metrics.market_cap < MIN_MARKET_CAP_THRESHOLD:
                self.logger.info(
                    f"Skipping token {token.symbol} due to low market cap: ${token.metrics.market_cap:.2f}")
                return False

            # Check if it's a potential meme coin
            is_potential_meme = await self._is_potential_meme_coin(token)
            token.is_meme_coin = is_potential_meme

            if is_potential_meme:
                self.logger.info(f"Discovered potential meme coin: {token.symbol} ({token_address})")

                # Get additional data
                await self._get_token_additional_data(token)

                # Calculate risk score
                token.risk_score = await self._calculate_risk_score(token)

                # Add to tracked tokens
                self.tracked_tokens[token_address] = token
                self.tokens_discovered_count += 1

                # Analyze creator wallet if available
                if token.creator_address:
                    await self._analyze_creator_wallet(token.creator_address)

                return True

            return False

        except Exception as e:
            self.logger.error(f"Error analyzing token {token_address}: {e}", exc_info=True)
            return False

    async def _get_token_metrics(self, token: Token):
        """Get token metrics like market cap, volume, etc."""
        try:
            # In a real implementation, would query Solana for actual metrics
            # For demo purposes, simulate with random but reasonable values

            # Get supply info
            supply_info = await self.solana_client.get_token_supply(token.address)
            circulating_supply = supply_info.get("circulating", 0)

            # Get price info
            price_data = await self.solana_client.get_token_price(token.address)
            price_usd = price_data.get("price_usd", 0)
            price_sol = price_data.get("price_sol", 0)

            # Calculate market cap
            market_cap = circulating_supply * price_usd

            # Add price to history
            token.price_history.append(TokenPrice(
                price_usd=price_usd,
                price_sol=price_sol
            ))

            # Set metrics
            token.metrics.market_cap = market_cap
            token.metrics.volume_24h = market_cap * random.uniform(0.05, 0.5)  # 5-50% of market cap
            token.metrics.price_change_24h = random.uniform(-0.3, 0.5)  # -30% to +50%
            token.metrics.liquidity = market_cap * random.uniform(0.1, 0.4)  # 10-40% of market cap
            token.metrics.holders_count = int(random.uniform(100, 5000))
            token.metrics.transaction_count_24h = int(random.uniform(50, 2000))
            token.metrics.average_transaction_size = token.metrics.volume_24h / max(token.metrics.transaction_count_24h,
                                                                                    1)
            token.metrics.last_updated = datetime.now()

            self.logger.info(
                f"Token {token.symbol} metrics: Market Cap: ${market_cap:.2f}, Volume: ${token.metrics.volume_24h:.2f}")

        except Exception as e:
            self.logger.error(f"Error getting metrics for token {token.address}: {e}", exc_info=True)

    async def _is_potential_meme_coin(self, token: Token) -> bool:
        """Determine if a token is potentially a meme coin based on various factors"""
        # Check name and symbol for meme-related keywords
        meme_keywords = [
            "doge", "shib", "dog", "cat", "pepe", "frog", "moon", "elon", "musk", "safe",
            "cum", "porn", "chad", "wojak", "ape", "based", "trump", "biden", "president",
            "rocket", "mars", "lambo", "tendies", "diamond", "hands", "hodl", "moon", "pump",
            "dump", "gem", "baby", "inu", "floki", "kiss", "meme", "coin", "token", "ai", "yolo"
        ]

        name_lower = token.name.lower()
        symbol_lower = token.symbol.lower()

        # Check for meme keywords in name or symbol
        for keyword in meme_keywords:
            if keyword in name_lower or keyword in symbol_lower:
                self.logger.info(f"Token {token.symbol} has meme keyword: {keyword}")
                return True

        # Check creator's history
        if token.creator_address and token.creator_address in self.known_deployers:
            deployer = self.known_deployers[token.creator_address]
            if len(deployer.created_tokens) > 2:
                # Creator has created multiple tokens before
                self.logger.info(
                    f"Token {token.symbol} creator has created {len(deployer.created_tokens)} other tokens")
                return True

        # Check token metrics
        if token.metrics.market_cap > 0:
            # Check for suspicious holder distribution
            if token.metrics.holders_count < 100 and token.metrics.market_cap > 1000000:
                # High market cap but few holders
                self.logger.info(f"Token {token.symbol} has suspicious holder distribution")
                return True

            # Check for unusual price action
            if token.metrics.price_change_24h > 1.0:  # >100% gain in 24h
                self.logger.info(
                    f"Token {token.symbol} has unusual price action: {token.metrics.price_change_24h * 100:.2f}% in 24h")
                return True

        # Not classified as a meme coin
        return False

    async def _get_token_additional_data(self, token: Token):
        """Get additional data for tokens identified as potential meme coins"""
        try:
            # Get social links
            social_data = await self.solana_client.get_token_social_links(token.address)

            token.website = social_data.get("website")
            token.twitter = social_data.get("twitter")
            token.telegram = social_data.get("telegram")
            token.discord = social_data.get("discord")

            # Get launch date
            launch_info = await self.solana_client.get_token_launch_info(token.address)
            token.launch_date = launch_info.get("launch_date")

            # Get trading pairs
            trading_pairs = await self.solana_client.get_token_trading_pairs(token.address)
            token.trading_pairs = trading_pairs

            self.logger.info(f"Retrieved additional data for {token.symbol}")

        except Exception as e:
            self.logger.error(f"Error getting additional data for token {token.address}: {e}", exc_info=True)

    async def _calculate_risk_score(self, token: Token) -> float:
        """Calculate a risk score for the token (0.0 = low risk, 1.0 = high risk)"""
        risk_score = 0.0
        risk_factors = 0

        # Factor 1: Lack of social presence
        if not token.website and not token.twitter and not token.telegram:
            risk_score += 0.2
            risk_factors += 1

        # Factor 2: Very recent launch (less than 24 hours)
        if token.launch_date and (datetime.now() - token.launch_date) < timedelta(hours=24):
            risk_score += 0.15
            risk_factors += 1

        # Factor 3: Suspicious token distribution
        if token.metrics.holders_count < 50:
            risk_score += 0.2
            risk_factors += 1

        # Factor 4: Unusual price action
        if abs(token.metrics.price_change_24h) > 2.0:  # >200% movement in 24h
            risk_score += 0.2
            risk_factors += 1

        # Factor 5: Creator history
        if token.creator_address and token.creator_address in self.known_deployers:
            creator = self.known_deployers[token.creator_address]
            if creator.profile.risk_score > 0.7:
                risk_score += 0.25
                risk_factors += 1

        # Normalize risk score based on number of factors
        if risk_factors > 0:
            risk_score = min(risk_score, 1.0)

        # Log risk assessment
        if risk_score > 0.7:
            risk_level = "HIGH"
        elif risk_score > 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        self.logger.info(f"Risk assessment for {token.symbol}: {risk_level} ({risk_score:.2f})")
        return risk_score

    async def _analyze_creator_wallet(self, wallet_address: str):
        """Analyze a token creator's wallet for additional insights"""
        try:
            # Skip if already known
            if wallet_address in self.known_deployers:
                return

            self.logger.info(f"Analyzing creator wallet {wallet_address}")

            # Get wallet info
            wallet_info = await self.solana_client.get_wallet_info(wallet_address)

            # Create wallet object
            wallet = Wallet(address=wallet_address)
            wallet.sol_balance = wallet_info.get("sol_balance", 0)
            wallet.first_seen = wallet_info.get("first_seen")

            # Get created tokens
            created_tokens = await self.solana_client.get_tokens_created_by_wallet(wallet_address)
            wallet.created_tokens = set(created_tokens)

            # Set profile flags
            wallet.profile.is_contract_deployer = len(wallet.created_tokens) > 0
            wallet.profile.is_developer = wallet.profile.is_contract_deployer
            wallet.profile.is_whale = wallet.sol_balance > 100  # >100 SOL

            # Calculate risk score
            risk_score = 0.0

            # Factor 1: Many token deployments
            if len(wallet.created_tokens) > 5:
                risk_score += min(len(wallet.created_tokens) / 20, 0.5)  # Up to 0.5 for many deployments

            # Factor 2: New wallet
            if wallet.first_seen and (datetime.now() - wallet.first_seen) < timedelta(days=30):
                risk_score += 0.3  # Newer wallets are higher risk

            # Set risk score
            wallet.profile.risk_score = min(risk_score, 1.0)

            # Store in known deployers
            self.known_deployers[wallet_address] = wallet

            self.logger.info(
                f"Wallet {wallet_address} analyzed: Created {len(wallet.created_tokens)} tokens, Risk: {wallet.profile.risk_score:.2f}")

        except Exception as e:
            self.logger.error(f"Error analyzing wallet {wallet_address}: {e}", exc_info=True)

    def _cleanup_old_data(self):
        """Clean up old data to prevent memory bloat"""
        now = datetime.now()

        # Clean up recently analyzed set (keep last 24 hours)
        self.recently_analyzed = {
            addr for addr in self.recently_analyzed
            if addr in self.tracked_tokens  # Keep if tracked
        }

        # Limit number of tracked tokens
        if len(self.tracked_tokens) > MAX_TOKENS_TO_TRACK:
            # Sort by trend score and keep only the top ones
            sorted_tokens = sorted(
                self.tracked_tokens.values(),
                key=lambda t: t.trend_score,
                reverse=True
            )

            # Keep the top tokens
            self.tracked_tokens = {
                token.address: token
                for token in sorted_tokens[:MAX_TOKENS_TO_TRACK]
            }

            self.logger.info(f"Cleaned up tracked tokens, now tracking {len(self.tracked_tokens)}")

    async def get_top_trending_tokens(self, limit: int = 10) -> List[Token]:
        """Get the top trending tokens with high potential"""
        # Sort by trend score
        sorted_tokens = sorted(
            self.tracked_tokens.values(),
            key=lambda t: t.trend_score,
            reverse=True
        )

        # Return top tokens
        return sorted_tokens[:limit]

    async def stop(self):
        """Stop the token scanning process"""
        self.logger.info("Stopping Sandworm Scanner")
        self.is_running = False