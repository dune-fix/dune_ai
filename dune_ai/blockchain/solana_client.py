import asyncio
import base58
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import aiohttp
import time
import random

from config.settings import (
    SOLANA_RPC_URL,
    SOLANA_WEBSOCKET_URL,
    SOLANA_COMMITMENT
)
from config.logging_config import get_logger
from dune_ai.models.token import Token, TokenPrice


class SolanaClient:
    """
    Client for interacting with the Solana blockchain.

    Provides methods for querying token information, account data,
    program interactions, and other blockchain data.
    """

    def __init__(self, rpc_url: str = SOLANA_RPC_URL):
        self.logger = get_logger("solana_client")
        self.rpc_url = rpc_url
        self.session = None
        self.websocket = None
        self.request_id = 0
        self.token_program_id = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
        self.token_metadata_program_id = "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s"

        # Cache for frequently accessed data
        self.token_cache: Dict[str, Dict[str, Any]] = {}
        self.account_cache: Dict[str, Dict[str, Any]] = {}
        self.price_cache: Dict[str, Tuple[float, datetime]] = {}  # token_address -> (price, timestamp)

        self.logger.info("SolanaClient initialized with RPC URL: %s", rpc_url)

    async def initialize(self):
        """Initialize the client session"""
        self.session = aiohttp.ClientSession()
        self.logger.info("Solana client session initialized")

    async def close(self):
        """Close the client session"""
        if self.session:
            await self.session.close()
            self.session = None

        if self.websocket:
            await self.websocket.close()
            self.websocket = None

        self.logger.info("Solana client session closed")

    async def _make_rpc_request(self, method: str, params: List[Any] = None) -> Dict[str, Any]:
        """Make an RPC request to the Solana node"""
        if not self.session:
            await self.initialize()

        self.request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params or []
        }

        try:
            async with self.session.post(self.rpc_url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.logger.error(f"RPC request failed: {response.status} - {error_text}")
                    return {"error": error_text}

                result = await response.json()

                if "error" in result:
                    self.logger.error(f"RPC error: {result['error']}")
                    return {"error": result["error"]}

                return result.get("result", {})

        except aiohttp.ClientError as e:
            self.logger.error(f"RPC client error: {e}")
            return {"error": str(e)}
        except Exception as e:
            self.logger.error(f"Unexpected error in RPC request: {e}")
            return {"error": str(e)}

    async def get_token_info(self, token_address: str) -> Dict[str, Any]:
        """Get information about a token from its mint address"""
        # Check cache first
        if token_address in self.token_cache:
            return self.token_cache[token_address]

        self.logger.info(f"Getting token info for {token_address}")

        try:
            # In a real implementation, would query the token metadata on-chain
            # For demo purposes, generate some simulated info

            # Get account info from RPC
            account_info = await self._make_rpc_request(
                "getAccountInfo",
                [token_address, {"encoding": "jsonParsed"}]
            )

            if "error" in account_info:
                self.logger.error(f"Error getting account info for {token_address}: {account_info['error']}")
                return {}

            # Check if it's a token mint
            data = account_info.get("data", {})
            if data.get("program") != "spl-token":
                self.logger.warning(f"Address {token_address} is not an SPL token")
                return {}

            # Parse token info
            parsed_data = data.get("parsed", {}).get("info", {})
            decimals = parsed_data.get("decimals", 9)

            # Get metadata (would be a separate request in real implementation)
            # Here we simulate it
            token_name = self._generate_token_name()
            token_symbol = token_name.split()[0].upper()

            # Get creator (would be from metadata in real implementation)
            creator = await self._get_token_creator(token_address)

            # Store result
            token_info = {
                "address": token_address,
                "name": token_name,
                "symbol": token_symbol,
                "decimals": decimals,
                "creator": creator
            }

            # Cache the result
            self.token_cache[token_address] = token_info

            return token_info

        except Exception as e:
            self.logger.error(f"Error getting token info for {token_address}: {e}", exc_info=True)
            return {}

    def _generate_token_name(self) -> str:
        """Generate a random token name for testing purposes"""
        prefixes = ["Space", "Moon", "Mars", "Doge", "Shib", "Baby", "Pepe", "Galaxy",
                    "Star", "Rocket", "Diamond", "Gold", "Silver", "Elon", "Safe",
                    "Floki", "Inu", "Solana", "Sol", "Wojak", "Chad", "Based", "AI",
                    "Trump", "Biden", "Pump", "Frog", "Lambo", "Ape", "Monkey"]

        suffixes = ["Coin", "Token", "Finance", "Moon", "Rocket", "Doge", "Inu",
                    "Protocol", "Network", "Chain", "DAO", "Swap", "Yield", "Cash",
                    "Money", "Dollar", "Elon", "Musk", "Capital", "Fund", "Gem",
                    "Diamond", "Hands", "HODL", "Pump", "Dump", "Ape", "Moon"]

        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)

        # Ensure prefix and suffix are different
        while prefix.lower() == suffix.lower():
            suffix = random.choice(suffixes)

        return f"{prefix} {suffix}"

    async def _get_token_creator(self, token_address: str) -> Optional[str]:
        """Get the creator address of a token (simulated for demo)"""
        # In a real implementation, would look up the token's metadata account
        # and extract the creators field

        # For demo, generate a random creator address
        random_bytes = bytearray(random.getrandbits(8) for _ in range(32))
        creator_address = base58.b58encode(bytes(random_bytes)).decode('ascii')

        return creator_address

    async def get_token_supply(self, token_address: str) -> Dict[str, float]:
        """Get the total and circulating supply of a token"""
        self.logger.info(f"Getting token supply for {token_address}")

        try:
            # In a real implementation, would query the token's supply from RPC
            # For demo, generate simulated supply data

            # Get decimals
            token_info = await self.get_token_info(token_address)
            decimals = token_info.get("decimals", 9)

            # Generate total supply (between 1M and 1T tokens)
            total_supply_raw = random.uniform(1e6, 1e12)

            # Calculate circulating supply (70-95% of total)
            circulating_percent = random.uniform(0.7, 0.95)
            circulating_supply_raw = total_supply_raw * circulating_percent

            # Adjust for decimals
            total_supply = total_supply_raw / (10 ** decimals)
            circulating_supply = circulating_supply_raw / (10 ** decimals)

            return {
                "total": total_supply,
                "circulating": circulating_supply,
                "percent_circulating": circulating_percent
            }

        except Exception as e:
            self.logger.error(f"Error getting token supply for {token_address}: {e}", exc_info=True)
            return {"total": 0, "circulating": 0, "percent_circulating": 0}

    async def get_token_price(self, token_address: str) -> Dict[str, float]:
        """Get the current price of a token in USD and SOL"""
        # Check cache (valid for 5 minutes)
        if token_address in self.price_cache:
            price, timestamp = self.price_cache[token_address]
            if (datetime.now() - timestamp) < timedelta(minutes=5):
                return {"price_usd": price, "price_sol": price / 100}  # Assume 1 SOL = $100

        self.logger.info(f"Getting token price for {token_address}")

        try:
            # In a real implementation, would query DEX pools or price oracles
            # For demo, generate simulated price data

            # Generate a price between $0.000001 and $100
            price_range = random.choice([
                (0.000001, 0.0001),  # Very cheap tokens
                (0.0001, 0.01),  # Cheap tokens
                (0.01, 1.0),  # Medium tokens
                (1.0, 100.0)  # Expensive tokens
            ])

            price_usd = random.uniform(price_range[0], price_range[1])

            # Assume 1 SOL = $100 for conversion
            sol_price_usd = 100.0
            price_sol = price_usd / sol_price_usd

            # Cache the result
            self.price_cache[token_address] = (price_usd, datetime.now())

            return {
                "price_usd": price_usd,
                "price_sol": price_sol
            }

        except Exception as e:
            self.logger.error(f"Error getting token price for {token_address}: {e}", exc_info=True)
            return {"price_usd": 0, "price_sol": 0}

    async def get_recent_token_creations(self, limit: int = 20) -> List[str]:
        """Get list of recently created tokens"""
        self.logger.info(f"Getting {limit} recent token creations")

        try:
            # In a real implementation, would query recent transactions for the token program
            # For demo, generate random token addresses

            token_addresses = []
            for _ in range(limit):
                random_bytes = bytearray(random.getrandbits(8) for _ in range(32))
                address = base58.b58encode(bytes(random_bytes)).decode('ascii')
                token_addresses.append(address)

            return token_addresses

        except Exception as e:
            self.logger.error(f"Error getting recent token creations: {e}", exc_info=True)
            return []

    async def get_top_tokens(self, limit: int = 100) -> List[Token]:
        """Get list of top tokens by market cap"""
        self.logger.info(f"Getting top {limit} tokens")

        try:
            # In a real implementation, would query a data service or index
            # For demo, generate random token data

            tokens = []
            for _ in range(limit):
                # Generate a random address
                random_bytes = bytearray(random.getrandbits(8) for _ in range(32))
                address = base58.b58encode(bytes(random_bytes)).decode('ascii')

                # Create token with random name/symbol
                name = self._generate_token_name()
                symbol = name.split()[0].upper()

                token = Token(
                    address=address,
                    name=name,
                    symbol=symbol,
                    decimals=9
                )

                # Generate metrics
                await self._get_token_metrics(token)

                tokens.append(token)

            # Sort by market cap
            tokens.sort(key=lambda t: t.metrics.market_cap, reverse=True)

            return tokens[:limit]

        except Exception as e:
            self.logger.error(f"Error getting top tokens: {e}", exc_info=True)
            return []

    async def _get_token_metrics(self, token: Token):
        """Generate metrics for a token (for demo purposes)"""
        # Generate market cap between $100K and $10B
        market_cap = random.uniform(1e5, 1e10)

        # Set metrics
        token.metrics.market_cap = market_cap
        token.metrics.volume_24h = market_cap * random.uniform(0.01, 0.3)
        token.metrics.price_change_24h = random.uniform(-0.2, 0.3)
        token.metrics.liquidity = market_cap * random.uniform(0.05, 0.4)
        token.metrics.holders_count = int(random.uniform(100, 100000))
        token.metrics.transaction_count_24h = int(random.uniform(10, 10000))

    async def get_token_social_links(self, token_address: str) -> Dict[str, str]:
        """Get social media links for a token"""
        self.logger.info(f"Getting social links for {token_address}")

        try:
            # In a real implementation, would query token metadata or an external API
            # For demo, generate random links

            # Randomly decide which links to include
            has_website = random.random() > 0.1  # 90% have website
            has_twitter = random.random() > 0.2  # 80% have Twitter
            has_telegram = random.random() > 0.3  # 70% have Telegram
            has_discord = random.random() > 0.4  # 60% have Discord

            result = {}

            if has_website:
                token_name = token_address[:8].lower()
                result["website"] = f"https://{token_name}.io"

            if has_twitter:
                token_name = token_address[:8].lower()
                result["twitter"] = f"https://twitter.com/{token_name}"

            if has_telegram:
                token_name = token_address[:8].lower()
                result["telegram"] = f"https://t.me/{token_name}official"

            if has_discord:
                token_name = token_address[:8].lower()
                result["discord"] = f"https://discord.gg/{token_name}"

            return result

        except Exception as e:
            self.logger.error(f"Error getting social links for {token_address}: {e}", exc_info=True)
            return {}

    async def get_token_launch_info(self, token_address: str) -> Dict[str, Any]:
        """Get launch information for a token"""
        self.logger.info(f"Getting launch info for {token_address}")

        try:
            # In a real implementation, would query first transactions or an external API
            # For demo, generate random launch date

            # Random launch date between 1 day and 6 months ago
            max_days_ago = 180
            days_ago = random.randint(1, max_days_ago)
            launch_date = datetime.now() - timedelta(days=days_ago)

            return {
                "launch_date": launch_date,
                "days_since_launch": days_ago
            }

        except Exception as e:
            self.logger.error(f"Error getting launch info for {token_address}: {e}", exc_info=True)
            return {}

    async def get_token_trading_pairs(self, token_address: str) -> Dict[str, str]:
        """Get trading pairs for a token"""
        self.logger.info(f"Getting trading pairs for {token_address}")

        try:
            # In a real implementation, would query DEXes for liquidity pools
            # For demo, generate random trading pairs

            # Common base tokens
            base_tokens = {
                "SOL": "So11111111111111111111111111111111111111112",
                "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                "USDT": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
                "wBTC": "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E",
                "wETH": "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs"
            }

            # Randomly select which pairs to include
            num_pairs = random.randint(1, len(base_tokens))
            selected_pairs = random.sample(list(base_tokens.items()), num_pairs)

            # Create trading pairs
            trading_pairs = {}
            for symbol, address in selected_pairs:
                # Generate a random pool address
                random_bytes = bytearray(random.getrandbits(8) for _ in range(32))
                pool_address = base58.b58encode(bytes(random_bytes)).decode('ascii')

                trading_pairs[symbol] = pool_address

            return trading_pairs

        except Exception as e:
            self.logger.error(f"Error getting trading pairs for {token_address}: {e}", exc_info=True)
            return {}

    async def get_wallet_info(self, wallet_address: str) -> Dict[str, Any]:
        """Get information about a wallet"""
        self.logger.info(f"Getting wallet info for {wallet_address}")

        try:
            # In a real implementation, would query account balance and history
            # For demo, generate random wallet data

            # Random SOL balance
            sol_balance = random.uniform(0.1, 1000)

            # Random first seen date
            max_days_ago = 365
            days_ago = random.randint(1, max_days_ago)
            first_seen = datetime.now() - timedelta(days=days_ago)

            return {
                "address": wallet_address,
                "sol_balance": sol_balance,
                "first_seen": first_seen
            }

        except Exception as e:
            self.logger.error(f"Error getting wallet info for {wallet_address}: {e}", exc_info=True)
            return {}

    async def get_tokens_created_by_wallet(self, wallet_address: str) -> List[str]:
        """Get list of tokens created by a wallet"""
        self.logger.info(f"Getting tokens created by wallet {wallet_address}")

        try:
            # In a real implementation, would query for token mints created by the wallet
            # For demo, generate random number of created tokens

            # Random number of created tokens (0-10)
            num_tokens = random.randint(0, 10)

            # Generate random token addresses
            created_tokens = []
            for _ in range(num_tokens):
                random_bytes = bytearray(random.getrandbits(8) for _ in range(32))
                token_address = base58.b58encode(bytes(random_bytes)).decode('ascii')
                created_tokens.append(token_address)

            return created_tokens

        except Exception as e:
            self.logger.error(f"Error getting tokens created by wallet {wallet_address}: {e}", exc_info=True)
            return []

    async def subscribe_to_program(self, program_id: str, callback):
        """Subscribe to program account updates"""
        self.logger.info(f"Subscribing to program {program_id}")

        try:
            # In a real implementation, would establish a WebSocket connection
            # and subscribe to program account updates

            # For demo, just log the subscription
            self.logger.info(f"Subscribed to program {program_id} (simulated)")

            # Return a subscription ID (would be used to unsubscribe)
            return random.randint(1000, 9999)

        except Exception as e:
            self.logger.error(f"Error subscribing to program {program_id}: {e}", exc_info=True)
            return None

    async def get_token_holders(self, token_address: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get top holders of a token"""
        self.logger.info(f"Getting top {limit} holders for token {token_address}")

        try:
            # In a real implementation, would query token accounts for the mint
            # For demo, generate random holder data

            # Generate total supply
            token_info = await self.get_token_info(token_address)
            supply_info = await self.get_token_supply(token_address)
            circulating_supply = supply_info.get("circulating", 0)
            decimals = token_info.get("decimals", 9)

            # Generate random holder count
            holder_count = random.randint(50, 10000)
            actual_limit = min(limit, holder_count)

            # Top holders typically follow a power law distribution
            # A few wallets hold most of the supply
            holders = []
            remaining_supply = circulating_supply

            for i in range(actual_limit):
                # Generate random wallet address
                random_bytes = bytearray(random.getrandbits(8) for _ in range(32))
                wallet_address = base58.b58encode(bytes(random_bytes)).decode('ascii')

                # Calculate balance (larger for earlier wallets)
                if i == 0:
                    # Top holder has 5-30% of supply
                    percent = random.uniform(0.05, 0.3)
                elif i < 5:
                    # Next holders have 1-10% each
                    percent = random.uniform(0.01, 0.1)
                elif i < 20:
                    # Next holders have 0.1-1% each
                    percent = random.uniform(0.001, 0.01)
                else:
                    # Remaining holders have smaller amounts
                    percent = random.uniform(0.0001, 0.001)

                # Ensure we don't exceed remaining supply
                balance = min(circulating_supply * percent, remaining_supply)
                remaining_supply -= balance

                holders.append({
                    "address": wallet_address,
                    "balance": balance,
                    "percent": balance / circulating_supply if circulating_supply > 0 else 0
                })

                if remaining_supply <= 0:
                    break

            return holders

        except Exception as e:
            self.logger.error(f"Error getting token holders for {token_address}: {e}", exc_info=True)
            return []

    async def get_historical_price(self, token_address: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get historical price data for a token"""
        self.logger.info(f"Getting {days} days of historical price data for {token_address}")

        try:
            # In a real implementation, would query historical price data from an API
            # For demo, generate random price history

            # Get current price
            current_price = await self.get_token_price(token_address)
            price_usd = current_price.get("price_usd", 0)

            # Generate price history
            history = []
            current_time = datetime.now()

            # Start price (50-200% of current price)
            start_factor = random.uniform(0.5, 2.0)
            start_price = price_usd * start_factor

            # Volatility factor (higher for meme coins)
            volatility = random.uniform(0.05, 0.3)

            for i in range(days, -1, -1):
                # Calculate date
                date = current_time - timedelta(days=i)

                # Calculate price with random walk
                if i == days:
                    # Start price
                    day_price = start_price
                else:
                    # Previous price with random change
                    prev_price = history[-1]["price_usd"]
                    change = random.normalvariate(0, volatility)
                    day_price = prev_price * (1 + change)

                # Ensure price is positive
                day_price = max(day_price, 0.000000001)

                # Add to history
                history.append({
                    "date": date.isoformat(),
                    "price_usd": day_price
                })

            return history

        except Exception as e:
            self.logger.error(f"Error getting historical price for {token_address}: {e}", exc_info=True)
            return []

    async def get_transaction_history(self, address: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get transaction history for an address (token or wallet)"""
        self.logger.info(f"Getting transaction history for {address}")

        try:
            # In a real implementation, would query the transaction history
            # For demo, generate random transactions

            history = []
            current_time = datetime.now()

            for i in range(limit):
                # Random time in the past (up to 30 days ago)
                time_delta = random.uniform(0, 30 * 24 * 60 * 60)  # seconds
                tx_time = current_time - timedelta(seconds=time_delta)

                # Random transaction type
                tx_type = random.choice([
                    "transfer", "swap", "mint", "burn", "stake", "unstake", "create"
                ])

                # Random transaction hash
                random_bytes = bytearray(random.getrandbits(8) for _ in range(32))
                tx_hash = base58.b58encode(bytes(random_bytes)).decode('ascii')

                # Random amount
                amount = random.uniform(1, 1000000)

                # Random status (mostly successful)
                status = "success" if random.random() > 0.1 else "failed"

                history.append({
                    "signature": tx_hash,
                    "time": tx_time.isoformat(),
                    "type": tx_type,
                    "amount": amount,
                    "status": status
                })

            # Sort by time (most recent first)
            history.sort(key=lambda tx: tx["time"], reverse=True)

            return history

        except Exception as e:
            self.logger.error(f"Error getting transaction history for {address}: {e}", exc_info=True)
            return []