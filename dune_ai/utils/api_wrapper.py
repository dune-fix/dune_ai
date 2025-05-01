import logging
import aiohttp
import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from config.logging_config import get_logger


class ApiWrapper:
    """
    Wrapper for various external APIs used by the DUNE AI system.

    Provides unified interfaces for interacting with social media,
    blockchain explorers, and market data services.
    """

    def __init__(self):
        self.logger = get_logger("api_wrapper")
        self.session = None
        self.rate_limits = {
            "twitter": {"calls": 0, "reset_time": datetime.now()},
            "telegram": {"calls": 0, "reset_time": datetime.now()},
            "coinmarketcap": {"calls": 0, "reset_time": datetime.now()},
            "coingecko": {"calls": 0, "reset_time": datetime.now()},
            "solscan": {"calls": 0, "reset_time": datetime.now()}
        }
        self.logger.info("ApiWrapper initialized")

    async def initialize(self):
        """Initialize the API wrapper"""
        self.session = aiohttp.ClientSession()
        self.logger.info("API session initialized")

    async def close(self):
        """Close the API session"""
        if self.session:
            await self.session.close()
            self.session = None
            self.logger.info("API session closed")

    async def check_rate_limit(self, api_name: str) -> bool:
        """Check if rate limit allows another call"""
        if api_name not in self.rate_limits:
            return True

        rate_info = self.rate_limits[api_name]

        # Check if reset time has passed
        if datetime.now() > rate_info["reset_time"]:
            # Reset counter
            rate_info["calls"] = 0

            # Set new reset time (typically 15-minute windows)
            rate_info["reset_time"] = datetime.now() + timedelta(minutes=15)

        # Check current count against limit
        max_calls = self._get_max_calls(api_name)

        if rate_info["calls"] >= max_calls:
            self.logger.warning(f"Rate limit reached for {api_name}")
            return False

        # Increment counter
        rate_info["calls"] += 1
        return True

    def _get_max_calls(self, api_name: str) -> int:
        """Get maximum calls per window for an API"""
        # Default limits (in 15-minute windows)
        limits = {
            "twitter": 300,  # Standard API limit
            "telegram": 100,  # Estimated reasonable limit
            "coinmarketcap": 30,
            "coingecko": 50,
            "solscan": 100
        }

        return limits.get(api_name, 100)  # Default to 100 if not specified

    async def get_twitter_data(self, query: str, count: int = 100) -> List[Dict[str, Any]]:
        """Get data from Twitter API"""
        self.logger.info(f"Getting Twitter data for query: {query}")

        if not await self.check_rate_limit("twitter"):
            self.logger.warning("Twitter rate limit exceeded, returning empty result")
            return []

        # In a real implementation, would make actual API calls
        # For demo, simulate response

        # Simulated tweets
        simulated_tweets = []

        for i in range(min(count, 20)):  # Simulate up to 20 tweets
            tweet = {
                "id": f"12345{i}",
                "text": f"This is a simulated tweet about {query} #{i}",
                "created_at": (datetime.now() - timedelta(hours=i)).isoformat(),
                "user": {
                    "id": f"user{i}",
                    "screen_name": f"user{i}",
                    "followers_count": 100 * (i + 1)
                },
                "retweet_count": i * 5,
                "favorite_count": i * 10
            }

            simulated_tweets.append(tweet)

        return simulated_tweets

    async def get_telegram_data(self, channel: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get data from Telegram channel"""
        self.logger.info(f"Getting Telegram data from channel: {channel}")

        if not await self.check_rate_limit("telegram"):
            self.logger.warning("Telegram rate limit exceeded, returning empty result")
            return []

        # In a real implementation, would make actual API calls
        # For demo, simulate response

        # Simulated messages
        simulated_messages = []

        for i in range(min(limit, 10)):  # Simulate up to 10 messages
            message = {
                "id": i,
                "text": f"Simulated Telegram message in {channel} #{i}",
                "date": int((datetime.now() - timedelta(hours=i)).timestamp()),
                "views": 100 * (i + 1),
                "forwards": i * 2
            }

            simulated_messages.append(message)

        return simulated_messages

    async def get_coinmarketcap_data(self, token_id: Optional[str] = None) -> Dict[str, Any]:
        """Get data from CoinMarketCap API"""
        self.logger.info(f"Getting CoinMarketCap data for token: {token_id or 'all'}")

        if not await self.check_rate_limit("coinmarketcap"):
            self.logger.warning("CoinMarketCap rate limit exceeded, returning empty result")
            return {}

        # In a real implementation, would make actual API calls
        # For demo, simulate response

        if token_id:
            # Simulated single token data
            return {
                "id": token_id,
                "name": f"Token {token_id}",
                "symbol": f"TK{token_id}",
                "rank": 100,
                "price_usd": 1.23,
                "market_cap_usd": 123000000,
                "volume_24h_usd": 45000000,
                "percent_change_24h": 5.67,
                "last_updated": datetime.now().isoformat()
            }
        else:
            # Simulated market data
            return {
                "total_market_cap_usd": 1234567890000,
                "total_24h_volume_usd": 98765432100,
                "bitcoin_dominance_percentage": 45.6,
                "last_updated": datetime.now().isoformat()
            }

    async def get_coingecko_data(self, token_id: Optional[str] = None) -> Dict[str, Any]:
        """Get data from CoinGecko API"""
        self.logger.info(f"Getting CoinGecko data for token: {token_id or 'all'}")

        if not await self.check_rate_limit("coingecko"):
            self.logger.warning("CoinGecko rate limit exceeded, returning empty result")
            return {}

        # In a real implementation, would make actual API calls
        # For demo, simulate response

        if token_id:
            # Simulated single token data
            return {
                "id": token_id,
                "name": f"Token {token_id}",
                "symbol": f"TK{token_id}",
                "current_price": 2.34,
                "market_cap": 234000000,
                "total_volume": 56000000,
                "price_change_percentage_24h": 6.78,
                "last_updated": datetime.now().isoformat()
            }
        else:
            # Simulated global data
            return {
                "active_cryptocurrencies": 12345,
                "total_market_cap": {
                    "usd": 2345678901000
                },
                "total_volume": {
                    "usd": 87654321000
                },
                "market_cap_percentage": {
                    "btc": 45.6,
                    "eth": 18.9
                },
                "last_updated": datetime.now().isoformat()
            }

    async def get_solscan_data(self, address: str) -> Dict[str, Any]:
        """Get data from Solscan API"""
        self.logger.info(f"Getting Solscan data for address: {address}")

        if not await self.check_rate_limit("solscan"):
            self.logger.warning("Solscan rate limit exceeded, returning empty result")
            return {}

        # In a real implementation, would make actual API calls
        # For demo, simulate response

        # Check if address format looks like a token vs wallet
        if len(address) > 40:  # Likely a token address
            return {
                "success": True,
                "data": {
                    "address": address,
                    "name": f"Token {address[:5]}",
                    "symbol": f"TK{address[:3]}",
                    "decimals": 9,
                    "supply": 1000000000,
                    "holder_count": 12345,
                    "volume_24h": 789000,
                    "price_sol": 0.0123,
                    "price_usd": 1.23
                }
            }
        else:  # Likely a wallet address
            return {
                "success": True,
                "data": {
                    "address": address,
                    "balance_sol": 123.45,
                    "token_count": 67,
                    "nft_count": 8,
                    "tx_count": 910,
                    "first_tx_time": (datetime.now() - timedelta(days=90)).isoformat()
                }
            }

    async def make_generic_request(self, url: str, method: str = "GET",
                                   data: Any = None, headers: Dict[str, str] = None) -> Dict[str, Any]:
        """Make a generic API request"""
        self.logger.info(f"Making {method} request to {url}")

        if not self.session:
            await self.initialize()

        try:
            if method == "GET":
                async with self.session.get(url, headers=headers) as response:
                    if response.status != 200:
                        self.logger.error(f"API request failed: {response.status}")
                        return {"error": f"Request failed with status {response.status}"}

                    return await response.json()

            elif method == "POST":
                async with self.session.post(url, json=data, headers=headers) as response:
                    if response.status != 200:
                        self.logger.error(f"API request failed: {response.status}")
                        return {"error": f"Request failed with status {response.status}"}

                    return await response.json()

            else:
                self.logger.error(f"Unsupported method: {method}")
                return {"error": f"Unsupported method: {method}"}

        except aiohttp.ClientError as e:
            self.logger.error(f"API client error: {e}")
            return {"error": str(e)}

        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return {"error": str(e)}