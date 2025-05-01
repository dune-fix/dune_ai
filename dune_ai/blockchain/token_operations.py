import logging
import random
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from config.logging_config import get_logger
from dune_ai.blockchain.solana_client import SolanaClient


class TokenOperations:
    """
    Utility class for performing operations related to Solana tokens,
    such as calculating metrics, analyzing price movements, and detecting
    patterns in token behavior.
    """

    def __init__(self, solana_client: SolanaClient):
        self.logger = get_logger("token_operations")
        self.solana_client = solana_client
        self.logger.info("TokenOperations initialized")

    async def calculate_token_metrics(self, token_address: str) -> Dict[str, Any]:
        """Calculate comprehensive metrics for a token"""
        self.logger.info(f"Calculating metrics for token {token_address}")

        try:
            # Get basic token info
            token_info = await self.solana_client.get_token_info(token_address)
            if not token_info:
                self.logger.warning(f"Could not retrieve info for token {token_address}")
                return {}

            # Get supply info
            supply_info = await self.solana_client.get_token_supply(token_address)
            circulating_supply = supply_info.get("circulating", 0)

            # Get price info
            price_data = await self.solana_client.get_token_price(token_address)
            price_usd = price_data.get("price_usd", 0)

            # Get holders
            holders = await self.solana_client.get_token_holders(token_address, limit=10)
            holder_count = len(holders)

            # Calculate market cap
            market_cap = circulating_supply * price_usd

            # Get historical price for calculations
            price_history = await self.solana_client.get_historical_price(token_address, days=7)

            # Calculate price change over 24h
            if len(price_history) >= 2:
                current_price = price_history[-1]["price_usd"]
                day_ago_price = price_history[-2]["price_usd"]
                price_change_24h = (current_price - day_ago_price) / day_ago_price if day_ago_price > 0 else 0
            else:
                price_change_24h = 0

            # Calculate volatility (standard deviation of daily returns)
            volatility = self._calculate_volatility(price_history)

            # Calculate liquidity (estimated)
            liquidity = market_cap * random.uniform(0.05, 0.3)  # Simplified estimation

            # Get transaction history to estimate volume
            transactions = await self.solana_client.get_transaction_history(token_address, limit=100)
            volume_24h = self._estimate_volume_from_transactions(transactions)

            # Calculate concentration (% held by top 5 holders)
            concentration = self._calculate_holder_concentration(holders, circulating_supply)

            # Calculate risk score
            risk_score = self._calculate_risk_score(
                volatility,
                concentration,
                market_cap,
                price_change_24h
            )

            # Return combined metrics
            return {
                "market_cap": market_cap,
                "price_usd": price_usd,
                "circulating_supply": circulating_supply,
                "holder_count": holder_count,
                "price_change_24h": price_change_24h,
                "volume_24h": volume_24h,
                "liquidity": liquidity,
                "volatility": volatility,
                "concentration": concentration,
                "risk_score": risk_score
            }

        except Exception as e:
            self.logger.error(f"Error calculating metrics for token {token_address}: {e}", exc_info=True)
            return {}

    def _calculate_volatility(self, price_history: List[Dict[str, Any]]) -> float:
        """Calculate price volatility from historical data"""
        if len(price_history) < 2:
            return 0

        # Calculate daily returns
        returns = []
        for i in range(1, len(price_history)):
            prev_price = price_history[i - 1]["price_usd"]
            curr_price = price_history[i]["price_usd"]

            if prev_price > 0:
                daily_return = (curr_price - prev_price) / prev_price
                returns.append(daily_return)

        # Calculate standard deviation of returns
        if returns:
            mean_return = sum(returns) / len(returns)
            squared_diffs = [(r - mean_return) ** 2 for r in returns]
            variance = sum(squared_diffs) / len(returns)
            volatility = variance ** 0.5
            return volatility

        return 0

    def _estimate_volume_from_transactions(self, transactions: List[Dict[str, Any]]) -> float:
        """Estimate 24h trading volume from transaction history"""
        # Filter transactions from last 24 hours
        now = datetime.now()
        day_ago = now - timedelta(days=1)

        recent_txs = [
            tx for tx in transactions
            if tx["type"] in ["transfer", "swap"] and
               datetime.fromisoformat(tx["time"]) > day_ago and
               tx["status"] == "success"
        ]

        # Sum amounts
        volume = sum(tx["amount"] for tx in recent_txs)

        return volume

    def _calculate_holder_concentration(self, holders: List[Dict[str, Any]], total_supply: float) -> float:
        """Calculate concentration of holdings (higher is more concentrated)"""
        if not holders or total_supply <= 0:
            return 0

        # Sum percentage held by top 5 holders
        top_holders = holders[:5]
        top_5_percent = sum(holder["percent"] for holder in top_holders)

        return top_5_percent

    def _calculate_risk_score(self, volatility: float, concentration: float,
                              market_cap: float, price_change_24h: float) -> float:
        """Calculate overall risk score from multiple factors (0.0-1.0)"""
        # Factor 1: Volatility (higher volatility = higher risk)
        volatility_score = min(volatility * 5, 1.0)  # Cap at 1.0

        # Factor 2: Concentration (higher concentration = higher risk)
        concentration_score = concentration

        # Factor 3: Market cap (lower market cap = higher risk)
        # Normalize market cap between 0 and 1 (0 for large caps, 1 for small caps)
        market_cap_score = 1.0
        if market_cap > 0:
            # Log scale for market cap
            log_market_cap = min(9, max(0, (market_cap / 1e6).bit_length()))
            market_cap_score = 1.0 - (log_market_cap / 9.0)

        # Factor 4: Extreme price movement
        price_movement_score = min(abs(price_change_24h), 1.0)

        # Weighted average of factors
        risk_score = (
                volatility_score * 0.3 +
                concentration_score * 0.3 +
                market_cap_score * 0.3 +
                price_movement_score * 0.1
        )

        # Ensure result is in range 0.0-1.0
        return max(0.0, min(1.0, risk_score))

    async def analyze_price_pattern(self, token_address: str) -> Dict[str, Any]:
        """Analyze price patterns for a token"""
        self.logger.info(f"Analyzing price patterns for token {token_address}")

        try:
            # Get price history (30 days)
            price_history = await self.solana_client.get_historical_price(token_address, days=30)

            if len(price_history) < 7:
                self.logger.warning(f"Insufficient price history for token {token_address}")
                return {"pattern": "unknown", "confidence": 0}

            # Extract prices
            prices = [entry["price_usd"] for entry in price_history]

            # Calculate trends
            trends = self._identify_price_trends(prices)

            # Calculate moving averages
            short_ma = self._calculate_moving_average(prices, 7)
            long_ma = self._calculate_moving_average(prices, 14)

            # Check for common patterns
            patterns = []

            # Pattern 1: Uptrend
            if trends["overall"] > 0.1:
                patterns.append(("uptrend", min(trends["overall"] * 5, 1.0)))

            # Pattern 2: Downtrend
            elif trends["overall"] < -0.1:
                patterns.append(("downtrend", min(abs(trends["overall"]) * 5, 1.0)))

            # Pattern 3: Sideways/Consolidation
            elif abs(trends["overall"]) <= 0.1 and trends["volatility"] < 0.1:
                patterns.append(("consolidation", 1.0 - trends["volatility"] * 5))

            # Pattern 4: Double bottom
            if self._check_double_bottom(prices):
                patterns.append(("double_bottom", 0.7))

            # Pattern 5: Head and shoulders
            if self._check_head_and_shoulders(prices):
                patterns.append(("head_and_shoulders", 0.6))

            # Pattern 6: Golden cross (short MA crosses above long MA)
            if self._check_moving_average_cross(short_ma, long_ma, "golden"):
                patterns.append(("golden_cross", 0.8))

            # Pattern 7: Death cross (short MA crosses below long MA)
            if self._check_moving_average_cross(short_ma, long_ma, "death"):
                patterns.append(("death_cross", 0.8))

            # Get the most confident pattern
            if patterns:
                patterns.sort(key=lambda x: x[1], reverse=True)
                top_pattern, confidence = patterns[0]
            else:
                top_pattern = "no_clear_pattern"
                confidence = 0.0

            return {
                "pattern": top_pattern,
                "confidence": confidence,
                "all_patterns": patterns,
                "trend": trends["overall"],
                "volatility": trends["volatility"]
            }

        except Exception as e:
            self.logger.error(f"Error analyzing price patterns for token {token_address}: {e}", exc_info=True)
            return {"pattern": "error", "confidence": 0}

    def _identify_price_trends(self, prices: List[float]) -> Dict[str, float]:
        """Identify trend direction and strength from price data"""
        if len(prices) < 2:
            return {"overall": 0, "recent": 0, "volatility": 0}

        # Calculate overall trend (first to last)
        first_price = prices[0]
        last_price = prices[-1]

        if first_price > 0:
            overall_change = (last_price - first_price) / first_price
        else:
            overall_change = 0

        # Calculate recent trend (last 7 days or fewer)
        recent_prices = prices[-min(7, len(prices)):]
        recent_first = recent_prices[0]
        recent_last = recent_prices[-1]

        if recent_first > 0:
            recent_change = (recent_last - recent_first) / recent_first
        else:
            recent_change = 0

        # Calculate volatility
        price_changes = []
        for i in range(1, len(prices)):
            if prices[i - 1] > 0:
                change = abs((prices[i] - prices[i - 1]) / prices[i - 1])
                price_changes.append(change)

        volatility = sum(price_changes) / len(price_changes) if price_changes else 0

        return {
            "overall": overall_change,
            "recent": recent_change,
            "volatility": volatility
        }

    def _calculate_moving_average(self, prices: List[float], window: int) -> List[float]:
        """Calculate simple moving average with given window"""
        ma = []

        for i in range(len(prices)):
            if i < window - 1:
                # Not enough data for full window
                ma.append(None)
            else:
                # Calculate average of window
                window_avg = sum(prices[i - (window - 1):i + 1]) / window
                ma.append(window_avg)

        return ma

    def _check_double_bottom(self, prices: List[float]) -> bool:
        """Check for double bottom pattern (W shape)"""
        # Simplified check - in real implementation would be more sophisticated
        if len(prices) < 10:
            return False

        # Divide data into quarters
        quarter_size = len(prices) // 4

        # Check if first and third quarters have lows
        q1_low = min(prices[:quarter_size * 2])
        q3_low = min(prices[quarter_size * 2:])

        # Check if middle has a peak
        q2_high = max(prices[quarter_size:quarter_size * 3])

        # Check pattern: low -> high -> similar low
        if q2_high > q1_low * 1.1 and abs(q1_low - q3_low) / q1_low < 0.1:
            return True

        return False

    def _check_head_and_shoulders(self, prices: List[float]) -> bool:
        """Check for head and shoulders pattern"""
        # Simplified check - in real implementation would be more sophisticated
        if len(prices) < 15:
            return False

        third = len(prices) // 3

        # Check for three peaks with middle one higher
        left = max(prices[:third])
        middle = max(prices[third:third * 2])
        right = max(prices[third * 2:])

        # Check if middle peak is higher and other peaks are similar
        if middle > left * 1.1 and middle > right * 1.1 and abs(left - right) / left < 0.2:
            return True

        return False

    def _check_moving_average_cross(self, short_ma: List[float], long_ma: List[float],
                                    cross_type: str) -> bool:
        """Check for moving average crossover"""
        # Need at least 2 valid points in both MAs
        valid_points = 0
        for i in range(len(short_ma)):
            if short_ma[i] is not None and long_ma[i] is not None:
                valid_points += 1

        if valid_points < 2:
            return False

        # Find last two valid points
        last_valid = []
        for i in range(len(short_ma) - 1, -1, -1):
            if short_ma[i] is not None and long_ma[i] is not None:
                last_valid.append((short_ma[i], long_ma[i]))
                if len(last_valid) >= 2:
                    break

        if len(last_valid) < 2:
            return False

        # Check for crossover
        if cross_type == "golden":
            # Short MA crosses above long MA
            return (last_valid[0][0] > last_valid[0][1] and  # Current: short > long
                    last_valid[1][0] < last_valid[1][1])  # Previous: short < long

        elif cross_type == "death":
            # Short MA crosses below long MA
            return (last_valid[0][0] < last_valid[0][1] and  # Current: short < long
                    last_valid[1][0] > last_valid[1][1])  # Previous: short > long

        return False

    async def compare_tokens(self, token_address1: str, token_address2: str) -> Dict[str, Any]:
        """Compare two tokens on various metrics"""
        self.logger.info(f"Comparing tokens {token_address1} and {token_address2}")

        try:
            # Get metrics for both tokens
            metrics1 = await self.calculate_token_metrics(token_address1)
            metrics2 = await self.calculate_token_metrics(token_address2)

            # Get token info
            token_info1 = await self.solana_client.get_token_info(token_address1)
            token_info2 = await self.solana_client.get_token_info(token_address2)

            # Get historical price for both
            history1 = await self.solana_client.get_historical_price(token_address1, days=30)
            history2 = await self.solana_client.get_historical_price(token_address2, days=30)

            # Calculate price correlation
            correlation = self._calculate_price_correlation(history1, history2)

            # Compare metrics
            comparison = {
                "token1": {
                    "address": token_address1,
                    "name": token_info1.get("name", "Unknown"),
                    "symbol": token_info1.get("symbol", "UNKNOWN"),
                    "metrics": metrics1
                },
                "token2": {
                    "address": token_address2,
                    "name": token_info2.get("name", "Unknown"),
                    "symbol": token_info2.get("symbol", "UNKNOWN"),
                    "metrics": metrics2
                },
                "comparison": {
                    "price_correlation": correlation,
                    "market_cap_ratio": metrics1.get("market_cap", 0) / max(metrics2.get("market_cap", 1), 1),
                    "volume_ratio": metrics1.get("volume_24h", 0) / max(metrics2.get("volume_24h", 1), 1),
                    "volatility_difference": metrics1.get("volatility", 0) - metrics2.get("volatility", 0),
                    "risk_difference": metrics1.get("risk_score", 0) - metrics2.get("risk_score", 0)
                }
            }

            return comparison

        except Exception as e:
            self.logger.error(f"Error comparing tokens: {e}", exc_info=True)
            return {}

    def _calculate_price_correlation(self, history1: List[Dict[str, Any]],
                                     history2: List[Dict[str, Any]]) -> float:
        """Calculate price correlation between two tokens"""
        # Extract prices
        dates1 = [entry["date"] for entry in history1]
        dates2 = [entry["date"] for entry in history2]

        # Find common dates
        common_dates = set(dates1).intersection(set(dates2))

        if len(common_dates) < 5:
            return 0  # Not enough data for correlation

        # Create price series for common dates
        prices1 = []
        prices2 = []

        for date in sorted(common_dates):
            # Find price for token 1
            for entry in history1:
                if entry["date"] == date:
                    prices1.append(entry["price_usd"])
                    break

            # Find price for token 2
            for entry in history2:
                if entry["date"] == date:
                    prices2.append(entry["price_usd"])
                    break

        # Calculate correlation
        if len(prices1) < 5:
            return 0

        # Calculate means
        mean1 = sum(prices1) / len(prices1)
        mean2 = sum(prices2) / len(prices2)

        # Calculate covariance and standard deviations
        cov = 0
        std1 = 0
        std2 = 0

        for i in range(len(prices1)):
            diff1 = prices1[i] - mean1
            diff2 = prices2[i] - mean2

            cov += diff1 * diff2
            std1 += diff1 ** 2
            std2 += diff2 ** 2

        if std1 == 0 or std2 == 0:
            return 0

        correlation = cov / ((std1 ** 0.5) * (std2 ** 0.5))

        return correlation