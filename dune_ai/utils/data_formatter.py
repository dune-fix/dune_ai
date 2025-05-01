import logging
import json
from typing import Dict, List, Any, Union
from datetime import datetime

from config.logging_config import get_logger


class DataFormatter:
    """
    Utility for formatting and standardizing data across the DUNE AI system.

    Provides methods for converting between different formats,
    standardizing timestamps, and preparing data for API responses.
    """

    def __init__(self):
        self.logger = get_logger("data_formatter")
        self.logger.info("DataFormatter initialized")

    def format_token_data(self, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format token data for API responses"""
        self.logger.debug(f"Formatting token data: {token_data.get('symbol', 'unknown')}")

        # Ensure all expected fields are present
        formatted = {
            "address": token_data.get("address", ""),
            "name": token_data.get("name", "Unknown"),
            "symbol": token_data.get("symbol", "UNKNOWN"),
            "decimals": token_data.get("decimals", 9),
            "price": {
                "usd": self._format_price(token_data.get("price_usd", 0)),
                "sol": self._format_price(token_data.get("price_sol", 0))
            },
            "market_data": {
                "market_cap": self._format_large_number(token_data.get("market_cap", 0)),
                "volume_24h": self._format_large_number(token_data.get("volume_24h", 0)),
                "price_change_24h": self._format_percentage(token_data.get("price_change_24h", 0)),
                "liquidity": self._format_large_number(token_data.get("liquidity", 0))
            },
            "social": {
                "website": token_data.get("website", ""),
                "twitter": token_data.get("twitter", ""),
                "telegram": token_data.get("telegram", ""),
                "discord": token_data.get("discord", "")
            },
            "metrics": {
                "holders_count": token_data.get("holders_count", 0),
                "transactions_24h": token_data.get("transactions_24h", 0),
                "is_meme_coin": token_data.get("is_meme_coin", False),
                "trend_score": self._format_score(token_data.get("trend_score", 0)),
                "risk_score": self._format_score(token_data.get("risk_score", 0))
            },
            "sentiment": {
                "overall": self._format_score(token_data.get("sentiment_overall", 0)),
                "positive": self._format_percentage(token_data.get("sentiment_positive", 0)),
                "negative": self._format_percentage(token_data.get("sentiment_negative", 0)),
                "neutral": self._format_percentage(token_data.get("sentiment_neutral", 0))
            },
            "timestamps": {
                "launch_date": self._format_timestamp(token_data.get("launch_date")),
                "last_updated": self._format_timestamp(token_data.get("last_updated", datetime.now()))
            }
        }

        return formatted

    def format_wallet_data(self, wallet_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format wallet data for API responses"""
        self.logger.debug(f"Formatting wallet data: {wallet_data.get('address', 'unknown')}")

        # Ensure all expected fields are present
        formatted = {
            "address": wallet_data.get("address", ""),
            "balance": {
                "sol": self._format_price(wallet_data.get("sol_balance", 0)),
                "tokens_count": wallet_data.get("token_count", 0)
            },
            "activity": {
                "transaction_count": wallet_data.get("transaction_count", 0),
                "first_activity": self._format_timestamp(wallet_data.get("first_activity")),
                "last_activity": self._format_timestamp(wallet_data.get("last_activity")),
                "transaction_volume_sol": self._format_price(wallet_data.get("transaction_volume_sol", 0))
            },
            "profile": {
                "is_whale": wallet_data.get("is_whale", False),
                "is_developer": wallet_data.get("is_developer", False),
                "is_contract_deployer": wallet_data.get("is_contract_deployer", False),
                "risk_score": self._format_score(wallet_data.get("risk_score", 0)),
                "influence_score": self._format_score(wallet_data.get("influence_score", 0))
            },
            "created_tokens_count": wallet_data.get("created_tokens_count", 0),
            "timestamps": {
                "first_seen": self._format_timestamp(wallet_data.get("first_seen")),
                "last_updated": self._format_timestamp(wallet_data.get("last_updated", datetime.now()))
            }
        }

        return formatted

    def format_trend_data(self, trend_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format trend analysis data for API responses"""
        self.logger.debug("Formatting trend data")

        # Ensure all expected fields are present
        formatted = {
            "summary": {
                "trend": trend_data.get("trend", "unknown"),
                "strength": self._format_score(trend_data.get("strength", 0)),
                "confidence": self._format_score(trend_data.get("confidence", 0))
            },
            "top_trending": [
                {
                    "symbol": token.get("symbol", "UNKNOWN"),
                    "address": token.get("address", ""),
                    "trend_score": self._format_score(token.get("trend_score", 0)),
                    "price_change_24h": self._format_percentage(token.get("price_change_24h", 0))
                }
                for token in trend_data.get("top_trending", [])
            ],
            "sentiment": {
                "overall": self._format_score(trend_data.get("sentiment_overall", 0)),
                "positive_ratio": self._format_percentage(trend_data.get("sentiment_positive", 0)),
                "negative_ratio": self._format_percentage(trend_data.get("sentiment_negative", 0))
            },
            "patterns": trend_data.get("patterns", []),
            "timestamps": {
                "analyzed_at": self._format_timestamp(trend_data.get("analyzed_at", datetime.now())),
                "data_period": trend_data.get("data_period", "24h")
            }
        }

        return formatted

    def format_sentiment_data(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format sentiment analysis data for API responses"""
        self.logger.debug("Formatting sentiment data")

        # Ensure all expected fields are present
        formatted = {
            "overall_score": self._format_score(sentiment_data.get("overall_score", 0)),
            "category": sentiment_data.get("category", "neutral"),
            "distribution": {
                "positive": self._format_percentage(sentiment_data.get("positive_ratio", 0)),
                "negative": self._format_percentage(sentiment_data.get("negative_ratio", 0)),
                "neutral": self._format_percentage(sentiment_data.get("neutral_ratio", 0))
            },
            "token_sentiments": [
                {
                    "symbol": token.get("token", "UNKNOWN"),
                    "sentiment_score": self._format_score(token.get("sentiment_score", 0)),
                    "mention_count": token.get("mention_count", 0),
                    "relative_sentiment": self._format_score(token.get("relative_sentiment", 0))
                }
                for token in sentiment_data.get("token_sentiments", [])
            ],
            "source_breakdown": sentiment_data.get("source_breakdown", {
                "twitter": self._format_percentage(0.6),
                "telegram": self._format_percentage(0.3),
                "discord": self._format_percentage(0.1)
            }),
            "timestamps": {
                "analyzed_at": self._format_timestamp(sentiment_data.get("analyzed_at", datetime.now())),
                "data_period": sentiment_data.get("data_period", "24h")
            }
        }

        return formatted

    def _format_price(self, price: Union[int, float]) -> str:
        """Format price values with appropriate precision"""
        if price is None:
            return "0.00"

        # Format based on price magnitude
        if price >= 1:
            # For larger prices, show 2 decimal places
            return f"{price:.2f}"
        elif price >= 0.01:
            # For medium prices, show 4 decimal places
            return f"{price:.4f}"
        elif price >= 0.0001:
            # For small prices, show 6 decimal places
            return f"{price:.6f}"
        else:
            # For very small prices, use scientific notation
            return f"{price:.8f}"

    def _format_large_number(self, number: Union[int, float]) -> str:
        """Format large numbers with appropriate suffixes (K, M, B, T)"""
        if number is None:
            return "0"

        # Handle negative numbers
        sign = "-" if number < 0 else ""
        number = abs(number)

        # Format based on magnitude
        if number >= 1e12:
            # Trillions
            return f"{sign}{number / 1e12:.2f}T"
        elif number >= 1e9:
            # Billions
            return f"{sign}{number / 1e9:.2f}B"
        elif number >= 1e6:
            # Millions
            return f"{sign}{number / 1e6:.2f}M"
        elif number >= 1e3:
            # Thousands
            return f"{sign}{number / 1e3:.2f}K"
        else:
            # Regular numbers
            return f"{sign}{number:.2f}"

    def _format_percentage(self, value: Union[int, float]) -> str:
        """Format a value as a percentage string"""
        if value is None:
            return "0.00%"

        # Convert to percentage and round to 2 decimal places
        return f"{value * 100:.2f}%"

    def _format_score(self, score: Union[int, float]) -> float:
        """Format a score value (ensure it's between 0.0 and 1.0)"""
        if score is None:
            return 0.0

        # Ensure score is between 0.0 and 1.0
        return max(0.0, min(float(score), 1.0))

    def _format_timestamp(self, timestamp: Union[str, datetime, None]) -> str:
        """Format timestamp to ISO 8601 format"""
        if timestamp is None:
            return ""

        # Convert string to datetime if needed
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                return timestamp

        # Format datetime to ISO 8601
        if isinstance(timestamp, datetime):
            return timestamp.isoformat()

        return ""

    def to_json(self, data: Any) -> str:
        """Convert data to JSON string"""
        try:
            # Convert to JSON with pretty formatting
            return json.dumps(data, indent=2, default=self._json_default)
        except Exception as e:
            self.logger.error(f"Error converting to JSON: {e}")
            return "{}"

    def _json_default(self, obj):
        """Handle non-serializable objects in JSON conversion"""
        if isinstance(obj, datetime):
            return obj.isoformat()

        return str(obj)

    def from_json(self, json_str: str) -> Any:
        """Convert JSON string to Python object"""
        try:
            return json.loads(json_str)
        except Exception as e:
            self.logger.error(f"Error parsing JSON: {e}")
            return {}

    def format_csv(self, data: List[Dict[str, Any]], columns: List[str] = None) -> str:
        """Format data as CSV string"""
        if not data:
            return ""

        # Determine columns if not specified
        if columns is None:
            columns = list(data[0].keys())

        # Create header row
        csv_rows = [",".join(columns)]

        # Add data rows
        for item in data:
            row = []
            for column in columns:
                # Get value and format for CSV
                value = item.get(column, "")

                # Handle special types
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)

                # Escape commas and quotes
                if isinstance(value, str):
                    if "," in value or '"' in value:
                        value = f'"{value.replace(\'"\', \'""\')})"'

                        row.append(str(value))

                        csv_rows.append(",".join(row))

        return "\n".join(csv_rows)