from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import uuid


@dataclass
class TokenPrice:
    """Token price information with timestamp"""
    price_usd: float
    price_sol: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TokenMetrics:
    """Metrics for a token including market data and trading statistics"""
    market_cap: float = 0.0
    volume_24h: float = 0.0
    price_change_24h: float = 0.0
    liquidity: float = 0.0
    holders_count: int = 0
    average_transaction_size: float = 0.0
    transaction_count_24h: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class TokenSentiment:
    """Sentiment analysis metrics for a token"""
    positive_score: float = 0.0
    negative_score: float = 0.0
    neutral_score: float = 0.0
    overall_sentiment: float = 0.0  # Range from -1.0 (negative) to 1.0 (positive)
    tweet_count: int = 0
    influential_mentions: int = 0
    sentiment_trend: float = 0.0  # Rate of change
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class Token:
    """Comprehensive representation of a Solana token"""
    address: str
    name: str
    symbol: str
    decimals: int

    # Optional fields that may be populated later
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    creator_address: Optional[str] = None
    launch_date: Optional[datetime] = None
    logo_url: Optional[str] = None
    website: Optional[str] = None
    twitter: Optional[str] = None
    telegram: Optional[str] = None
    discord: Optional[str] = None

    # Related data
    price_history: List[TokenPrice] = field(default_factory=list)
    metrics: TokenMetrics = field(default_factory=TokenMetrics)
    sentiment: TokenSentiment = field(default_factory=TokenSentiment)

    # Trading pairs
    trading_pairs: Dict[str, str] = field(default_factory=dict)  # market: pair_address

    # Analysis tags
    is_meme_coin: bool = False
    trend_score: float = 0.0  # Higher value means stronger trend
    similarity_tokens: List[str] = field(default_factory=list)  # Similar tokens by pattern
    risk_score: float = 0.0  # 0.0 (safe) to 1.0 (high risk)

    def current_price_usd(self) -> Optional[float]:
        """Return the most recent USD price if available"""
        if not self.price_history:
            return None
        return self.price_history[-1].price_usd

    def current_price_sol(self) -> Optional[float]:
        """Return the most recent SOL price if available"""
        if not self.price_history:
            return None
        return self.price_history[-1].price_sol

    def to_dict(self) -> Dict:
        """Convert the token object to a dictionary for API responses"""
        return {
            "id": self.id,
            "address": self.address,
            "name": self.name,
            "symbol": self.symbol,
            "decimals": self.decimals,
            "creator_address": self.creator_address,
            "launch_date": self.launch_date.isoformat() if self.launch_date else None,
            "current_price_usd": self.current_price_usd(),
            "current_price_sol": self.current_price_sol(),
            "market_cap": self.metrics.market_cap,
            "volume_24h": self.metrics.volume_24h,
            "price_change_24h": self.metrics.price_change_24h,
            "holders_count": self.metrics.holders_count,
            "sentiment_score": self.sentiment.overall_sentiment,
            "trend_score": self.trend_score,
            "risk_score": self.risk_score,
            "is_meme_coin": self.is_meme_coin,
            "social_links": {
                "website": self.website,
                "twitter": self.twitter,
                "telegram": self.telegram,
                "discord": self.discord,
            }
        }