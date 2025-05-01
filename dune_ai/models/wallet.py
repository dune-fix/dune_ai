from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from datetime import datetime


@dataclass
class WalletActivity:
    """Represents activity of a wallet"""
    transaction_count: int = 0
    first_activity: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    transaction_volume_sol: float = 0.0
    token_interactions: Dict[str, int] = field(default_factory=dict)  # token address to count


@dataclass
class TokenBalance:
    """Represents a token balance in a wallet"""
    token_address: str
    token_symbol: str
    amount: float
    value_usd: Optional[float] = None
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class WalletProfile:
    """Measures of wallet behavior and classification"""
    is_whale: bool = False
    is_developer: bool = False
    is_contract_deployer: bool = False
    is_active_trader: bool = False
    risk_score: float = 0.0  # 0.0 to 1.0, higher means riskier
    influence_score: float = 0.0  # 0.0 to 1.0, higher means more influential
    cluster_id: Optional[int] = None  # For grouping similar wallets


@dataclass
class WalletRelationship:
    """Represents relationships between wallets"""
    related_wallet: str
    relationship_type: str  # "transaction", "co-ownership", "similar-pattern", etc.
    strength: float  # 0.0 to 1.0
    transaction_count: int = 0


@dataclass
class Wallet:
    """Comprehensive representation of a Solana wallet"""
    address: str

    # Balance information
    sol_balance: float = 0.0
    token_balances: List[TokenBalance] = field(default_factory=list)

    # Activity data
    activity: WalletActivity = field(default_factory=WalletActivity)

    # Profiling and behavior
    profile: WalletProfile = field(default_factory=WalletProfile)

    # Relationships
    relationships: List[WalletRelationship] = field(default_factory=list)

    # Token creation and deployment
    created_tokens: Set[str] = field(default_factory=set)

    # Tracking data
    first_seen: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.now)

    def add_token_interaction(self, token_address: str):
        """Record an interaction with a token"""
        if token_address not in self.activity.token_interactions:
            self.activity.token_interactions[token_address] = 0
        self.activity.token_interactions[token_address] += 1

    def update_sol_balance(self, new_balance: float):
        """Update SOL balance and record the update time"""
        self.sol_balance = new_balance
        self.last_updated = datetime.now()

    def update_token_balance(self, token_address: str, token_symbol: str, amount: float,
                             value_usd: Optional[float] = None):
        """Update a token balance in the wallet"""
        # Find existing balance if any
        for balance in self.token_balances:
            if balance.token_address == token_address:
                balance.amount = amount
                balance.value_usd = value_usd
                balance.last_updated = datetime.now()
                return

        # Add new balance if not found
        self.token_balances.append(TokenBalance(
            token_address=token_address,
            token_symbol=token_symbol,
            amount=amount,
            value_usd=value_usd
        ))

    def calculate_total_value_usd(self) -> float:
        """Calculate the total USD value of all tokens in the wallet"""
        total = 0.0

        # Add value of SOL (would need current SOL price from elsewhere)
        # total += self.sol_balance * sol_price_usd

        # Add value of tokens
        for balance in self.token_balances:
            if balance.value_usd is not None:
                total += balance.value_usd

        return total

    def to_dict(self) -> Dict:
        """Convert wallet data to dictionary for API responses"""
        return {
            "address": self.address,
            "sol_balance": self.sol_balance,
            "token_count": len(self.token_balances),
            "created_tokens_count": len(self.created_tokens),
            "activity": {
                "transaction_count": self.activity.transaction_count,
                "first_activity": self.activity.first_activity.isoformat() if self.activity.first_activity else None,
                "last_activity": self.activity.last_activity.isoformat() if self.activity.last_activity else None,
                "transaction_volume_sol": self.activity.transaction_volume_sol,
            },
            "profile": {
                "is_whale": self.profile.is_whale,
                "is_developer": self.profile.is_developer,
                "is_contract_deployer": self.profile.is_contract_deployer,
                "is_active_trader": self.profile.is_active_trader,
                "risk_score": self.profile.risk_score,
                "influence_score": self.profile.influence_score,
            },
            "last_updated": self.last_updated.isoformat(),
        }