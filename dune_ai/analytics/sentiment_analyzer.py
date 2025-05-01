import logging
import re
from typing import Dict, List, Any, Optional, Tuple
import random
from textblob import TextBlob

from config.logging_config import get_logger


class SentimentAnalyzer:
    """
    Sentiment analyzer for crypto-related text content.

    Analyzes text from social media, news, and other sources to determine
    sentiment toward specific tokens or the overall cryptocurrency market.
    """

    def __init__(self):
        self.logger = get_logger("sentiment_analyzer")

        # Lists of positive and negative words specific to crypto
        self.crypto_positive_words = [
            "bullish", "moon", "lambo", "hodl", "diamond hands", "gains", "profit",
            "rally", "surge", "breakout", "strength", "support", "buy", "accumulate",
            "adoption", "institutional", "partnership", "launch", "upgrade", "milestone",
            "staking", "rewards", "passive", "innovative", "revolutionary", "potential",
            "winner", "gem", "undervalued", "legit", "solid", "strong", "growth"
        ]

        self.crypto_negative_words = [
            "bearish", "scam", "rugpull", "dump", "crash", "sell", "weak", "fud",
            "fear", "correction", "dip", "volatile", "risky", "fake", "fraud",
            "investigation", "sec", "regulation", "ban", "restriction", "hack",
            "exploit", "vulnerability", "stolen", "inflation", "overvalued",
            "bubble", "capitulation", "suspicious", "sketchy", "ponzi"
        ]

        # Emoji sentiment
        self.positive_emojis = ["🚀", "🌕", "💎", "🙌", "📈", "🔥", "💰", "👍", "✅", "🤑", "💯"]
        self.negative_emojis = ["📉", "🔻", "👎", "❌", "💩", "🙅", "🤡", "💸", "😱", "🆘"]

        self.logger.info("SentimentAnalyzer initialized")

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of a text"""
        self.logger.debug(f"Analyzing sentiment of text (length: {len(text)})")

        # Clean text
        cleaned_text = self._clean_text(text)

        # Basic sentiment analysis using TextBlob
        blob = TextBlob(cleaned_text)
        base_sentiment = blob.sentiment.polarity  # Range from -1.0 to 1.0

        # Adjust for crypto-specific terms
        adjusted_sentiment = self._adjust_for_crypto_terms(cleaned_text, base_sentiment)

        # Analyze emoji sentiment
        emoji_sentiment = self._analyze_emoji_sentiment(text)

        # Combine scores (giving more weight to crypto-specific adjustments)
        combined_sentiment = base_sentiment * 0.3 + adjusted_sentiment * 0.5 + emoji_sentiment * 0.2

        # Determine sentiment category
        if combined_sentiment > 0.2:
            category = "positive"
        elif combined_sentiment < -0.2:
            category = "negative"
        else:
            category = "neutral"

        # Extract mentioned tokens
        token_mentions = self._extract_token_mentions(text)

        return {
            "sentiment_score": combined_sentiment,
            "sentiment_category": category,
            "base_sentiment": base_sentiment,
            "crypto_adjusted_sentiment": adjusted_sentiment,
            "emoji_sentiment": emoji_sentiment,
            "token_mentions": token_mentions
        }

    def _clean_text(self, text: str) -> str:
        """Clean text for analysis"""
        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)

        # Remove mentions and hashtags but keep the text
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)

        # Remove non-alphanumeric characters except emoji
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F700-\U0001F77F"  # alchemical symbols
            u"\U0001F780-\U0001F7FF"  # Geometric Shapes
            u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            u"\U0001FA00-\U0001FA6F"  # Chess Symbols
            u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            u"\U00002702-\U000027B0"  # Dingbats
            u"\U000024C2-\U0001F251"
            "]+"
        )

        # Extract emojis for later analysis
        self.emojis = emoji_pattern.findall(text)

        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def _adjust_for_crypto_terms(self, text: str, base_sentiment: float) -> float:
        """Adjust sentiment score based on crypto-specific terms"""
        words = text.split()

        # Count occurrences of positive and negative crypto terms
        positive_count = 0
        negative_count = 0

        for word in words:
            # Check for full matches
            if word in self.crypto_positive_words:
                positive_count += 1
            elif word in self.crypto_negative_words:
                negative_count += 1

        # Check for phrases
        for phrase in self.crypto_positive_words:
            if ' ' in phrase and phrase in text:
                positive_count += 2  # Give more weight to phrases

        for phrase in self.crypto_negative_words:
            if ' ' in phrase and phrase in text:
                negative_count += 2  # Give more weight to phrases

        # Calculate adjustment
        total_count = len(words)
        if total_count > 0:
            positive_ratio = positive_count / total_count
            negative_ratio = negative_count / total_count

            # Adjust sentiment
            adjustment = (positive_ratio - negative_ratio) * 0.5
            adjusted_sentiment = base_sentiment + adjustment

            # Ensure result is in range [-1.0, 1.0]
            return max(-1.0, min(1.0, adjusted_sentiment))

        return base_sentiment

    def _analyze_emoji_sentiment(self, text: str) -> float:
        """Analyze sentiment based on emojis"""
        if not hasattr(self, 'emojis') or not self.emojis:
            return 0.0  # No emojis found

        positive_count = 0
        negative_count = 0

        for emoji in self.emojis:
            if emoji in self.positive_emojis:
                positive_count += 1
            elif emoji in self.negative_emojis:
                negative_count += 1

        total_count = len(self.emojis)
        if total_count > 0:
            # Calculate sentiment score from -1.0 to 1.0
            emoji_sentiment = (positive_count - negative_count) / total_count
            return emoji_sentiment

        return 0.0

    def _extract_token_mentions(self, text: str) -> List[str]:
        """Extract token symbols from text"""
        # Pattern for token symbols like $BTC, $SOL, $DOGE
        token_pattern = r'\$([A-Z]{2,10})'

        # Find all matches
        matches = re.findall(token_pattern, text)

        # Return unique symbols
        return list(set(matches))

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze sentiment for a batch of texts"""
        self.logger.info(f"Analyzing sentiment for batch of {len(texts)} texts")

        results = []
        for text in texts:
            sentiment = self.analyze(text)
            results.append(sentiment)

        return results

    def get_aggregate_sentiment(self, sentiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate sentiment from multiple analyses"""
        if not sentiments:
            return {
                "overall_score": 0.0,
                "category": "neutral",
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "neutral_ratio": 0.0
            }

        # Count categories
        categories = {"positive": 0, "negative": 0, "neutral": 0}

        # Sum scores
        total_score = 0.0

        for sentiment in sentiments:
            category = sentiment["sentiment_category"]
            categories[category] += 1
            total_score += sentiment["sentiment_score"]

        # Calculate overall score
        overall_score = total_score / len(sentiments)

        # Determine overall category
        if overall_score > 0.2:
            overall_category = "positive"
        elif overall_score < -0.2:
            overall_category = "negative"
        else:
            overall_category = "neutral"

        # Calculate ratios
        total = len(sentiments)
        positive_ratio = categories["positive"] / total
        negative_ratio = categories["negative"] / total
        neutral_ratio = categories["neutral"] / total

        return {
            "overall_score": overall_score,
            "category": overall_category,
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
            "neutral_ratio": neutral_ratio
        }

    def analyze_token_sentiment(self, texts: List[str], token_symbol: str) -> Dict[str, Any]:
        """Analyze sentiment specifically for a token"""
        self.logger.info(f"Analyzing sentiment for token {token_symbol}")

        # Filter texts that mention the token
        token_mentions = []
        other_texts = []

        for text in texts:
            if f"${token_symbol}" in text.upper() or f" {token_symbol} " in f" {text.upper()} ":
                token_mentions.append(text)
            else:
                other_texts.append(text)

        # Analyze token-specific mentions
        token_sentiments = self.analyze_batch(token_mentions)

        # Analyze other texts
        other_sentiments = self.analyze_batch(other_texts)

        # Calculate aggregates
        token_aggregate = self.get_aggregate_sentiment(token_sentiments)
        other_aggregate = self.get_aggregate_sentiment(other_sentiments)

        # Calculate relative sentiment (compared to overall market)
        relative_sentiment = 0.0
        if token_aggregate["overall_score"] != 0 or other_aggregate["overall_score"] != 0:
            relative_sentiment = token_aggregate["overall_score"] - other_aggregate["overall_score"]

        return {
            "token": token_symbol,
            "mention_count": len(token_mentions),
            "sentiment_score": token_aggregate["overall_score"],
            "sentiment_category": token_aggregate["category"],
            "relative_sentiment": relative_sentiment,
            "positive_ratio": token_aggregate["positive_ratio"],
            "negative_ratio": token_aggregate["negative_ratio"],
            "texts_analyzed": len(texts)
        }