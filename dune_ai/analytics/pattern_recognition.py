import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from config.logging_config import get_logger


class PatternRecognition:
    """
    Pattern recognition for cryptocurrency price movements and market trends.

    Identifies common chart patterns, correlations between tokens,
    and emerging market trends to inform investment decisions.
    """

    def __init__(self):
        self.logger = get_logger("pattern_recognition")
        self.logger.info("PatternRecognition initialized")

    def identify_chart_pattern(self, prices: List[float]) -> Dict[str, Any]:
        """Identify common chart patterns in price data"""
        self.logger.debug(f"Identifying chart pattern from {len(prices)} price points")

        if len(prices) < 10:
            return {"pattern": "insufficient_data", "confidence": 0.0}

        # Calculate price changes
        changes = []
        for i in range(1, len(prices)):
            if prices[i - 1] > 0:
                change = (prices[i] - prices[i - 1]) / prices[i - 1]
                changes.append(change)

        # Identify possible patterns
        patterns = []

        # Check for uptrend
        uptrend_confidence = self._check_uptrend(prices)
        if uptrend_confidence > 0.5:
            patterns.append(("uptrend", uptrend_confidence))

        # Check for downtrend
        downtrend_confidence = self._check_downtrend(prices)
        if downtrend_confidence > 0.5:
            patterns.append(("downtrend", downtrend_confidence))

        # Check for sideways/consolidation
        consolidation_confidence = self._check_consolidation(prices)
        if consolidation_confidence > 0.5:
            patterns.append(("consolidation", consolidation_confidence))

        # Check for double bottom (W pattern)
        double_bottom_confidence = self._check_double_bottom(prices)
        if double_bottom_confidence > 0.5:
            patterns.append(("double_bottom", double_bottom_confidence))

        # Check for double top (M pattern)
        double_top_confidence = self._check_double_top(prices)
        if double_top_confidence > 0.5:
            patterns.append(("double_top", double_top_confidence))

        # Check for head and shoulders
        head_shoulders_confidence = self._check_head_and_shoulders(prices)
        if head_shoulders_confidence > 0.5:
            patterns.append(("head_and_shoulders", head_shoulders_confidence))

        # Check for inverse head and shoulders
        inv_head_shoulders_confidence = self._check_inverse_head_and_shoulders(prices)
        if inv_head_shoulders_confidence > 0.5:
            patterns.append(("inverse_head_and_shoulders", inv_head_shoulders_confidence))

        # Check for cup and handle
        cup_handle_confidence = self._check_cup_and_handle(prices)
        if cup_handle_confidence > 0.5:
            patterns.append(("cup_and_handle", cup_handle_confidence))

        # Return the pattern with highest confidence
        if patterns:
            patterns.sort(key=lambda x: x[1], reverse=True)
            top_pattern, confidence = patterns[0]

            return {
                "pattern": top_pattern,
                "confidence": confidence,
                "all_patterns": patterns
            }

        return {"pattern": "no_clear_pattern", "confidence": 0.0}

    def _check_uptrend(self, prices: List[float]) -> float:
        """Check for uptrend pattern"""
        if len(prices) < 5:
            return 0.0

        # Linear regression
        x = np.array(range(len(prices)))
        y = np.array(prices)

        slope, _ = np.polyfit(x, y, 1)

        # Normalize slope to confidence value
        price_range = max(prices) - min(prices)
        if price_range == 0:
            return 0.0

        normalized_slope = slope * len(prices) / price_range

        # Convert to confidence (0.0-1.0)
        if normalized_slope > 0:
            confidence = min(normalized_slope, 1.0)
        else:
            confidence = 0.0

        return confidence

    def _check_downtrend(self, prices: List[float]) -> float:
        """Check for downtrend pattern"""
        if len(prices) < 5:
            return 0.0

        # Linear regression
        x = np.array(range(len(prices)))
        y = np.array(prices)

        slope, _ = np.polyfit(x, y, 1)

        # Normalize slope to confidence value
        price_range = max(prices) - min(prices)
        if price_range == 0:
            return 0.0

        normalized_slope = slope * len(prices) / price_range

        # Convert to confidence (0.0-1.0)
        if normalized_slope < 0:
            confidence = min(-normalized_slope, 1.0)
        else:
            confidence = 0.0

        return confidence

    def _check_consolidation(self, prices: List[float]) -> float:
        """Check for sideways/consolidation pattern"""
        if len(prices) < 5:
            return 0.0

        # Calculate linear regression
        x = np.array(range(len(prices)))
        y = np.array(prices)

        slope, _ = np.polyfit(x, y, 1)

        # Calculate price volatility
        price_mean = np.mean(prices)
        price_std = np.std(prices)

        if price_mean == 0:
            return 0.0

        # Calculate coefficient of variation
        cv = price_std / price_mean

        # Check if slope is close to zero and volatility is low
        slope_factor = 1.0 - min(abs(slope) * 20, 1.0)  # Lower slope = higher factor
        volatility_factor = 1.0 - min(cv * 5, 1.0)  # Lower volatility = higher factor

        # Combine factors
        confidence = (slope_factor * 0.7 + volatility_factor * 0.3)

        return confidence

    def _check_double_bottom(self, prices: List[float]) -> float:
        """Check for double bottom (W) pattern"""
        if len(prices) < 10:
            return 0.0

        # Find local minima
        minima = self._find_local_extrema(prices, "min")

        if len(minima) < 2:
            return 0.0

        # For a double bottom:
        # 1. Need at least 2 minima
        # 2. Minima should be at similar price levels
        # 3. Minima should be separated by an intermediate peak

        # Try all pairs of minima
        best_confidence = 0.0

        for i in range(len(minima) - 1):
            for j in range(i + 1, len(minima)):
                idx1, val1 = minima[i]
                idx2, val2 = minima[j]

                # Check if minima are separated by reasonable distance
                if idx2 - idx1 < len(prices) // 4:
                    continue

                # Check if minima are at similar levels
                price_range = max(prices) - min(prices)
                if price_range == 0:
                    continue

                similarity = 1.0 - abs(val1 - val2) / price_range

                # Check if there's a peak in between
                intermediate_prices = prices[idx1 + 1:idx2]
                if not intermediate_prices:
                    continue

                intermediate_max = max(intermediate_prices)

                # Calculate how much higher the peak is
                peak_height = (intermediate_max - (val1 + val2) / 2) / price_range

                if peak_height <= 0:
                    continue

                # Calculate overall pattern confidence
                confidence = similarity * 0.6 + min(peak_height, 1.0) * 0.4

                if confidence > best_confidence:
                    best_confidence = confidence

        return best_confidence

    def _check_double_top(self, prices: List[float]) -> float:
        """Check for double top (M) pattern"""
        if len(prices) < 10:
            return 0.0

        # Find local maxima
        maxima = self._find_local_extrema(prices, "max")

        if len(maxima) < 2:
            return 0.0

        # For a double top:
        # 1. Need at least 2 maxima
        # 2. Maxima should be at similar price levels
        # 3. Maxima should be separated by an intermediate dip

        # Try all pairs of maxima
        best_confidence = 0.0

        for i in range(len(maxima) - 1):
            for j in range(i + 1, len(maxima)):
                idx1, val1 = maxima[i]
                idx2, val2 = maxima[j]

                # Check if maxima are separated by reasonable distance
                if idx2 - idx1 < len(prices) // 4:
                    continue

                # Check if maxima are at similar levels
                price_range = max(prices) - min(prices)
                if price_range == 0:
                    continue

                similarity = 1.0 - abs(val1 - val2) / price_range

                # Check if there's a dip in between
                intermediate_prices = prices[idx1 + 1:idx2]
                if not intermediate_prices:
                    continue

                intermediate_min = min(intermediate_prices)

                # Calculate how much lower the dip is
                dip_depth = ((val1 + val2) / 2 - intermediate_min) / price_range

                if dip_depth <= 0:
                    continue

                # Calculate overall pattern confidence
                confidence = similarity * 0.6 + min(dip_depth, 1.0) * 0.4

                if confidence > best_confidence:
                    best_confidence = confidence

        return best_confidence

    def _check_head_and_shoulders(self, prices: List[float]) -> float:
        """Check for head and shoulders pattern"""
        if len(prices) < 15:
            return 0.0

        # Find local maxima
        maxima = self._find_local_extrema(prices, "max")

        if len(maxima) < 3:
            return 0.0

        # For a head and shoulders:
        # 1. Need 3 peaks (left shoulder, head, right shoulder)
        # 2. Head should be higher than shoulders
        # 3. Shoulders should be at similar levels
        # 4. Pattern should have neckline connecting the troughs

        best_confidence = 0.0

        for i in range(len(maxima) - 2):
            idx1, val1 = maxima[i]  # Left shoulder
            idx2, val2 = maxima[i + 1]  # Head
            idx3, val3 = maxima[i + 2]  # Right shoulder

            # Check if peaks are in the right order
            if not (idx1 < idx2 < idx3):
                continue

            # Check if head is higher than shoulders
            if not (val2 > val1 and val2 > val3):
                continue

            # Check if shoulders are at similar levels
            price_range = max(prices) - min(prices)
            if price_range == 0:
                continue

            shoulder_similarity = 1.0 - abs(val1 - val3) / price_range

            # Check head height relative to shoulders
            head_height = (val2 - (val1 + val3) / 2) / price_range

            # Calculate confidence based on pattern characteristics
            confidence = shoulder_similarity * 0.5 + min(head_height * 2, 1.0) * 0.5

            if confidence > best_confidence:
                best_confidence = confidence

        return best_confidence

    def _check_inverse_head_and_shoulders(self, prices: List[float]) -> float:
        """Check for inverse head and shoulders pattern"""
        if len(prices) < 15:
            return 0.0

        # Find local minima
        minima = self._find_local_extrema(prices, "min")

        if len(minima) < 3:
            return 0.0

        # For an inverse head and shoulders:
        # 1. Need 3 troughs (left shoulder, head, right shoulder)
        # 2. Head should be lower than shoulders
        # 3. Shoulders should be at similar levels
        # 4. Pattern should have neckline connecting the peaks

        best_confidence = 0.0

        for i in range(len(minima) - 2):
            idx1, val1 = minima[i]  # Left shoulder
            idx2, val2 = minima[i + 1]  # Head
            idx3, val3 = minima[i + 2]  # Right shoulder

            # Check if troughs are in the right order
            if not (idx1 < idx2 < idx3):
                continue

            # Check if head is lower than shoulders
            if not (val2 < val1 and val2 < val3):
                continue

            # Check if shoulders are at similar levels
            price_range = max(prices) - min(prices)
            if price_range == 0:
                continue

            shoulder_similarity = 1.0 - abs(val1 - val3) / price_range

            # Check head depth relative to shoulders
            head_depth = ((val1 + val3) / 2 - val2) / price_range

            # Calculate confidence based on pattern characteristics
            confidence = shoulder_similarity * 0.5 + min(head_depth * 2, 1.0) * 0.5

            if confidence > best_confidence:
                best_confidence = confidence

        return best_confidence

    def _check_cup_and_handle(self, prices: List[float]) -> float:
        """Check for cup and handle pattern"""
        if len(prices) < 20:
            return 0.0

        # Cup and handle requires:
        # 1. A rounded bottom (cup)
        # 2. A small dip/consolidation after the cup (handle)
        # 3. Typically bullish continuation pattern

        # Simplistic approach - check for "U" shape followed by a small dip

        # Split the data in half
        mid_point = len(prices) // 2

        # First half should form a cup (rounded bottom)
        cup_confidence = self._check_rounded_bottom(prices[:mid_point * 3 // 2])

        if cup_confidence < 0.4:
            return 0.0

        # Last quarter should form a handle (small dip)
        handle_start = len(prices) * 3 // 4
        handle_confidence = self._check_handle(prices[handle_start:])

        if handle_confidence < 0.4:
            return 0.0

        # Overall pattern confidence
        return (cup_confidence * 0.7 + handle_confidence * 0.3)

    def _check_rounded_bottom(self, prices: List[float]) -> float:
        """Check for a rounded bottom pattern (cup)"""
        if len(prices) < 10:
            return 0.0

        # For a rounded bottom:
        # 1. Starts high
        # 2. Dips in the middle
        # 3. Returns to similar level at the end

        # Check if start and end are at similar levels
        start_price = prices[0]
        end_price = prices[-1]

        price_range = max(prices) - min(prices)
        if price_range == 0:
            return 0.0

        similarity = 1.0 - abs(start_price - end_price) / price_range

        # Check if middle forms a bottom
        mid_idx = len(prices) // 2
        mid_section = prices[mid_idx - 2:mid_idx + 3]  # Middle section

        if not mid_section:
            return 0.0

        mid_min = min(mid_section)
        mid_min_idx = prices.index(mid_min)

        # Check if minimum is in the middle section
        if abs(mid_min_idx - mid_idx) > len(prices) // 4:
            return 0.0

        # Check depth of the cup
        depth = (start_price + end_price) / 2 - mid_min
        relative_depth = depth / price_range

        # Check roundness (using quadratic fit)
        try:
            x = np.array(range(len(prices)))
            y = np.array(prices)

            # Fit quadratic curve
            coeffs = np.polyfit(x, y, 2)

            # For cup, first coefficient should be positive (upward curvature)
            if coeffs[0] <= 0:
                return 0.0

            # Calculate overall confidence
            curvature = min(abs(coeffs[0]) * 1000, 1.0)  # Scale appropriately
            confidence = similarity * 0.4 + relative_depth * 0.3 + curvature * 0.3

            return confidence

        except Exception:
            return 0.0

    def _check_handle(self, prices: List[float]) -> float:
        """Check for a handle pattern (small dip after cup)"""
        if len(prices) < 5:
            return 0.0

        # Handle should be:
        # 1. A small dip (much smaller than the cup)
        # 2. Relatively short in duration
        # 3. Typically has lower volume

        # Check for a small dip
        start_price = prices[0]
        min_price = min(prices)
        end_price = prices[-1]

        price_range = max(prices) - min(prices)
        if price_range == 0:
            return 0.0

        # Handle depth (as % of price range)
        depth = (start_price - min_price) / price_range

        # End price should be similar to or higher than start
        end_factor = max(0, min((end_price - min_price) / (start_price - min_price), 1.0))

        # Ideal handle depth is 30-50% of the price range
        if depth < 0.1 or depth > 0.6:
            return 0.0

        # Calculate confidence
        confidence = depth * 0.7 + end_factor * 0.3

        return confidence

    def _find_local_extrema(self, prices: List[float], extrema_type: str) -> List[Tuple[int, float]]:
        """Find local maxima or minima in price data"""
        extrema = []

        for i in range(1, len(prices) - 1):
            if extrema_type == "max":
                # Local maximum
                if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
                    extrema.append((i, prices[i]))
            else:
                # Local minimum
                if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
                    extrema.append((i, prices[i]))

        return extrema

    def calculate_price_correlation(self, prices1: List[float], prices2: List[float]) -> float:
        """Calculate correlation between two price series"""
        self.logger.debug(f"Calculating correlation between price series (lengths: {len(prices1)}, {len(prices2)})")

        # Ensure equal length
        min_len = min(len(prices1), len(prices2))
        if min_len < 5:
            return 0.0

        # Truncate to equal length
        p1 = np.array(prices1[-min_len:])
        p2 = np.array(prices2[-min_len:])

        # Calculate correlation coefficient
        try:
            # Calculate means
            mean1 = np.mean(p1)
            mean2 = np.mean(p2)

            # Calculate deviations
            dev1 = p1 - mean1
            dev2 = p2 - mean2

            # Calculate covariance and variances
            cov = np.sum(dev1 * dev2)
            var1 = np.sum(dev1 ** 2)
            var2 = np.sum(dev2 ** 2)

            # Calculate correlation
            if var1 == 0 or var2 == 0:
                return 0.0

            correlation = cov / np.sqrt(var1 * var2)

            return correlation

        except Exception as e:
            self.logger.error(f"Error calculating correlation: {e}")
            return 0.0

    def identify_token_clusters(self, token_data: Dict[str, List[float]]) -> Dict[str, List[str]]:
        """Identify clusters of tokens with similar price movements"""
        self.logger.info(f"Identifying token clusters from {len(token_data)} tokens")

        if len(token_data) < 2:
            return {"cluster_1": list(token_data.keys())}

        # Calculate correlation matrix
        tokens = list(token_data.keys())
        n_tokens = len(tokens)

        correlation_matrix = np.zeros((n_tokens, n_tokens))

        for i in range(n_tokens):
            for j in range(i, n_tokens):
                token1 = tokens[i]
                token2 = tokens[j]

                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    correlation = self.calculate_price_correlation(
                        token_data[token1], token_data[token2]
                    )

                    correlation_matrix[i, j] = correlation
                    correlation_matrix[j, i] = correlation

        # Apply simple clustering (based on correlation threshold)
        clusters = {}
        assigned = set()

        # Threshold for strong correlation
        threshold = 0.7

        # Find highly correlated groups
        cluster_id = 1

        for i in range(n_tokens):
            if tokens[i] in assigned:
                continue

            # Start a new cluster
            cluster_name = f"cluster_{cluster_id}"
            clusters[cluster_name] = [tokens[i]]
            assigned.add(tokens[i])

            # Find correlated tokens
            for j in range(n_tokens):
                if i == j or tokens[j] in assigned:
                    continue

                if correlation_matrix[i, j] >= threshold:
                    clusters[cluster_name].append(tokens[j])
                    assigned.add(tokens[j])

            # Only count as a cluster if it has at least 2 tokens
            if len(clusters[cluster_name]) > 1:
                cluster_id += 1
            else:
                # Single token - remove from clusters and assigned
                assigned.remove(tokens[i])
                del clusters[cluster_name]

        # Assign remaining tokens to their most correlated cluster
        for token in tokens:
            if token not in assigned:
                best_cluster = None
                best_correlation = -1.0

                token_idx = tokens.index(token)

                for cluster_name, cluster_tokens in clusters.items():
                    # Calculate average correlation with cluster
                    cluster_correlations = []

                    for cluster_token in cluster_tokens:
                        cluster_token_idx = tokens.index(cluster_token)
                        cluster_correlations.append(correlation_matrix[token_idx, cluster_token_idx])

                    if cluster_correlations:
                        avg_correlation = sum(cluster_correlations) / len(cluster_correlations)

                        if avg_correlation > best_correlation:
                            best_correlation = avg_correlation
                            best_cluster = cluster_name

                # Assign to best cluster or create a new one
                if best_cluster and best_correlation > 0.3:
                    clusters[best_cluster].append(token)
                else:
                    new_cluster = f"cluster_{cluster_id}"
                    clusters[new_cluster] = [token]
                    cluster_id += 1

        return clusters

    def detect_market_trends(self, token_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Detect overall market trends from multiple token price data"""
        self.logger.info(f"Detecting market trends from {len(token_data)} tokens")

        if not token_data:
            return {
                "trend": "unknown",
                "strength": 0.0,
                "confidence": 0.0
            }

        # Calculate trends for each token
        token_trends = {}

        for token, prices in token_data.items():
            if len(prices) < 5:
                continue

            # Calculate linear regression
            x = np.array(range(len(prices)))
            y = np.array(prices)

            slope, _ = np.polyfit(x, y, 1)

            # Normalize slope
            price_range = max(prices) - min(prices)
            if price_range > 0:
                normalized_slope = slope * len(prices) / price_range
            else:
                normalized_slope = 0.0

            token_trends[token] = normalized_slope

        if not token_trends:
            return {
                "trend": "unknown",
                "strength": 0.0,
                "confidence": 0.0
            }

        # Calculate average trend
        avg_trend = sum(token_trends.values()) / len(token_trends)

        # Calculate standard deviation (for confidence)
        trend_values = list(token_trends.values())
        std_dev = np.std(trend_values)

        # Calculate market trend strength
        trend_strength = abs(avg_trend)

        # Determine trend direction
        if avg_trend > 0.1:
            trend = "bullish"
        elif avg_trend < -0.1:
            trend = "bearish"
        else:
            trend = "sideways"

        # Calculate confidence (inverse of standard deviation)
        # Higher deviation = lower confidence
        confidence = 1.0 / (1.0 + std_dev)

        # Get top gainers and losers
        sorted_trends = sorted(token_trends.items(), key=lambda x: x[1], reverse=True)

        top_gainers = sorted_trends[:5]
        top_losers = sorted_trends[-5:]

        return {
            "trend": trend,
            "strength": min(trend_strength, 1.0),
            "confidence": confidence,
            "avg_trend": avg_trend,
            "top_gainers": top_gainers,
            "top_losers": top_losers
        }

    def predict_future_movement(self, prices: List[float], days_ahead: int = 7) -> Dict[str, Any]:
        """Predict future price movement based on historical data"""
        self.logger.debug(f"Predicting future movement {days_ahead} days ahead")

        if len(prices) < 10:
            return {
                "prediction": "insufficient_data",
                "confidence": 0.0,
                "predicted_values": []
            }

        try:
            # Simple linear prediction
            x = np.array(range(len(prices)))
            y = np.array(prices)

            # Fit linear model
            slope, intercept = np.polyfit(x, y, 1)

            # Predict future values
            future_x = np.array(range(len(prices), len(prices) + days_ahead))
            predicted_values = slope * future_x + intercept

            # Ensure no negative prices
            predicted_values = np.maximum(predicted_values, 0)

            # Calculate recent trend strength
            recent_prices = prices[-min(14, len(prices)):]
            recent_x = np.array(range(len(recent_prices)))
            recent_slope, _ = np.polyfit(recent_x, recent_prices, 1)

            price_range = max(recent_prices) - min(recent_prices)
            if price_range > 0:
                normalized_slope = recent_slope * len(recent_prices) / price_range
            else:
                normalized_slope = 0.0

            # Determine prediction direction
            if normalized_slope > 0.1:
                prediction = "upward"
            elif normalized_slope < -0.1:
                prediction = "downward"
            else:
                prediction = "sideways"

            # Calculate confidence based on fit quality
            residuals = y - (slope * x + intercept)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)

            if ss_tot == 0:
                r_squared = 0
            else:
                r_squared = 1 - (ss_res / ss_tot)

            # Confidence is based on R-squared value
            confidence = max(0.0, min(r_squared, 1.0))

            return {
                "prediction": prediction,
                "confidence": confidence,
                "predicted_values": predicted_values.tolist(),
                "trend_strength": abs(normalized_slope)
            }

        except Exception as e:
            self.logger.error(f"Error predicting future movement: {e}")
            return {
                "prediction": "error",
                "confidence": 0.0,
                "predicted_values": []
            }

    def find_divergences(self, prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """Find price-volume divergences (potential reversal signals)"""
        self.logger.debug(f"Finding divergences in {len(prices)} price points")

        if len(prices) < 10 or len(volumes) < 10:
            return {
                "divergences": [],
                "count": 0
            }

        # Ensure equal lengths
        min_len = min(len(prices), len(volumes))
        prices = prices[-min_len:]
        volumes = volumes[-min_len:]

        # Find local price extrema
        price_maxima = self._find_local_extrema(prices, "max")
        price_minima = self._find_local_extrema(prices, "min")

        # Find divergences
        divergences = []

        # Check for bullish divergences (price makes lower low, volume makes higher low)
        for i in range(len(price_minima) - 1):
            idx1, val1 = price_minima[i]
            idx2, val2 = price_minima[i + 1]

            # Check if price made lower low
            if val2 < val1:
                # Check volume at these points
                vol1 = volumes[idx1]
                vol2 = volumes[idx2]

                # Bullish divergence if volume made higher low
                if vol2 > vol1:
                    divergences.append({
                        "type": "bullish",
                        "position": idx2,
                        "strength": (vol2 / vol1) * (val1 / val2)
                    })

        # Check for bearish divergences (price makes higher high, volume makes lower high)
        for i in range(len(price_maxima) - 1):
            idx1, val1 = price_maxima[i]
            idx2, val2 = price_maxima[i + 1]

            # Check if price made higher high
            if val2 > val1:
                # Check volume at these points
                vol1 = volumes[idx1]
                vol2 = volumes[idx2]

                # Bearish divergence if volume made lower high
                if vol2 < vol1:
                    divergences.append({
                        "type": "bearish",
                        "position": idx2,
                        "strength": (vol1 / vol2) * (val2 / val1)
                    })

        # Sort by strength
        divergences.sort(key=lambda x: x["strength"], reverse=True)

        return {
            "divergences": divergences,
            "count": len(divergences)
        }