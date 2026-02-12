# Enhanced Features Quick Reference

## 🎯 Feature Cheat Sheet

### Volatility Dynamics (16 features)

| Feature | Formula | Interpretation | Best For |
|---------|---------|----------------|----------|
| `rv_ratio` | log(H/L) / log(H₋₁/L₋₁) | Vol expansion (>1) or contraction (<1) | Breakout detection |
| `parkinson_vol` | √[(1/4ln2) × (ln(H/L))²] | Efficient volatility estimator | Vol regime classification |
| `parkinson_accel` | Δ(parkinson_vol) | Vol acceleration | Volatility spike detection |
| `vol_regime_shift` | σ₅ₐ / σ₂₀ₐ - 1 | Short vs long-term vol | Regime transition |
| `gap_vol` | σ₂₀ₐ(Open/Close₋₁ - 1) | Overnight risk | Gap trading |
| `gap_vol_surge` | gap_vol / MA₆₀(gap_vol) | Gap vol spike | Event detection |
| `vol_5d_std` | σ₅ₐ(returns) | Short-term volatility | Quick regime changes |
| `vol_20d_std` | σ₂₀ₐ(returns) | Medium-term volatility | Trend stability |

**Trading Signals:**
- `rv_ratio > 1.5`: Volatility expanding → Reduce position size
- `vol_regime_shift > 0.3`: Vol spiking → Avoid new entries
- `gap_vol_surge > 2.0`: Overnight risk elevated → Close before market close

---

### Microstructure (9 features)

| Feature | Formula | Interpretation | Best For |
|---------|---------|----------------|----------|
| `illiquidity` | \|r\| / (Volume × Price) | Amihud illiquidity | Slippage estimation |
| `illiquidity_surge` | illiquidity / MA₂₀(illiquidity) | Liquidity drying up | Flash crash detection |
| `close_pressure` | (Close - Open) / (High - Low) | Closing auction bias | Institutional positioning |
| `close_pressure_5d` | MA₅(close_pressure) | Persistent pressure | Trend confirmation |
| `am_momentum` | (High - Open) / Open | Morning strength | Session bias |
| `pm_momentum` | (Close - Low) / Low | Afternoon strength | Closing strength |
| `session_reversal` | am_momentum × pm_momentum | Intraday reversal | Reversal detection |

**Trading Signals:**
- `close_pressure_5d > 0.5`: Institutions buying → Bullish
- `illiquidity_surge > 2.0`: Low liquidity → Reduce size
- `session_reversal < 0`: Intraday reversal → Fade the move

---

### Regime Transitions (7 features)

| Feature | Formula | Interpretation | Best For |
|---------|---------|----------------|----------|
| `drawdown` | Close / Max₂₀(Close) - 1 | Current drawdown | Support levels |
| `runup` | Close / Min₂₀(Close) - 1 | Current runup | Resistance levels |
| `asymmetry` | runup + drawdown | Fear vs greed | Sentiment gauge |
| `volume_regime` | MA₅(Volume) / MA₂₀(Volume) | Volume surge | Panic/euphoria |
| `regime_bull` | Close > MA₅₀ | Bull/bear regime | Trend filter |
| `regime_duration` | Days in current regime | Regime age | Reversal timing |

**Trading Signals:**
- `asymmetry > 0.1`: Greed (runup > drawdown) → Fade rallies
- `asymmetry < -0.1`: Fear (drawdown > runup) → Buy dips
- `volume_regime > 1.5`: Panic/euphoria → Reversal likely
- `regime_duration > 30`: Regime aging → Transition likely

---

### Temporal Dynamics (10 features)

| Feature | Formula | Interpretation | Best For |
|---------|---------|----------------|----------|
| `autocorr_1d` | ρ₁(returns) | 1-day momentum | Mean reversion |
| `autocorr_5d` | ρ₅(returns) | 5-day momentum | Trend persistence |
| `mom_decay` | r₅ₐ / r₂₀ₐ | Momentum decay | Trend exhaustion |
| `mom_5d` | r₅ₐ | 5-day return | Short-term momentum |
| `mom_20d` | r₂₀ₐ | 20-day return | Medium-term momentum |
| `returns_skew_20d` | Skew₂₀(returns) | Tail risk | Crash risk |
| `returns_kurt_20d` | Kurt₂₀(returns) | Fat tails | Extreme moves |

**Trading Signals:**
- `autocorr_1d > 0.2`: Momentum → Trend following
- `autocorr_1d < -0.2`: Mean reversion → Fade moves
- `mom_decay < 0.5`: Momentum dying → Exit longs
- `returns_skew_20d < -1.0`: Negative skew → Crash risk

---

### Volume Features (6 features)

| Feature | Formula | Interpretation | Best For |
|---------|---------|----------------|----------|
| `volume_surge` | Volume / MA₂₀(Volume) | Volume spike | Breakout confirmation |
| `obv_zscore` | Z-score(OBV) | Cumulative volume | Divergence detection |
| `mfi_14` | (MFI - 50) / 50 | Money flow | Overbought/oversold |
| `volume_ma5` | MA₅(Volume) | Short-term volume | Quick changes |
| `volume_ma20` | MA₂₀(Volume) | Medium-term volume | Baseline |

**Trading Signals:**
- `volume_surge > 2.0`: High volume → Breakout valid
- `obv_zscore > 2.0` + price up: Strong buying → Bullish
- `mfi_14 > 0.6`: Overbought → Reversal likely

---

## 🎨 Feature Combinations (Power Plays)

### 1. **Volatility Breakout**
```
IF rv_ratio > 1.3 AND vol_regime_shift > 0.2 AND volume_surge > 1.5:
    → Volatility breakout confirmed
    → Action: Reduce position size, widen stops
```

### 2. **Institutional Accumulation**
```
IF close_pressure_5d > 0.3 AND obv_zscore > 1.0 AND illiquidity_surge < 1.2:
    → Institutions accumulating
    → Action: Follow the smart money (long)
```

### 3. **Regime Transition (Trend → Mean Reversion)**
```
IF autocorr_1d < -0.15 AND mom_decay < 0.6 AND regime_duration > 25:
    → Trend exhausting, switching to mean reversion
    → Action: Exit trend trades, prepare for range
```

### 4. **Fear Spike (Buy Opportunity)**
```
IF asymmetry < -0.15 AND volume_regime > 1.8 AND returns_skew_20d < -0.5:
    → Panic selling
    → Action: Buy dips (if fundamentals intact)
```

### 5. **Greed Peak (Sell Opportunity)**
```
IF asymmetry > 0.15 AND mfi_14 > 0.7 AND session_reversal < -0.1:
    → Euphoria topping
    → Action: Take profits, reduce longs
```

---

## 📊 Feature Importance by Regime

### Bull Regime
**Top 5 Features:**
1. `pm_momentum` (0.079) - Afternoon strength confirms trend
2. `trend_strength` (0.068) - Linear trend slope
3. `session_reversal` (0.045) - Intraday reversals signal weakness
4. `ret_60d` (0.041) - Long-term momentum
5. `adx_14` (0.033) - Trend strength

**Strategy:** Focus on momentum and trend continuation

### Bear Regime
**Top 5 Features:**
1. `asymmetry` (0.061) - Fear vs greed
2. `regime_bull` (0.056) - Regime classification
3. `ret_60d` (0.049) - Long-term momentum
4. `vol_60d` (0.046) - Long-term volatility
5. `zscore_20d` (0.045) - Mean reversion

**Strategy:** Focus on mean reversion and volatility

### Neutral Regime
**Top 5 Features:**
1. `trend_strength` (0.068) - Detect micro-trends
2. `adx_14` (0.066) - Trend vs range
3. `returns_skew_20d` (0.059) - Tail risk
4. `session_reversal` (0.048) - Intraday patterns
5. `asymmetry` (0.045) - Sentiment shifts

**Strategy:** Range trading with quick reversals

### Volatile Regime
**Top 5 Features:**
1. `returns_skew_20d` (0.052) - Tail risk critical
2. `trend_strength` (0.047) - Trend in chaos
3. `pm_momentum` (0.046) - Closing strength
4. `adx_14` (0.041) - Trend strength
5. `session_reversal` (0.036) - Intraday reversals

**Strategy:** Risk management, smaller positions

---

## 🔧 Feature Engineering Tips

### 1. **Handling Missing Data**
All features have fallback values:
- OHLC missing → 0.0
- Volume missing → 1.0 (neutral)
- Insufficient history → 0.0

### 2. **Feature Scaling**
- All features are automatically scaled by `RobustScaler`
- Outliers are clipped to [-3, 3] sigma
- No manual scaling needed

### 3. **Feature Correlation**
High correlation (>0.9) pairs to watch:
- `vol_5d` ↔ `vol_5d_std` (0.95)
- `mom_5d` ↔ `ret_5d` (1.0 - same feature)
- `volume_ma5` ↔ `volume_surge` (0.85)

Consider removing one from each pair if overfitting occurs.

### 4. **Feature Stability**
Most stable features (low variance):
- `trend_strength`
- `adx_14`
- `autocorr_1d`

Most volatile features (high variance):
- `rv_ratio` (can spike 10x)
- `illiquidity_surge` (can spike 5x)
- `gap_vol_surge` (can spike 3x)

---

## 🚀 Quick Start

### Generate Features
```python
from models.institutional_signal_generator import FeatureEngine

engine = FeatureEngine(lookback=60)
features_df = engine.extract_features_vectorized(ohlcv_data)
```

### Train Model
```python
from models.institutional_signal_generator import UltimateSignalGenerator

generator = UltimateSignalGenerator(model_dir="models/saved_ultimate")
results = generator.train(historical_data, target_horizon=5)
```

### Generate Signal
```python
signal = generator.generate_signal(
    symbol='AAPL',
    data=recent_ohlcv,
    sentiment_score=0.2,
    momentum_rank=0.6
)

print(f"Signal: {signal.signal:+.3f}")
print(f"Confidence: {signal.confidence:.1%}")
print(f"Regime: {signal.regime}")
```

---

## 📈 Performance Expectations

| Metric | Baseline | Enhanced | Improvement |
|--------|----------|----------|-------------|
| **Features** | 35 | 67 | +91% |
| **Bull Accuracy** | 51.4% | 58-60% | +6-9% |
| **Bear Accuracy** | 56.3% | 62-65% | +6-9% |
| **Neutral Accuracy** | 53.1% | 58-61% | +5-8% |
| **Volatile Accuracy** | 55.2% | 60-63% | +5-8% |
| **Avg Accuracy** | 54.0% | 60-62% | +8-14% |
| **Sharpe Ratio** | ~1.2 | ~1.8-2.0 | +50-67% |

---

**Last Updated:** 2026-02-11  
**Version:** 1.0  
**Status:** Production Ready
