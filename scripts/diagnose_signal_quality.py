#!/usr/bin/env python3
"""Diagnostic: measure signal quality on synthetic data to confirm no-alpha baseline."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Patch market data
from scripts.run_backtest_with_synthetic_data import patch_market_data, generate_synthetic_ohlcv
patch_market_data()

import numpy as np
import pandas as pd
from config import ApexConfig
from models.god_level_signal_generator import GodLevelSignalGenerator

gen = GodLevelSignalGenerator()

# Train on synthetic data
symbols = ApexConfig.SYMBOLS[:20]
train_data = {}
for sym in symbols:
    df = generate_synthetic_ohlcv(sym, n_days=504)
    if len(df) >= 252:
        train_data[sym] = df

gen.train_models(train_data)

# Now measure signal vs forward return correlation (IC)
ics = []
signal_strengths = []
confidence_levels = []
forward_returns = []
correct_direction = 0
total_signals = 0

for sym, df in train_data.items():
    prices = df['Close']
    # Use last 100 days as out-of-sample
    for i in range(len(prices) - 100, len(prices) - 5):
        hist_prices = prices.iloc[:i]
        if len(hist_prices) < 120:
            continue
        sig_data = gen.generate_ml_signal(sym, hist_prices)
        signal = sig_data['signal']
        confidence = sig_data['confidence']

        # 5-day forward return
        fwd_ret = (prices.iloc[i+5] - prices.iloc[i]) / prices.iloc[i]

        if abs(signal) > 0.05:
            signal_strengths.append(signal)
            confidence_levels.append(confidence)
            forward_returns.append(fwd_ret)
            total_signals += 1
            if (signal > 0 and fwd_ret > 0) or (signal < 0 and fwd_ret < 0):
                correct_direction += 1

if signal_strengths:
    ic = np.corrcoef(signal_strengths, forward_returns)[0, 1]
    accuracy = correct_direction / total_signals
    print(f"\n{'='*60}")
    print(f"SIGNAL QUALITY DIAGNOSTIC (Synthetic Data)")
    print(f"{'='*60}")
    print(f"  Total signals evaluated:     {total_signals}")
    print(f"  IC (signal vs 5d return):    {ic:.4f}")
    print(f"  Directional accuracy:        {accuracy:.1%}")
    print(f"  Avg |signal|:                {np.mean(np.abs(signal_strengths)):.4f}")
    print(f"  Avg confidence:              {np.mean(confidence_levels):.4f}")
    print(f"  Signal std:                  {np.std(signal_strengths):.4f}")
    print(f"  Forward return std:          {np.std(forward_returns):.4f}")
    print(f"\n  BASELINE CHECK:")
    if abs(ic) < 0.05 and abs(accuracy - 0.50) < 0.05:
        print(f"  CONFIRMED: No exploitable signal in synthetic GBM data.")
        print(f"  This is the expected null result — synthetic data has no alpha.")
        print(f"  The NO-GO verdict is correct for this data environment.")
    elif ic > 0.05 or accuracy > 0.55:
        print(f"  WARNING: Positive signal detected — possible look-ahead bias.")
    else:
        print(f"  Marginal signal quality — below threshold for profitable trading.")
else:
    print("  No signals generated.")
