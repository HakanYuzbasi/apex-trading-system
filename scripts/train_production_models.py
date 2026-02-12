#!/usr/bin/env python3
"""
Production Model Training Script with Enhanced Features
========================================================
Retrains all production models with the new 67-feature set.

Usage:
    python scripts/train_production_models.py [--symbols SYMBOLS] [--days DAYS]
    
Options:
    --symbols: Comma-separated list of symbols (default: from config)
    --days: Number of days of historical data (default: 1500)
    --force: Force retrain even if models exist
    --validate: Run validation after training
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import logging
import yfinance as yf

from models.institutional_signal_generator import UltimateSignalGenerator
from core.logging_config import setup_logging

logger = setup_logging(level="INFO", log_file="logs/model_training.log", 
                       json_format=False, console_output=True)

# Default symbols for training
DEFAULT_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'SPY', 'QQQ',
    'BTC-USD', 'ETH-USD'
]


def fetch_historical_data(symbols: List[str], days: int = 2000) -> Dict[str, pd.DataFrame]:
    """Fetch historical OHLCV data for training using yfinance.
    
    Default 2000 days (~5.5 years) ensures we capture:
    - 2020 COVID crash (bear regime)
    - 2021-2022 bull run
    - 2022 crypto winter (bear regime)
    - 2023-2024 recovery
    """
    
    historical_data = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    for symbol in symbols:
        try:
            logger.info(f"  Fetching {symbol}...")
            
            # Convert crypto symbols for yfinance
            yf_symbol = symbol
            if '/' in symbol:
                # Crypto: BTC/USD -> BTC-USD
                yf_symbol = symbol.replace('/', '-')
            
            # Fetch data
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if df is not None and len(df) > 100:
                # Standardize column names
                df = df.rename(columns={
                    'Open': 'Open',
                    'High': 'High',
                    'Low': 'Low',
                    'Close': 'Close',
                    'Volume': 'Volume'
                })
                historical_data[symbol] = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                logger.info(f"    ✅ {symbol}: {len(df)} bars")
            else:
                logger.warning(f"    ⚠️  {symbol}: Insufficient data")
                
        except Exception as e:
            logger.error(f"    ❌ {symbol}: {e}")
    
    logger.info(f"\n✅ Fetched data for {len(historical_data)}/{len(symbols)} symbols")
    return historical_data


def train_production_models(
    historical_data: Dict[str, pd.DataFrame],
    model_dir: str = "models/saved_ultimate",
    force: bool = False,
    enable_deep_learning: bool = False,
) -> UltimateSignalGenerator:
    """Train production models with enhanced features."""
    
    # Check if models already exist
    if os.path.exists(f"{model_dir}/metadata.pkl") and not force:
        logger.warning(f"Models already exist in {model_dir}")
        logger.warning("Use --force to retrain")
        
        # Load existing models
        generator = UltimateSignalGenerator(model_dir=model_dir)
        if generator.loadModels():
            logger.info("✅ Loaded existing models")
            return generator
        else:
            logger.warning("Failed to load existing models, will retrain")
    
    # Initialize generator
    logger.info("\n" + "="*80)
    logger.info("INITIALIZING PRODUCTION MODEL TRAINING")
    logger.info("="*80)
    
    generator = UltimateSignalGenerator(
        model_dir=model_dir,
        lookback=60,
        n_cv_splits=4,
        purge_gap=5,
        embargo_gap=2,
        random_seed=42,
        enable_deep_learning=enable_deep_learning,
    )
    
    # Train
    logger.info("\n🚀 Starting training with enhanced features...")
    logger.info(f"   Total symbols: {len(historical_data)}")
    logger.info(f"   Total features: 67 (35 baseline + 32 enhanced)")
    logger.info(f"   Deep learning: {enable_deep_learning}")
    logger.info(f"   Model directory: {model_dir}")
    
    results = generator.train(
        historical_data=historical_data,
        target_horizon=5,
        min_samples_per_regime=200
    )
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING SUMMARY")
    logger.info("="*80)
    
    for regime, metrics_list in results.items():
        if not metrics_list:
            continue
        
        avg_acc = sum(m.directional_accuracy for m in metrics_list) / len(metrics_list)
        avg_train_mse = sum(m.train_mse for m in metrics_list) / len(metrics_list)
        avg_val_mse = sum(m.val_mse for m in metrics_list) / len(metrics_list)
        
        logger.info(f"\n{regime.upper():12s}:")
        logger.info(f"  Directional Accuracy: {avg_acc:.1%}")
        logger.info(f"  Train MSE:            {avg_train_mse:.6f}")
        logger.info(f"  Val MSE:              {avg_val_mse:.6f}")
        logger.info(f"  Overfit Ratio:        {avg_val_mse/avg_train_mse:.2f}")
        
        # Top features
        if metrics_list[0].feature_importance:
            top_features = sorted(
                metrics_list[0].feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            logger.info(f"  Top 5 Features:")
            for feat, imp in top_features:
                logger.info(f"    {feat:25s}: {imp:.4f}")
    
    return generator


def validate_models(generator: UltimateSignalGenerator, test_data: Dict[str, pd.DataFrame]):
    """Validate trained models on test data."""
    logger.info("\n" + "="*80)
    logger.info("MODEL VALIDATION")
    logger.info("="*80)
    
    results = []
    
    for symbol, data in test_data.items():
        try:
            signal = generator.generate_signal(
                symbol=symbol,
                data=data,
                sentiment_score=0.0,
                momentum_rank=0.5
            )
            
            results.append({
                'symbol': symbol,
                'signal': signal.signal,
                'confidence': signal.confidence,
                'regime': signal.regime,
                'data_quality': signal.data_quality
            })
            
        except Exception as e:
            logger.error(f"Validation failed for {symbol}: {e}")
    
    # Summary
    if results:
        df_results = pd.DataFrame(results)
        logger.info(f"\n✅ Validated {len(results)} symbols")
        logger.info(f"   Avg Confidence: {df_results['confidence'].mean():.2%}")
        logger.info(f"   Avg Data Quality: {df_results['data_quality'].mean():.2%}")
        logger.info(f"\n   Regime Distribution:")
        for regime, count in df_results['regime'].value_counts().items():
            logger.info(f"     {regime:12s}: {count:3d} ({count/len(results):.1%})")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train production models with enhanced features")
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols')
    parser.add_argument('--days', type=int, default=2000, help='Days of historical data (default: 2000 for bear market coverage)')
    parser.add_argument('--force', action='store_true', help='Force retrain')
    parser.add_argument('--validate', action='store_true', help='Run validation')
    parser.add_argument('--model-dir', type=str, default='models/saved_ultimate',
                       help='Model directory')
    parser.add_argument('--enable-deep-learning', action='store_true',
                       help='Enable LSTM and Transformer models (requires torch)')

    args = parser.parse_args()
    
    # Get symbols
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    else:
        symbols = DEFAULT_SYMBOLS
        logger.info(f"Using default symbols: {symbols}")
    
    logger.info("="*80)
    logger.info("PRODUCTION MODEL TRAINING WITH ENHANCED FEATURES")
    logger.info("="*80)
    logger.info(f"Timestamp: {datetime.now()}")
    logger.info(f"Symbols: {len(symbols)}")
    logger.info(f"Historical days: {args.days}")
    logger.info(f"Model directory: {args.model_dir}")
    logger.info(f"Force retrain: {args.force}")
    logger.info(f"Deep learning: {args.enable_deep_learning}")
    
    try:
        # Fetch data
        historical_data = fetch_historical_data(symbols, args.days)
        
        if len(historical_data) < 5:
            logger.error(f"Insufficient data: only {len(historical_data)} symbols")
            logger.error("Need at least 5 symbols for robust training")
            return 1
        
        # Train models
        generator = train_production_models(
            historical_data=historical_data,
            model_dir=args.model_dir,
            force=args.force,
            enable_deep_learning=args.enable_deep_learning,
        )
        
        # Validate if requested
        if args.validate:
            # Use last 200 bars of each symbol for validation
            test_data = {
                symbol: df.iloc[-200:] 
                for symbol, df in historical_data.items()
            }
            validate_models(generator, test_data)
        
        # Final summary
        logger.info("\n" + "="*80)
        logger.info("✅ TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"Models saved to: {args.model_dir}")
        logger.info(f"Training date: {generator.training_date}")
        logger.info(f"\n🚀 Next steps:")
        logger.info(f"   1. Review training metrics above")
        logger.info(f"   2. Update main.py to use new models")
        logger.info(f"   3. Monitor live performance for 1-2 weeks")
        logger.info(f"   4. Compare against baseline (51-56% accuracy)")
        logger.info(f"   5. Document results in ENHANCED_FEATURES.md")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
