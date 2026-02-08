"""Bootstrap existing IBKR positions into performance tracker"""
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).parent))
from monitoring.performance_tracker import PerformanceTracker
from config import ApexConfig

def bootstrap_existing_positions():
    print("=" * 80)
    print("📊 BOOTSTRAPPING EXISTING POSITIONS")
    print("=" * 80)
    
    tracker = PerformanceTracker()
    
    # Your ACTUAL positions from IBKR
    positions = [
        {'symbol': 'ADBE', 'qty': 6, 'avg_price': 329.867, 'current_price': 329.10},
        {'symbol': 'AMAT', 'qty': 40, 'avg_price': 269.829, 'current_price': 287.42},
        {'symbol': 'CRM', 'qty': 16, 'avg_price': 252.983, 'current_price': 257.34},
        {'symbol': 'DG', 'qty': 19, 'avg_price': 136.893, 'current_price': 136.29},
    ]
    
    print("\n📈 Recording BUY trades:")
    for pos in positions:
        tracker.record_trade(pos['symbol'], 'BUY', pos['qty'], pos['avg_price'])
        print(f"   ✅ BUY {pos['qty']} {pos['symbol']} @ ${pos['avg_price']:.2f}")
    
    starting_capital = 1_000_000.00
    current_value = 1_128_574.00
    tracker.record_equity(current_value)
    print(f"\n💰 Portfolio Value: ${current_value:,.2f}")
    
    print("\n📁 Exporting to data files...")
    data_dir = ApexConfig.DATA_DIR
    data_dir.mkdir(exist_ok=True)
    
    # Export trades.csv
    trades_data = []
    for trade in tracker.trades:
        trades_data.append({
            'timestamp': trade['timestamp'],
            'symbol': trade['symbol'],
            'side': trade['side'],
            'quantity': trade['quantity'],
            'price': trade['price'],
            'pnl': 0
        })
    pd.DataFrame(trades_data).to_csv(data_dir / "trades.csv", index=False)
    print(f"   ✅ trades.csv created ({len(trades_data)} trades)")
    
    # Export equity_curve.csv - SIMPLE VERSION
    equity_data = [{
        'timestamp': datetime.now().isoformat(),
        'equity': current_value,
        'drawdown': 0.0
    }]
    pd.DataFrame(equity_data).to_csv(data_dir / "equity_curve.csv", index=False)
    print(f"   ✅ equity_curve.csv created")
    
    # Export trading_state.json
    state = {
        'timestamp': datetime.now().isoformat(),
        'capital': current_value,
        'starting_capital': starting_capital,
        'positions': {},
        'daily_pnl': 792.00,
        'total_pnl': current_value - starting_capital,
        'max_drawdown': 0.0,
        'sharpe_ratio': 0.0,
        'win_rate': 0.0,
        'total_trades': len(positions),
        'open_positions': len(positions)
    }
    for pos in positions:
        state['positions'][pos['symbol']] = {
            'qty': pos['qty'],
            'avg_price': pos['avg_price'],
            'current_price': pos['current_price']
        }
    with open(data_dir / "trading_state.json", 'w') as f:
        json.dump(state, f, indent=2)
    print(f"   ✅ trading_state.json created")
    
    print("\n" + "=" * 80)
    print("✅ BOOTSTRAP COMPLETE!")
    print("=" * 80)
    print(f"📊 Trades Recorded: {len(positions)}")
    print(f"💼 Portfolio Value: ${current_value:,.2f}")
    print(f"📈 Total Return: +{((current_value/starting_capital - 1)*100):.2f}%")
    print(f"💰 Total P&L: ${(current_value - starting_capital):+,.2f}")
    print("\n🎯 Dashboard should now show all data correctly!")
    print("   Refresh Streamlit to see updates.")
    print("=" * 80)

if __name__ == "__main__":
    bootstrap_existing_positions()
