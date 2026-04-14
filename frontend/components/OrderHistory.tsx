"use client";

import React from 'react';
import { ShoppingCart, History, TrendingUp, TrendingDown } from 'lucide-react';

interface Order {
  time: string;
  symbol: string;
  side: string;
  qty: number;
  price: number;
}

interface OrderHistoryProps {
  orders: Order[];
}

export default function OrderHistory({ orders }: OrderHistoryProps) {
  return (
    <div className="glass-card rounded-2xl border border-slate-800 bg-slate-900/30 overflow-hidden">
      <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/50">
        <div className="flex items-center gap-2">
          <History size={16} className="text-slate-400" />
          <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">Order Execution History</h3>
        </div>
        <span className="text-[10px] font-mono text-slate-600 bg-slate-950 px-2 py-0.5 rounded border border-slate-800">
          Last 50 Fills
        </span>
      </div>
      
      <div className="overflow-x-auto max-h-[350px] overflow-y-auto custom-scrollbar">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-slate-800/50 bg-slate-950/30">
              <th className="px-6 py-3 text-[10px] font-bold uppercase tracking-widest text-slate-500">Time</th>
              <th className="px-6 py-3 text-[10px] font-bold uppercase tracking-widest text-slate-500">Symbol</th>
              <th className="px-6 py-3 text-[10px] font-bold uppercase tracking-widest text-slate-500">Side</th>
              <th className="px-6 py-3 text-[10px] font-bold uppercase tracking-widest text-slate-500 text-right">Quantity</th>
              <th className="px-6 py-3 text-[10px] font-bold uppercase tracking-widest text-slate-500 text-right">Price</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/30">
            {orders.length > 0 ? (
              orders.map((order, i) => {
                const isBuy = order.side.toLowerCase() === 'buy';
                return (
                  <tr key={i} className="hover:bg-slate-800/30 transition-colors group">
                    <td className="px-6 py-3 text-[11px] font-mono text-slate-500">{order.time}</td>
                    <td className="px-6 py-3 text-[11px] font-bold text-slate-200">{order.symbol}</td>
                    <td className="px-6 py-3">
                      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-tighter ${
                        isBuy ? 'bg-emerald-500/10 text-emerald-500' : 'bg-rose-500/10 text-rose-500'
                      }`}>
                        {isBuy ? <TrendingUp size={10} /> : <TrendingDown size={10} />}
                        {order.side}
                      </span>
                    </td>
                    <td className="px-6 py-3 text-[11px] font-mono text-slate-300 text-right tabular-nums">
                      {order.qty.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 4 })}
                    </td>
                    <td className="px-6 py-3 text-[11px] font-mono text-emerald-400/80 text-right tabular-nums group-hover:text-emerald-400 transition-colors">
                      ${order.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 })}
                    </td>
                  </tr>
                );
              })
            ) : (
              <tr>
                <td colSpan={5} className="px-6 py-12 text-center">
                  <div className="flex flex-col items-center gap-2 opacity-30">
                    <ShoppingCart size={32} className="text-slate-500" />
                    <p className="text-[10px] uppercase font-bold tracking-widest text-slate-400">Waiting for live executions...</p>
                  </div>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      
      {/* Table Footer / Visual Fade */}
      <div className="h-2 bg-gradient-to-t from-slate-950/20 to-transparent"></div>
    </div>
  );
}
