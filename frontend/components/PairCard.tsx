import React from 'react';
import { ArrowUpRight, ArrowDownLeft, Activity, Pause, Play } from 'lucide-react';

interface PairCardProps {
  pairName: string;
  zScore: number;
  status: string;
  pnl: number;
  onToggle: (name: string) => void;
}

export default function PairCard({ pairName, zScore, status, pnl, onToggle }: PairCardProps) {
  const isPositive = pnl >= 0;
  const isPaused = status === "PAUSED";
  
  return (
    <div className={`glass-card p-5 rounded-xl border transition-all group relative overflow-hidden ${
      isPaused 
        ? 'border-slate-800 bg-slate-950/40 opacity-60 grayscale-[0.5]' 
        : 'border-slate-800 bg-slate-900/50 hover:bg-slate-900/80'
    }`}>
      {/* Paused Overlay Pattern */}
      {isPaused && (
        <div className="absolute inset-0 pointer-events-none opacity-5 bg-[radial-gradient(#475569_1px,transparent_1px)] [background-size:16px_16px]" />
      )}

      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-slate-400 text-xs font-bold uppercase tracking-widest">{pairName}</h3>
          <div className="flex items-center gap-2 mt-1">
            <span className={`h-2 w-2 rounded-full ${
              isPaused ? 'bg-amber-500' : 'bg-emerald-500 animate-pulse'
            }`}></span>
            <span className={`text-sm font-medium ${isPaused ? 'text-amber-500/80' : 'text-slate-200'}`}>
              {status}
            </span>
          </div>
        </div>
        <div className={`p-2 rounded-lg ${
          isPaused ? 'bg-slate-800 text-slate-500' : (isPositive ? 'bg-emerald-500/10 text-emerald-500' : 'bg-rose-500/10 text-rose-500')
        }`}>
          <Activity size={18} />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 mt-6">
        <div>
          <p className="text-[10px] text-slate-500 uppercase font-bold tracking-tighter">Kalman Z-Score</p>
          <p className={`text-xl font-mono font-bold mt-1 ${
            isPaused ? 'text-slate-600' : (Math.abs(zScore) > 2 ? 'text-amber-400' : 'text-slate-100')
          }`}>
            {zScore > 0 ? '+' : ''}{zScore.toFixed(2)}
          </p>
        </div>
        <div className="text-right">
          <p className="text-[10px] text-slate-500 uppercase font-bold tracking-tighter">Unrealized PnL</p>
          <div className={`flex items-center justify-end gap-1 text-xl font-mono font-bold mt-1 ${
            isPaused ? 'text-slate-600' : (isPositive ? 'text-emerald-400' : 'text-rose-400')
          }`}>
            {isPositive ? <ArrowUpRight size={16} /> : <ArrowDownLeft size={16} />}
            ${Math.abs(pnl).toLocaleString(undefined, { minimumFractionDigits: 2 })}
          </div>
        </div>
      </div>

      <div className="mt-6 pt-4 border-t border-slate-800/50 flex justify-between items-center">
        <span className="text-[10px] text-slate-600 font-mono italic">PPO Execution: {isPaused ? 'HALTED' : 'ACTIVE'}</span>
        <button 
          onClick={() => onToggle(pairName)}
          className={`flex items-center gap-1.5 text-[10px] px-3 py-1.5 rounded-md transition-all border uppercase font-bold tracking-wider hover:scale-105 active:scale-95 ${
            isPaused 
              ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400 hover:bg-emerald-500/20' 
              : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
          }`}
        >
          {isPaused ? <Play size={12} fill="currentColor" /> : <Pause size={12} fill="currentColor" />}
          {isPaused ? 'Resume' : 'Pause'}
        </button>
      </div>
    </div>
  );
}
