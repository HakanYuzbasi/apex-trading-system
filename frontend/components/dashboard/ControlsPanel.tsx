"use client";

import { ReactNode, useState, useEffect, useRef } from "react";
import { createPortal } from "react-dom";

export type ControlsPanelProps = {
  children?: ReactNode;
  onKillSwitch?: () => void;
};

// --- SLIDE-TO-CONFIRM KILL SWITCH ---
function SlideToConfirm({ onConfirm }: { onConfirm: () => void }) {
  const [progress, setProgress] = useState(0);
  const isDragging = useRef(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMove = (clientX: number) => {
    if (!isDragging.current || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    let newProgress = ((clientX - rect.left) / rect.width) * 100;
    newProgress = Math.max(0, Math.min(newProgress, 100));
    setProgress(newProgress);

    if (newProgress > 95) {
      setProgress(100);
      isDragging.current = false;
      onConfirm();
    }
  };

  const stopDrag = () => {
    isDragging.current = false;
    if (progress < 100) setProgress(0);
  };

  useEffect(() => {
    const handleMouseUp = () => stopDrag();
    const handleMouseMove = (e: MouseEvent) => handleMove(e.clientX);
    const handleTouchMove = (e: TouchEvent) => handleMove(e.touches[0].clientX);
    const handleTouchEnd = () => stopDrag();

    window.addEventListener("mouseup", handleMouseUp);
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("touchend", handleTouchEnd);
    window.addEventListener("touchmove", handleTouchMove);

    return () => {
      window.removeEventListener("mouseup", handleMouseUp);
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("touchend", handleTouchEnd);
      window.removeEventListener("touchmove", handleTouchMove);
    };
  }, [progress]);

  return (
    <div
      ref={containerRef}
      className="relative w-56 h-10 bg-slate-950 rounded-full overflow-hidden flex items-center border border-red-900/50 ring-1 ring-black shadow-inner"
    >
      <div className="absolute inset-0 flex items-center justify-center text-[11px] font-bold text-slate-500 tracking-widest select-none z-0">
        SLIDE TO HALT
      </div>
      <div
        className="absolute left-0 top-0 bottom-0 bg-red-600/20 z-0"
        style={{ width: `${progress}%` }}
      />
      <button
        onMouseDown={() => (isDragging.current = true)}
        onTouchStart={() => (isDragging.current = true)}
        style={{ left: `${progress}%`, transform: `translateX(-${progress}%)` }}
        className="absolute h-10 w-12 bg-red-600 rounded-full flex items-center justify-center text-white cursor-grab active:cursor-grabbing z-10 shadow-[0_0_15px_rgba(220,38,38,0.6)] transition-transform duration-75 hover:bg-red-500"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m13 17 5-5-5-5M6 17l5-5-5-5" /></svg>
      </button>
    </div>
  );
}

// --- COMMAND PALETTE (CMD+K) ---
function CommandPalette({ isOpen, setIsOpen }: { isOpen: boolean, setIsOpen: (v: boolean) => void }) {
  const [query, setQuery] = useState("");

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setIsOpen(!isOpen);
      }
      if (e.key === "Escape") setIsOpen(false);
    };
    document.addEventListener("keydown", down);
    return () => document.removeEventListener("keydown", down);
  }, [isOpen, setIsOpen]);

  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  if (!isOpen || !mounted) return null;

  return createPortal(
    <div className="fixed inset-0 z-[9999] bg-black/60 backdrop-blur-sm flex items-start justify-center pt-[20vh]">
      <div className="bg-slate-900 w-full max-w-lg rounded-xl shadow-2xl border border-slate-700 overflow-hidden">
        <div className="p-4 border-b border-slate-800 flex items-center gap-3">
          <svg className="w-5 h-5 text-slate-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8" /><path d="m21 21-4.3-4.3" /></svg>
          <input
            type="text"
            autoFocus
            placeholder="Type a command (e.g. /halt, /flatten)..."
            className="w-full bg-transparent text-white outline-none placeholder:text-slate-500"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <span className="text-xs text-slate-500 border border-slate-700 rounded px-1.5 py-0.5">ESC</span>
        </div>
        <div className="p-2 max-h-64 overflow-y-auto text-sm">
          <div className="px-3 py-1.5 text-xs font-semibold text-slate-500 uppercase tracking-wider">System Actions</div>
          <button onClick={() => setIsOpen(false)} className="w-full text-left px-3 py-2.5 text-slate-300 hover:bg-slate-800 hover:text-white rounded-md flex items-center gap-3 transition-colors">
            <span className="bg-red-500/20 text-red-400 p-1.5 rounded"><svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 6 6 18M6 6l12 12" /></svg></span>
            Trigger Fail-Safe Mode
          </button>
          <button onClick={() => setIsOpen(false)} className="w-full text-left px-3 py-2.5 text-slate-300 hover:bg-slate-800 hover:text-white rounded-md flex items-center gap-3 transition-colors">
            <span className="bg-blue-500/20 text-blue-400 p-1.5 rounded"><svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8" /><path d="M21 3v5h-5" /></svg></span>
            Flatten Portfolio (Liquidate)
          </button>
          <button onClick={() => setIsOpen(false)} className="w-full text-left px-3 py-2.5 text-slate-300 hover:bg-slate-800 hover:text-white rounded-md flex items-center gap-3 transition-colors">
            <span className="bg-emerald-500/20 text-emerald-400 p-1.5 rounded"><svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" /></svg></span>
            Force Macro Risk Check
          </button>
        </div>
      </div>
    </div>,
    document.body
  );
}

// --- MAIN EXPORT ---
export default function ControlsPanel({ children, onKillSwitch }: ControlsPanelProps) {
  const [cmdOpen, setCmdOpen] = useState(false);

  const handleKillSwitch = () => {
    console.warn("🚨 EMERGENCY HALT INITIATED FROM UI");
    if (onKillSwitch) onKillSwitch();
  };

  return (
    <div className="flex flex-wrap items-center justify-between gap-4 w-full bg-slate-900/50 p-4 rounded-xl border border-slate-800 backdrop-blur-sm">
      <div className="flex items-center gap-3">
        {children}
        <button
          onClick={() => setCmdOpen(true)}
          className="flex items-center gap-2 px-3 py-2 bg-slate-800/80 hover:bg-slate-700 text-slate-300 text-sm rounded-lg transition-colors border border-slate-700 shadow-sm"
        >
          <svg className="w-4 h-4 text-slate-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="m21 21-4.3-4.3" /><circle cx="11" cy="11" r="8" /></svg>
          Command Palette
          <kbd className="ml-2 font-mono text-[10px] text-slate-400 bg-slate-950 px-1.5 py-0.5 rounded border border-slate-700">⌘K</kbd>
        </button>
      </div>

      <div className="flex items-center">
        <SlideToConfirm onConfirm={handleKillSwitch} />
      </div>

      <CommandPalette isOpen={cmdOpen} setIsOpen={setCmdOpen} />
    </div>
  );
}
