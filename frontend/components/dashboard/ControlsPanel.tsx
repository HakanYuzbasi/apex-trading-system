"use client";

import { ReactNode, useState, useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { Command, X, ShieldAlert, Zap, Search, Activity, CornerDownLeft } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

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
      className="relative w-64 h-12 bg-background/40 backdrop-blur-md rounded-full overflow-hidden flex items-center border border-negative/30 shadow-inner group"
    >
      <div className="absolute inset-0 flex items-center justify-center text-[10px] font-bold text-muted-foreground tracking-[0.2em] select-none z-0 group-hover:text-negative/40 transition-colors">
        DRAG TO KILL SESSION
      </div>
      <div
        className="absolute left-0 top-0 bottom-0 bg-negative/20 backdrop-blur-sm z-0"
        style={{ width: `${progress}%` }}
      />
      <button
        onMouseDown={() => (isDragging.current = true)}
        onTouchStart={() => (isDragging.current = true)}
        style={{ left: `${progress}%`, transform: `translateX(-${progress}%)` }}
        className="absolute h-10 w-12 bg-negative text-white rounded-full flex items-center justify-center cursor-grab active:cursor-grabbing z-10 shadow-[0_0_15px_rgba(var(--negative),0.4)] transition-transform duration-75 hover:scale-105"
      >
        <Zap className="h-4 w-4 fill-current" />
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
    <div className="fixed inset-0 z-[9999] bg-background/80 backdrop-blur-md flex items-start justify-center pt-[15vh]">
      <div className="glass-card w-full max-w-2xl rounded-2xl shadow-2xl border border-border/50 overflow-hidden animate-in fade-in zoom-in-95 duration-200">
        <div className="p-6 border-b border-border/40 flex items-center gap-4 bg-background/20">
          <Search className="w-5 h-5 text-muted-foreground" />
          <input
            type="text"
            autoFocus
            placeholder="Type a command or search metrics..."
            className="w-full bg-transparent text-foreground text-lg outline-none placeholder:text-muted-foreground"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <Badge variant="outline" className="text-[10px] font-mono opacity-50">ESC</Badge>
        </div>
        <div className="p-3 max-h-[400px] overflow-y-auto custom-scrollbar">
          <div className="px-3 py-2 text-[10px] font-bold text-muted-foreground uppercase tracking-widest">System Overrides</div>
          
          <div className="space-y-1">
            <button onClick={() => setIsOpen(false)} className="w-full text-left px-4 py-3 text-foreground hover:bg-primary/10 rounded-xl flex items-center justify-between group transition-all">
              <div className="flex items-center gap-3">
                <div className="bg-negative/20 text-negative p-2 rounded-lg group-hover:scale-110 transition-transform">
                  <ShieldAlert className="w-4 h-4" />
                </div>
                <div>
                  <p className="font-bold text-sm">Emergency Halt</p>
                  <p className="text-xs text-muted-foreground">Kill all active sessions and order routers</p>
                </div>
              </div>
              <CornerDownLeft className="h-4 w-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
            </button>

            <button onClick={() => setIsOpen(false)} className="w-full text-left px-4 py-3 text-foreground hover:bg-primary/10 rounded-xl flex items-center justify-between group transition-all">
              <div className="flex items-center gap-3">
                <div className="bg-primary/20 text-primary p-2 rounded-lg group-hover:scale-110 transition-transform">
                  <Zap className="w-4 h-4" />
                </div>
                <div>
                  <p className="font-bold text-sm">Force Flatten</p>
                  <p className="text-xs text-muted-foreground">Market liquidate all open positions instantly</p>
                </div>
              </div>
              <CornerDownLeft className="h-4 w-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
            </button>

            <button onClick={() => setIsOpen(false)} className="w-full text-left px-4 py-3 text-foreground hover:bg-primary/10 rounded-xl flex items-center justify-between group transition-all">
              <div className="flex items-center gap-3">
                <div className="bg-positive/20 text-positive p-2 rounded-lg group-hover:scale-110 transition-transform">
                  <Activity className="w-4 h-4" />
                </div>
                <div>
                  <p className="font-bold text-sm">Risk Re-validation</p>
                  <p className="text-xs text-muted-foreground">Trigger full portfolio stress/margin check</p>
                </div>
              </div>
              <CornerDownLeft className="h-4 w-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
            </button>
          </div>
        </div>
        <div className="p-4 bg-muted/20 border-t border-border/40 flex items-center justify-between text-[10px] text-muted-foreground uppercase tracking-widest font-bold">
           <span>Select Item: ↑↓</span>
           <span>Confirm: ↵</span>
           <span>Close: ESC</span>
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
    <div className="flex flex-wrap items-center justify-between gap-6 w-full glass-card p-4 rounded-2xl border border-border/40 backdrop-blur-xl animate-in fade-in duration-500">
      <div className="flex items-center gap-4">
        {children}
        <Button
          variant="outline"
          onClick={() => setCmdOpen(true)}
           className="h-10 px-4 flex items-center gap-2.5 bg-background/40 hover:bg-background/80 border-border/40 hover:border-primary/40 rounded-xl transition-all shadow-sm group"
        >
          <Search className="w-4 h-4 text-muted-foreground group-hover:text-primary transition-colors" />
          <span className="text-sm font-bold text-foreground/80">Command Palette</span>
          <kbd className="ml-2 font-mono text-[10px] text-muted-foreground bg-muted/40 px-2 py-0.5 rounded-md border border-border/40 shadow-inner">
            ⌘K
          </kbd>
        </Button>
      </div>

      <div className="flex items-center">
        <SlideToConfirm onConfirm={handleKillSwitch} />
      </div>

      <CommandPalette isOpen={cmdOpen} setIsOpen={setCmdOpen} />
    </div>
  );
}
