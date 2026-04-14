"use client";

import { useEffect, useState } from "react";
import { Loader2, MailOpen, Activity } from "lucide-react";
import ReactMarkdown from "react-markdown";

interface Briefing {
  timestamp: string;
  date: string;
  content: string;
}

export default function BriefingArchivePanel() {
  const [briefings, setBriefings] = useState<Briefing[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchBriefings() {
      try {
        const res = await fetch("/api/v1/briefings");
        if (!res.ok) {
          throw new Error(`Failed to load briefings: ${res.status}`);
        }
        const data = await res.json();
        setBriefings(data.briefings || []);
      } catch (err) {
        setError(String(err));
      } finally {
        setLoading(false);
      }
    }
    fetchBriefings();
  }, []);

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center rounded-xl border border-border/80 bg-background/50">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-xl border border-negative/30 bg-negative/10 p-6 text-negative">
        <h3 className="mb-2 font-semibold">Error Loading Archive</h3>
        <p className="text-sm">{error}</p>
      </div>
    );
  }

  if (briefings.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 rounded-xl border border-border/80 bg-background/50 text-muted-foreground">
        <Activity className="h-8 w-8 mb-4 opacity-50" />
        <h3 className="font-semibold text-foreground">No Briefings Found</h3>
        <p className="text-sm">The Chief of Staff script has not persisted any daily briefings yet.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-foreground flex items-center gap-2">
          <MailOpen className="h-5 w-5 text-primary" />
          Chief of Staff Archive
        </h2>
        <span className="text-sm text-muted-foreground">{briefings.length} reports available</span>
      </div>

      <div className="grid gap-6 md:grid-cols-1 lg:grid-cols-2">
        {briefings.map((briefing, idx) => (
          <article
            key={idx}
            className="flex flex-col rounded-xl border border-border/80 bg-background/70 shadow-sm transition-all hover:border-primary/30"
          >
            <header className="border-b border-border/50 bg-secondary/20 px-5 py-3">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-foreground">Daily Briefing — {briefing.date}</h3>
                <span className="text-xs text-muted-foreground font-mono">
                  {new Date(briefing.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                </span>
              </div>
            </header>
            <div className="p-5 prose prose-sm prose-invert max-w-none text-muted-foreground prose-headings:text-foreground prose-a:text-primary">
              <ReactMarkdown>{briefing.content}</ReactMarkdown>
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}
