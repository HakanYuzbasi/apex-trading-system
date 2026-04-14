"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { BarChart3, Bitcoin, Layers, ShieldAlert } from "lucide-react";

const sessionLinks = [
  { href: "/dashboard", label: "Overview", icon: Layers },
  { href: "/dashboard/core", label: "Core Strategy", icon: BarChart3 },
  { href: "/dashboard/crypto", label: "Crypto Sleeve", icon: Bitcoin },
  { href: "/dashboard/broker-sync", label: "Broker Sync", icon: ShieldAlert },
];

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();

  return (
    <div className="min-h-screen bg-background">
      {/* Session Navigation Bar */}
      <nav className="sticky top-0 z-40 border-b border-border/60 bg-background/95 backdrop-blur-sm">
        <div className="mx-auto flex max-w-[1600px] items-center gap-1 px-4 py-2">
          <span className="mr-4 text-sm font-semibold text-foreground/70">
            APEX Terminal
          </span>
          {sessionLinks.map(({ href, label, icon: Icon }) => {
            const isActive =
              href === "/dashboard"
                ? pathname === "/dashboard"
                : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-primary/10 text-primary"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground"
                }`}
              >
                <Icon className="h-4 w-4" />
                {label}
              </Link>
            );
          })}
        </div>
      </nav>
      {children}
    </div>
  );
}
