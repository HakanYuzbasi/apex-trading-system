import { ReactNode } from "react";

export type ControlsPanelProps = {
  children: ReactNode;
};

export default function ControlsPanel({ children }: ControlsPanelProps) {
  return <div className="flex flex-wrap items-center gap-2">{children}</div>;
}
