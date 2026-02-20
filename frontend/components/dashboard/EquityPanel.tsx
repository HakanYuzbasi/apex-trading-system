import { ReactNode } from "react";

export type EquityPanelProps = {
  children: ReactNode;
};

export default function EquityPanel({ children }: EquityPanelProps) {
  return <section className="apex-density-grid apex-stagger grid grid-cols-2 gap-3 lg:grid-cols-4 xl:grid-cols-7">{children}</section>;
}
