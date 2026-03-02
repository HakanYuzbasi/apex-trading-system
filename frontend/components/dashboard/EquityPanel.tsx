import { ReactNode } from "react";

export type EquityPanelProps = {
  children: ReactNode;
};

export default function EquityPanel({ children }: EquityPanelProps) {
  return <section className="apex-density-grid apex-stagger grid grid-cols-2 gap-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-5">{children}</section>;
}
