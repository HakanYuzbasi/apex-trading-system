import { ReactNode } from "react";

export type PositionsTableProps = {
  children: ReactNode;
};

export default function PositionsTable({ children }: PositionsTableProps) {
  return <article className="apex-panel apex-fade-up rounded-2xl p-5">{children}</article>;
}
