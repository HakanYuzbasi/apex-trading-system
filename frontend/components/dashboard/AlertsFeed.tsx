import { ReactNode } from "react";

export type AlertsFeedProps = {
  children: ReactNode;
};

export default function AlertsFeed({ children }: AlertsFeedProps) {
  return <section className="sticky top-2 z-30" aria-live="polite" aria-atomic="false">{children}</section>;
}
