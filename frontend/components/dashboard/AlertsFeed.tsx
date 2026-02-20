import { ReactNode } from "react";

export type AlertsFeedProps = {
  children: ReactNode;
};

export default function AlertsFeed({ children }: AlertsFeedProps) {
  return <section className="sticky top-2 z-30">{children}</section>;
}
