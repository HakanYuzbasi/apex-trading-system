import type { HTMLAttributes } from "react";

export function Skeleton({ className, ...props }: HTMLAttributes<HTMLSpanElement>) {
  const classes = ["apex-skeleton", "inline-block", "align-middle", className].filter(Boolean).join(" ");
  return <span aria-hidden="true" className={classes} {...props} />;
}
