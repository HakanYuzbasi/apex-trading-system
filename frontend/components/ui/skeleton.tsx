export function Skeleton({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div aria-hidden="true" className={`apex-skeleton ${className ?? ""}`} {...props} />;
}
