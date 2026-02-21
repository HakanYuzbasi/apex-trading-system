/**
 * @jest-environment jsdom
 */
import { render, screen } from "@testing-library/react";
import { Skeleton } from "../ui/skeleton";

describe("Skeleton component", () => {
  it("renders a div with apex-skeleton class", () => {
    const { container } = render(<Skeleton />);
    const el = container.firstChild as HTMLElement;
    expect(el).toBeInTheDocument();
    expect(el.tagName).toBe("DIV");
    expect(el.className).toContain("apex-skeleton");
  });

  it("is hidden from screen readers via aria-hidden", () => {
    const { container } = render(<Skeleton />);
    const el = container.firstChild as HTMLElement;
    expect(el).toHaveAttribute("aria-hidden", "true");
  });

  it("applies additional className", () => {
    const { container } = render(<Skeleton className="h-5 w-20" />);
    const el = container.firstChild as HTMLElement;
    expect(el.className).toContain("apex-skeleton");
    expect(el.className).toContain("h-5 w-20");
  });

  it("passes through extra HTML attributes", () => {
    render(<Skeleton data-testid="test-skeleton" style={{ height: 20 }} />);
    const el = screen.getByTestId("test-skeleton");
    expect(el).toBeInTheDocument();
    expect(el.style.height).toBe("20px");
  });

  it("renders without className gracefully", () => {
    const { container } = render(<Skeleton />);
    const el = container.firstChild as HTMLElement;
    // className should be "apex-skeleton " (no extra classes)
    expect(el.className).toBe("apex-skeleton ");
  });
});
