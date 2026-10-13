import { render, screen } from "@testing-library/react";
import App from "@/App";

describe("App scaffold", () => {
  it("renders the vo-rag portal heading", () => {
    render(<App />);
    expect(screen.getByRole("heading", { name: /vo-rag portal/i })).toBeInTheDocument();
  });

  it("applies Tailwind classes on main element", () => {
    const { container } = render(<App />);
    const main = container.querySelector("main");
    expect(main?.className).toMatch(/min-h-screen/);
    expect(main?.className).toMatch(/bg-zinc-950/);
  });
});
