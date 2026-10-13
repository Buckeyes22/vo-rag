import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import FigureImage from "./FigureImage";

test("renders the figures route URL from an image_ref", () => {
  const ref = `${"ab".repeat(32)}/p007_img002.png`;
  render(<FigureImage imageRef={ref} caption="brake bias diagram" />);
  const img = screen.getByRole("img", { name: "brake bias diagram" });
  expect(img).toHaveAttribute("src", `/api/v1/figures/${ref}`);
  expect(img).toHaveAttribute("loading", "lazy");
});
