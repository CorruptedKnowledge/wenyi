import { expect, test, type Page } from "@playwright/test";
import { fakeApi, pid } from "../../sandbox/fixtures";
import { chooseOption } from "./select-helper";

/** Whether the Desktop host must keep running this suite as well.
 *
 * Bulk glossary cleanup is shared page logic over plain HTTP requests; it uses no native
 * dialog, file or IPC capability. The Desktop bootstrap smoke still renders the page.
 */
export const desktopHostSensitive = false;

export function registerTests({ apiOrigin }: { apiOrigin?: string } = {}) {
  const base = apiOrigin ?? "**/api";
  const term = (source: string, type = "term") => ({
    source, target: `Translation ${source}`, type, reading: "", note: "", gender: "", aliases: [],
  });
  const conflict = (id: number, source: string) => ({
    id, source, existing_target: `Current ${source}`, proposed_target: `Other ${source}`,
    chapter: 1, note: null,
  });

  async function posted(page: Page, path: string, json: unknown) {
    const bodies: unknown[] = [];
    await page.route(`${base}/projects/${pid}/glossary/${path}`, (route) => {
      bodies.push(route.request().postDataJSON());
      return route.fulfill({ json });
    });
    return bodies;
  }

  test("keep current for all closes every conflict after confirmation", async ({ page }) => {
    let open = Array.from({ length: 150 }, (_, i) => conflict(i + 1, `Word ${i % 120}`));
    await fakeApi(page, { [`/projects/${pid}/glossary/terms`]: [term("Word 0")] }, apiOrigin);
    await page.route(`${base}/projects/${pid}/glossary/conflicts`, (route) =>
      route.fulfill({ json: open }));
    const calls: string[] = [];
    await page.route(`${base}/projects/${pid}/glossary/conflicts/keep-current`, (route) => {
      calls.push(route.request().method());
      open = [];
      return route.fulfill({ json: { terms: 120, conflicts: 150 } });
    });
    const prompts: string[] = [];
    page.on("dialog", (dialog) => {
      prompts.push(dialog.message());
      return dialog.accept();
    });
    await page.goto(`/projects/${pid}/glossary`);
    await expect(page.getByText("Open conflicts (150)", { exact: true })).toBeVisible();
    // Long conflict lists page like the glossary instead of rendering every row.
    const pager = page.getByRole("navigation", { name: "Conflict pages", exact: true });
    await expect(pager).toContainText("Page 1 / 2");
    await expect(page.getByRole("button", { name: "Keep current", exact: true })).toHaveCount(100);
    await pager.getByRole("button", { name: "Next page", exact: true }).click();
    await expect(page.getByRole("button", { name: "Keep current", exact: true })).toHaveCount(50);

    await page.getByRole("button", { name: "Keep current for all", exact: true }).click();
    await expect(page.getByText("Current translations kept: 120")).toBeVisible();
    expect(prompts).toEqual([expect.stringContaining("(120)")]);
    expect(calls).toEqual(["POST"]);
    await expect(page.getByText(/Open conflicts/)).toHaveCount(0);
  });

  test("delete all matching removes every page of the current filter", async ({ page }) => {
    const terms = [
      ...Array.from({ length: 250 }, (_, i) => term(`Common ${String(i).padStart(3, "0")}`)),
      term("Lance", "person"),
    ];
    await fakeApi(page, { [`/projects/${pid}/glossary/terms`]: terms }, apiOrigin);
    const listed: string[] = [];
    await page.route(`${base}/projects/${pid}/glossary/sources?**`, (route) => {
      listed.push(new URL(route.request().url()).search);
      const generic = terms.filter((entry) => entry.type === "term");
      return route.fulfill({ json: generic.map((entry) => entry.source) });
    });
    const deletions = await posted(page, "bulk-delete", { deleted: 250 });
    page.on("dialog", (dialog) => dialog.accept());
    await page.goto(`/projects/${pid}/glossary`);
    await expect(page.getByRole("table").locator("tbody tr")).toHaveCount(100);
    // Without a search or filter there is nothing narrower than the whole glossary.
    await expect(page.getByRole("button", { name: /Delete all matching/ })).toHaveCount(0);

    await chooseOption(page.getByRole("combobox", { name: "Type", exact: true }), "Terms");
    await page.getByRole("button", { name: "Delete all matching (250)", exact: true }).click();
    await expect(page.getByText("Terms deleted: 250")).toBeVisible();
    expect(listed).toEqual(["?type=term"]);
    expect(deletions).toHaveLength(1);
    const { sources } = deletions[0] as { sources: string[] };
    expect(sources).toHaveLength(250);
    expect(sources).not.toContain("Lance");
  });

  test("delete all matching stops when the matching list changed", async ({ page }) => {
    const terms = Array.from({ length: 3 }, (_, i) => term(`Common ${i}`));
    await fakeApi(page, { [`/projects/${pid}/glossary/terms`]: terms }, apiOrigin);
    await page.route(`${base}/projects/${pid}/glossary/sources?**`, (route) =>
      route.fulfill({ json: ["Common 0", "Common 1", "Common 2", "Added meanwhile"] }));
    const deletions = await posted(page, "bulk-delete", { deleted: 0 });
    page.on("dialog", (dialog) => dialog.accept());
    await page.goto(`/projects/${pid}/glossary`);
    await page.getByRole("main").getByRole("textbox").fill("Common");
    await page.getByRole("button", { name: "Delete all matching (3)", exact: true }).click();
    await expect(page.getByText("The matching terms changed. Check the list and try again."))
      .toBeVisible();
    expect(deletions).toEqual([]);
  });

  test("one-off cleanup previews the terms and deletes exactly those", async ({ page }) => {
    const oneOff = Array.from({ length: 25 }, (_, i) => term(`Once ${String(i).padStart(2, "0")}`));
    await fakeApi(page, { [`/projects/${pid}/glossary/terms`]: [...oneOff, term("Lance")] },
      apiOrigin);
    let release!: () => void;
    const gate = new Promise<void>((resolve) => { release = resolve; });
    await page.route(`${base}/projects/${pid}/glossary/one-off`, async (route) => {
      await gate;
      return route.fulfill({ json: oneOff });
    });
    const deletions = await posted(page, "bulk-delete", { deleted: 25 });
    await page.goto(`/projects/${pid}/glossary`);
    await page.getByRole("button", { name: "Remove one-off terms", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "Remove one-off terms", exact: true });
    await expect(dialog.getByRole("status")).toContainText("Checking every chapter");
    const remove = dialog.getByRole("button", { name: /Delete these terms/ });
    await expect(remove).toBeDisabled();
    release();
    await expect(dialog).toContainText("fewer than twice in the whole book: 25");
    await expect(dialog.getByRole("listitem")).toHaveCount(20);
    await expect(dialog).toContainText("More not shown: 5");
    await dialog.getByRole("button", { name: "Delete these terms (25)", exact: true }).click();
    await expect(page.getByText("Terms deleted: 25")).toBeVisible();
    await expect(dialog).toHaveCount(0);
    expect(deletions).toEqual([{ sources: oneOff.map((entry) => entry.source) }]);
  });

  test("one-off cleanup offers nothing to delete when every term recurs", async ({ page }) => {
    await fakeApi(page, { [`/projects/${pid}/glossary/terms`]: [term("Lance")] }, apiOrigin);
    await page.route(`${base}/projects/${pid}/glossary/one-off`, (route) =>
      route.fulfill({ json: [] }));
    await page.goto(`/projects/${pid}/glossary`);
    await page.getByRole("button", { name: "Remove one-off terms", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "Remove one-off terms", exact: true });
    await expect(dialog).toContainText("Every term occurs at least twice in the book.");
    await expect(dialog.getByRole("button", { name: "Delete these terms (0)", exact: true }))
      .toBeDisabled();
  });
}
