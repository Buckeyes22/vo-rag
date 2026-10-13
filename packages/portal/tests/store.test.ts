import { useAppStore } from "@/stores/app";

describe("useAppStore", () => {
  beforeEach(() => {
    useAppStore.setState({
      queryMode: "internal",
      safePublicationMode: true,
      csrfToken: "",
      sessionEmail: "",
      authSource: "",
    });
  });

  it("setQueryMode accepts the internal backend mode", () => {
    useAppStore.getState().setQueryMode("internal");
    expect(useAppStore.getState().queryMode).toBe("internal");
  });

  it("default queryMode is internal", () => {
    expect(useAppStore.getState().queryMode).toBe("internal");
  });

  it("setSession stores non-persisted Access session fields", () => {
    useAppStore.getState().setSession({
      csrfToken: "csrf",
      email: "admin@example.com",
      authSource: "cloudflare_access",
    });
    expect(useAppStore.getState().csrfToken).toBe("csrf");
    expect(useAppStore.getState().sessionEmail).toBe("admin@example.com");
  });
});
