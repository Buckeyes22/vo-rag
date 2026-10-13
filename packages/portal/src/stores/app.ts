import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { QueryMode } from "@/schemas";

type AppState = {
  queryMode: QueryMode;
  setQueryMode: (m: QueryMode) => void;
  safePublicationMode: boolean;
  setSafePublicationMode: (enabled: boolean) => void;
  csrfToken: string;
  sessionEmail: string;
  authSource: string;
  setSession: (session: {
    csrfToken?: string | null;
    email: string;
    authSource: string;
  }) => void;
};

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      queryMode: "internal" as const,
      setQueryMode: (queryMode) => set({ queryMode }),
      safePublicationMode: true,
      setSafePublicationMode: (safePublicationMode) => set({ safePublicationMode }),
      csrfToken: "",
      sessionEmail: "",
      authSource: "",
      setSession: ({ csrfToken, email, authSource }) =>
        set({
          csrfToken: csrfToken ?? "",
          sessionEmail: email,
          authSource,
        }),
    }),
    {
      name: "vorag-portal",
      version: 2,
      migrate: (persisted) => {
        const state = persisted as Partial<AppState> | undefined;
        return {
          queryMode: "internal" as const,
          safePublicationMode: state?.safePublicationMode ?? true,
        };
      },
      partialize: (state) => ({
        queryMode: state.queryMode,
        safePublicationMode: state.safePublicationMode,
      }),
    },
  ),
);
