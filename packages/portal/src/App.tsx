import { Navigate, Route, Routes, useInRouterContext } from "react-router-dom";
import Shell from "@/components/layout/Shell";
import BookRag from "@/routes/BookRag";
import Ingest from "@/routes/Ingest";
import Library from "@/routes/Library";
import LibraryDetail from "@/routes/LibraryDetail";
import Ops from "@/routes/Ops";
import Overview from "@/routes/Overview";
import Pipelines from "@/routes/Pipelines";
import Query from "@/routes/Query";
import Settings from "@/routes/Settings";
import System from "@/routes/System";

function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<Shell />}>
        <Route index element={<Navigate to="/overview" replace />} />
        <Route path="overview" element={<Overview />} />
        <Route path="ingest" element={<Ingest />} />
        <Route path="library" element={<Library />} />
        <Route path="library/:id" element={<LibraryDetail />} />
        <Route path="query" element={<Query />} />
        <Route path="pipelines" element={<Pipelines />} />
        <Route path="book-rag" element={<BookRag />} />
        <Route path="system" element={<System />} />
        <Route path="ops" element={<Ops />} />
        <Route path="settings" element={<Settings />} />
      </Route>
    </Routes>
  );
}

export default function App() {
  const inRouter = useInRouterContext();
  if (inRouter) {
    return <AppRoutes />;
  }

  return (
    <main className="min-h-screen bg-zinc-950 text-zinc-100 p-8">
      <h1 className="text-3xl font-bold text-blue-400">vo-rag Portal</h1>
      <p className="mt-2 text-zinc-400">Operator console</p>
    </main>
  );
}
