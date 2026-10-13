# Development

Use Python 3.12 and uv:

    uv sync --locked --extra dev --extra llamaindex
    make test
    make lint

For the portal: `corepack enable && pnpm install && pnpm -r test`.

Integration tests need a disposable Postgres/Qdrant instance; `docker-compose.testinfra.yml` starts loopback-only test services. Their credentials are intentionally public and must never be reused elsewhere.

Identifiers of the form `VOR-####` in code comments refer to the maintainer's internal issue tracker, which is not public.
