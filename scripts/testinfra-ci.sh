#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

run_key="${GITHUB_RUN_ID:-local-$$}-${GITHUB_RUN_ATTEMPT:-1}"
project="vo-rag-testinfra-${run_key}"
compose=(docker compose --project-name "$project" -f docker-compose.testinfra.yml)

# Port zero asks Docker for a free loopback port, so concurrent branch jobs and
# a developer's fixed-port `make up` stack cannot cross-connect or collide.
export VORAG_TEST_QDRANT_PORT="${VORAG_TEST_QDRANT_PORT:-0}"
export VORAG_TEST_POSTGRES_PORT="${VORAG_TEST_POSTGRES_PORT:-0}"

# The `default` runner label is served by both host-mode and containerized
# (act image + host docker socket) runners.  In a containerized job the
# compose stack's 127.0.0.1-published ports live on the daemon host, not in
# this network namespace, so the job must join the stack's network and speak
# to the services by DNS name instead.
in_container=0
if [ -f /.dockerenv ]; then
  in_container=1
  self_id="$(cat /etc/hostname)"
fi
network_joined=0

cleanup() {
  local status=$?
  if [ "$status" != "0" ]; then
    # Server-side evidence for the failure (stalls, crashes) before teardown.
    "${compose[@]}" logs --no-color --tail=200 qdrant postgres || true
  fi
  if [ "$network_joined" = "1" ]; then
    docker network disconnect "${project}_default" "$self_id" >/dev/null 2>&1 || true
  fi
  "${compose[@]}" down --volumes --remove-orphans
}
trap cleanup EXIT INT TERM

# A cancelled job's container is killed outright, so the EXIT trap above never
# runs and its stack stays up (orphans ran for days on localhost). Before
# starting, remove other CI stacks (vo-rag-testinfra-<run>-<attempt>) on this
# daemon that are older than any job can run. Local `local-<pid>` stacks and
# developer `make up` stacks never match the name pattern.
sweep_stale_stacks() {
  local max_age="${VORAG_TESTINFRA_STALE_S:-7200}" now name created
  now="$(date +%s)"
  docker ps -a --format '{{.Names}}' \
    | sed -nE 's/^(vo-rag-testinfra-[0-9]+-[0-9]+)-[a-z]+-[0-9]+$/\1/p' \
    | sort -u \
    | while read -r name; do
        [ "$name" = "$project" ] && continue
        created="$(docker ps -a --format '{{.CreatedAt}}' --filter "name=^${name}-" | head -1 | cut -d' ' -f1-3)"
        [ -n "$created" ] || continue
        if [ $((now - $(date -d "$created" +%s))) -ge "$max_age" ]; then
          echo "removing stale testinfra stack $name (created $created)"
          docker ps -a --format '{{.Names}}' | grep "^${name}-" | xargs -r docker rm -f >/dev/null || true
          docker volume ls --format '{{.Name}}' | grep "^${name}_" | xargs -r docker volume rm >/dev/null || true
          docker network ls --format '{{.Name}}' | grep "^${name}_" | xargs -r docker network rm >/dev/null || true
        fi
      done
}
sweep_stale_stacks

echo "== testinfra bootstrap =="
uv sync --locked --extra dev --extra llamaindex

echo "== testinfra services =="
"${compose[@]}" up -d --wait

if [ "$in_container" = "1" ]; then
  docker network connect "${project}_default" "$self_id"
  network_joined=1
  qdrant_host="qdrant"
  qdrant_port=6333
  postgres_host="postgres"
  postgres_port=5432
else
  qdrant_address=$("${compose[@]}" port qdrant 6333)
  postgres_address=$("${compose[@]}" port postgres 5432)
  qdrant_host="127.0.0.1"
  postgres_host="127.0.0.1"
  qdrant_port="${qdrant_address##*:}"
  postgres_port="${postgres_address##*:}"
fi

echo "== migration + qdrant integration =="
test_database_url="postgresql://vorag:vorag@${postgres_host}:${postgres_port}/vorag_test"  # pragma: allowlist secret
VORAG_REQUIRE_TESTINFRA=1 \
VORAG_TEST_DATABASE_URL="${test_database_url}" \
VORAG_TEST_QDRANT_URL="http://${qdrant_host}:${qdrant_port}" \
uv run --no-sync pytest -q -rs \
  tests/core/db/test_postgres_integration.py \
  tests/core/db/test_cloud_cost_schema.py \
  tests/core/db/test_source_publication_integration.py \
  tests/core/db/test_source_publication_qdrant_admission.py \
  tests/core/storage/test_qdrant_integration.py \
  tests/core/storage/test_source_revision_writer.py \
  tests/core/retrieval/test_hybrid_integration.py \
  tests/core/graph/test_persistence_integration.py \
  tests/core/corpus/test_purge_integration.py \
  tests/core/corpus/test_health_integration.py \
  tests/core/corpus/test_duplicates_integration.py \
  tests/core/corpus/test_boilerplate_integration.py \
  tests/core/corpus/test_tf_store_integration.py \
  tests/core/observability/test_ingest_metrics_integration.py \
  tests/pipelines/book_rag/test_dispatcher_claim.py
