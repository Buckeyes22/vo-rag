-- 032: swap the cloud_cost workload CHECK to the static-router vocabulary.
-- Non-destructive: the table itself and every existing row stay put. Only the
-- inline CHECK constraint (which PostgreSQL auto-named when 015 ran) is
-- dropped and replaced with one bound to the static-router workload set:
--   text-gen / vlm / dsi / embedding-batch / misc
-- PostgreSQL gives the inline workload CHECK from 015 the deterministic name
-- cloud_cost_workload_check. Drop that exact name so both the legacy schema and
-- an already-correct restored schema converge without expression-string matching.
-- See tests/core/db/test_cloud_cost_schema.py for the contract.

ALTER TABLE public.cloud_cost
    DROP CONSTRAINT IF EXISTS cloud_cost_workload_check;

ALTER TABLE public.cloud_cost
    ADD CONSTRAINT cloud_cost_workload_check
    CHECK (workload IN ('text-gen', 'vlm', 'dsi', 'embedding-batch', 'misc'))
    NOT VALID;

-- A restored production database contains historical rows written under the
-- previous vocabulary. NOT VALID preserves those immutable cost records while
-- still enforcing the new vocabulary for every new/updated row. Fresh or
-- already-converged databases validate the constraint immediately.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM public.cloud_cost
        WHERE workload NOT IN ('text-gen', 'vlm', 'dsi', 'embedding-batch', 'misc')
    ) THEN
        ALTER TABLE public.cloud_cost
            VALIDATE CONSTRAINT cloud_cost_workload_check;
    END IF;
END
$$;
