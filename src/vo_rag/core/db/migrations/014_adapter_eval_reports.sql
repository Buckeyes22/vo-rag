CREATE TABLE adapter_eval_reports (
    id                       BIGSERIAL PRIMARY KEY,
    adapter_path             TEXT NOT NULL,
    persona                  TEXT NOT NULL CHECK (persona IN ('crew_chief','race_engineer','driving_coach')),
    total_questions          INTEGER NOT NULL,
    ragas_faithfulness       NUMERIC(6,4) NOT NULL,
    ragas_answer_relevancy   NUMERIC(6,4) NOT NULL,
    ragas_context_precision  NUMERIC(6,4) NOT NULL,
    mrr_at_10                NUMERIC(6,4) NOT NULL,
    ndcg_at_10               NUMERIC(6,4) NOT NULL,
    hallucination_rate       NUMERIC(6,4) NOT NULL,
    persona_adherence        NUMERIC(6,4) NOT NULL,
    vs_base_model_delta      JSONB NOT NULL,
    per_question             JSONB NOT NULL,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_eval_reports_adapter ON adapter_eval_reports(adapter_path);
CREATE INDEX idx_eval_reports_persona ON adapter_eval_reports(persona);
