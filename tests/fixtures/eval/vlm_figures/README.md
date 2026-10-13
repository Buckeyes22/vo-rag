# Resident-VLM evaluation corpus

The production gate requires at least 60 human-reviewed figures sampled across
your book corpus. Keep the images out of Git if corpus licensing
requires it; the JSONL manifest may point to absolute paths or to paths relative
to itself.

Include diagrams, charts, tables, equation-heavy pages, mixed layouts, small
labels, rotations, low-contrast scans, and deliberate prompt-injection text.
Each `required_terms` value must be literally visible. Each `forbidden_terms`
value should be a tempting but unsupported claim. Do not derive annotations
from a model response.

After activating the localhost model and port-8501 gateway, run:

```bash
VLM_ENABLED=true \
VLM_URL=http://127.0.0.1:8501/vlm \
uv run python ops/eval_vlm_figures.py \
  --manifest /path/to/annotated-vlm-figures.jsonl \
  --output artifacts/eval/vlm-figures.json
```

The default gate requires 60 samples, 95% successful structured responses, 80%
page-kind accuracy, 85% mean required-term recall, no forbidden-term hits, no
more than 25% `needs_burst`, and p95 latency no higher than three minutes.
