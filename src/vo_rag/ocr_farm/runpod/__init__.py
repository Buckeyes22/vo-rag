"""RunPod OCR farm as a versioned job (VOR-1434).

``vo-rag corpus ocr-farm runpod plan|run|collect|verify`` runs the farm as a
job. The pieces:

- ``pitwall.py``: pod create/get/terminate, catalogue price and remaining
  budget through the local ``pitwall`` CLI (an injected runner; tests use fakes).
- ``job.py``: the job directory, the cost estimate and the budget refusal,
  pod start-up (bootstrap, unit push, pool start) and result pulls.
- ``stage.py``: source books -> <=400-page unit PDFs, sharded across pods.
- ``merge_upload.py``: unit JSONs -> one docling.json per book, uploaded as
  ``ocr/<sha256>/docling.json`` then ``meta.json`` (``--overwrite`` for re-dos).
- ``verify.py``: every book's cache is in Garage with ``pipeline: ocr``.
- ``bootstrap.sh`` / ``run_units.sh``: what runs on the pod.
"""
