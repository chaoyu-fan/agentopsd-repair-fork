# Experimental patches retained for audit

These patch files document source-level probes performed during the single-card
investigation. They are **not** part of the paper-result claim and are not applied by
the rerun wrappers in `tools/`.

- `fsdp_lora_dtype.patch`: aligns newly injected PEFT tensors with the actor dtype for
  an FSDP1/LoRA probe.
- `fsdp_lora_orig.patch`: switches an FSDP1 LoRA probe to `use_orig_params`.
- `hf_rollout.diff` and `hf_rollout.patch`: diagnostic variants that omit explicit
  `position_ids` for Qwen2 HF generation; the final evidence treats HF rollout as
  non-claimable because the behavior was not validated as a paper path.

The validated compatibility change for the reported 3B vLLM smoke is separate: it is
the FSDP2-safe wrapped-module fallback recorded in
`evidence/fsdp2_vllm_fail_manifest.json` and in the reproduction matrix.
