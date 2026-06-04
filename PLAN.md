# LibreDiffusion — Iterative Model-Expansion & Validation Plan

> **⚠️ LIVE STATUS lives in the fork tracker, not here.** This file is the original strategic plan
> (the phases/strategy below remain valid). For current execution state, always read:
> `/home/jcelerier/ossia/daydream-streamdiffusion/LIBREDIFFUSION_PROGRESS.md` (the live tracker),
> plus `validation/PHASE0_6_PLAN.md`. Project memory mirrors it under `…/memory/` (esp.
> `validation-and-export-scope.md`).
>
> **Status as of 2026-06-04:**
> - Fork at `/home/jcelerier/ossia/daydream-streamdiffusion` (branch `librediffusion`). uv/CUDA 13.3.
> - **Phase 0 COMPLETE**: daydream validated in pure PyTorch. Exhaustive matrix **1352/1352**, 0 errors.
>   Harness (numeric+perceptual), ControlNet, guidance/neg-prompt, prompt-interp parity, v-prediction
>   (SD2.1-768) all green. **9 daydream bugs found & fixed** (IP-Adapter deferred — fork kvo_cache conflict).
>   Phase 3.x TensorRT optimizations ported (incl. TRT-11/cu13 compat: tactic sources, fp16/tf32 flags, polygraphy).
> - **Phase 1 IN PROGRESS**: new exporter `train-lora-daydream.py` (original preserved as `train-lora.orig.py`,
>   vendored as `tensorrt_orig/`). **SD1.5 engine set builds**, C++-compatible I/O verified; orig-vs-new
>   numerically equivalent (PSNR 52.5dB). **Benchmark**: SD1.5 LCM 2-step @512 → PyTorch 25.7 / TensorRT 52.3 img/s.
> - **SDXL**: our original NEVER traced SDXL from scratch — it consumes the PREBUILT ONNX from
>   `stabilityai/sdxl-turbo-tensorrt` (via SDXLUNetPrebuilt). The SDXL path = build TRT from that prebuilt ONNX.
> - **NEXT**: (c) SDXL via prebuilt ONNX; (d) load engines in the C++ engine + first C++-vs-Python compare.

Goal: grow model support (SD2.1, v-prediction, Hyper-SD/Lightning, SDXS/Vega, ControlNet,
IP-Adapter, StreamV2V) while being able to **validate every step against a Python reference**.

Strategy decided:
- **Reference = Python only** (daydreamlive/StreamDiffusion), perceptual + numeric tolerance.
- **Validate daydream first** with Python "unit tests" before migrating off the bundled copy.
- **First C++ model added = SDXS / Segmind-Vega** (trivial: same SD1.5 UNet I/O).
- **Build a reusable validation harness first.**
- Adopt daydream **vendor-pinned, migrate per-feature** (each feature gated by the harness).

Reference repos (analyzed, in `/tmp/sd-refs` — re-clone as needed):
- `daydreamlive/StreamDiffusion` — migration target; superset of our bundled StreamDiffusion.
- `leejet/stable-diffusion.cpp` — architecture reference (model-type/prediction/scheduler abstraction).
- `moxin-org/Ominix-SD.cpp` — not useful (stale sd.cpp fork). Ignore.

Key coupling fact: the C++ engine couples to the **TensorRT I/O contract** (tensor names,
`2*batch` CFG-doubled batch, fp16, opset-17). Core names (`sample`/`timestep`/
`encoder_hidden_states`/`latent`) are identical between bundled and daydream, so existing
SD1.5/Turbo/SDXL engines keep working. New adapter/V2V engines add inputs the current
`UNetWrapper` rejects (fixed name list + `allInputDimensionsSpecified()`), so adopting each
new export wrapper forces the matching C++ binding work — done per-feature, validated each time.

---

## Phase 0 — Validate daydream as the reference (Python only, no C++ yet)

Objective: prove daydream produces correct, reproducible output across every model × step ×
guidance × noise/batch combination, works under `uv`, and covers our prompt-interpolation
feature — BEFORE we depend on it.

### 0.0 Port our TensorRT optimizations into the daydream fork (CRITICAL — daydream LACKS them)
daydream's TRT build path is **materially weaker** than ours. Adopting it as-is would regress
perf/quality. Our "Phase 3.x" optimizations (absent in daydream) must be ported into the fork:
- [ ] `onnx_opset=19` (daydream=17).
- [ ] `builder_optimization_level=5`, `tiling_optimization_level=FULL`, `tf32=True`, explicit
      `tactic_sources` incl. EDGE_MASK_CONVOLUTIONS (daydream defaults to `tactic_sources=[]` →
      disables tactics), VRAM-tiered workspace (50–75% free), per-engine named timing cache.
- [ ] TF32 torch backends (`matmul/cudnn.conv fp32_precision='tf32'`, CC≥8.0 gated).
- [ ] Advanced ONNX passes in `BaseModel.optimize`: `onnxsim.simplify` + `onnxoptimizer`
      fusion passes (daydream only does fold_constants+cleanup+toposort).
- [ ] Runtime: `PYTORCH_CUDA_ALLOC_CONF` expandable_segments, OMP/MKL/torch thread clamp.
- [ ] TensorRT-RTX path: `USE_TRT_RTX` env + fp16_flag-off-when-RTX + pre-converted fp16.
- [ ] **Add a PERF gate** (latency/fps) to the harness — numeric+perceptual alone won't catch
      a build-optimization regression. Re-validate numeric tolerance after the ONNX fusion passes
      (they alter graph numerics slightly).
- Exit: forked daydream builds engines at parity with (or better than) our current path on perf.

### 0.1 Make daydream installable with uv
daydream is **not uv-ready**: its `pyproject.toml` is ruff-only; deps live in `setup.py`,
expect torch pre-installed, and pin a **forked diffusers** (`varshith15/diffusers@3e3b72f`)
and a **forked IP-Adapter** (`livepeer/Diffusers_IPAdapter@405f87da`).
- [ ] Author a `[project]`/uv `pyproject.toml` for a pinned daydream checkout, mirroring our
      existing torch-cu130 / tensorrt index setup (see bundled `pyproject.toml`).
- [ ] Carry the forked-diffusers and forked-IPAdapter git pins (load-bearing).
- [ ] `uv sync` + import smoke test (`from streamdiffusion import StreamDiffusion`).
- [ ] Vendor-pin a specific daydream commit; record it.
- Exit: `uv run python -c "import streamdiffusion"` succeeds; TensorRT extras resolve.

### 0.2 Build the reusable validation harness (the foundation everything reuses)
A small Python package + CLI that:
- [ ] **Pins determinism**: single seed → `generator.manual_seed`; fix `init_noise`/`stock_noise`;
      seed/replace the ungenerator `torch.randn_like` in the non-batched LCM/TCD loop
      (daydream `pipeline.py` ~954) so Python is reproducible.
- [ ] **Dumps tensors at every pipeline boundary** as `.npy` (fp32): `encode_image` →
      per-step UNet `model_pred` → `scheduler_step_batch` x0 → `x_t_latent_buffer` refill →
      RCFG `stock_noise` update → `decode_image`, plus CLIP embeds (and SDXL
      `text_embeds`/`time_ids`), final RGBA image.
- [ ] **Compare module**: numeric (cosine ≥ ~0.999, max-abs-diff / rel-err thresholds tuned
      for fp16-vs-fp32) on latents/embeds + perceptual (PSNR/SSIM, optional LPIPS) on the
      decoded image. One `assert_matches(ref, got, profile)` with named tolerance profiles.
- [ ] **Fixed fixtures**: a few prompts, a couple of negative prompts, fixed input images for
      img2img, fixed seeds. Stored under `validation/fixtures/`.
- [ ] Golden dumps are produced by the SAME run that exports engines later (one pipeline).
- Exit: harness can dump + self-compare a single PyTorch run to itself (sanity), and dump
        a PyTorch run vs a TensorRT run of the same daydream model (the real gate).

### 0.3 The "unit test" matrix (Python, daydream PyTorch path = source of truth)
For each cell, run daydream and assert output is sane + reproducible across two runs, and (where
a TRT engine is built) PyTorch-vs-TRT matches within tolerance. Matrix axes:
- **Model kind**: SD1.5(LCM), SD-Turbo, SDXL-Turbo, SD2.1 (epsilon-512 and v-pred-768),
  + at least one distilled (SDXS or Vega) and one LoRA-accel (Hyper-SD / SDXL-Lightning / TCD).
- **Steps**: 1, 2, 3, 4.
- **Guidance type (cfg_type)**: none, full, self, initialize. Guidance scale: =1 and >1.
- **add_noise**: off / on.
- **denoising_batch**: off / on. (Note: TCD forces sequential — assert it disables batch.)
- **Mode**: txt2img and img2img.
- [ ] Generate this matrix programmatically (skip invalid combos, e.g. turbo single-step + RCFG).
- [ ] Snapshot golden outputs per cell; CI re-runs and compares.
- Exit: every valid cell is green and reproducible run-to-run.

### 0.4 Prompt-interpolation parity (our branch's special feature)
Our branch: `update_prompts(List[(str,float)])` = normalized **weighted-sum (linear)** blend
(drives C++ `parse_input_string`/`blend_embeds`). Daydream: `prompt_blending`/`prompt_list`
with **linear AND slerp** + seed blending — a superset but different default (slerp).
- [ ] Reproduce our weighted-sum behavior via daydream's **linear** prompt blending; assert the
      blended embedding matches our bundled `update_prompts` numerically.
- [ ] Document the slerp option as a new capability; decide whether the C++ should expose it.
- Exit: a unit test shows daydream-linear == bundled weighted-sum within tolerance.

### 0.5 Go/no-go on migration
- [ ] All of 0.1–0.4 green ⇒ proceed to Phase 1. Otherwise file gaps and either fix in the
      pinned daydream fork or keep the bundled copy for the affected feature.

---

## Phase 1 — Adopt daydream + land the first new model (SDXS / Segmind-Vega)

Objective: prove the end-to-end C++ path against daydream with the cheapest possible new model
(same SD1.5 UNet I/O — no kernel/binding change), exercising the full export→engine→C++→compare loop.

### 1.1 Wire the C++↔Python validation bridge
- [ ] Add a C++ debug/dump mode to librediffusion that writes the same boundary tensors as the
      harness (CLIP embeds, per-step latents, x0, decoded image) for a fixed seed/prompt/input.
- [ ] Extend the harness compare module to diff C++ dumps vs daydream PyTorch dumps.
- Exit: existing SD1.5/Turbo/SDXL C++ output matches daydream within tolerance (regression baseline).

### 1.2 SDXS / Segmind-Vega
- [ ] Export engines from daydream's exporter (Vega = pruned SDXL I/O; SDXS = SD1.5 I/O + tiny VAE).
- [ ] Wire the scheduler table (an unused `SDXS-512-DreamShaper.hpp` already exists) — `#include`
      + `case` in `LibreDiffusion.cpp::updateScheduler`; reuse `MODEL_SD_15`/SDXL config.
- [ ] Add the score-side `Workflow` entry + model-type mapping.
- [ ] Validate C++ vs daydream across the relevant 0.3 matrix subset.
- Exit: SDXS/Vega green in C++ vs Python; export→engine→C++→compare loop proven.

---

## Phase 2 — Generalize model support (refactor toward sd.cpp's abstraction)

Objective: stop hardcoding per-model behavior; make prediction-type and scheduler orthogonal.
Lessons from `stable-diffusion.cpp`: flat variant enum + trait predicates; one denoising contract
`noised = x*c_in; denoised = pred*c_out + x*c_skip`; prediction type = a 3-float `get_scalings`.

### 2.1 SD2.1 + RG-LCM (epsilon, 1024-dim) — easy
- [ ] Add `MODEL_SD_21` (C API + native enum), `text_hidden_dim=1024`, pad token, Workflow entry.
- [ ] Generate SD2.1 LCM scheduler table; add `case` in `updateScheduler`.
- [ ] No kernel change (epsilon). Validate vs daydream (RG-LCM-512 / SD2.1-base).

### 2.2 v-prediction branch — moderate
- [ ] Add `prediction_type` to config + C API; branch `SchedulerStepFunctor` (`kernels.cu:34`):
      v-pred `x0 = alpha*xt - beta*v` vs epsilon `(xt - beta*eps)/alpha`. Mirror in the
      SD-Turbo open-coded path (`librediffusion.workflows.cpp:144-158`).
- [ ] Validate vs daydream on SD2.1-768 (v-pred) and any v-pred LCM.

### 2.3 Refactor scheduler/prediction into orthogonal axes (optional but recommended)
- [ ] Introduce a `get_scalings(idx) -> {c_skip, c_out, c_in}` seam; collapse epsilon/v/flow.
- [ ] Port sd.cpp scalar formulas as documentation/reference (not as runtime dep).

---

## Phase 3 — Accel LoRAs (Hyper-SD, SDXL-Lightning, DMD2, TCD)

Objective: few-step quality. Mostly data + scheduler-generator work; some need prediction-type/sampler branches.

- [ ] Extend `tools/generate_scheduler_tables.py` beyond LCM: DDIM-trailing, Euler-trailing
      (Lightning; 1-step uses `prediction_type=sample`), TCD (stochastic; sequential path).
- [ ] DMD2: reuses existing LCM table (fixed timesteps {999,749,499,249}) — but **CC-BY-NC-SA
      (non-commercial)**: confirm licensing before shipping.
- [ ] Hyper-SD / SDXL-Lightning: merge LoRA pre-export (daydream exporter), add table, validate.
- [ ] TCD: replicate daydream's sequential non-batched path; do not reuse LCM batched buffer math.
- Validate each vs daydream across the step/guidance matrix.

---

## Phase 4 — Conditioning adapters

Objective: ControlNet / IP-Adapter. These break the fixed UNet input signature; adopt daydream's
export wrappers + matching C++ binding, per-feature, validated.

### 4.1 IP-Adapter (recommended first adapter — cheapest at runtime)
- [ ] Use daydream's `unet_ipadapter_export` (attention procs baked at export). New UNet input
      `image_embeds`; `encoder_hidden_states` extended by `num_image_tokens` (4/16); runtime
      `ipadapter_scale` is a **per-layer vector**, not a scalar.
- [ ] C++: new `UNetWrapper::forward_ipadapter` binding the extra input by name; CLIP-image
      encoder wrapper (embed computed once per ref image); config fields; `CachedEngine` member;
      reference-image port in `inputs_t`. (`SD_*_IPADAPTER` Workflow enums already reserved.)
- [ ] Validate vs daydream `IPcompositionHyperSD*` style pipelines.

### 4.2 ControlNet — hard
- [ ] Use daydream's controlnet export (separate `cnet.engine`; UNet inputs `input_control_00..NN`
      + `input_control_middle`; SD1.5 = 12 down + 1 mid, SDXL = 9 + 1).
- [ ] C++: `ControlNetWrapper`, residual binding in a new UNet forward overload, control-image
      texture port, C-API + cache + per-step orchestration (ControlNet runs every step).
- [ ] Reference injection math: sd.cpp `unet.hpp:540-559` (scaled add into bottleneck + reversed
      skip residuals). Validate vs daydream `controlnet*` pipelines.

---

## Phase 5 — StreamV2V completion (fix the two real bugs)

The existing `MODE_TEMPORAL_V2V` is currently **inert** (audited): (1) feature injection runs
AFTER `unet_->forward()` so it's discarded; (2) it caches attention OUTPUTS not K/V (real
StreamV2V concatenates banked K/V pre-softmax — Extended Attention — entirely missing); no
token-merge bank update; `warp_noise` unused.
- [ ] Export a real `UNetV2V` engine emitting `attention_0..15` / `kvo_cache_*` (daydream supports it).
- [ ] Make injection take effect (inputs consumed mid-graph) and add Extended Attention (K/V concat).
- [ ] Validate vs daydream's cached-attention V2V on a fixed video clip (temporal-consistency metric).

---

## Phase 6 — Beyond (separate engine projects, not drop-ins)

DiT / flow-matching (SD3.5, Z-Image), temporal/3D (T2V-Turbo, AnimateDiff), Wan2.1 autoregressive
(CausVid/Self-Forcing). Not real-time on consumer GPUs and/or require a new engine + new scheduler
math + non-CLIP text encoders. Track but defer.

---

## Cross-cutting: the diffusers→TensorRT converter

Keep Python (weights + architecture graph live in diffusers; no mature non-Python export path
that fuses LoRAs and traces to ONNX). The upgrade is a **better-structured** converter, which
daydream already provides: `model_detection.py` (arch fingerprint) → parameterized export
wrappers → documented TRT I/O contract. Refactor our conversion around it, and have the converter
emit golden validation dumps in the same run. This is also the canonical source to merge newer
model features from.
