# CUDA Graph Integration Plan — 1-step SD / ControlNet / IP-Adapter

**Status:** spec for the `.so` (built by the user at `/media/data1/trt-install/lrd-build`). Grounded in
measured benchmarks (see memory `cudagraph-trt11-bench`) + code inspection at the line numbers below.

## Why (measured)

CUDA-graph gain ≈ `launch_overhead / frame_time`. Measured on RTX 4090, TRT 11.0.0.114, full pipeline
(encode → predict → decode), `sd_fps_harness`:

| Workload | baseline | whole-frame graph | gain | capturable as-is? |
|---|---|---|---|---|
| **SD-Turbo 1-step 512** | 81.3 fps | **95.8 fps** | **+17.8%** (p99 +17.9%) | ✅ yes (proven) |
| SD1.5 **2-step** 512 | 57.1 fps | — | 0% | ❌ multi-step host logic blocks capture |
| SDXL 1024 | 12.7 fps | 12.8 | ~0% | compute-bound |
| klein (4B) | 6.9–9.8 fps | — | ≤2% (est) | ❌ + not worth it (compute-bound) |

Graph output is **byte-equivalent** to per-call enqueue (verified: identical min/max/mean, 0 NaN).

**Scope = the 1-step single-step path only** (`predict_x0_batch_impl_single_step`), which the node hits when
`denoising_steps == 1` (SD-Turbo, SDXS, 1-step Hyper, and their ControlNet/IP-Adapter variants). Multi-step,
SDXL@1024, and klein are explicitly OUT (no/negative payoff; multi-step also can't capture without a GPU-side
scheduler refactor).

## Capture preconditions (CUDA rule: no malloc/free/sync/host-branch inside the captured region)

What's already satisfied in the single-step path:
- **Persistent buffers + stable addresses.** `predict_x0_batch_*` and the wrapper `*_buffer_` tensors are
  allocated once and reused (`setTensorAddress` to fixed pointers). This is the hard precondition and it's met.
- **The 8 redundant pre-enqueue `cudaStreamSynchronize` are already removed** from `tensorrt_wrappers.cpp`
  (lines 197/357/448/548/663/864/1007/1067 → comments `LRD-PERF`) plus the DEBUG sync in
  `librediffusion.unet.cpp:181`. These were same-stream-redundant (proven: 3/3 seams still PASS). Their removal
  is what makes the region sync-free.
- **Branch selection is config-fixed**, not per-step: `config_.cfg_type`, `controlnet_enabled_`,
  `ipadapter_enabled_` are set at configure time and pick the SAME branch every frame → safe to capture.
- `set_input_shape`/`setTensorAddress` run every call but with identical values (static shape, persistent
  buffers) → set them once before capture; they need not (and must not) be re-run inside replay.

### The 3 blockers to fix (per-frame stack `CUDATensor` allocations → illegal `cudaMalloc` during capture)

All three are in `librediffusion.unet.cpp` `predict_x0_batch_impl_single_step`:

1. **IP-Adapter** `ext_ehs` — line **1442**: `CUDATensor<__half> ext_ehs((size_t)unet_batch_size * ext_row);`
2. **SDXL-turbo** `tiled_text_embeds` / `tiled_time_ids` — lines **1388–1389** (only matters if you graph an
   SDXL-*turbo* 1-step variant; not needed for plain SD-Turbo/CN/IP).

Fix = hoist each to a **persistent member** allocated/grown on first use, exactly like
`ipadapter_scale_buffer_` already does in `UNetWrapper::forward_ipadapter` (tensorrt_wrappers.cpp:26-27):
```cpp
// member (LibreDiffusionPipeline): std::unique_ptr<CUDATensor<__half>> ip_ext_ehs_;
// in the IP branch, replacing line 1442:
const size_t ext_n = (size_t)unet_batch_size * ext_row;
if(!ip_ext_ehs_ || ip_ext_ehs_->size() < ext_n)
  ip_ext_ehs_ = std::make_unique<CUDATensor<__half>>(ext_n);
auto* ext_ehs = ip_ext_ehs_->data();   // then use ext_ehs (a __half*) below instead of ext_ehs.data()
```
(Same shape every frame for a fixed config, so the grow path runs once.) Do the analogous hoist for the two
SDXL-turbo tiles if/when graphing SDXL-turbo. **Plain SD-Turbo and ControlNet need NO code change** — their
single-step branches already use only persistent buffers + stream-ordered memcpyAsync/enqueue.

> Note: ControlNet's `run_controlnets` → `ControlNetWrapper::forward` → `forward_controlnet` chain was verified
> to contain no `cudaMalloc`/`cudaStreamSynchronize` (the forward_* syncs were among the 8 removed). The extra
> ControlNet engine `enqueueV3` is stream-ordered and captures fine.

## Design: capture-once / replay-per-frame, keyed on a capture signature

Add a small capturer owned by `LibreDiffusionPipeline` (single stream — the pipeline's existing stream).

### New members (LibreDiffusionPipeline)
```cpp
bool          graph_enabled_   = false;   // gate (config flag, default off until validated per-bundle)
bool          graph_ready_     = false;
cudaGraph_t   graph_           = nullptr;
cudaGraphExec_t graph_exec_    = nullptr;
uint64_t      graph_sig_       = 0;       // invalidation key (see below)
std::unique_ptr<CUDATensor<__half>> ip_ext_ehs_;  // the hoisted IP buffer
```

### Where to wrap
`predict_x0_batch_impl_single_step` is the capture region. Restructure its tail so the **device work** (all
the `launch_*`, `cudaMemcpyAsync`, `unet_->forward*`, `run_controlnets`) is one callable, with the
shape/address setup done before capture and the host-side `temporal_state_`/cache logic (the V2V block at
1476+) left OUTSIDE (that block has host branches + reads; V2V is multi-frame anyway and out of scope — only
graph when `config_.mode != TEMPORAL_V2V`).

```cpp
void LibreDiffusionPipeline::predict_x0_batch_impl_single_step(const __half* in, __half* out, cudaStream_t s)
{
  if(!graph_enabled_ || config_.mode == PipelineMode::TEMPORAL_V2V) { run_single_step_body(in,out,s); return; }

  const uint64_t sig = capture_signature();      // see below
  if(!graph_ready_ || sig != graph_sig_) {
    if(graph_exec_) { cudaGraphExecDestroy(graph_exec_); graph_exec_=nullptr; }
    if(graph_)      { cudaGraphDestroy(graph_);          graph_=nullptr; }
    // WARM: a real enqueue first (TRT requires it before capture; also primes shapes/addresses).
    run_single_step_body(in,out,s);
    cudaStreamSynchronize(s);
    // CAPTURE
    cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
    run_single_step_body(in,out,s);
    cudaStreamEndCapture(s, &graph_);
    cudaGraphInstantiate(&graph_exec_, graph_, 0);
    graph_sig_ = sig; graph_ready_ = true;
    return;                                       // this frame already produced a valid result (the warm run)
  }
  cudaGraphLaunch(graph_exec_, s);                // steady state: one launch replaces all the enqueues+kernels
}
```

`run_single_step_body` = the current body of `predict_x0_batch_impl_single_step` (the cfg-concat + the
UNet/CN/IP branch), with the `ext_ehs` hoist applied. **Input handling caveat (important):** the body starts
with `predict_x0_batch_x_t_latent->load_d2d(x_t_latent_in, …)` — a `cudaMemcpyAsync` from the **caller's**
`x_t_latent_in` pointer. For replay to be correct, the caller must write the new frame's latent into the SAME
device address each frame (the graph baked that source pointer). Two options:
- **(A, recommended)** stage the input into a fixed pipeline-owned buffer *before* the captured region: the
  capture copies from `predict_x0_batch_x_t_latent` (persistent) → so the first op in the body should be a copy
  from a FIXED staging buffer that the caller fills, not from the volatile `x_t_latent_in`. I.e. before
  `predict_x0_batch_impl_single_step`, do `staging_in_->load_d2d(x_t_latent_in)` (outside capture) and have the
  body copy from `staging_in_`. Same for the output: the body writes to `predict_x0_batch_unet_output`/
  `x_0_pred_out`; ensure `x_0_pred_out` is a fixed buffer, then copy out after `cudaGraphLaunch`.
- (B) require callers to pass stable pointers — brittle; prefer (A).

### Invalidation signature
Recapture when anything baked into the graph changes. `capture_signature()` hashes:
`config_.batch_size, latent_height, latent_width, denoising_steps(==1), cfg_type, model_type,
controlnet_enabled_, ipadapter_enabled_, ipadapter_num_tokens_, text_seq_len, text_hidden_dim`, **and** the
device addresses of every persistent buffer the body touches (sample/timestep/ehs/output + ext_ehs + CN cond
buffers). If a buffer is reallocated (resolution change, engine swap, IP image change that resizes tokens),
the address changes → signature changes → recapture. This mirrors the Python `allocate_buffers()` rule
(utilities.py:347-352) that already destroys the graph when buffers move.

NOTE: graph bakes the *timestep* too (it's a fixed 1-step value copied from `sub_timesteps_`). For 1-step that
is constant per config → fine. (This is precisely why multi-step can't be a single graph: t changes per step.)

## C-API / gating
- Add `librediffusion_config_set_cuda_graph(cfg, int enable)` → sets `graph_enabled_`. Default **off**;
  the node turns it on for 1-step workflows once validated. (Conservative: never auto-enable.)
- The node/harness already calls `predict_x0_batch` per frame — no call-site change beyond the flag.

## Validation (must pass before shipping each bundle)
1. **Correctness:** run the C++ seam harness (`harness`) on a 1-step cell with graph ON vs OFF — the
   `decode_image` / `predict_x0_batch` seams must still PASS at the same tolerances (PSNR≥30, cos≥0.999). The
   microbench already showed graph replay is byte-identical; this confirms it end-to-end through the pipeline.
2. **Perf:** `sd_fps_harness --label graph-on` vs `graph-off` on the same 1-step bundle (SD-Turbo expected
   ~+18%; ControlNet a bit less; IP-Adapter ~similar after the hoist).
3. **Invalidation:** change resolution mid-run (forces buffer realloc) and confirm it recaptures (no stale
   pointer / no crash) and output stays correct.
4. **V2V untouched:** confirm `TEMPORAL_V2V` mode still runs the non-graph path (graph gated off for it).

## Effort / risk
- **SD-Turbo, ControlNet (1-step):** zero body changes — just the capturer + signature + flag. Lowest risk.
- **IP-Adapter (1-step):** + the one `ext_ehs` hoist (≈6 lines). Low risk.
- **SDXL-turbo (1-step):** + the two tile hoists (1388-89) if desired.
- Everything is gated behind `graph_enabled_` (default off) → zero behavior change until explicitly enabled and
  validated. Fully reversible.
- All edits are in the `.so` (`librediffusion.unet.cpp`, `tensorrt_wrappers.cpp` header for members, C-API).
  The user builds the `.so`; this plan does not touch the score build or the Python export toolchain.

## Out of scope (measured rationale)
- **Multi-step SD (≥2):** per-step host scheduler branches + changing timestep ⇒ can't be one graph; also
  these workloads are GPU-compute-bound at 2-step/512 (sync removal gave 0%). A per-step sub-graph is possible
  but low ROI.
- **SDXL @ 1024 / klein 4B:** compute-bound (klein 102-145ms/frame); launch overhead ≈1.5% → ≤2% ceiling. Not
  worth the malloc-hoist refactor. Klein's levers are FP8 (already its base) + RIFE/pacing.
