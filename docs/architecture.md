# Depth Anything V2 C runtime architecture

## Delivery boundary

The runtime package is:

- `depth_anything_v2.dll`;
- one model file for each installed encoder (`vits`, `vitb`, or `vitl`);
- the public C header for native integration.

Python, OpenCV, TorchVision, training code, visualization, video processing,
and metric-depth training/evaluation are outside the runtime boundary.

## Compatibility contract

`dav2_infer_bgr8` reproduces `DepthAnythingV2.infer_image`:

1. accept OpenCV-compatible interleaved BGR8;
2. convert to RGB in `[0, 1]`;
3. resize with cubic interpolation, preserving aspect ratio, using `input_size`
   as a lower bound and rounding both dimensions to multiples of the 14-pixel
   patch size;
4. normalize with ImageNet mean and standard deviation;
5. execute the selected DINOv2 + DPT model on Vulkan;
6. resize the network depth map to the source dimensions with bilinear
   interpolation and aligned corners.

`dav2_infer_tensor_f32` bypasses image processing so neural-network parity can
be measured independently.

Activations and accumulators remain FP32. At first use, representative
workloads select convolution tiles, linear tiles, packed-FP16 weights, and
packed-FP16 attention-score storage independently. Selection is based on
measured performance, not adapter names or vendor IDs, and is cached per
context (and per token count when shape changes the tradeoff). Packed values
are unpacked before FP32 arithmetic; attention softmax and matrix
accumulations remain FP32.

An adapter that selects only FP32 paths can remain bit-identical to the
reference Vulkan graph. An adapter that benefits from packed storage has a
small controlled drift. PyTorch CPU is the cross-backend accuracy authority,
using relative L1 as the acceptance metric:

`sum(abs(native - cpu)) / sum(abs(cpu))`

Every tested image must remain below 1%. Maximum absolute error, MAE, and RMSE
are reported as diagnostics, but are not normalized percentages.

### PyTorch CPU reference

PyTorch CPU is authoritative because its Vulkan backend is experimental. All
22 repository asset images (the 20 examples, project teaser, and DA-2K poster)
were tested at input size 518 using PyTorch 2.13 CPU and the DLL on an AMD
Radeon RX 9070:

| Encoder | Mean relative L1 | Worst image relative L1 | Mean MAE | Mean RMSE | Maximum absolute error |
|---|---:|---:|---:|---:|---:|
| ViT-S | 0.1285% | 0.3162% | 0.002696 | 0.003718 | 0.291724 |
| ViT-B | 0.1698% | 0.3333% | 0.007092 | 0.010779 | 2.042236 |
| ViT-L | 0.0881% | 0.1945% | 0.129391 | 0.187141 | 8.018814 |

Every model and image remains below the 1% relative-L1 requirement. CPU and
GPU results are numerically close but not bitwise identical; backend-specific
reductions, transcendental implementations, and selected packed-FP16 storage
accumulate small differences through the transformer.

The InferBridge worker-compatibility path is independently validated at the
smallest supported size (140) across all 22 assets and the RX 9070, GTX 1080,
and RX 6700 XT. It uses the exact erf GELU required by PyTorch, forces FP32
attention scores, and retains per-adapter weight autotuning. Worst relative L1
deviations are 0.1224% for ViT-S, 0.2014% for ViT-B, and 0.1089% for ViT-L.
The maximum absolute deviations on the normalized [0, 1] output are 0.004924,
0.008126, and 0.001926 respectively. All remain below the 1% requirement.
`tools/compare_inferbridge.py` reproduces this 198-image/model/device
comparison and can select individual precision flags for tuning experiments.

Keeping only attention scores in FP32 recovers safe packed-weight wins where
the device's measured kernels justify them. At ViT-B/ViT-L 140, matched
end-to-end harness medians on the RX 6700 XT improved from 42.2/167.0 ms to
39.3/126.7 ms. The GTX 1080's marginal packed DPT result was rejected by
requiring a 10% isolated convolution advantage; its ViT-L median remained
approximately 347 ms instead of regressing to 366 ms. This rule is based on
measured candidate times and model scale, not adapter names or vendor IDs.

The same runs, including each context's first-use autotuning and the poster's
extreme aspect ratio, took 4.449/8.388/23.537 seconds in the DLL versus
25.232/68.725/197.102 seconds in PyTorch CPU for ViT-S/B/L: aggregate speedups
of 5.67x, 8.19x, and 8.37x.

`tools/compare_assets.py` reproduces the test and writes per-image CSV results
and native depth previews. It accepts ordinary state dictionaries and the
development ViT-S TorchScript checkpoint.

## Runtime implementation strategy

The DLL owns its complete inference stack:

- a bounded flat model-file parser;
- Vulkan instance, adapter, queue, memory, descriptor, pipeline, and command
  submission management;
- embedded SPIR-V compute shaders;
- tensor layouts and inference-only operators required by DINOv2 and DPT;
- deterministic CPU image preprocessing and final depth resize.

No tensor or inference framework is linked. PyTorch is only a development
reference used to convert official checkpoints and produce comparison outputs.
Relevant Vulkan kernels from the local PyTorch source may be adapted into this
runtime, but the dispatcher, Tensor implementation, JIT, autograd, Python
binding, and unrelated operators are not included.

On Windows the C/C++ runtime is linked statically. The only non-system API
loaded by the resulting DLL is `vulkan-1.dll`, supplied by the installed GPU
driver.

For development profiling, setting `DAV2_VULKAN_PROFILE=1` disables command
batching and measures every compute dispatch with a reusable Vulkan timestamp
query pool. When the context is destroyed, the runtime prints total, average,
maximum, and dispatch count grouped by embedded kernel name. The query pool and
per-dispatch synchronization do not exist on the normal execution path when
the variable is unset.

## Model files

The development-only `tools/export_model.py` converts the official state
dictionary to a bounded, memory-mappable file:

- fixed magic, format version, endian tag, and encoder identifier;
- a sorted tensor directory with rank, dimensions, offset, byte count, and
  CRC-32 for every tensor;
- optional fixed derivation metadata containing the canonical checkpoint
  SHA-256, converter identity, DAV2 format version, and encoder;
- 64-byte-aligned contiguous FP32 tensor payloads.

The loader rejects truncated files, integer overflows, overlapping or
out-of-range payloads, invalid tensor shapes, checksum failures, duplicate
names, and a model whose encoder does not match `dav2_create_options`.

Current converted model sizes are approximately 99 MB for ViT-S, 390 MB for
ViT-B, and 1.34 GB for ViT-L. The files are memory-mapped and tensors are
uploaded directly to Vulkan device memory during context creation. Precision
autotuning initially needs FP32 and packed-FP16 candidates for eligible
weights; immediately after selection, the unused copy is destroyed instead of
remaining in the Vulkan allocation pool.

New exports use a cache identity of:

`canonical SHA-256 + converter ID + DAV2 format version + encoder`

The metadata is inserted between the tensor directory and aligned tensor data.
The original format version remains readable: older files have a zero metadata
offset, and older readers ignore the new offset and padding. The official
PyTorch checkpoint remains the canonical artifact; `.dav2` is a derived,
content-addressed representation rather than a second model identity.

Metric exports use `dav2-export-pytorch-metric-v1` and extend that cache
identity with the depth mode and maximum depth. Their validated derivation
metadata binds the official checkpoint to either HyperSim (`20` metres) or
VKITTI (`80` metres). The graph reads this metadata directly: relative models
retain the ReLU head, while metric models execute `sigmoid(logit) * max_depth`
in FP32. Caller parameters cannot change a container's output semantics.
`dav2_get_model_info` exposes the validated semantic type to harnesses so
metric outputs are leased as `DEPTH_METRIC_FLOAT32` without min/max preview
normalization.

## External GPU inference

The internal Windows Vulkan layer probes external D3D12 resource and fence
import support on the selected physical device and matches that device to a
D3D12 adapter by LUID. When both imports and the matching D3D12 device are
available, the public additive GPU ABI can:

- duplicate and import a shared D3D12 buffer as dedicated Vulkan memory;
- duplicate and import a shared D3D12 fence as a Vulkan semaphore with an
  explicit wait value;
- resize BGRA8 or RGBA8 buffers with bicubic filtering and normalize directly
  into device-local planar RGB FP32;
- record preprocessing, the full DINOv2 encoder, the DPT head, and final
  source-size resize into one asynchronous Vulkan command batch;
- allocate a shared D3D12 depth buffer and fence, import both into Vulkan,
  write row-major float32 depth, and signal the explicit fence value without
  queue-idle waits or host readback;
- perform explicit external-to-compute and compute-to-external Vulkan queue
  family ownership transfers for both shared buffers;
- directly import shared D3D12 `TEXTURE2D` capture resources in
  `B8G8R8A8_UNORM` or `R8G8B8A8_UNORM` as sampled Vulkan images;
- run bicubic normalization from the sampled image and write the final
  source-size depth directly to an imported shared D3D12
  `DXGI_FORMAT_R32_FLOAT` texture with a storage-image shader.

The AMD RX 9070 driver supports D3D12-to-Vulkan resource and fence import but
does not expose the inverse Vulkan export capabilities. DAV2 therefore creates
the output resource and fence with D3D12 and imports them into Vulkan. This is
still a single physical GPU allocation and does not stage pixels or depth
through host memory.

`dav2_submit_d3d12` duplicates/imports the caller's borrowed resource and fence
handles before returning. `dav2_submit_d3d12_texture` provides the equivalent
direct image path and validates the opened D3D12 texture's dimensions, format,
array size, mip count, and sample count before importing it. Jobs retain every
descriptor set, intermediate buffer, image view, sampler, imported semaphore,
command buffer, and completion fence until the submission completes. Acquired
buffer or texture output descriptors contain borrowed shared D3D12
resource/fence handles which remain valid through the explicit output lease.
Releasing a job handle does not invalidate a live lease. Cancellation does not
attempt unsafe command-buffer preemption: it makes the output unavailable and
retained resources are reclaimed after the submitted fence completes. Each
context owns three reusable output slots. A slot's shared resource and fence
handles remain stable across same-kind, same-dimension reuse, while its fence
value increases for each submission. A live job or output lease retains only
its slot; a fourth submission returns `DAV2_STATUS_INVALID_STATE` so callers
can apply a latest-frame drop policy. A free slot is recreated only when its
output kind or dimensions change. The lightweight Vulkan memory/image import
is scoped to each job so no Vulkan layout state crosses a consumer lease, while
the physical D3D12 allocation and exported handles remain stable. Submissions
share one Vulkan queue and may serialize there without `vkQueueWaitIdle`.

Capability flags are returned only when the complete input, graph, output, and
synchronization path is available. `dav2_probe_gpu_capabilities` performs the
same truthful device probe without loading model weights, so a harness can
answer its pre-runtime capability query. The transfer counters cover tensor
upload/download staging and are used by the hardware test to prove that a GPU
submission performs neither.

## Native ABI

The exported surface is deliberately small:

- create and destroy a model context;
- calculate the aspect-preserving network shape;
- infer from interleaved BGR8 with Python-compatible preprocessing;
- infer directly from normalized RGB CHW float32;
- query the complete D3D12/Vulkan interop capability;
- asynchronously submit shared D3D12 BGRA8/RGBA8 buffers with a shared-fence
  wait, poll/cancel the correlated job, and acquire/release a shared D3D12
  float32 depth output lease;
- asynchronously submit shared D3D12 BGRA8/RGBA8 textures and acquire a shared
  D3D12 `R32_FLOAT` texture plus completion fence through the same job and
  lease lifecycle;
- obtain stable status strings and a thread-local detailed error.

The DLL additionally exports InferBridge harness ABI 1.6 without adding a
runtime dependency on InferBridge. Host memory retains the existing worker's
BGRA/BGR, legacy-nearest, transposed processed-shape behavior. The Windows
GPU resource path maps the common ABI directly onto the proven standalone
D3D12 texture/fence API: imported BGRA capture, official preprocessing and
full graph, and a leased shared `R32_FLOAT` source-size texture. It preserves
source frame/timestamp correlation, duplicates input handles before submit
returns, exposes the signal fence/value, and never falls back to host staging.
The hardware canary exercises the public harness entry points and the same
native path's zero upload/download counters for all relative and metric
encoders.

Every input dimension is checked before allocation or dispatch. Network tensor
dimensions must be positive multiples of the 14-pixel patch size. A context is
not safe for concurrent inference calls; applications should use one context
per simultaneously executing stream.
