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

## Model files

The development-only `tools/export_model.py` converts the official state
dictionary to a bounded, memory-mappable file:

- fixed magic, format version, endian tag, and encoder identifier;
- a sorted tensor directory with rank, dimensions, offset, byte count, and
  CRC-32 for every tensor;
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

## Native ABI

The exported surface is deliberately small:

- create and destroy a model context;
- calculate the aspect-preserving network shape;
- infer from interleaved BGR8 with Python-compatible preprocessing;
- infer directly from normalized RGB CHW float32;
- obtain stable status strings and a thread-local detailed error.

Every input dimension is checked before allocation or dispatch. Network tensor
dimensions must be positive multiples of the 14-pixel patch size. A context is
not safe for concurrent inference calls; applications should use one context
per simultaneously executing stream.
