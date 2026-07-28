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

On the development Vulkan device, the complete native graph is bit-identical
to the repository's Python model running through the reference PyTorch Vulkan
backend. This was checked for ViT-S, ViT-B, and ViT-L at both the native
518-by-518 positional grid and a non-square 70-by-56 grid. A separate
280-by-182 non-zero input comparison for ViT-S also had zero differing float32
values across all 50,960 output pixels. The complete BGR8 API, including cubic
preprocessing and the final Vulkan resize back to a non-square source image,
was also bit-identical across all tested output pixels.

Floating-point operations are not necessarily bitwise reproducible across
different GPU architectures or driver shader compilers. Compatibility mode
always uses FP32, preserves the reference operation order, and never enables
reduced-precision storage or arithmetic.

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
uploaded directly to Vulkan device memory during context creation.

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
