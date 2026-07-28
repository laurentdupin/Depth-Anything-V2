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

Floating-point operations on different GPU architectures are not generally
bitwise reproducible. Validation therefore reports exact equality separately
from maximum absolute, mean absolute, and relative error. The initial target is
FP32 Vulkan output within a documented tolerance of the FP32 Python reference;
reduced-precision storage or arithmetic is never enabled in compatibility mode.

## Runtime implementation strategy

The first executor uses the repository owner's Vulkan-enabled PyTorch build as
the semantic reference. It loads an inference-only serialized model and keeps
weights and intermediate tensors on the selected Vulkan adapter. This proves
the ABI, model graph, dynamic-shape behavior, and output parity before runtime
trimming.

The release build statically links only the dispatcher, serialization, tensor,
and Vulkan operator closure reached by the three supported model variants. The
selected-op manifest is generated from model execution coverage. This avoids
shipping the Python binding, autograd, optimizers, distributed code, training
kernels, CUDA, ROCm, or unrelated operators while preserving the already
validated Vulkan kernels.
