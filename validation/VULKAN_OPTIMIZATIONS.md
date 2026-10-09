# Vulkan kernel optimization validation

Validated on Windows with Vulkan SDK 1.4.363.0 and on Ubuntu through WSL.
These are kernel and inference measurements, not headset frame-rate measurements.

## Enabled behavior

- KHR cooperative matrices can request a 32-lane compute subgroup on devices
  whose default subgroup is 64. Both Vulkan memory-model features required by
  the shaders are enabled and checked. Unsupported devices retain their existing
  paths. The shared pipeline helper requests a subgroup only for cooperative
  matrix SPIR-V, and only when the context enabled that request.
- The shared KHR linear and attention-score kernels use a 64-by-32 workgroup
  tile, two accumulators per subgroup, vector loads, and a 32-element K step.
- Arithmetic-subgroup devices use a row-relative, two-pass LayerNorm and FP32
  subgroup softmax. The LayerNorm change addresses cancellation at large offsets;
  it is an accuracy improvement, not a consistent timing improvement.
- Implicit 3-by-3 FP16 convolution is an additional candidate in the existing
  DPT tuner. It must beat the existing candidates by 10% to be selected. The
  tested Small model retained its existing tiled SIMT convolution.
- INT8 preserves the previous normalization, softmax, and attention-score
  ordering to avoid changing quantization behavior.

The FP16 subgroup softmax and online cooperative attention shaders are built only
with `DAV2_BUILD_TOOLS`. They are not runtime defaults: their speedups were
inconsistent. Larger NVIDIA cooperative-matrix tiles were not changed because
the remote RTX machine was unreachable. GTX 1080 fallback behavior was tested,
but that does not validate NVIDIA cooperative-matrix kernels.

## Paired kernel measurements

RX 9070, validation layer enabled, alternating AB/BA batches, warmup excluded.
Times include host submission and completion of repeated batched dispatches.
Original and candidate shader fixtures were compiled with the same SDK.

| Fixture | Original ms | Candidate ms |
| --- | ---: | ---: |
| Linear, 401 rows, K=384, N=1152, original single subgroup | 0.158 | 0.078 |
| Linear, 1370 rows, K=1024, N=4096, original single subgroup | 2.352 | 0.863 |
| Linear, same small shape, original four-subgroup tall kernel | 0.125 | 0.084 |
| Linear, same large shape, original four-subgroup tall kernel | 1.762 | 0.996 |
| Attention scores, 401 tokens, 6 heads | 0.081 | 0.035 |
| Attention scores, 1370 tokens, 16 heads | 1.303 | 0.554 |
| FP32 softmax, 401 columns | 0.052 | 0.047 |
| FP32 softmax, 1370 columns | 0.352 | 0.285 |
| Implicit convolution, 80-by-80, 64 channels, versus naive FP16 | 1.025 | 0.232 |
| Implicit convolution, 148-by-148, 128 channels, versus naive FP16 | 5.600 | 1.744 |

The convolution comparison above is against the naive FP16 shader, not the best
existing SIMT candidate; it does not justify replacing the tuner.

Online attention improved the small fixture from 0.259 to 0.147 ms, but regressed
the large fixture from 1.764 to 3.406 ms. Its relative L1 error was approximately
0.068%. This is why it remains a probe rather than a production path.

At a LayerNorm input offset of 65536, the old fixture's maximum error was about
498–514; the new row-relative implementation reduced this to 1.4e-7–3.1e-7.
The fixtures include tail dimensions and CPU numerical references. All eight
kernel gates passed without Vulkan validation errors.

## Model and application checks

- 17 relative-depth comparisons: RX 9070, RX 6700 XT, and GTX 1080; FP32,
  supported FP16, and INT8; sizes 280 and 518; sinusoidal, gradient, and checker
  inputs. Maximum relative L1 change against the original runtime was 0.1107%,
  below the 0.33% compatibility gate. GTX 1080 continued to reject unsupported
  FP16 consistently in both versions.
- Six metric comparisons: Hypersim and VKITTI Small, three precisions. FP16
  changes were below 0.028%; INT8 outputs were unchanged. Hypersim INT8 on the
  synthetic gradient already failed the probe's FP32-quality check in the
  original runtime and still does; this is an existing limitation, not a passing
  accuracy result.
- Godot's real native GPU path completed warmup and 12 depth frames in each
  precision using D3D12 input/output sharing and Vulkan inference. Size 280 with
  a 640-by-360 source produced a 504-by-280 depth texture. A final FP16 run with
  Vulkan validation enabled completed with no reported validation errors.
- All 14 Windows model libraries rebuilt. All Linux shared shader targets and
  all 26 Vulkan/operator translation units compiled. DAv2's Linux library and
  kernel probe linked. No full Linux Godot editor build was required.

`dav2_kernel_probe` performs paired timings and numerical gates. Its arguments
are original SPIR-V, candidate SPIR-V, device index, and fixture mode (`16`, `64`,
`attention`, `implicit`, `norm`, `softmax`, `softmax-half`, or `flash`).
`dav2_precision_probe` additionally accepts warmup count, raw FP32 output path,
and an optional normalized FP32 input file for deterministic model comparisons.

The shared kernels live in InferBridge's native-support SDK. Other model
executors do not automatically opt into DAv2's precision policy or new dispatch
geometry; their existing behavior is retained.
