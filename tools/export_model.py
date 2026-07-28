"""Export a Depth Anything V2 checkpoint for the native Vulkan runtime.

This is a development/conversion tool. Python is not used by the produced DLL
or required beside the exported model file at deployment time.
"""

from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

import torch
import torch.nn.functional as F


MODEL_CONFIGS = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
}


def _ensure_torchvision_import_compatibility() -> None:
    try:
        torch._C._dispatch_find_schema_or_throw("torchvision::nms", "")
    except RuntimeError:
        library = torch.library.Library("torchvision", "DEF")
        library.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
        # Torch libraries are deregistered when the object is collected.
        globals()["_torchvision_compat_library"] = library


def _dynamic_dpt_head(self, out_features, patch_height, patch_width):
    outputs = []
    for index, feature in enumerate(out_features):
        feature = feature[0]
        feature = feature.permute(0, 2, 1).reshape(
            feature.shape[0],
            feature.shape[-1],
            patch_height,
            patch_width,
        )
        feature = self.projects[index](feature)
        outputs.append(self.resize_layers[index](feature))

    layer_1, layer_2, layer_3, layer_4 = outputs
    layer_1 = self.scratch.layer1_rn(layer_1)
    layer_2 = self.scratch.layer2_rn(layer_2)
    layer_3 = self.scratch.layer3_rn(layer_3)
    layer_4 = self.scratch.layer4_rn(layer_4)

    path_4 = self.scratch.refinenet4(layer_4, size=layer_3.shape[2:])
    path_3 = self.scratch.refinenet3(path_4, layer_3, size=layer_2.shape[2:])
    path_2 = self.scratch.refinenet2(path_3, layer_2, size=layer_1.shape[2:])
    path_1 = self.scratch.refinenet1(path_2, layer_1)

    output = self.scratch.output_conv1(path_1)
    output = F.interpolate(
        output,
        size=(patch_height * 14, patch_width * 14),
        mode="bilinear",
        align_corners=True,
    )
    return self.scratch.output_conv2(output)


class _ExportableModel(torch.nn.Module):
    """Inference-only graph with positional interpolation supplied by C++."""

    def __init__(self, model):
        super().__init__()
        pretrained = model.pretrained
        self.patch_embed = pretrained.patch_embed
        self.cls_token = pretrained.cls_token
        # Retained in the model file for the C++ executor. Forward receives its
        # interpolated form explicitly so the 0.1 scale offset remains dynamic.
        self.base_pos_embed = pretrained.pos_embed
        self.blocks = pretrained.blocks
        self.norm = pretrained.norm
        self.depth_head = model.depth_head
        self.capture_indices = model.intermediate_layer_idx[model.encoder]

    def forward(self, image, position_encoding):
        patch_height = image.shape[-2] // 14
        patch_width = image.shape[-1] // 14
        tokens = self.patch_embed(image)
        tokens = torch.cat(
            (self.cls_token.expand(tokens.shape[0], -1, -1), tokens), dim=1
        )
        tokens = tokens + position_encoding

        features = []
        for index, block in enumerate(self.blocks):
            tokens = block(tokens)
            if index in self.capture_indices:
                normalized = self.norm(tokens)
                features.append((normalized[:, 1:], normalized[:, 0]))
        depth = self.depth_head(features, patch_height, patch_width)
        return F.relu(depth).squeeze(1)


def _load_model(encoder: str, checkpoint: Path):
    _ensure_torchvision_import_compatibility()
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(**MODEL_CONFIGS[encoder]).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model


def _adapt_head_for_dynamic_trace(model) -> None:
    # The upstream head converts its symbolic output size to Python int.
    model.depth_head.forward = types.MethodType(_dynamic_dpt_head, model.depth_head)


def _position_encoding(model, image):
    tokens = model.pretrained.patch_embed(image)
    tokens = torch.cat(
        (
            model.pretrained.cls_token.expand(tokens.shape[0], -1, -1),
            tokens,
        ),
        dim=1,
    )
    return model.pretrained.interpolate_pos_encoding(
        tokens, image.shape[-2], image.shape[-1]
    )


def _validate_dynamic_trace(
    reference_model, traced, shapes: list[tuple[int, int]]
) -> None:
    generator = torch.Generator().manual_seed(0xDA22)
    with torch.inference_mode():
        for height, width in shapes:
            sample = torch.randn(1, 3, height, width, generator=generator)
            position = _position_encoding(reference_model, sample)
            expected = reference_model(sample)
            actual = traced(sample, position)
            if expected.shape != actual.shape:
                raise RuntimeError(
                    f"dynamic trace shape mismatch at {height}x{width}: "
                    f"{tuple(expected.shape)} != {tuple(actual.shape)}"
                )
            maximum = float((expected - actual).abs().max())
            if maximum != 0.0:
                raise RuntimeError(
                    f"dynamic trace changed CPU output at {height}x{width}: "
                    f"max_abs={maximum}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=MODEL_CONFIGS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-height", type=int, default=140)
    parser.add_argument("--trace-width", type=int, default=196)
    args = parser.parse_args()

    for name, value in (
        ("trace height", args.trace_height),
        ("trace width", args.trace_width),
    ):
        if value <= 0 or value % 14:
            parser.error(f"{name} must be a positive multiple of 14")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")

    model = _load_model(args.encoder, args.checkpoint)
    _adapt_head_for_dynamic_trace(model)
    exportable = _ExportableModel(model).eval()
    example = torch.zeros(
        1, 3, args.trace_height, args.trace_width, dtype=torch.float32
    )
    with torch.inference_mode():
        example_position = _position_encoding(model, example)
        traced = torch.jit.trace(
            exportable,
            (example, example_position),
            strict=False,
            check_trace=False,
        )
    validation_shapes = [
        (args.trace_height, args.trace_width),
        (182, 280),
        (280, 504),
    ]
    _validate_dynamic_trace(model, traced, validation_shapes)

    manifest = json.dumps(
        {
            "format": "depth-anything-v2-torchscript",
            "version": 1,
            "encoder": args.encoder,
            "dtype": "float32",
            "patch_size": 14,
            "dynamic_spatial_shapes": True,
            "external_position_encoding": True,
            "position_interpolate_offset": 0.1,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(
        str(args.output),
        _extra_files={"dav2_manifest.json": manifest},
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "bytes": args.output.stat().st_size,
                "encoder": args.encoder,
                "validated_shapes": validation_shapes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
