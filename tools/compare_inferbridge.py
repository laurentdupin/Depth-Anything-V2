"""Validate the native InferBridge image contract against the Python worker.

Python and its packages are validation-only dependencies. They are not linked
or shipped with the native runtime.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, value):
        for transform in self.transforms:
            value = transform(value)
        return value


torchvision = types.ModuleType("torchvision")
torchvision_transforms = types.ModuleType("torchvision.transforms")
torchvision_transforms.Compose = Compose
torchvision.transforms = torchvision_transforms
sys.modules["torchvision"] = torchvision
sys.modules["torchvision.transforms"] = torchvision_transforms

from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402
from depth_anything_v2.util.transform import Resize  # noqa: E402


CONFIGS = {
    "vits": (0, 64, [48, 96, 192, 384]),
    "vitb": (1, 128, [96, 192, 384, 768]),
    "vitl": (2, 256, [256, 512, 1024, 1024]),
}


class CreateOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("encoder", ctypes.c_int),
        ("vulkan_device_index", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
    ]


class ImageShape(ctypes.Structure):
    _fields_ = [("width", ctypes.c_int32), ("height", ctypes.c_int32)]


def configure_dll(path: Path):
    dll = ctypes.CDLL(str(path.resolve()))
    dll.dav2_create.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(CreateOptions),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    dll.dav2_create.restype = ctypes.c_int
    dll.dav2_destroy.argtypes = [ctypes.c_void_p]
    dll.dav2_get_inferbridge_shape.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ImageShape),
    ]
    dll.dav2_get_inferbridge_shape.restype = ctypes.c_int
    dll.dav2_inferbridge_bgra8_f32.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_ssize_t,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    dll.dav2_inferbridge_bgra8_f32.restype = ctypes.c_int
    dll.dav2_last_error.restype = ctypes.c_char_p
    return dll


def load_model(encoder: str, checkpoint: Path) -> DepthAnythingV2:
    _, features, channels = CONFIGS[encoder]
    model = DepthAnythingV2(
        encoder=encoder, features=features, out_channels=channels
    )
    try:
        loaded = torch.load(
            checkpoint, map_location="cpu", weights_only=True
        )
    except RuntimeError as error:
        if "TorchScript archives" not in str(error):
            raise
        loaded = torch.jit.load(str(checkpoint), map_location="cpu")
    if isinstance(loaded, torch.jit.ScriptModule):
        scripted = loaded
        state = {}
        encoder_prefixes = (
            "cls_token",
            "patch_embed.",
            "blocks.",
            "norm.",
        )
        for name, value in scripted.state_dict().items():
            if name == "base_pos_embed":
                mapped = "pretrained.pos_embed"
            elif name.startswith(encoder_prefixes):
                mapped = "pretrained." + name
            else:
                mapped = name
            state[mapped] = value
    else:
        state = loaded
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or missing not in ([], ["pretrained.mask_token"]):
        raise RuntimeError(
            f"checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    return model.eval()


def reference(
    model: DepthAnythingV2, bgra: np.ndarray, size: int
) -> np.ndarray:
    resize = Resize(
        width=size,
        height=size,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_NEAREST,
    )
    width, height = resize.get_size(bgra.shape[1], bgra.shape[0])
    image = torch.from_numpy(bgra[:, :, :3])
    image = image.unsqueeze(0).permute(0, 3, 1, 2)
    # Preserve the worker's (width, height) order and default legacy-nearest.
    image = functional.interpolate(image, (width, height))
    image = image[0].permute(1, 2, 0).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    image = ((image - mean) / std).permute(2, 0, 1).unsqueeze(0)
    with torch.inference_mode():
        depth = model(image)[0]
        depth = (depth - depth.min()) / (depth.max() - depth.min())
    return depth.numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=CONFIGS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--size", type=int, default=140)
    parser.add_argument("--create-flags", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    model = load_model(args.encoder, args.checkpoint)
    dll = configure_dll(args.dll)
    options = CreateOptions(
        ctypes.sizeof(CreateOptions),
        1,
        CONFIGS[args.encoder][0],
        args.device,
        args.create_flags,
    )
    context = ctypes.c_void_p()
    status = dll.dav2_create(
        str(args.model.resolve()).encode(), ctypes.byref(options),
        ctypes.byref(context)
    )
    if status != 0:
        raise RuntimeError(dll.dav2_last_error().decode())

    paths = sorted(
        p for p in args.assets.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    worst_max = 0.0
    worst_mean = 0.0
    worst_relative_l1 = 0.0
    try:
        for path in paths:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"cannot decode {path}")
            bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
            shape = ImageShape()
            status = dll.dav2_get_inferbridge_shape(
                bgra.shape[1], bgra.shape[0], args.size,
                ctypes.byref(shape)
            )
            if status != 0:
                raise RuntimeError(dll.dav2_last_error().decode())
            native = np.empty((shape.height, shape.width), np.float32)
            status = dll.dav2_inferbridge_bgra8_f32(
                context,
                bgra.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                bgra.shape[1],
                bgra.shape[0],
                bgra.strides[0],
                args.size,
                native.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                native.size,
            )
            if status != 0:
                raise RuntimeError(dll.dav2_last_error().decode())
            expected = reference(model, bgra, args.size)
            if native.shape != expected.shape:
                raise RuntimeError(
                    f"{path}: shape {native.shape} != {expected.shape}"
                )
            difference = np.abs(native - expected)
            worst_max = max(worst_max, float(difference.max()))
            worst_mean = max(worst_mean, float(difference.mean()))
            denominator = float(np.abs(expected, dtype=np.float64).sum())
            relative_l1 = (
                float(difference.astype(np.float64).sum()) / denominator
                if denominator > 0.0
                else float(difference.max())
            )
            worst_relative_l1 = max(worst_relative_l1, relative_l1)
            if args.verbose:
                print(
                    f"{path.name}: max_abs={float(difference.max()):.8f} "
                    f"mean_abs={float(difference.mean()):.8f} "
                    f"relative_l1={100.0 * relative_l1:.4f}% "
                    f"native=[{float(native.min()):.6f},"
                    f"{float(native.max()):.6f}] "
                    f"cpu=[{float(expected.min()):.6f},"
                    f"{float(expected.max()):.6f}]"
                )
    finally:
        dll.dav2_destroy(context)

    print(
        f"{args.encoder} GPU {args.device}: images={len(paths)} "
        f"max_abs={worst_max:.8f} worst_mean_abs={worst_mean:.8f} "
        f"worst_relative_l1={100.0 * worst_relative_l1:.4f}%"
    )
    if worst_relative_l1 > 0.01:
        raise SystemExit("worst relative L1 deviation exceeds 1%")


if __name__ == "__main__":
    main()
