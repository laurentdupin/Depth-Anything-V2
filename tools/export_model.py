"""Convert an official Depth Anything V2 checkpoint to the flat DAV2 format.

This development tool is not part of deployment. The resulting .dav2 file is
read directly by the dependency-free native DLL.
"""

from __future__ import annotations

import argparse
import json
import struct
import zlib
from pathlib import Path

import torch


MAGIC = b"DAV2MOD\0"
FORMAT_VERSION = 1
ENDIAN_TAG = 0x01020304
DTYPE_FLOAT32 = 1
HEADER = struct.Struct("<8sIIIIQQQQQ")
RECORD = struct.Struct("<112sII4QQQQIIQ")
ALIGNMENT = 64
ENCODERS = {"vits": 0, "vitb": 1, "vitl": 2}


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) & ~(ALIGNMENT - 1)


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.dtype != torch.float32:
        raise TypeError(f"unsupported tensor dtype: {tensor.dtype}")
    return tensor.detach().cpu().contiguous().numpy().tobytes(order="C")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=ENCODERS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise TypeError("checkpoint is not a non-empty state dictionary")

    tensors: list[
        tuple[str, tuple[int, ...], torch.Tensor, int, int]
    ] = []
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state entry is not a tensor: {name}")
        encoded_name = name.encode("utf-8")
        if len(encoded_name) >= 112:
            raise ValueError(f"tensor name is too long: {name}")
        if not 1 <= value.ndim <= 4:
            raise ValueError(f"unsupported tensor rank for {name}: {value.ndim}")
        payload = tensor_bytes(value)
        tensors.append(
            (
                name,
                tuple(value.shape),
                value,
                len(payload),
                zlib.crc32(payload),
            )
        )

    directory_offset = HEADER.size
    directory_bytes = len(tensors) * RECORD.size
    data_offset = align(directory_offset + directory_bytes)
    cursor = data_offset
    records = []
    for name, shape, _, payload_bytes, checksum in tensors:
        cursor = align(cursor)
        dimensions = list(shape) + [0] * (4 - len(shape))
        element_count = 1
        for dimension in shape:
            element_count *= dimension
        if element_count * 4 != payload_bytes:
            raise RuntimeError(f"tensor byte count mismatch: {name}")
        records.append(
            RECORD.pack(
                name.encode("utf-8"),
                DTYPE_FLOAT32,
                len(shape),
                *dimensions,
                cursor,
                payload_bytes,
                element_count,
                checksum,
                0,
                0,
            )
        )
        cursor += payload_bytes
    file_bytes = cursor
    header = HEADER.pack(
        MAGIC,
        FORMAT_VERSION,
        ENDIAN_TAG,
        ENCODERS[args.encoder],
        len(tensors),
        directory_offset,
        directory_bytes,
        data_offset,
        file_bytes,
        0,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        output.write(header)
        for record in records:
            output.write(record)
        output.write(b"\0" * (data_offset - output.tell()))
        for (_, _, tensor, payload_bytes, _), record in zip(tensors, records):
            record_values = RECORD.unpack(record)
            tensor_offset = record_values[7]
            output.write(b"\0" * (tensor_offset - output.tell()))
            payload = tensor_bytes(tensor)
            if len(payload) != payload_bytes:
                raise RuntimeError("tensor changed while exporting")
            output.write(payload)
        if output.tell() != file_bytes:
            raise RuntimeError(
                f"file length mismatch: {output.tell()} != {file_bytes}"
            )

    print(
        json.dumps(
            {
                "format": "DAV2MOD",
                "version": FORMAT_VERSION,
                "encoder": args.encoder,
                "tensor_count": len(tensors),
                "bytes": file_bytes,
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
