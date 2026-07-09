from __future__ import annotations

import struct
from pathlib import Path


class ImageReadError(ValueError):
    """Raised when an image header cannot be read."""


def read_image_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(32)
        if len(header) < 24:
            raise ImageReadError(f"Image file is too small: {path}")

        if header[:2] == b"BM":
            width, height = struct.unpack_from("<ii", header, 18)
            return abs(width), abs(height)

        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            width, height = struct.unpack(">II", header[16:24])
            return width, height

        if header[:2] == b"\xff\xd8":
            return _read_jpeg_size(path)

    raise ImageReadError(f"Unsupported image format: {path}")


def _read_jpeg_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        if handle.read(2) != b"\xff\xd8":
            raise ImageReadError(f"Not a JPEG file: {path}")
        while True:
            marker_start = handle.read(1)
            if not marker_start:
                break
            if marker_start != b"\xff":
                continue
            marker = handle.read(1)
            while marker == b"\xff":
                marker = handle.read(1)
            if marker in (b"\xc0", b"\xc1", b"\xc2", b"\xc3", b"\xc5", b"\xc6", b"\xc7", b"\xc9", b"\xca", b"\xcb", b"\xcd", b"\xce", b"\xcf"):
                segment = handle.read(7)
                if len(segment) != 7:
                    break
                height, width = struct.unpack(">HH", segment[3:7])
                return width, height
            length_bytes = handle.read(2)
            if len(length_bytes) != 2:
                break
            length = struct.unpack(">H", length_bytes)[0]
            if length < 2:
                break
            handle.seek(length - 2, 1)
    raise ImageReadError(f"Could not read JPEG dimensions: {path}")
