from __future__ import annotations

import base64
import csv
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"\x7fELF"
    except OSError:
        return False


def wheel_record_path(root: Path) -> Path:
    records = list(root.glob("*.dist-info/RECORD"))
    if len(records) != 1:
        raise RuntimeError(f"Expected one wheel RECORD file, found {len(records)}")
    return records[0]


def record_hash(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
    return f"sha256={digest}", str(len(data))


def rewrite_record(root: Path) -> None:
    record = wheel_record_path(root)
    rows: list[list[str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if path == record:
            rows.append([rel, "", ""])
        else:
            digest, size = record_hash(path)
            rows.append([rel, digest, size])
    with record.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def repack(root: Path, wheel: Path) -> None:
    temp_wheel = wheel.with_suffix(".tmp.whl")
    with zipfile.ZipFile(temp_wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(root).as_posix())
    os.replace(temp_wheel, wheel)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: strip_gnu_property.py WHEEL", file=sys.stderr)
        return 2
    wheel = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="strip-gnu-property-") as temp_name:
        root = Path(temp_name) / "wheel"
        root.mkdir()
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(root)
        stripped = 0
        for path in root.rglob("*"):
            if path.is_file() and is_elf(path):
                subprocess.run(["objcopy", "--remove-section", ".note.gnu.property", str(path)], check=True)
                stripped += 1
        rewrite_record(root)
        repack(root, wheel)
    print(f"stripped .note.gnu.property from {stripped} ELF files in {wheel.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
