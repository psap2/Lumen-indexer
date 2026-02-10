#!/usr/bin/env python3
"""
Compile scip.proto → scip_pb2.py using grpc_tools.

Run:
    python -m lumen.proto.compile

This is idempotent — safe to call during CI or first-time setup.
"""

from __future__ import annotations

import sys
from pathlib import Path


def compile_proto() -> None:
    proto_dir = Path(__file__).resolve().parent
    proto_file = proto_dir / "scip.proto"
    output_file = proto_dir / "scip_pb2.py"

    if not proto_file.exists():
        print(f"[lumen] ERROR: {proto_file} not found.", file=sys.stderr)
        sys.exit(1)

    try:
        from grpc_tools import protoc  # type: ignore[import-untyped]
    except ImportError:
        print(
            "[lumen] ERROR: grpcio-tools is required to compile protobuf.\n"
            "  pip install grpcio-tools",
            file=sys.stderr,
        )
        sys.exit(1)

    result = protoc.main(
        [
            "grpc_tools.protoc",
            f"--proto_path={proto_dir}",
            f"--python_out={proto_dir}",
            str(proto_file),
        ]
    )

    if result != 0:
        print(f"[lumen] ERROR: protoc exited with code {result}.", file=sys.stderr)
        sys.exit(result)

    if output_file.exists():
        print(f"[lumen] Compiled {proto_file.name} → {output_file.name}")
    else:
        print("[lumen] ERROR: compilation succeeded but output file not found.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    compile_proto()
