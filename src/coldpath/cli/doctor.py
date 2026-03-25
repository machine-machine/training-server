"""
ColdPath Doctor - verify dependencies, connectivity, and system health.

Usage:
    python -m coldpath.cli.doctor
"""

import importlib
import os
import socket
import sqlite3
import struct
import sys
import tempfile
from pathlib import Path

# Result type: (check_name, passed, detail)
CheckResult = tuple[str, bool, str]

hotpath_socket: str = os.getenv(
    "HOTPATH_SOCKET", os.path.join(tempfile.gettempdir(), "dexy-engine.sock")
)


def check_python_version() -> CheckResult:
    v = sys.version_info
    ok = v >= (3, 11)
    detail = f"{v.major}.{v.minor}.{v.micro}"
    if not ok:
        detail += " (requires >= 3.11)"
    return ("Python version", ok, detail)


def check_dependency(module_name: str, package_name: str | None = None) -> CheckResult:
    display = package_name or module_name
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", getattr(mod, "VERSION", "installed"))
        return (f"Dependency: {display}", True, str(version))
    except ImportError as e:
        return (f"Dependency: {display}", False, str(e))


def check_dependencies() -> list[CheckResult]:
    deps = [
        ("grpc", "grpcio"),
        ("google.protobuf", "protobuf"),
        ("numpy", None),
        ("pandas", None),
        ("sklearn", "scikit-learn"),
        ("httpx", None),
        ("polars", None),
        ("scipy", None),
        ("numba", None),
        ("anthropic", None),
        ("fastapi", None),
        ("psutil", None),
    ]
    return [check_dependency(mod, pkg) for mod, pkg in deps]


def check_database() -> CheckResult:
    db_path = os.getenv("DEXY_DB_PATH", "sniperdesk.db")
    p = Path(db_path)
    if not p.exists():
        return ("Database", False, f"{db_path} not found")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        if not tables:
            return ("Database", True, f"{db_path} (empty, no tables yet)")
        return (
            "Database",
            True,
            f"{db_path} ({len(tables)} tables: {', '.join(tables[:5])}"
            f"{'...' if len(tables) > 5 else ''})",
        )
    except Exception as e:
        return ("Database", False, f"{db_path}: {e}")


def check_hotpath_socket() -> CheckResult:
    socket_path = hotpath_socket
    p = Path(socket_path)
    if not p.exists():
        return ("HotPath Unix socket", False, f"{socket_path} not found (engine not running?)")

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(3)
        sock.connect(socket_path)
        # Send a status ping: length-prefixed JSON
        msg = b'{"id":"doctor","command":"get_status","params":{}}'
        sock.sendall(struct.pack(">I", len(msg)) + msg)
        # Read response length
        length_bytes = sock.recv(4)
        if len(length_bytes) == 4:
            resp_len = struct.unpack(">I", length_bytes)[0]
            resp = sock.recv(min(resp_len, 4096))
            sock.close()
            return (
                "HotPath Unix socket",
                True,
                f"{socket_path} - connected, got {len(resp)} bytes",
            )
        sock.close()
        return ("HotPath Unix socket", True, f"{socket_path} - connected (no response)")
    except TimeoutError:
        return ("HotPath Unix socket", False, f"{socket_path} exists but timed out")
    except ConnectionRefusedError:
        return ("HotPath Unix socket", False, f"{socket_path} exists but connection refused")
    except Exception as e:
        return ("HotPath Unix socket", False, f"{socket_path}: {e}")


def check_hotpath_grpc() -> CheckResult:
    host = os.getenv("HOTPATH_GRPC_HOST", "localhost")
    port = int(os.getenv("HOTPATH_GRPC_PORT", "50051"))
    addr = f"{host}:{port}"

    try:
        import grpc

        channel = grpc.insecure_channel(addr)
        # Quick connectivity check
        try:
            grpc.channel_ready_future(channel).result(timeout=3)
            channel.close()
            return ("HotPath gRPC", True, f"{addr} - channel ready")
        except grpc.FutureTimeoutError:
            channel.close()
            return ("HotPath gRPC", False, f"{addr} - timeout (engine not running?)")
    except ImportError:
        return ("HotPath gRPC", False, "grpcio not installed")
    except Exception as e:
        return ("HotPath gRPC", False, f"{addr}: {e}")


def check_model_artifacts() -> CheckResult:
    artifact_dir = os.getenv("MODEL_DIR", "models")
    p = Path(artifact_dir)
    if not p.exists():
        return (
            "Model artifacts",
            True,
            f"{artifact_dir}/ not found (will be created on first distillation)",
        )

    model_types = ["fraud", "slippage", "inclusion", "profitability", "regime"]
    found = []
    for mt in model_types:
        mt_dir = p / mt
        if mt_dir.exists():
            versions = sorted(mt_dir.glob("v*.json"))
            if versions:
                found.append(f"{mt}(v{len(versions)})")

    if not found:
        return ("Model artifacts", True, f"{artifact_dir}/ exists but no models yet")
    return ("Model artifacts", True, f"{', '.join(found)}")


def check_env_vars() -> CheckResult:
    important = {
        "ANTHROPIC_API_KEY": "required for AI features",
        "HELIUS_API_KEY": "required for Solana token discovery",
        "BITQUERY_API_KEY": "optional for OHLCV data",
    }
    set_vars = []
    missing = []
    for var, desc in important.items():
        if os.getenv(var):
            set_vars.append(var)
        else:
            missing.append(f"{var} ({desc})")

    if missing:
        detail = (
            f"Set: {', '.join(set_vars) if set_vars else 'none'}. Missing: {', '.join(missing)}"
        )
        # Not a hard failure since some are optional
        return ("Environment variables", len(set_vars) > 0, detail)
    return ("Environment variables", True, f"All set: {', '.join(set_vars)}")


def check_system_resources() -> CheckResult:
    try:
        import psutil

        mem = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        avail_gb = mem.available / (1024**3)
        total_gb = mem.total / (1024**3)
        ok = avail_gb > 1.0  # Need at least 1GB free
        detail = f"{avail_gb:.1f}/{total_gb:.1f} GB RAM available, {cpu_count} CPUs"
        if not ok:
            detail += " (low memory!)"
        return ("System resources", ok, detail)
    except ImportError:
        return ("System resources", True, "psutil not available, skipping")


def check_gpu() -> CheckResult:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            return ("GPU/CUDA", True, f"{name} ({mem:.1f} GB)")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return ("GPU/MPS", True, "Apple Silicon MPS available")
        else:
            return ("GPU/CUDA", True, "No GPU detected (CPU-only, OK for distillation)")
    except ImportError:
        return ("GPU/CUDA", True, "PyTorch not installed (not required)")


def run_doctor() -> bool:
    """Run all doctor checks and print results. Returns True if all critical checks pass."""
    print("=" * 60)
    print("  ColdPath Doctor - System Health Check")
    print("=" * 60)
    print()

    all_results: list[CheckResult] = []

    # 1. Python version
    all_results.append(check_python_version())

    # 2. Dependencies
    all_results.extend(check_dependencies())

    # 3. Database
    all_results.append(check_database())

    # 4. HotPath connectivity
    all_results.append(check_hotpath_socket())
    all_results.append(check_hotpath_grpc())

    # 5. Model artifacts
    all_results.append(check_model_artifacts())

    # 6. Environment variables
    all_results.append(check_env_vars())

    # 7. System resources
    all_results.append(check_system_resources())

    # 8. GPU
    all_results.append(check_gpu())

    # Print results
    passed = 0
    failed = 0
    for name, ok, detail in all_results:
        icon = "[OK]" if ok else "[!!]"
        print(f"  {icon} {name}: {detail}")
        if ok:
            passed += 1
        else:
            failed += 1

    print()
    print("-" * 60)
    print(f"  Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("  All checks passed. ColdPath is ready.")
    else:
        print("  Some checks failed. Review above for details.")

    print("=" * 60)
    return failed == 0


def main():
    ok = run_doctor()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
