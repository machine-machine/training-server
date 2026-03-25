#!/usr/bin/env python3
"""
Kick off a training-server job end-to-end:
1) Generate a small synthetic CSV
2) Upload it via /data/upload
3) Queue a job via /jobs/train
4) Poll /jobs/{id}/status until completed/failed

Designed to run on the GPU server (or from code-server) without extra deps.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _http_json(
    method: str,
    url: str,
    api_key: str,
    body: bytes | None = None,
    content_type: str | None = None,
    timeout_sec: int = 15,
) -> Tuple[int, Dict[str, Any]]:
    headers = {"Authorization": f"Bearer {api_key}"}
    if content_type:
        headers["Content-Type"] = content_type
    req = Request(url, data=body, method=method, headers=headers)
    try:
        with urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read()
            return resp.status, json.loads(raw.decode("utf-8") or "{}")
    except HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {url}: {raw}") from e
    except (URLError, TimeoutError) as e:
        raise RuntimeError(f"Request failed {url}: {e}") from e


def _http_multipart_upload(
    url: str,
    api_key: str,
    file_path: Path,
    name: str,
    description: str,
    label_column: str,
    timeout_sec: int = 60,
) -> Dict[str, Any]:
    boundary = f"----dexy{uuid.uuid4().hex}"
    crlf = "\r\n"

    parts: list[bytes] = []

    def add_field(field_name: str, value: str) -> None:
        parts.append(f"--{boundary}{crlf}".encode("utf-8"))
        parts.append(f'Content-Disposition: form-data; name="{field_name}"{crlf}{crlf}'.encode("utf-8"))
        parts.append(value.encode("utf-8"))
        parts.append(crlf.encode("utf-8"))

    def add_file(field_name: str, path: Path) -> None:
        parts.append(f"--{boundary}{crlf}".encode("utf-8"))
        parts.append(
            (
                f'Content-Disposition: form-data; name="{field_name}"; filename="{path.name}"{crlf}'
                f"Content-Type: text/csv{crlf}{crlf}"
            ).encode("utf-8")
        )
        parts.append(path.read_bytes())
        parts.append(crlf.encode("utf-8"))

    add_file("file", file_path)
    add_field("name", name)
    add_field("description", description)
    add_field("label_column", label_column)
    parts.append(f"--{boundary}--{crlf}".encode("utf-8"))

    body = b"".join(parts)
    content_type = f"multipart/form-data; boundary={boundary}"
    status, data = _http_json("POST", url, api_key, body=body, content_type=content_type, timeout_sec=timeout_sec)
    if status not in (200, 201):
        raise RuntimeError(f"Unexpected upload status: {status} {data}")
    return data


def _write_synthetic_ensemble_csv(out_path: Path, rows: int, *, seed: int) -> None:
    """
    Generate a numeric-only dataset suitable for the ensemble trainer:
    - Must include 'profitable' label by default
    - Remaining columns can be arbitrary numeric features
    """
    rng = random.Random(seed)
    feature_names = [
        "liquidity_usd",
        "fdv_usd",
        "holder_count",
        "top_holder_pct",
        "age_seconds",
        "volume_24h",
        "price_change_1h",
        "mev_sandwich_count",
        "avg_slippage_bps",
        "volatility",
    ]

    # Simple synthetic rule: more liquidity/holders, lower slippage => more likely profitable.
    lines = []
    header = ["profitable"] + feature_names
    lines.append(",".join(header))

    for _ in range(rows):
        liquidity = rng.uniform(1_000, 2_000_000)
        fdv = rng.uniform(10_000, 50_000_000)
        holders = rng.randint(10, 50_000)
        top_pct = rng.uniform(1.0, 90.0)
        age = rng.randint(60, 7 * 24 * 3600)
        vol24 = rng.uniform(0, 5_000_000)
        chg1h = rng.uniform(-80.0, 200.0)
        mev = rng.randint(0, 50)
        slip = rng.uniform(10.0, 2500.0)
        vol = rng.uniform(0.01, 2.5)

        score = (
            (liquidity / 2_000_000.0) * 0.35
            + (holders / 50_000.0) * 0.20
            + (vol24 / 5_000_000.0) * 0.15
            + (max(chg1h, -80.0) / 200.0) * 0.10
            + (1.0 - min(slip, 2500.0) / 2500.0) * 0.15
            + (1.0 - min(top_pct, 90.0) / 90.0) * 0.05
        )
        score -= min(mev / 50.0, 1.0) * 0.10
        score -= min(vol / 2.5, 1.0) * 0.05

        # Stochastic label from score.
        p = max(0.0, min(1.0, 0.15 + score))
        profitable = 1 if rng.random() < p else 0

        row = [
            str(profitable),
            f"{liquidity:.2f}",
            f"{fdv:.2f}",
            str(holders),
            f"{top_pct:.4f}",
            str(age),
            f"{vol24:.2f}",
            f"{chg1h:.4f}",
            str(mev),
            f"{slip:.4f}",
            f"{vol:.6f}",
        ]
        lines.append(",".join(row))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_synthetic_fraud_csv(out_path: Path, rows: int, *, seed: int) -> None:
    """Generate a fraud/rug dataset matching the worker's expected schema."""
    rng = random.Random(seed)
    header = [
        "is_rug",
        "liquidity_usd",
        "fdv_usd",
        "holder_count",
        "top_holder_pct",
        "mint_authority_enabled",
        "freeze_authority_enabled",
        "lp_burn_pct",
        "age_seconds",
        "volume_24h",
        "price_change_1h",
    ]
    lines = [",".join(header)]

    for _ in range(rows):
        liquidity = rng.uniform(500, 800_000)
        fdv = rng.uniform(5_000, 50_000_000)
        holders = rng.randint(5, 30_000)
        top_holder_pct = rng.uniform(5.0, 99.0)
        mint_auth = 1 if rng.random() < 0.55 else 0
        freeze_auth = 1 if rng.random() < 0.35 else 0
        lp_burn_pct = max(0.0, min(100.0, rng.gauss(65, 25)))
        age_seconds = rng.randint(60, 14 * 24 * 3600)
        volume_24h = rng.uniform(0, 4_000_000)
        price_change_1h = rng.uniform(-95.0, 300.0)

        risk = 0.0
        risk += (1.0 - min(liquidity / 800_000.0, 1.0)) * 0.22
        risk += (1.0 - min(holders / 30_000.0, 1.0)) * 0.15
        risk += min(top_holder_pct / 99.0, 1.0) * 0.18
        risk += mint_auth * 0.12 + freeze_auth * 0.12
        risk += (1.0 - min(lp_burn_pct / 100.0, 1.0)) * 0.12
        risk += (1.0 - min(age_seconds / (14 * 24 * 3600), 1.0)) * 0.10
        risk += (1.0 - min(volume_24h / 4_000_000.0, 1.0)) * 0.06
        if price_change_1h < -40:
            risk += 0.08

        p_rug = max(0.02, min(0.98, 0.05 + risk))
        is_rug = 1 if rng.random() < p_rug else 0

        lines.append(
            ",".join(
                [
                    str(is_rug),
                    f"{liquidity:.2f}",
                    f"{fdv:.2f}",
                    str(holders),
                    f"{top_holder_pct:.4f}",
                    str(mint_auth),
                    str(freeze_auth),
                    f"{lp_burn_pct:.4f}",
                    str(age_seconds),
                    f"{volume_24h:.2f}",
                    f"{price_change_1h:.4f}",
                ]
            )
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://localhost:8000", help="Base URL, e.g. http://localhost:8000")
    parser.add_argument("--api-key", default="", help="Bearer token (required)")
    parser.add_argument("--job-type", default="ensemble", choices=["ensemble", "fraud", "rl", "bandit", "slippage", "regime"])
    parser.add_argument("--gpu-id", type=int, default=None, help="Optional GPU index (0 or 1); omit for auto")
    parser.add_argument("--label-column", default="profitable")
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--hyperparams-json",
        default="",
        help="Optional JSON object merged into hyperparams. Example: '{\"lstm_epochs\":0,\"seed\":42}'",
    )
    parser.add_argument("--poll-sec", type=float, default=3.0)
    parser.add_argument("--timeout-sec", type=int, default=3600)
    args = parser.parse_args()

    if not args.api_key:
        print("Missing --api-key (or set API_KEY env and pass it explicitly).", file=sys.stderr)
        return 2

    api_base = args.api_base.rstrip("/")
    tmp_csv = Path(f"/tmp/dexy_kickoff_{args.job_type}_{int(time.time())}.csv")
    if args.job_type == "fraud":
        if args.label_column == "profitable":
            args.label_column = "is_rug"
        _write_synthetic_fraud_csv(tmp_csv, rows=args.rows, seed=args.seed)
    else:
        _write_synthetic_ensemble_csv(tmp_csv, rows=args.rows, seed=args.seed)

    upload_url = f"{api_base}/data/upload"
    train_url = f"{api_base}/jobs/train"

    dataset = _http_multipart_upload(
        upload_url,
        api_key=args.api_key,
        file_path=tmp_csv,
        name=f"kickoff-{args.job_type}-{int(time.time())}",
        description="Auto-generated synthetic dataset for kickoff training job",
        label_column=args.label_column,
    )
    dataset_id = dataset["dataset_id"]

    hyperparams: Dict[str, Any] = {"label_column": args.label_column}
    if args.hyperparams_json:
        try:
            parsed = json.loads(args.hyperparams_json)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid --hyperparams-json: {e}") from e
        if not isinstance(parsed, dict):
            raise RuntimeError("--hyperparams-json must decode to a JSON object")
        hyperparams.update(parsed)

    train_req = {
        "job_type": args.job_type,
        "dataset_id": dataset_id,
        "hyperparams": hyperparams,
        "gpu_id": args.gpu_id,
    }
    _, queued = _http_json(
        "POST",
        train_url,
        api_key=args.api_key,
        body=json.dumps(train_req).encode("utf-8"),
        content_type="application/json",
        timeout_sec=30,
    )
    job_id = queued["job_id"]
    print(json.dumps({"dataset_id": dataset_id, "job_id": job_id}, indent=2))

    status_url = f"{api_base}/jobs/{job_id}/status"
    deadline = time.time() + args.timeout_sec
    while time.time() < deadline:
        _, st = _http_json("GET", status_url, api_key=args.api_key, timeout_sec=15)
        state = st.get("status")
        if state in ("completed", "failed"):
            print(json.dumps(st, indent=2))
            return 0 if state == "completed" else 1
        time.sleep(args.poll_sec)

    raise RuntimeError(f"Timed out waiting for job {job_id}")


if __name__ == "__main__":
    raise SystemExit(main())
