"""Create deterministic ALFWorld placeholder parquet files for AgentOPSD runs.

The environment manager supplies the actual ALFWorld observation after reset;
the parquet rows only provide the dataset cardinality expected by the trainer.
This utility deliberately records hashes and the reduced-run label so that the
generated files cannot be mistaken for an official offline dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def rows(split: str, count: int) -> list[dict]:
    return [
        {
            "data_source": "alfworld",
            "prompt": [{"role": "user", "content": "Begin the ALFWorld task."}],
            "ability": "agent",
            "extra_info": {"split": split, "index": i},
        }
        for i in range(count)
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, required=True)
    parser.add_argument("--val-rows", type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    train = args.output / "train.parquet"
    val = args.output / "val.parquet"
    pq.write_table(pa.Table.from_pylist(rows("train", args.train_rows)), train)
    pq.write_table(pa.Table.from_pylist(rows("test", args.val_rows)), val)
    manifest = {
        "kind": "agentopsd_alfworld_placeholder_dataset",
        "label": "engineering_approximation_not_paper_dataset",
        "reason": "AlfWorldEnvironmentManager resets the actual task; parquet rows provide cardinality only.",
        "train_rows": args.train_rows,
        "val_rows": args.val_rows,
        "train.parquet_sha256": sha256(train),
        "val.parquet_sha256": sha256(val),
        "train.parquet_bytes": train.stat().st_size,
        "val.parquet_bytes": val.stat().st_size,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
