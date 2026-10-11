"""Export full-quality masters from explicitly reviewed preview candidates.

The preview batch and decisions CSV can be moved between machines. Original
source videos must be present in --source-dir when this command is run.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

from src.data.preparation.cut_clips import (
    ACCEPTED_FIELDS,
    CutConfig,
    _sha256,
    _write_csv_atomic,
    _write_json_atomic,
    cut_clip_with_fallback,
    stable_clip_id,
)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def validate_reviews(candidates, decisions):
    ids = [row.get("clip_id", "").strip() for row in candidates]
    reviewed_ids = [row.get("clip_id", "").strip() for row in decisions]
    if not ids or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("preview_candidates.csv thiếu clip hoặc có clip_id rỗng/trùng")
    if any(not value for value in reviewed_ids) or len(reviewed_ids) != len(set(reviewed_ids)):
        raise ValueError("review decisions có clip_id rỗng/trùng")
    if set(ids) != set(reviewed_ids):
        raise ValueError(
            f"review chưa phủ đủ candidates: missing={len(set(ids)-set(reviewed_ids))}, "
            f"unknown={len(set(reviewed_ids)-set(ids))}"
        )
    result = {}
    for row in decisions:
        decision = row.get("decision", "").strip().lower()
        if decision not in {"keep", "reject"}:
            raise ValueError(f"decision phải là keep/reject: {row.get('clip_id')}")
        result[row["clip_id"].strip()] = decision
    return result


def source_map(inventory, source_dir):
    source_dir = Path(source_dir).resolve()
    mapped = {}
    for row in inventory:
        filename = row.get("filename", "")
        if not filename:
            raise ValueError("input_inventory.csv thiếu filename")
        source = (source_dir / filename).resolve()
        try:
            source.relative_to(source_dir)
        except ValueError as exc:
            raise ValueError(f"filename vượt source_dir: {filename}") from exc
        stem = Path(filename).stem
        if stem in mapped:
            raise ValueError(f"source stem trùng: {stem}")
        mapped[stem] = source
    return mapped


def export_reviewed(preview_batch, decisions_csv, source_dir, output_root, run_id):
    preview_batch = Path(preview_batch).resolve()
    candidates = read_csv(preview_batch / "preview_candidates.csv")
    preview_summary = json.loads((preview_batch / "run_summary.json").read_text(encoding="utf-8"))
    if preview_summary.get("output_mode") != "preview" or preview_summary.get("accepted_clips") != len(candidates):
        raise ValueError("Preview candidates không khớp run_summary.json")
    decisions = read_csv(decisions_csv)
    reviews = validate_reviews(candidates, decisions)
    inventory = read_csv(preview_batch / "input_inventory.csv")
    sources = source_map(inventory, source_dir)
    source_sizes = {Path(row["filename"]).stem: int(row["size_bytes"]) for row in inventory}
    config = CutConfig.from_json(preview_batch / "config.json")
    if config.preview_height == 0:
        raise ValueError("Chỉ nhận batch được tạo ở chế độ preview")
    full_config = replace(
        config, preview_height=0, dataset_dir=str(Path(source_dir).resolve()),
        output_root=str(Path(output_root).resolve()), run_id=run_id,
    )
    full_config.validate()
    kept = []
    for row in candidates:
        if reviews[row["clip_id"]] != "keep":
            continue
        source = sources.get(row["source_video"])
        if source is None:
            raise ValueError(f"Không tìm thấy source_video: {row['source_video']}")
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.stat().st_size != source_sizes[row["source_video"]]:
            raise ValueError(f"Kích thước video gốc đã đổi: {source}")
        start, end = float(row["start_time"]), float(row["end_time"])
        if end <= start or row["clip_id"] != stable_clip_id(row["source_video"], start, end):
            raise ValueError(f"Mốc/clip_id không khớp: {row['clip_id']}")
        kept.append((row, source, start, end))
    output = Path(output_root).resolve() / f"cut_{run_id}" / config.tier / "reviewed"
    if output.exists():
        raise FileExistsError(f"Output bất biến đã tồn tại: {output}")
    media = output / "media"
    media.mkdir(parents=True)
    started = time.perf_counter()
    accepted = []
    for row, source, start, end in kept:
        clip_path = media / f"{row['clip_id']}.mp4"
        cut = cut_clip_with_fallback(source, clip_path, start, end, 0, full_config)
        if not cut.ok:
            raise RuntimeError(f"Xuất master lỗi: {row['clip_id']} / {cut.attempts}")
        accepted.append({**row, "file_path": str(clip_path),
                         "cut_backend": cut.backend, "run_id": run_id})
    _write_csv_atomic(output / "accepted_clips.csv", accepted, ACCEPTED_FIELDS)
    _write_csv_atomic(
        output / "review_decisions.csv",
        [{"clip_id": row["clip_id"], "decision": reviews[row["clip_id"]]}
         for row in candidates],
        ["clip_id", "decision"],
    )
    _write_json_atomic(output / "config.json", asdict(full_config))
    summary = {
        "schema": "reviewed_master_export_v1",
        "preview_batch": str(preview_batch), "review_file": str(Path(decisions_csv).resolve()),
        "source_dir": str(Path(source_dir).resolve()),
        "candidate_previews": len(candidates), "reviewed_keep": len(kept),
        "reviewed_reject": len(candidates) - len(kept),
        "masters_exported": len(accepted),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    _write_json_atomic(output / "run_summary.json", summary)
    if full_config.hash_media:
        paths = sorted(path for path in output.rglob("*") if path.is_file())
        (output / "SHA256SUMS").write_text(
            "".join(f"{_sha256(path)}  {path.relative_to(output).as_posix()}\n" for path in paths),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Batch output: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-batch", required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    export_reviewed(args.preview_batch, args.decisions, args.source_dir,
                    args.output_root, args.run_id)


if __name__ == "__main__":
    main()
