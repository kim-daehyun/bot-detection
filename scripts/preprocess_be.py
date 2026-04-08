from __future__ import annotations

from pathlib import Path
import csv
import json
import math
import re
from typing import Any


REQ_INPUT_DIR = Path("/Users/daehyun/Desktop/실무통합/bot_detection_project/data/BE/BE_server_request_log/rawdata")
EVT_INPUT_DIR = Path("/Users/daehyun/Desktop/실무통합/bot_detection_project/data/BE/BE_domain_event_log/rawdata")
OUTPUT_CSV = Path("/Users/daehyun/Desktop/실무통합/bot_detection_project/data/BE/feature/be_preprocess.csv")


# =========================
# 최종 핵심 컬럼
# =========================
CSV_COLUMNS = [
    "ts_payment_ready",
    "ts_whole_session",
    "req_interval_cv_pre_hold",
    "req_interval_cv_hold_gap",
    "req_source_file",
    "evt_source_file",
    "label",
]

HEADER_LINE = ",".join(CSV_COLUMNS)


# /api/ticketing/{id}/hold/seat
HOLD_SEAT_PATTERN = re.compile(r"^/api/ticketing/[^/]+/hold/seat$")


def infer_label_from_filename(filename: str) -> int | str:
    if filename.startswith("[bot]"):
        return 1
    if filename.startswith("[human]"):
        return 0
    return ""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
            except json.JSONDecodeError:
                print(f"[WARN] JSON decode error: {path.name}:{line_no}")
                continue

    return records


def normalize_filename_key(filename: str) -> str:
    """
    req/evt 파일명 매칭용 키 정규화
    예:
    [bot][raw]be_req_a_001.jsonl -> a_001
    [bot][raw]be_evt_a_001.jsonl -> a_001
    """
    name = filename
    name = re.sub(r"^\[(bot|human|unknown)\]", "", name)
    name = re.sub(r"^\[raw\]", "", name)

    if name.endswith(".jsonl"):
        name = name[:-6]

    name = re.sub(r"^be_req_", "", name)
    name = re.sub(r"^be_evt_", "", name)
    return name


def build_evt_map(evt_dir: Path) -> dict[str, Path]:
    evt_map: dict[str, Path] = {}
    for evt_path in sorted(evt_dir.glob("*.jsonl")):
        key = normalize_filename_key(evt_path.name)
        evt_map[key] = evt_path
    return evt_map


def find_matching_evt_path(req_path: Path, evt_map: dict[str, Path]) -> Path | None:
    req_key = normalize_filename_key(req_path.name)
    return evt_map.get(req_key)


def load_existing_processed_files(csv_path: Path) -> tuple[set[str], set[str]]:
    existing_req_files: set[str] = set()
    existing_evt_files: set[str] = set()

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return existing_req_files, existing_evt_files

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            req_file = (row.get("req_source_file") or "").strip()
            evt_file = (row.get("evt_source_file") or "").strip()

            if req_file:
                existing_req_files.add(req_file)
            if evt_file:
                existing_evt_files.add(evt_file)

    return existing_req_files, existing_evt_files


def ensure_csv_header(csv_path: Path) -> None:
    """
    CSV가 없으면 정확한 헤더 생성.
    CSV가 있는데 첫 줄 헤더가 다르면 기존 내용을 살리고 맨 위 헤더를 정확히 교정.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            f.write(HEADER_LINE + "\n")
        return

    content = csv_path.read_text(encoding="utf-8")
    if not content:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            f.write(HEADER_LINE + "\n")
        return

    first_line, *rest = content.splitlines()
    if first_line.strip() == HEADER_LINE:
        return

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write(HEADER_LINE + "\n")
        if first_line.strip() and first_line.strip() != HEADER_LINE:
            f.write(first_line.rstrip("\n") + "\n")
        for line in rest:
            f.write(line.rstrip("\n") + "\n")


def is_hold_seat(record: dict[str, Any]) -> bool:
    path = record.get("path", "")
    return isinstance(path, str) and bool(HOLD_SEAT_PATTERN.match(path))


def extract_numeric_timestamps(records: list[dict[str, Any]]) -> list[float]:
    timestamps: list[float] = []
    for r in records:
        ts = r.get("tsServer")
        if isinstance(ts, (int, float)):
            timestamps.append(float(ts))
    return timestamps


def calc_cv_from_timestamps(timestamps: list[float]) -> float:
    """
    인접 요청 간격(dt)의 CV = std / mean
    """
    if len(timestamps) < 2:
        return 0.0

    timestamps = sorted(timestamps)

    intervals: list[float] = []
    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i - 1]
        if dt > 0:
            intervals.append(dt)

    if not intervals:
        return 0.0

    mean_dt = sum(intervals) / len(intervals)
    if mean_dt == 0:
        return 0.0

    variance = sum((x - mean_dt) ** 2 for x in intervals) / len(intervals)
    std_dt = math.sqrt(variance)
    return std_dt / mean_dt


def split_pre_post_hold_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    첫 번째 hold/seat를 경계로 pre / post 분리
    - pre_hold  : 첫 hold/seat 직전까지
    - post_hold : 첫 hold/seat 포함 이후
    """
    first_hold_idx = None

    for idx, record in enumerate(records):
        if is_hold_seat(record):
            first_hold_idx = idx
            break

    if first_hold_idx is None:
        return records[:], []

    pre_records = records[:first_hold_idx]
    post_records = records[first_hold_idx:]

    return pre_records, post_records


def calc_req_interval_cv_pre_hold(records: list[dict[str, Any]]) -> float:
    pre_records, _ = split_pre_post_hold_records(records)
    timestamps = extract_numeric_timestamps(pre_records)
    return calc_cv_from_timestamps(timestamps)


def calc_req_interval_cv_post_hold(records: list[dict[str, Any]]) -> float:
    _, post_records = split_pre_post_hold_records(records)
    timestamps = extract_numeric_timestamps(post_records)
    return calc_cv_from_timestamps(timestamps)


def calc_req_interval_cv_hold_gap(records: list[dict[str, Any]]) -> float:
    """
    선점 후 CV - 선점 전 CV
    음수면: 선점 후가 더 안정적/규칙적
    양수면: 선점 후가 더 불규칙
    """
    pre_hold_cv = calc_req_interval_cv_pre_hold(records)
    post_hold_cv = calc_req_interval_cv_post_hold(records)
    return abs(post_hold_cv - pre_hold_cv)


def calc_ts_payment_ready(records: list[dict[str, Any]]) -> float | str:
    """
    EVT 파일 안의 tsServer 값들 중
    가장 이른 시각과 가장 늦은 시각의 차이
    """
    timestamps = []
    for r in records:
        ts = r.get("tsServer")
        if isinstance(ts, (int, float)):
            timestamps.append(float(ts))

    if len(timestamps) < 2:
        return ""

    timestamps.sort()
    return timestamps[-1] - timestamps[0]


def calc_ts_whole_session(records: list[dict[str, Any]]) -> float | str:
    """
    req 파일에서:
    - 첫 번째 /api/auth/login 의 tsServer
    - 마지막 /api/payments/confirm 의 tsServer
    차이를 계산
    """
    login_ts = None
    confirm_ts = None

    for r in records:
        path = r.get("path")
        ts = r.get("tsServer")

        if not isinstance(path, str):
            continue
        if not isinstance(ts, (int, float)):
            continue

        if login_ts is None and path == "/api/auth/login":
            login_ts = float(ts)

        if path == "/api/payments/confirm":
            confirm_ts = float(ts)

    if login_ts is None or confirm_ts is None:
        return ""

    return confirm_ts - login_ts


def build_row(req_path: Path, evt_map: dict[str, Path]) -> dict[str, Any] | None:
    matched_evt_path = find_matching_evt_path(req_path, evt_map)
    if matched_evt_path is None:
        print(f"[SKIP] matching evt file not found for req: {req_path.name}")
        return None

    req_records = load_jsonl(req_path)
    evt_records = load_jsonl(matched_evt_path)

    req_interval_cv_pre_hold = calc_req_interval_cv_pre_hold(req_records)
    req_interval_cv_hold_gap = calc_req_interval_cv_hold_gap(req_records)
    ts_payment_ready = calc_ts_payment_ready(evt_records)
    ts_whole_session = calc_ts_whole_session(req_records)
    label = infer_label_from_filename(req_path.name)

    row = {
        "ts_payment_ready": ts_payment_ready,
        "ts_whole_session": ts_whole_session,
        "req_interval_cv_pre_hold": round(req_interval_cv_pre_hold, 6),
        "req_interval_cv_hold_gap": round(req_interval_cv_hold_gap, 6),
        "req_source_file": req_path.name,
        "evt_source_file": matched_evt_path.name,
        "label": label,
    }
    return row


def append_rows_to_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("[INFO] No new BE rows to append.")
        return

    ensure_csv_header(csv_path)

    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        for row in rows:
            writer.writerow(row)

    print(f"[DONE] Appended {len(rows)} new rows -> {csv_path}")


def main() -> None:
    if not REQ_INPUT_DIR.exists():
        raise FileNotFoundError(f"[ERROR] Request input directory not found: {REQ_INPUT_DIR}")

    if not EVT_INPUT_DIR.exists():
        raise FileNotFoundError(f"[ERROR] Event input directory not found: {EVT_INPUT_DIR}")

    ensure_csv_header(OUTPUT_CSV)

    evt_map = build_evt_map(EVT_INPUT_DIR)
    existing_req_files, existing_evt_files = load_existing_processed_files(OUTPUT_CSV)

    new_rows: list[dict[str, Any]] = []

    for req_path in sorted(REQ_INPUT_DIR.glob("*.jsonl")):
        if req_path.name in existing_req_files:
            print(f"[SKIP] already processed req file: {req_path.name}")
            continue

        matched_evt_path = find_matching_evt_path(req_path, evt_map)
        if matched_evt_path is None:
            print(f"[SKIP] no matching evt file for req: {req_path.name}")
            continue

        if matched_evt_path.name in existing_evt_files:
            print(f"[SKIP] already processed evt file: {matched_evt_path.name}")
            continue

        row = build_row(req_path, evt_map)
        if row is None:
            continue

        new_rows.append(row)
        print(
            f"[ADD] req={row['req_source_file']}, "
            f"evt={row['evt_source_file']}, "
            f"label={row['label']}, "
            f"pre_hold_cv={row['req_interval_cv_pre_hold']}, "
            f"hold_gap={row['req_interval_cv_hold_gap']}, "
            f"ts_payment_ready={row['ts_payment_ready']}, "
            f"ts_whole_session={row['ts_whole_session']}"
        )

    append_rows_to_csv(OUTPUT_CSV, new_rows)


if __name__ == "__main__":
    main()