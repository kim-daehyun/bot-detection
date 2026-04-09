from __future__ import annotations

from pathlib import Path
import csv
import json
import math
from typing import Any


# =========================
# 1) 입출력 경로
# =========================
INPUT_DIR = Path("/Users/daehyun/Desktop/실무통합/bot_detection_project/data/FE/rawdata")
OUTPUT_CSV = Path("/Users/daehyun/Desktop/실무통합/bot_detection_project/data/FE/feature/fe_preprocess.csv")


# =========================
# 2) teleport 판정 기준
# =========================
TELEPORT_DT_MS_THRESHOLD = 20
TELEPORT_NORM_DIST_THRESHOLD = 0.003
TELEPORT_NORM_SPEED_THRESHOLD = 0.002


# =========================
# 3) 최종 CSV 컬럼
# =========================
CSV_COLUMNS = [
    "duration_ms",
    "mousemove_teleport_count",
    "mousemove_count",
    "source_file",
    "label",
]


def get_label_from_filename(path: Path) -> int:
    """
    파일명 머릿말 기준 라벨링
    [bot]   -> 1
    [human] -> 0
    """
    name = path.name

    if name.startswith("[bot]"):
        return 1
    if name.startswith("[human]"):
        return 0

    raise ValueError(
        f"[ERROR] 파일명 머릿말이 [bot] 또는 [human] 형식이 아닙니다: {name}"
    )


def calc_duration_ms(page_enter_ts: int, page_leave_ts: int) -> int:
    return max(0, int(page_leave_ts) - int(page_enter_ts))


def calc_mousemove_teleport_count(
    events: list[dict[str, Any]],
    viewport_width: int,
    viewport_height: int,
) -> int:
    """
    teleport 판정 기준:
      (dt < 20ms and norm_dist > 0.12) or (norm_speed > 0.006)
    """
    if len(events) < 2:
        return 0

    teleport_count = 0

    for i in range(1, len(events)):
        prev_evt = events[i - 1]
        curr_evt = events[i]

        prev_ts = prev_evt.get("timestamp")
        curr_ts = curr_evt.get("timestamp")
        prev_x = prev_evt.get("x")
        prev_y = prev_evt.get("y")
        curr_x = curr_evt.get("x")
        curr_y = curr_evt.get("y")

        if None in (prev_ts, curr_ts, prev_x, prev_y, curr_x, curr_y):
            continue

        dt = curr_ts - prev_ts
        if dt <= 0:
            continue

        dx = curr_x - prev_x
        dy = curr_y - prev_y

        norm_dx = dx / viewport_width if viewport_width > 0 else 0.0
        norm_dy = dy / viewport_height if viewport_height > 0 else 0.0
        norm_dist = math.sqrt(norm_dx ** 2 + norm_dy ** 2)
        norm_speed = norm_dist / dt

        is_teleport = (
            (dt < TELEPORT_DT_MS_THRESHOLD and norm_dist > TELEPORT_NORM_DIST_THRESHOLD)
            or (norm_speed > TELEPORT_NORM_SPEED_THRESHOLD)
        )

        if is_teleport:
            teleport_count += 1

    return teleport_count


def load_existing_source_files(csv_path: Path) -> set[str]:
    """
    기존 CSV에 이미 들어간 source_file 목록을 읽어와
    동일 파일은 다시 처리하지 않도록 한다.
    """
    if not csv_path.exists():
        return set()

    existing_files: set[str] = set()

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_file = (row.get("source_file") or "").strip()
            if source_file:
                existing_files.add(source_file)

    return existing_files


def parse_jsonl_file(path: Path) -> list[dict[str, Any]]:
    """
    .jsonl 파일: 한 줄당 JSON 1개
    """
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON decode error - {path.name}:{line_no} - {e}")
                continue

            if not isinstance(obj, dict):
                print(f"[WARN] JSON object 아님 - {path.name}:{line_no}")
                continue

            rows.append(obj)

    return rows


def build_fe_rows_from_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    파일 1개(.jsonl) 안의 각 JSON line을 모두 읽어서 row 생성
    """
    label = get_label_from_filename(path)
    log_objects = parse_jsonl_file(path)

    built_rows: list[dict[str, Any]] = []

    for idx, data in enumerate(log_objects, start=1):
        request_body = data.get("requestBody", {})
        if not isinstance(request_body, dict):
            print(f"[WARN] requestBody 없음/비정상 - {path.name}:{idx}")
            continue

        page_enter_ts = request_body.get("page_enter_ts")
        page_leave_ts = request_body.get("page_leave_ts")
        mousemove_events = request_body.get("mousemove", [])
        mousemove_count = request_body.get("mousemove_count", 0)
        viewport_width = request_body.get("viewport_width", 0)
        viewport_height = request_body.get("viewport_height", 0)

        if page_enter_ts is None or page_leave_ts is None:
            print(f"[WARN] page_enter_ts / page_leave_ts 누락 - {path.name}:{idx}")
            continue

        if not isinstance(mousemove_events, list):
            mousemove_events = []

        duration_ms = calc_duration_ms(page_enter_ts, page_leave_ts)

        mousemove_teleport_count = calc_mousemove_teleport_count(
            events=mousemove_events,
            viewport_width=int(viewport_width or 0),
            viewport_height=int(viewport_height or 0),
        )

        row = {
            "duration_ms": duration_ms,
            "mousemove_teleport_count": int(mousemove_teleport_count),
            "mousemove_count": int(mousemove_count or 0),
            "source_file": path.name,
            "label": label,
        }
        built_rows.append(row)

    return built_rows


def append_rows_to_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("[INFO] No new FE rows to append.")
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)

        if not file_exists:
            writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(f"[DONE] Appended {len(rows)} new rows -> {csv_path}")


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"[ERROR] Input directory not found: {INPUT_DIR}")

    existing_source_files = load_existing_source_files(OUTPUT_CSV)
    new_rows: list[dict[str, Any]] = []

    # rawdata 하위 모든 폴더의 .jsonl 파일 탐색
    jsonl_files = sorted(INPUT_DIR.rglob("*.jsonl"))

    if not jsonl_files:
        print(f"[INFO] .jsonl files not found under: {INPUT_DIR}")
        return

    for path in jsonl_files:
        # 이미 같은 파일명이 처리되어 있으면 skip
        if path.name in existing_source_files:
            print(f"[SKIP] already processed source_file: {path.name}")
            continue

        try:
            rows = build_fe_rows_from_jsonl(path)

            if not rows:
                print(f"[WARN] no valid rows: {path.name}")
                continue

            new_rows.extend(rows)
            print(f"[ADD] {path.name} -> {len(rows)} rows")

        except Exception as e:
            print(f"[ERROR] {path.name}: {e}")

    append_rows_to_csv(OUTPUT_CSV, new_rows)


if __name__ == "__main__":
    main()