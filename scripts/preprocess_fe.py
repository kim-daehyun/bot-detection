from __future__ import annotations

from pathlib import Path
import csv
import json
import math
from typing import Any


# =========================
# 1) 라벨 규칙
# 여기 리스트만 수정하면 라벨링 변경 가능
# =========================
HUMAN_USER_IDS = ["a"]
BOT_USER_IDS = ["b"]

LABEL_MAP = {user_id: 0 for user_id in HUMAN_USER_IDS}
LABEL_MAP.update({user_id: 1 for user_id in BOT_USER_IDS})


# =========================
# 2) 입출력 경로
# =========================
INPUT_DIR = Path("./data/normalized/client_telemetry_log_FE")
OUTPUT_CSV = Path("./data/feature/preprocess_fe.csv")


# =========================
# 3) teleport 판정 기준
# =========================
TELEPORT_DT_MS_THRESHOLD = 20
TELEPORT_NORM_DIST_THRESHOLD = 0.12
TELEPORT_NORM_SPEED_THRESHOLD = 0.006


CSV_COLUMNS = [
    "session_id",
    "user_id",
    "label",
    "event_type",
    "duration_ms",
    "mousemove_count_raw",
    "mousemove_count_actual",
    "mousemove_teleport_count",
    "mouse_activity_rate",
    "mouse_teleport_rate",
    "viewport_width",
    "viewport_height",
    "source_file",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_label(user_id: str) -> int | str:
    return LABEL_MAP.get(user_id, "")


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def calc_mouse_activity_rate(mousemove_count_actual: int, duration_ms: int) -> float:
    return safe_div(mousemove_count_actual, duration_ms / 1000.0)


def calc_mousemove_teleport_count(
    events: list[dict[str, Any]],
    viewport_width: int,
    viewport_height: int,
) -> int:
    """
    사람처럼 연속 이동한 것이 아니라,
    비정상적으로 크게 점프한 이동 횟수를 계산.
    """
    if len(events) < 2:
        return 0

    teleport_count = 0

    for i in range(1, len(events)):
        prev_evt = events[i - 1]
        curr_evt = events[i]

        dt = curr_evt["timestamp"] - prev_evt["timestamp"]
        if dt <= 0:
            continue

        dx = curr_evt["x"] - prev_evt["x"]
        dy = curr_evt["y"] - prev_evt["y"]

        norm_dx = dx / viewport_width if viewport_width > 0 else 0.0
        norm_dy = dy / viewport_height if viewport_height > 0 else 0.0
        norm_dist = math.sqrt(norm_dx**2 + norm_dy**2)
        norm_speed = norm_dist / dt

        is_teleport = (
            (dt < TELEPORT_DT_MS_THRESHOLD and norm_dist > TELEPORT_NORM_DIST_THRESHOLD)
            or (norm_speed > TELEPORT_NORM_SPEED_THRESHOLD)
        )

        if is_teleport:
            teleport_count += 1

    return teleport_count


def build_fe_row(path: Path) -> dict[str, Any]:
    data = load_json(path)

    session_id = data["session_id"]
    user_id = data["user_id"]
    event_type = data["event_type"]
    duration_ms = data["duration_ms"]
    mousemove_count_raw = data["mousemove_count_raw"]
    mousemove_count_actual = data["mousemove_count_actual"]
    events = data["mousemove_events_sorted"]
    viewport_width = data["viewport_width"]
    viewport_height = data["viewport_height"]

    mousemove_teleport_count = calc_mousemove_teleport_count(
        events=events,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
    )

    mouse_activity_rate = calc_mouse_activity_rate(
        mousemove_count_actual=mousemove_count_actual,
        duration_ms=duration_ms,
    )

    mouse_teleport_rate = safe_div(mousemove_teleport_count, mousemove_count_actual)

    row = {
        "session_id": session_id,
        "user_id": user_id,
        "label": get_label(user_id),
        "event_type": event_type,
        "duration_ms": duration_ms,
        "mousemove_count_raw": mousemove_count_raw,
        "mousemove_count_actual": mousemove_count_actual,
        "mousemove_teleport_count": mousemove_teleport_count,
        "mouse_activity_rate": round(mouse_activity_rate, 6),
        "mouse_teleport_rate": round(mouse_teleport_rate, 6),
        "viewport_width": viewport_width,
        "viewport_height": viewport_height,
        "source_file": path.name,
    }
    return row


def load_existing_session_ids(csv_path: Path) -> set[str]:
    """
    기존 CSV가 있으면 이미 처리된 session_id를 읽어온다.
    append-only 방식 유지용.
    """
    if not csv_path.exists():
        return set()

    existing_ids: set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            session_id = row.get("session_id", "")
            if session_id:
                existing_ids.add(session_id)

    return existing_ids


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

    existing_session_ids = load_existing_session_ids(OUTPUT_CSV)
    new_rows: list[dict[str, Any]] = []

    for path in sorted(INPUT_DIR.glob("*.json")):
        row = build_fe_row(path)
        session_id = row["session_id"]

        if session_id in existing_session_ids:
            print(f"[SKIP] already exists in CSV: {session_id} ({path.name})")
            continue

        new_rows.append(row)
        print(f"[ADD] new FE row: {session_id} ({path.name})")

    append_rows_to_csv(OUTPUT_CSV, new_rows)


if __name__ == "__main__":
    main()