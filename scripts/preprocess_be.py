from __future__ import annotations

from pathlib import Path
import csv
import json
import math
from typing import Any


# =========================
# 1) 라벨 규칙
# 여기 리스트만 수정하면 라벨 변경 가능
# =========================
HUMAN_USER_IDS = ["a"]
BOT_USER_IDS = ["b"]

LABEL_MAP = {user_id: 0 for user_id in HUMAN_USER_IDS}
LABEL_MAP.update({user_id: 1 for user_id in BOT_USER_IDS})


# =========================
# 2) 입출력 경로
# =========================
REQ_INPUT_DIR = Path("./data/normalized/server_request_log_BE")
EVT_INPUT_DIR = Path("./data/normalized/domain_event_log_BE")
OUTPUT_CSV = Path("./data/feature/preprocess_be.csv")


CSV_COLUMNS = [
    "session_id",
    "user_id",
    "label",
    "showScheduleId",
    "orderId",
    "request_count",
    "endpoint_burst_max_1s",
    "req_interval_cv",
    "target_retry_count",
    "payment_ready_to_terminal_ms",
    "req_source_file",
    "evt_source_file",
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


def calc_endpoint_burst_max_1s(requests_sorted: list[dict[str, Any]]) -> int:
    """
    같은 endpoint 기준, 1초(1000ms) 슬라이딩 윈도우 내 최대 호출 수
    """
    max_burst = 0

    endpoints: dict[str, list[int]] = {}
    for req in requests_sorted:
        endpoint = req["endpoint"]
        ts = req["ts_ms_server"]
        endpoints.setdefault(endpoint, []).append(ts)

    for _, timestamps in endpoints.items():
        left = 0
        for right in range(len(timestamps)):
            while timestamps[right] - timestamps[left] > 1000:
                left += 1
            window_count = right - left + 1
            if window_count > max_burst:
                max_burst = window_count

    return max_burst


def calc_req_interval_cv(requests_sorted: list[dict[str, Any]]) -> float:
    """
    연속 요청 간 시간 차(dt)의 CV = 표준편차 / 평균
    """
    if len(requests_sorted) < 2:
        return 0.0

    intervals: list[float] = []
    for i in range(1, len(requests_sorted)):
        dt = requests_sorted[i]["ts_ms_server"] - requests_sorted[i - 1]["ts_ms_server"]
        if dt > 0:
            intervals.append(float(dt))

    if not intervals:
        return 0.0

    mean_dt = sum(intervals) / len(intervals)
    if mean_dt == 0:
        return 0.0

    variance = sum((x - mean_dt) ** 2 for x in intervals) / len(intervals)
    std_dt = math.sqrt(variance)
    return std_dt / mean_dt


def calc_target_retry_count(target_keys: list[str]) -> int:
    """
    현재 샘플/정규화 기준에서는 target_keys 자체가 세션 내 목표 목록이므로
    일단 target 개수의 총합을 사용.
    샘플에 이미 중복 타깃이 반영되어 있지 않으므로,
    이후 raw 구조 확장 시 requests 단위 target 출현 횟수 기반으로 고도화 가능.
    """
    return len(target_keys)


def load_event_map(evt_dir: Path) -> dict[str, dict[str, Any]]:
    event_map: dict[str, dict[str, Any]] = {}
    for path in sorted(evt_dir.glob("*.json")):
        data = load_json(path)
        session_id = data["session_id"]
        data["_source_file"] = path.name
        event_map[session_id] = data
    return event_map


def build_be_row(req_path: Path, evt_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    req_data = load_json(req_path)

    session_id = req_data["session_id"]
    user_id = req_data["user_id"]
    show_schedule_id = req_data["showScheduleId"]
    order_id = req_data["orderId"]
    target_keys = req_data["target_keys"]
    requests_sorted = req_data["requests_sorted"]

    request_count = len(requests_sorted)
    endpoint_burst_max_1s = calc_endpoint_burst_max_1s(requests_sorted)
    req_interval_cv = calc_req_interval_cv(requests_sorted)
    target_retry_count = calc_target_retry_count(target_keys)

    evt_data = evt_map.get(session_id)
    payment_ready_to_terminal_ms = ""
    evt_source_file = ""

    if evt_data is not None:
        payment_ready_to_terminal_ms = evt_data.get("payment_ready_to_terminal_ms", "")
        evt_source_file = evt_data.get("_source_file", "")

    row = {
        "session_id": session_id,
        "user_id": user_id,
        "label": get_label(user_id),
        "showScheduleId": show_schedule_id,
        "orderId": order_id,
        "request_count": request_count,
        "endpoint_burst_max_1s": endpoint_burst_max_1s,
        "req_interval_cv": round(req_interval_cv, 6),
        "target_retry_count": target_retry_count,
        "payment_ready_to_terminal_ms": payment_ready_to_terminal_ms,
        "req_source_file": req_path.name,
        "evt_source_file": evt_source_file,
    }
    return row


def load_existing_session_ids(csv_path: Path) -> set[str]:
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
        print("[INFO] No new BE rows to append.")
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
    if not REQ_INPUT_DIR.exists():
        raise FileNotFoundError(f"[ERROR] Request input directory not found: {REQ_INPUT_DIR}")

    if not EVT_INPUT_DIR.exists():
        raise FileNotFoundError(f"[ERROR] Event input directory not found: {EVT_INPUT_DIR}")

    evt_map = load_event_map(EVT_INPUT_DIR)
    existing_session_ids = load_existing_session_ids(OUTPUT_CSV)
    new_rows: list[dict[str, Any]] = []

    for req_path in sorted(REQ_INPUT_DIR.glob("*.json")):
        row = build_be_row(req_path, evt_map)
        session_id = row["session_id"]

        if session_id in existing_session_ids:
            print(f"[SKIP] already exists in CSV: {session_id} ({req_path.name})")
            continue

        new_rows.append(row)
        print(f"[ADD] new BE row: {session_id} ({req_path.name})")

    append_rows_to_csv(OUTPUT_CSV, new_rows)


if __name__ == "__main__":
    main()