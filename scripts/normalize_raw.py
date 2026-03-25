from pathlib import Path
import json
from typing import Any


FE_RAW_DIR = Path("./data/FE/rawdata")
BE_EVENT_RAW_DIR = Path("./data/BE/BE_domain_event_log/rawdata")
BE_REQUEST_RAW_DIR = Path("./data/BE/BE_server_request_log/rawdata")

FE_NORM_DIR = Path("./data/FE/normalized")
BE_EVENT_NORM_DIR = Path("./data/BE/BE_domain_event_log/normalized")
BE_REQUEST_NORM_DIR = Path("./data/BE/BE_server_request_log/normalized")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalized_filename(path: Path) -> str:
    return f"[norm]{path.name}"


def normalize_fe(data: dict[str, Any]) -> dict[str, Any]:
    page_enter_ts = data["page_enter_ts"]
    page_leave_ts = data["page_leave_ts"]
    mousemove_events = sorted(data["mousemove_events"], key=lambda x: x["timestamp"])

    duration_ms = page_leave_ts - page_enter_ts

    normalized = {
        "session_id": data["session_id"],
        "user_id": data["user_id"],
        "event_type": data["event_type"],
        "page_enter_ts": page_enter_ts,
        "page_leave_ts": page_leave_ts,
        "duration_ms": duration_ms,
        "mousemove_count_raw": data["mousemove_count"],
        "mousemove_count_actual": len(mousemove_events),
        "mousemove_events_sorted": mousemove_events,
        "viewport_width": data["viewport_width"],
        "viewport_height": data["viewport_height"],
    }
    return normalized


def normalize_be_request(data: dict[str, Any]) -> dict[str, Any]:
    requests = sorted(data["server_request_log"], key=lambda x: x["ts_ms_server"])

    seat_ids = data["seatIds"]
    show_schedule_id = data["showScheduleId"]

    target_keys = [f"{show_schedule_id}_{seat_id}" for seat_id in seat_ids]

    normalized = {
        "session_id": data["UUID"],
        "user_id": data["X-User-Id"],
        "session_ticket": data["X-Session-Ticket"],
        "showScheduleId": show_schedule_id,
        "seatIds": seat_ids,
        "orderId": data["orderId"],
        "target_keys": target_keys,
        "requests_sorted": requests,
    }
    return normalized


def normalize_be_event(data: dict[str, Any]) -> dict[str, Any]:
    ts_payment_ready = data["ts_payment_ready"]
    ts_terminal = data["ts_approvedAt_or_ts_fail"]

    normalized = {
        "session_id": data["UUID"],
        "reservationNumber": data["reservationNumber"],
        "orderId": data["orderId"],
        "ts_payment_ready": ts_payment_ready,
        "ts_terminal": ts_terminal,
        "payment_ready_to_terminal_ms": ts_terminal - ts_payment_ready,
    }
    return normalized


def main() -> None:
    if not FE_RAW_DIR.exists():
        raise FileNotFoundError(f"[ERROR] FE raw directory not found: {FE_RAW_DIR}")

    if not BE_EVENT_RAW_DIR.exists():
        raise FileNotFoundError(f"[ERROR] BE event raw directory not found: {BE_EVENT_RAW_DIR}")

    if not BE_REQUEST_RAW_DIR.exists():
        raise FileNotFoundError(f"[ERROR] BE request raw directory not found: {BE_REQUEST_RAW_DIR}")

    for path in sorted(FE_RAW_DIR.glob("*.json")):
        data = load_json(path)
        normalized = normalize_fe(data)
        out_name = normalized_filename(path)
        save_json(normalized, FE_NORM_DIR / out_name)
        print(f"[OK] normalized FE: {out_name}")

    for path in sorted(BE_REQUEST_RAW_DIR.glob("*.json")):
        data = load_json(path)
        normalized = normalize_be_request(data)
        out_name = normalized_filename(path)
        save_json(normalized, BE_REQUEST_NORM_DIR / out_name)
        print(f"[OK] normalized BE request: {out_name}")

    for path in sorted(BE_EVENT_RAW_DIR.glob("*.json")):
        data = load_json(path)
        normalized = normalize_be_event(data)
        out_name = normalized_filename(path)
        save_json(normalized, BE_EVENT_NORM_DIR / out_name)
        print(f"[OK] normalized BE event: {out_name}")

    print("\n[DONE] Raw normalization completed.")


if __name__ == "__main__":
    main()