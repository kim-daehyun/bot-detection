from pathlib import Path
import json
from typing import Any


RAW_BASE = Path("./data/rawdata")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_required_fields(data: dict[str, Any], required_fields: list[str], file_path: Path) -> None:
    missing = [field for field in required_fields if field not in data]
    if missing:
        raise ValueError(f"[ERROR] {file_path} missing fields: {missing}")


def check_type(value: Any, expected_type: type, field_name: str, file_path: Path) -> None:
    if not isinstance(value, expected_type):
        raise TypeError(
            f"[ERROR] {file_path} field '{field_name}' type mismatch: "
            f"expected {expected_type.__name__}, got {type(value).__name__}"
        )


def check_fe_schema(file_path: Path) -> None:
    data = load_json(file_path)

    required_fields = [
        "session_id",
        "user_id",
        "event_type",
        "page_enter_ts",
        "page_leave_ts",
        "mousemove_events",
        "mousemove_count",
        "viewport_width",
        "viewport_height",
    ]
    check_required_fields(data, required_fields, file_path)

    check_type(data["session_id"], str, "session_id", file_path)
    check_type(data["user_id"], str, "user_id", file_path)
    check_type(data["event_type"], str, "event_type", file_path)
    check_type(data["page_enter_ts"], int, "page_enter_ts", file_path)
    check_type(data["page_leave_ts"], int, "page_leave_ts", file_path)
    check_type(data["mousemove_events"], list, "mousemove_events", file_path)
    check_type(data["mousemove_count"], int, "mousemove_count", file_path)
    check_type(data["viewport_width"], int, "viewport_width", file_path)
    check_type(data["viewport_height"], int, "viewport_height", file_path)

    for i, evt in enumerate(data["mousemove_events"]):
        if not isinstance(evt, dict):
            raise TypeError(f"[ERROR] {file_path} mousemove_events[{i}] must be dict")
        for k in ["timestamp", "x", "y"]:
            if k not in evt:
                raise ValueError(f"[ERROR] {file_path} mousemove_events[{i}] missing '{k}'")
            if not isinstance(evt[k], int):
                raise TypeError(f"[ERROR] {file_path} mousemove_events[{i}]['{k}'] must be int")

    print(f"[OK] FE schema: {file_path}")


def check_be_request_schema(file_path: Path) -> None:
    data = load_json(file_path)

    required_fields = [
        "UUID",
        "X-User-Id",
        "X-Session-Ticket",
        "showScheduleId",
        "seatIds",
        "orderId",
        "server_request_log",
    ]
    check_required_fields(data, required_fields, file_path)

    check_type(data["UUID"], str, "UUID", file_path)
    check_type(data["X-User-Id"], str, "X-User-Id", file_path)
    check_type(data["X-Session-Ticket"], str, "X-Session-Ticket", file_path)
    check_type(data["showScheduleId"], int, "showScheduleId", file_path)
    check_type(data["seatIds"], list, "seatIds", file_path)
    check_type(data["orderId"], str, "orderId", file_path)
    check_type(data["server_request_log"], list, "server_request_log", file_path)

    for i, req in enumerate(data["server_request_log"]):
        if not isinstance(req, dict):
            raise TypeError(f"[ERROR] {file_path} server_request_log[{i}] must be dict")
        for k in ["endpoint", "ts_ms_server"]:
            if k not in req:
                raise ValueError(f"[ERROR] {file_path} server_request_log[{i}] missing '{k}'")
        if not isinstance(req["endpoint"], str):
            raise TypeError(f"[ERROR] {file_path} server_request_log[{i}]['endpoint'] must be str")
        if not isinstance(req["ts_ms_server"], int):
            raise TypeError(f"[ERROR] {file_path} server_request_log[{i}]['ts_ms_server'] must be int")

    print(f"[OK] BE request schema: {file_path}")


def check_be_event_schema(file_path: Path) -> None:
    data = load_json(file_path)

    required_fields = [
        "UUID",
        "reservationNumber",
        "orderId",
        "ts_payment_ready",
        "ts_approvedAt_or_ts_fail",
    ]
    check_required_fields(data, required_fields, file_path)

    check_type(data["UUID"], str, "UUID", file_path)
    check_type(data["reservationNumber"], str, "reservationNumber", file_path)
    check_type(data["orderId"], str, "orderId", file_path)
    check_type(data["ts_payment_ready"], int, "ts_payment_ready", file_path)
    check_type(data["ts_approvedAt_or_ts_fail"], int, "ts_approvedAt_or_ts_fail", file_path)

    print(f"[OK] BE event schema: {file_path}")


def main() -> None:
    fe_dir = RAW_BASE / "client_telemetry_log_FE"
    be_req_dir = RAW_BASE / "server_request_log_BE"
    be_evt_dir = RAW_BASE / "domain_event_log_BE"

    for path in sorted(fe_dir.glob("*.json")):
        check_fe_schema(path)

    for path in sorted(be_req_dir.glob("*.json")):
        check_be_request_schema(path)

    for path in sorted(be_evt_dir.glob("*.json")):
        check_be_event_schema(path)

    print("\n[DONE] All schema checks passed.")


if __name__ == "__main__":
    main()