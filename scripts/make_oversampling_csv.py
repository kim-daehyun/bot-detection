from __future__ import annotations

from pathlib import Path
import random
import math
import csv


random.seed(42)

FE_INPUT = Path("./data/FE/feature/fe_preprocess.csv")
BE_INPUT = Path("./data/BE/feature/be_preprocess.csv")

FE_OUTPUT = Path("./data/FE/feature/[over_sampling]fe_preprocess.csv")
BE_OUTPUT = Path("./data/BE/feature/[over_sampling]be_preprocess.csv")

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_int(v: str) -> int:
    return int(float(v))


def to_float(v: str) -> float:
    return float(v)


def clip_int(v: float, low: int | None = None, high: int | None = None) -> int:
    x = int(round(v))
    if low is not None:
        x = max(low, x)
    if high is not None:
        x = min(high, x)
    return x


def clip_float(v: float, low: float | None = None, high: float | None = None) -> float:
    x = float(v)
    if low is not None:
        x = max(low, x)
    if high is not None:
        x = min(high, x)
    return x


def make_fe_rows(base_rows: list[dict[str, str]], n: int = 100) -> list[dict]:
    out = []

    event_types = sorted({row["event_type"] for row in base_rows})
    source_files = sorted({row["source_file"] for row in base_rows})

    for i in range(1, n + 1):
        base = random.choice(base_rows)

        label = random.randint(0, 1)
        user_id = "a" if label == 0 else "b"

        duration_ms = clip_int(
            to_int(base["duration_ms"]) * random.uniform(0.8, 1.2),
            low=300
        )

        mousemove_count_actual = clip_int(
            to_int(base["mousemove_count_actual"]) * random.uniform(0.75, 1.25),
            low=1
        )

        mousemove_count_raw = clip_int(
            mousemove_count_actual + random.uniform(-2, 2),
            low=1
        )

        mousemove_teleport_count = clip_int(
            to_int(base["mousemove_teleport_count"]) + random.uniform(-1, 2),
            low=0,
            high=mousemove_count_actual
        )

        mouse_activity_rate = round(mousemove_count_actual / (duration_ms / 1000.0), 6)
        mouse_teleport_rate = round(
            mousemove_teleport_count / mousemove_count_actual if mousemove_count_actual > 0 else 0.0,
            6
        )

        viewport_width = random.choice([1366, 1440, 1536, 1600, 1728, 1920])
        viewport_height = random.choice([768, 864, 900, 960, 1080])

        out.append({
            "session_id": f"os_fe_{i:03d}",
            "user_id": user_id,
            "label": label,
            "event_type": random.choice(event_types) if event_types else "seatmap",
            "duration_ms": duration_ms,
            "mousemove_count_raw": mousemove_count_raw,
            "mousemove_count_actual": mousemove_count_actual,
            "mousemove_teleport_count": mousemove_teleport_count,
            "mouse_activity_rate": mouse_activity_rate,
            "mouse_teleport_rate": mouse_teleport_rate,
            "viewport_width": viewport_width,
            "viewport_height": viewport_height,
            "source_file": random.choice(source_files) if source_files else "",
        })

    return out


def make_be_rows(base_rows: list[dict[str, str]], n: int = 100) -> list[dict]:
    out = []

    show_schedule_ids = sorted({row["showScheduleId"] for row in base_rows})
    req_source_files = sorted({row["req_source_file"] for row in base_rows})
    evt_source_files = sorted({row["evt_source_file"] for row in base_rows})

    for i in range(1, n + 1):
        base = random.choice(base_rows)

        label = random.randint(0, 1)
        user_id = "a" if label == 0 else "b"

        request_count = clip_int(
            to_int(base["request_count"]) * random.uniform(0.8, 1.25),
            low=1
        )

        endpoint_burst_max_1s = clip_int(
            to_int(base["endpoint_burst_max_1s"]) + random.uniform(-1, 2),
            low=1,
            high=request_count
        )

        req_interval_cv = round(
            clip_float(
                to_float(base["req_interval_cv"]) * random.uniform(0.75, 1.3),
                low=0.0
            ),
            6
        )

        target_retry_count = clip_int(
            to_int(base["target_retry_count"]) + random.uniform(-1, 2),
            low=1
        )

        payment_ready_to_terminal_ms = clip_int(
            to_int(base["payment_ready_to_terminal_ms"]) * random.uniform(0.8, 1.25),
            low=100
        )

        out.append({
            "session_id": f"os_be_{i:03d}",
            "user_id": user_id,
            "label": label,
            "showScheduleId": random.choice(show_schedule_ids) if show_schedule_ids else "",
            "orderId": f"ORD-OS-{i:04d}",
            "request_count": request_count,
            "endpoint_burst_max_1s": endpoint_burst_max_1s,
            "req_interval_cv": req_interval_cv,
            "target_retry_count": target_retry_count,
            "payment_ready_to_terminal_ms": payment_ready_to_terminal_ms,
            "req_source_file": random.choice(req_source_files) if req_source_files else "",
            "evt_source_file": random.choice(evt_source_files) if evt_source_files else "",
        })

    return out


def main() -> None:
    fe_base_rows = read_csv(FE_INPUT)
    be_base_rows = read_csv(BE_INPUT)

    fe_rows = make_fe_rows(fe_base_rows, n=100)
    be_rows = make_be_rows(be_base_rows, n=100)

    fe_fieldnames = [
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

    be_fieldnames = [
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

    write_csv(FE_OUTPUT, fe_rows, fe_fieldnames)
    write_csv(BE_OUTPUT, be_rows, be_fieldnames)

    print(f"[DONE] created: {FE_OUTPUT}")
    print(f"[DONE] created: {BE_OUTPUT}")


if __name__ == "__main__":
    main()