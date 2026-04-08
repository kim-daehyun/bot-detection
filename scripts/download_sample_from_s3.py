from pathlib import Path
import json
import re
import shutil
import boto3
from botocore.exceptions import ClientError


BUCKET_NAME = "truve-dev-raw-data"
S3_PREFIX = "BE/"

BE_DOMAIN_EVENT_DIR = Path("/Users/daehyun/Desktop/실무통합/bot_detection_project/data/BE/BE_domain_event_log/rawdata")
BE_SERVER_REQUEST_DIR = Path("/Users/daehyun/Desktop/실무통합/bot_detection_project/data/BE/BE_server_request_log/rawdata")
FE_RAW_DIR = Path("/Users/daehyun/Desktop/실무통합/bot_detection_project/data/FE/rawdata")

DOWNLOAD_TMP_DIR = Path("./downloads/raw_logs")

LOGIN_PATH = "/api/auth/login"
PAYMENTS_CONFIRM_PATH = "/api/payments/confirm"
FE_TELEMETRY_PATH = "/telemetry"
PAYMENTS_READY_PATTERN = re.compile(r"^/api/bookings/[^/]+/payment-ready$")


def get_actor_prefix(user_id) -> str:
    if user_id == "49e7bc75-7bcb-44d9-ab8e-8d71a67df937": #제 계정 
        return "[bot]"
    if user_id == "64683eca-4333-477f-ba2d-c63566162d5a": #test계정
        return "[human]"
    return "[unknown]"


def add_raw_prefix(filename: str) -> str:
    return f"[raw]{filename}"


def should_download(local_path: Path, s3_size: int) -> bool:
    if not local_path.exists():
        return True
    return local_path.stat().st_size != s3_size


def reset_output_dirs() -> None:
    """
    매 실행 시 기존 rawdata 산출물을 비우고 다시 생성
    """
    for target_dir in [BE_SERVER_REQUEST_DIR, BE_DOMAIN_EVENT_DIR, FE_RAW_DIR]:
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)


def download_all_logs_from_prefix(
    s3_client,
    bucket_name: str,
    prefix: str,
    download_dir: Path,
) -> list[Path]:
    download_dir.mkdir(parents=True, exist_ok=True)

    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    downloaded_files: list[Path] = []
    found_any = False

    for page in pages:
        contents = page.get("Contents", [])
        for obj in contents:
            key = obj["Key"]
            size = obj["Size"]

            if key.endswith("/"):
                continue
            if not key.lower().endswith(".log"):
                continue

            found_any = True
            local_path = download_dir / Path(key).name

            if should_download(local_path, size):
                print(f"[DOWNLOAD] s3://{bucket_name}/{key} -> {local_path}")
                s3_client.download_file(bucket_name, key, str(local_path))
            else:
                print(f"[SKIP] already exists and same size: {local_path}")

            downloaded_files.append(local_path)

    if not found_any:
        print(f"[INFO] No .log objects found under prefix: {prefix}")

    return downloaded_files


def safe_json_loads(line: str):
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def load_and_merge_all_records(log_files: list[Path]) -> list[dict]:
    all_records: list[dict] = []

    for log_file in log_files:
        with log_file.open("r", encoding="utf-8") as f:
            for raw_line in f:
                record = safe_json_loads(raw_line)
                if record is None:
                    continue
                if isinstance(record, dict):
                    all_records.append(record)

    all_records.sort(key=lambda x: x.get("tsServer", 0))
    return all_records


def is_login(record: dict) -> bool:
    return record.get("path") == LOGIN_PATH


def is_payments_confirm(record: dict) -> bool:
    return record.get("path") == PAYMENTS_CONFIRM_PATH


def is_payments_ready(record: dict) -> bool:
    path = record.get("path", "")
    return bool(PAYMENTS_READY_PATTERN.match(path))


def is_fe_telemetry(record: dict) -> bool:
    return record.get("path") == FE_TELEMETRY_PATH


def infer_session_user_id(records: list[dict]) -> str | None:
    for record in records:
        user_id = record.get("userId")
        if user_id is not None:
            return user_id
    return None


def make_output_filename(user_id: str | None, session_index: int) -> str:
    actor_prefix = get_actor_prefix(user_id)
    return f"{actor_prefix}{add_raw_prefix(f'session_{session_index:03d}.jsonl')}"


def write_jsonl(records: list[dict], output_dir: Path, filename: str) -> None:
    if not records:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[WRITE] {output_path} ({len(records)} rows)")


def has_numeric_tsserver(record: dict) -> bool:
    return isinstance(record.get("tsServer"), (int, float))


def validate_complete_server_session(session_records: list[dict]) -> tuple[bool, str]:
    """
    server_request_log 스키마 검증
    - 첫 행은 /api/auth/login
    - 마지막 행은 /api/payments/confirm
    - login 1회
    - confirm 1회
    - 모든 행에 숫자형 tsServer 존재
    """
    if not session_records:
        return False, "empty session"

    if not is_login(session_records[0]):
        return False, "first row is not /api/auth/login"

    if not is_payments_confirm(session_records[-1]):
        return False, "last row is not /api/payments/confirm"

    login_count = sum(1 for r in session_records if is_login(r))
    confirm_count = sum(1 for r in session_records if is_payments_confirm(r))

    if login_count != 1:
        return False, f"login count != 1 ({login_count})"

    if confirm_count != 1:
        return False, f"confirm count != 1 ({confirm_count})"

    for idx, record in enumerate(session_records, start=1):
        if not has_numeric_tsserver(record):
            return False, f"invalid tsServer at row {idx}"

    return True, "ok"


def split_valid_server_sessions(all_records: list[dict]) -> list[list[dict]]:
    """
    핵심 규칙
    - login 만나면 세션 시작
    - 세션 도중 login이 다시 나오면:
      이전 세션은 미완성으로 버리고,
      새 login부터 새 세션 시작
    - confirm 나오면 세션 종료 후 검증
    """
    valid_sessions: list[list[dict]] = []

    in_session = False
    current_session: list[dict] = []

    for record in all_records:
        if not in_session:
            if is_login(record):
                in_session = True
                current_session = [record]
            continue

        # 세션 진행 중인데 login이 또 나오면:
        # 이전 current_session은 미완성으로 폐기하고, 새 login부터 다시 시작
        if is_login(record):
            print("[DISCARD] incomplete session dropped before new login")
            current_session = [record]
            in_session = True
            continue

        current_session.append(record)

        if is_payments_confirm(record):
            is_valid, reason = validate_complete_server_session(current_session)
            if is_valid:
                valid_sessions.append(current_session[:])
                print(f"[VALID] complete session accepted ({len(current_session)} rows)")
            else:
                print(f"[DISCARD] invalid session: {reason}")

            current_session = []
            in_session = False

    if current_session:
        print(
            f"[DISCARD] trailing incomplete session "
            f"(started with login but no {PAYMENTS_CONFIRM_PATH})"
        )

    return valid_sessions


def extract_domain_records_from_server_session(session_records: list[dict]) -> list[dict]:
    return [
        r for r in session_records
        if is_payments_confirm(r) or is_payments_ready(r)
    ]


def extract_fe_records_from_server_session(session_records: list[dict]) -> list[dict]:
    return [
        r for r in session_records
        if is_fe_telemetry(r)
    ]


def write_clean_sessions(valid_sessions: list[list[dict]]) -> None:
    """
    반드시 검증 통과한 BE_server_request_log를 먼저 저장하고,
    그 same session 기준으로 evt / fe를 파생 저장
    """
    for session_index, session_records in enumerate(valid_sessions, start=1):
        user_id = infer_session_user_id(session_records)
        filename = make_output_filename(user_id, session_index)

        # 1) 정제 완료된 BE server request rawdata 저장
        write_jsonl(session_records, BE_SERVER_REQUEST_DIR, filename)

        # 2) 위 정상 req 세션 기준으로 downstream 파생
        be_domain_records = extract_domain_records_from_server_session(session_records)
        fe_records = extract_fe_records_from_server_session(session_records)

        write_jsonl(be_domain_records, BE_DOMAIN_EVENT_DIR, filename)
        write_jsonl(fe_records, FE_RAW_DIR, filename)


def main():
    s3_client = boto3.client("s3")

    try:
        s3_client.head_bucket(Bucket=BUCKET_NAME)
        print(f"[OK] Connected to bucket: {BUCKET_NAME}")
    except ClientError as e:
        print(f"[ERROR] Cannot access bucket '{BUCKET_NAME}': {e}")
        return

    downloaded_files = download_all_logs_from_prefix(
        s3_client=s3_client,
        bucket_name=BUCKET_NAME,
        prefix=S3_PREFIX,
        download_dir=DOWNLOAD_TMP_DIR,
    )

    if not downloaded_files:
        print("[INFO] No downloaded log files to process.")
        return

    print("\n=== Merging all .log files ===")
    all_records = load_and_merge_all_records(downloaded_files)
    print(f"[INFO] Total merged records: {len(all_records)}")

    print("\n=== Reset output dirs ===")
    reset_output_dirs()

    print("\n=== Step 1. Validate and split BE_server_request_log sessions ===")
    valid_sessions = split_valid_server_sessions(all_records)
    print(f"[INFO] Valid complete sessions: {len(valid_sessions)}")

    print("\n=== Step 2. Write clean BE_server_request_log and derive downstream rawdata ===")
    write_clean_sessions(valid_sessions)

    print("\n[DONE] Clean split completed.")


if __name__ == "__main__":
    main()