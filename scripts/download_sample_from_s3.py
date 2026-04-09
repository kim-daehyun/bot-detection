from pathlib import Path
import json
import re
import boto3
from botocore.exceptions import ClientError


BUCKET_NAME = "truve-dev-raw-data"

# 사용자가 말한 값 기준으로 정의하되,
# 실제 S3 paginate에는 앞 슬래시 제거한 prefix를 사용
BE_S3_PREFIX = "/BE"
FE_S3_PREFIX = "/FE"

BE_DOMAIN_EVENT_DIR = Path("/Users/daehyun/Desktop/실무통합/bot_detection_project/data/BE/BE_domain_event_log/rawdata")
BE_SERVER_REQUEST_DIR = Path("/Users/daehyun/Desktop/실무통합/bot_detection_project/data/BE/BE_server_request_log/rawdata")
FE_RAW_DIR = Path("/Users/daehyun/Desktop/실무통합/bot_detection_project/data/FE/rawdata")

DOWNLOAD_TMP_ROOT_DIR = Path("./downloads/raw_logs")
DOWNLOAD_TMP_BE_DIR = DOWNLOAD_TMP_ROOT_DIR / "BE"
DOWNLOAD_TMP_FE_DIR = DOWNLOAD_TMP_ROOT_DIR / "FE"

LOGIN_PATH = "/api/auth/login"
PAYMENTS_CONFIRM_PATH = "/api/payments/confirm"
PAYMENTS_READY_PATTERN = re.compile(r"^/api/bookings/[^/]+/payment-ready$")
FE_TELEMETRY_PATH = "/telemetry"

BOT_USER_ID = "49e7bc75-7bcb-44d9-ab8e-8d71a67df937"      # 제 계정
HUMAN_USER_ID = "64683eca-4333-477f-ba2d-c63566162d5a"    # test 계정


def normalize_s3_prefix(prefix: str) -> str:
    """
    '/BE' -> 'BE/'
    '/FE' -> 'FE/'
    """
    prefix = prefix.strip().lstrip("/")
    if not prefix:
        raise ValueError("S3 prefix must not be empty")
    if not prefix.endswith("/"):
        prefix += "/"
    return prefix


def get_actor_prefix(user_id: str | None) -> str:
    if user_id == BOT_USER_ID:
        return "[bot]"
    if user_id == HUMAN_USER_ID:
        return "[human]"
    return "[unknown]"


def add_raw_prefix(filename: str) -> str:
    return f"[raw]{filename}"


def should_download(local_path: Path, s3_size: int) -> bool:
    if not local_path.exists():
        return True
    return local_path.stat().st_size != s3_size


def ensure_output_dirs() -> None:
    for target_dir in [
        BE_SERVER_REQUEST_DIR,
        BE_DOMAIN_EVENT_DIR,
        FE_RAW_DIR,
        DOWNLOAD_TMP_BE_DIR,
        DOWNLOAD_TMP_FE_DIR,
    ]:
        target_dir.mkdir(parents=True, exist_ok=True)


def download_updated_logs_from_prefix(
    s3_client,
    bucket_name: str,
    prefix: str,
    download_dir: Path,
) -> list[Path]:
    """
    - 새 파일 또는 크기 변경 파일만 로컬 tmp에 저장
    - 이번 실행에서 실제 다운로드/업데이트된 파일만 반환
    """
    download_dir.mkdir(parents=True, exist_ok=True)

    normalized_prefix = normalize_s3_prefix(prefix)
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=normalized_prefix)

    updated_files: list[Path] = []
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
                updated_files.append(local_path)
            else:
                print(f"[SKIP] already exists and same size: {local_path}")

    if not found_any:
        print(f"[INFO] No .log objects found under prefix: {normalized_prefix}")

    return updated_files


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


def infer_user_id_from_records(records: list[dict]) -> str | None:
    for record in records:
        user_id = record.get("userId")
        if user_id is not None:
            return user_id
    return None


def has_numeric_tsserver(record: dict) -> bool:
    return isinstance(record.get("tsServer"), (int, float))


def validate_complete_server_session(session_records: list[dict]) -> tuple[bool, str]:
    """
    server_request_log 검증
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
      이전 세션은 미완성으로 버리고, 새 login부터 다시 시작
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
        print(f"[DISCARD] trailing incomplete session (started with login but no {PAYMENTS_CONFIRM_PATH})")

    return valid_sessions


def extract_domain_records_from_server_session(session_records: list[dict]) -> list[dict]:
    return [
        r for r in session_records
        if is_payments_confirm(r) or is_payments_ready(r)
    ]


def write_jsonl(records: list[dict], output_dir: Path, filename: str) -> None:
    if not records:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    if output_path.exists():
        print(f"[SKIP] already exists: {output_path}")
        return

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[WRITE] {output_path} ({len(records)} rows)")


def write_single_line_jsonl(record: dict, output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    if output_path.exists():
        print(f"[SKIP] already exists: {output_path}")
        return

    with output_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[WRITE] {output_path} (1 row)")


def extract_max_index_from_dir(target_dir: Path, pattern: re.Pattern) -> int:
    if not target_dir.exists():
        return 0

    max_idx = 0
    for file_path in target_dir.glob("*.jsonl"):
        match = pattern.search(file_path.name)
        if not match:
            continue
        idx = int(match.group(1))
        if idx > max_idx:
            max_idx = idx
    return max_idx


BE_SESSION_FILENAME_PATTERN = re.compile(r"session_(\d+)\.jsonl$")
FE_TELEMETRY_FILENAME_PATTERN = re.compile(r"telemetry_(\d+)\.jsonl$")


def next_be_session_index() -> int:
    max_server_idx = extract_max_index_from_dir(BE_SERVER_REQUEST_DIR, BE_SESSION_FILENAME_PATTERN)
    max_domain_idx = extract_max_index_from_dir(BE_DOMAIN_EVENT_DIR, BE_SESSION_FILENAME_PATTERN)
    return max(max_server_idx, max_domain_idx) + 1


def next_fe_record_index() -> int:
    max_idx = extract_max_index_from_dir(FE_RAW_DIR, FE_TELEMETRY_FILENAME_PATTERN)
    return max_idx + 1


def make_be_output_filename(user_id: str | None, session_index: int) -> str:
    actor_prefix = get_actor_prefix(user_id)
    return f"{actor_prefix}{add_raw_prefix(f'session_{session_index:03d}.jsonl')}"


def make_fe_output_filename(user_id: str | None, record_index: int) -> str:
    actor_prefix = get_actor_prefix(user_id)
    return f"{actor_prefix}{add_raw_prefix(f'telemetry_{record_index:03d}.jsonl')}"


def write_clean_be_sessions(valid_sessions: list[list[dict]]) -> None:
    """
    - 검증 통과한 BE server request rawdata 저장
    - 같은 세션 기준으로 domain_event_log 저장
    - 기존 파일은 유지, 새 index부터 추가 저장
    """
    session_index = next_be_session_index()

    for session_records in valid_sessions:
        user_id = infer_user_id_from_records(session_records)
        filename = make_be_output_filename(user_id, session_index)

        write_jsonl(session_records, BE_SERVER_REQUEST_DIR, filename)

        be_domain_records = extract_domain_records_from_server_session(session_records)
        write_jsonl(be_domain_records, BE_DOMAIN_EVENT_DIR, filename)

        session_index += 1


def validate_fe_telemetry_record(record: dict) -> tuple[bool, str]:
    """
    FE telemetry 정상 스키마 검증
    기준:
    - path == /telemetry
    - tsServer 숫자형
    - userId 존재
    - requestBody dict
    - page_stage 존재
    - page_enter_ts / page_leave_ts 숫자형
    - page_leave_ts >= page_enter_ts
    - mousemove list
    - mousemove_count 숫자형
    - viewport_width / viewport_height 숫자형 및 > 0
    """
    if not is_fe_telemetry(record):
        return False, "path is not /telemetry"

    if not has_numeric_tsserver(record):
        return False, "invalid tsServer"

    user_id = record.get("userId")
    if user_id is None:
        return False, "missing userId"

    request_body = record.get("requestBody")
    if not isinstance(request_body, dict):
        return False, "requestBody is missing or not dict"

    page_stage = request_body.get("page_stage")
    if not isinstance(page_stage, str) or not page_stage.strip():
        return False, "missing page_stage"

    page_enter_ts = request_body.get("page_enter_ts")
    page_leave_ts = request_body.get("page_leave_ts")
    if not isinstance(page_enter_ts, (int, float)):
        return False, "invalid page_enter_ts"
    if not isinstance(page_leave_ts, (int, float)):
        return False, "invalid page_leave_ts"
    if page_leave_ts < page_enter_ts:
        return False, "page_leave_ts < page_enter_ts"

    mousemove = request_body.get("mousemove")
    if not isinstance(mousemove, list):
        return False, "mousemove is not list"

    mousemove_count = request_body.get("mousemove_count")
    if not isinstance(mousemove_count, (int, float)):
        return False, "invalid mousemove_count"

    viewport_width = request_body.get("viewport_width")
    viewport_height = request_body.get("viewport_height")
    if not isinstance(viewport_width, (int, float)) or viewport_width <= 0:
        return False, "invalid viewport_width"
    if not isinstance(viewport_height, (int, float)) or viewport_height <= 0:
        return False, "invalid viewport_height"

    return True, "ok"


def filter_valid_fe_telemetry_records(all_records: list[dict]) -> list[dict]:
    valid_records: list[dict] = []

    for idx, record in enumerate(all_records, start=1):
        is_valid, reason = validate_fe_telemetry_record(record)
        if is_valid:
            valid_records.append(record)
        else:
            print(f"[DISCARD] invalid FE telemetry record #{idx}: {reason}")

    return valid_records


def write_fe_records_one_line_per_file(fe_records: list[dict]) -> None:
    """
    FE는:
    - FE prefix에서 받은 파일들을 전부 merge
    - 정상 스키마 telemetry record만 필터링
    - record 1개당 jsonl 1파일
    - 파일 내부는 1줄만 저장
    """
    record_index = next_fe_record_index()

    for record in fe_records:
        user_id = record.get("userId")
        filename = make_fe_output_filename(user_id, record_index)
        write_single_line_jsonl(record, FE_RAW_DIR, filename)
        record_index += 1


def main():
    s3_client = boto3.client("s3")
    ensure_output_dirs()

    try:
        s3_client.head_bucket(Bucket=BUCKET_NAME)
        print(f"[OK] Connected to bucket: {BUCKET_NAME}")
    except ClientError as e:
        print(f"[ERROR] Cannot access bucket '{BUCKET_NAME}': {e}")
        return

    print("\n=== Step 1. Download updated BE logs ===")
    be_updated_files = download_updated_logs_from_prefix(
        s3_client=s3_client,
        bucket_name=BUCKET_NAME,
        prefix=BE_S3_PREFIX,
        download_dir=DOWNLOAD_TMP_BE_DIR,
    )

    print("\n=== Step 2. Download updated FE logs ===")
    fe_updated_files = download_updated_logs_from_prefix(
        s3_client=s3_client,
        bucket_name=BUCKET_NAME,
        prefix=FE_S3_PREFIX,
        download_dir=DOWNLOAD_TMP_FE_DIR,
    )

    if not be_updated_files and not fe_updated_files:
        print("[INFO] No updated BE/FE log files to process.")
        return

    if be_updated_files:
        print("\n=== Step 3. Merge updated BE logs ===")
        be_all_records = load_and_merge_all_records(be_updated_files)
        print(f"[INFO] Total merged BE records: {len(be_all_records)}")

        print("\n=== Step 4. Validate and split BE server sessions ===")
        valid_sessions = split_valid_server_sessions(be_all_records)
        print(f"[INFO] Valid complete BE sessions: {len(valid_sessions)}")

        print("\n=== Step 5. Write BE rawdata ===")
        write_clean_be_sessions(valid_sessions)
    else:
        print("\n[INFO] No updated BE files to process.")

    if fe_updated_files:
        print("\n=== Step 6. Merge updated FE logs ===")
        fe_all_records = load_and_merge_all_records(fe_updated_files)
        print(f"[INFO] Total merged FE records: {len(fe_all_records)}")

        print("\n=== Step 7. Validate FE telemetry schema ===")
        fe_valid_records = filter_valid_fe_telemetry_records(fe_all_records)
        print(f"[INFO] Valid FE telemetry records: {len(fe_valid_records)}")

        print("\n=== Step 8. Write FE rawdata (1 valid record = 1 jsonl = 1 line) ===")
        write_fe_records_one_line_per_file(fe_valid_records)
    else:
        print("\n[INFO] No updated FE files to process.")

    print("\n[DONE] Incremental BE/FE rawdata split completed.")


if __name__ == "__main__":
    main()