from pathlib import Path
import boto3
from botocore.exceptions import ClientError

# 추후 확인 필요
BUCKET_NAME = "sample-rawdata-bot"

# S3 prefix -> 로컬 저장 폴더 매핑
PREFIX_TO_LOCAL_DIR = {
    "client_telemetry_log_FE/": Path("./data/FE/rawdata"),
    "domain_event_log_BE/": Path("./data/BE/BE_domain_event_log/rawdata"),
    "server_request_log_BE/": Path("./data/BE/BE_server_request_log/rawdata"),
}


def add_raw_prefix(filename: str) -> str:
    return f"[raw]{filename}"


def should_download(local_path: Path, s3_size: int) -> bool:
    """
    로컬 파일이 없으면 다운로드.
    로컬 파일이 있더라도 크기가 다르면 다시 다운로드.
    로컬 파일이 있고 크기도 같으면 다운로드 생략.
    """
    if not local_path.exists():
        return True

    local_size = local_path.stat().st_size
    if local_size != s3_size:
        return True

    return False


def build_local_path(prefix: str, key: str) -> Path:
    """
    예:
      prefix = "client_telemetry_log_FE/"
      key    = "client_telemetry_log_FE/fe_a_001.json"

    저장 위치:
      ./data/FE/rawdata/[raw]fe_a_001.json
    """
    base_dir = PREFIX_TO_LOCAL_DIR[prefix]

    # key에서 prefix를 제거하고 실제 파일명만 추출
    filename = Path(key).name
    output_filename = add_raw_prefix(filename)

    return base_dir / output_filename


def download_prefix(s3_client, bucket_name: str, prefix: str) -> None:
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    found_any = False

    for page in pages:
        contents = page.get("Contents", [])
        for obj in contents:
            key = obj["Key"]
            size = obj["Size"]

            # S3 "폴더" placeholder는 건너뜀
            if key.endswith("/"):
                continue

            found_any = True

            local_path = build_local_path(prefix, key)
            local_path.parent.mkdir(parents=True, exist_ok=True)

            if should_download(local_path, size):
                print(f"[DOWNLOAD] s3://{bucket_name}/{key} -> {local_path}")
                s3_client.download_file(bucket_name, key, str(local_path))
            else:
                print(f"[SKIP] already exists and same size: {local_path}")

    if not found_any:
        print(f"[INFO] No objects found under prefix: {prefix}")


def main():
    s3_client = boto3.client("s3")

    try:
        # 버킷 접근 가능한지 간단 확인
        s3_client.head_bucket(Bucket=BUCKET_NAME)
        print(f"[OK] Connected to bucket: {BUCKET_NAME}")
    except ClientError as e:
        print(f"[ERROR] Cannot access bucket '{BUCKET_NAME}': {e}")
        return

    for prefix in PREFIX_TO_LOCAL_DIR.keys():
        print(f"\n=== Downloading prefix: {prefix} ===")
        download_prefix(s3_client, BUCKET_NAME, prefix)

    print("\n[DONE] Download completed.")


if __name__ == "__main__":
    main()