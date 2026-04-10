# Bot Detection Project

티켓 예매 과정에서 사람과 봇을 구분하기 위해, 프론트엔드(FE) 행동 로그와 백엔드(BE) 요청 로그를 각각 전처리하고 분류 모델로 학습하는 프로젝트입니다. 발표 자료는 이 문서를 기준으로 구성할 수 있도록, 현재 저장소 상태와 실제 코드 기준으로 파이프라인과 전처리 식을 정리했습니다.

## 1. 프로젝트 한눈에 보기

- 목표: 티켓 예매 시나리오에서 `human(0)` 과 `bot(1)` 을 구분하는 이진 분류 모델 구축
- 입력 데이터:
  - FE: 브라우저 telemetry 로그
  - BE: 서버 request 로그 + domain event 로그
- 출력 데이터:
  - FE feature CSV, BE feature CSV
  - train / valid / test 분할 CSV
  - FE / BE 분류 모델 파일 (`.pkl`)

전체 흐름은 아래와 같습니다.

```text
Raw JSONL logs
-> FE / BE feature engineering
-> train / valid / test split
-> preprocessing (median imputation + standard scaling)
-> classifier training
-> saved model (.pkl)
```

## 2. 현재 저장소 기준 데이터 현황

### FE 데이터

| 구분 | 행 수 | human(0) | bot(1) |
|---|---:|---:|---:|
| `data/FE/feature/fe_preprocess.csv` | 201 | 92 | 109 |
| `data/FE/dataset/trainset/fe_trainset.csv` | 160 | 73 | 87 |
| `data/FE/dataset/validset/fe_validset.csv` | 20 | 9 | 11 |
| `data/FE/dataset/testset/fe_testset.csv` | 21 | 10 | 11 |

### BE 데이터

| 구분 | 행 수 | human(0) | bot(1) |
|---|---:|---:|---:|
| `data/BE/feature/be_preprocess.csv` | 191 | 84 | 107 |
| `data/BE/dataset/trainset/be_trainset.csv` | 152 | 67 | 85 |
| `data/BE/dataset/validset/be_validset.csv` | 19 | 8 | 11 |
| `data/BE/dataset/testset/be_testset.csv` | 20 | 9 | 11 |

데이터 분할은 `scripts/build_dataset.py` 에서 stratified 8:1:1 비율로 수행합니다.

## 3. 저장소 구조

```text
bot_detection_project/
├── data/
│   ├── FE/
│   │   ├── rawdata/
│   │   ├── feature/fe_preprocess.csv
│   │   └── dataset/{trainset,validset,testset}/
│   └── BE/
│       ├── BE_server_request_log/rawdata/
│       ├── BE_domain_event_log/rawdata/
│       ├── feature/be_preprocess.csv
│       └── dataset/{trainset,validset,testset}/
├── scripts/
│   ├── preprocess_fe.py
│   ├── preprocess_be.py
│   ├── build_dataset.py
│   ├── model_fe.py
│   └── model_be.py
├── model/
│   ├── FE/
│   └── BE/
└── README.md
```

## 4. FE 모델 정리

### 4.1 FE raw log 입력 필드

FE 전처리는 `data/FE/rawdata/*.jsonl` 의 각 JSON line에서 아래 필드를 읽습니다.

- `requestBody.page_enter_ts`
- `requestBody.page_leave_ts`
- `requestBody.mousemove`
- `requestBody.mousemove_count`
- `requestBody.viewport_width`
- `requestBody.viewport_height`
- 파일명 prefix: `[bot]`, `[human]`

라벨은 파일명 기준으로 부여됩니다.

- `[bot]...` -> `label = 1`
- `[human]...` -> `label = 0`

### 4.2 FE 전처리 결과 컬럼

`data/FE/feature/fe_preprocess.csv` 에 저장되는 컬럼은 아래 5개입니다.

| 컬럼명 | 의미 | 모델 사용 여부 |
|---|---|---|
| `duration_ms` | 페이지 체류 시간 | 사용 |
| `mousemove_teleport_count` | 비정상적으로 큰 마우스 점프 횟수 | 사용 |
| `mousemove_count` | 마우스 이동 이벤트 수 | 사용 |
| `source_file` | 원본 파일명 | 미사용 |
| `label` | 사람/봇 라벨 | 타깃 |

### 4.3 FE 전처리 식

#### 1) 체류 시간

```text
duration_ms = max(0, page_leave_ts - page_enter_ts)
```

#### 2) 마우스 teleport 판정

연속된 두 mousemove 이벤트 `(i-1, i)` 에 대해 아래 값을 계산합니다.

```text
dt_i       = timestamp_i - timestamp_(i-1)
dx_i       = x_i - x_(i-1)
dy_i       = y_i - y_(i-1)
norm_dx_i  = dx_i / viewport_width
norm_dy_i  = dy_i / viewport_height
norm_dist_i = sqrt(norm_dx_i^2 + norm_dy_i^2)
norm_speed_i = norm_dist_i / dt_i
```

현재 코드 기준 teleport 조건은 다음과 같습니다.

```text
teleport_i =
    (dt_i < 20 and norm_dist_i > 0.003)
    or
    (norm_speed_i > 0.002)
```

최종 teleport 횟수는 모든 연속 구간에 대한 합입니다.

```text
mousemove_teleport_count = sum(teleport_i)
```

#### 3) 마우스 이동 수

```text
mousemove_count = requestBody.mousemove_count
```

### 4.4 FE 모델이 실제 사용하는 입력 변수

`scripts/model_fe.py` 에서 실제 학습에 사용하는 변수는 아래 3개입니다.

- `duration_ms`
- `mousemove_teleport_count`
- `mousemove_count`

사용하지 않는 컬럼은 아래와 같습니다.

- `source_file`

즉, FE 모델 입력 벡터는 아래처럼 정리할 수 있습니다.

```text
X_FE = [
  duration_ms,
  mousemove_teleport_count,
  mousemove_count
]
```

## 5. BE 모델 정리

### 5.1 BE raw log 입력 필드

BE 전처리는 두 종류의 로그를 함께 사용합니다.

- Request log: `data/BE/BE_server_request_log/rawdata/*.jsonl`
- Event log: `data/BE/BE_domain_event_log/rawdata/*.jsonl`

주요 입력 필드는 아래와 같습니다.

- `tsServer`
- `path`
- 파일명 prefix: `[bot]`, `[human]`

라벨은 request 파일명 기준으로 부여됩니다.

- `[bot]...` -> `label = 1`
- `[human]...` -> `label = 0`

### 5.2 req / evt 파일 매칭 방식

`scripts/preprocess_be.py` 는 request 로그와 event 로그를 파일명으로 매칭합니다.

예를 들어 아래 접두어를 제거한 뒤 같은 key로 대응합니다.

- `[bot]`, `[human]`, `[unknown]`
- `[raw]`
- `.jsonl`
- `be_req_`
- `be_evt_`

즉, `req` 와 `evt` 파일이 동일 세션이라고 판단되면 하나의 학습 row로 합쳐집니다.

### 5.3 BE 전처리 결과 컬럼

`data/BE/feature/be_preprocess.csv` 에 저장되는 컬럼은 아래 7개입니다.

| 컬럼명 | 의미 | 모델 사용 여부 |
|---|---|---|
| `ts_payment_ready` | event 로그의 시작~끝 시간 차 | 사용 |
| `ts_whole_session` | 로그인부터 결제 확인까지의 세션 시간 | 사용 |
| `req_interval_cv_pre_hold` | 첫 hold 이전 요청 간격의 CV | 사용 |
| `req_interval_cv_hold_gap` | hold 전후 CV 차이의 절대값 | 사용 |
| `req_source_file` | request 원본 파일명 | 미사용 |
| `evt_source_file` | event 원본 파일명 | 미사용 |
| `label` | 사람/봇 라벨 | 타깃 |

### 5.4 BE 전처리 식

#### 1) hold 기준 구간 분리

첫 번째 아래 패턴의 API를 경계로 요청 로그를 분리합니다.

```text
/api/ticketing/{id}/hold/seat
```

정규식으로는 아래와 같습니다.

```text
^/api/ticketing/[^/]+/hold/seat$
```

분리 방식은 다음과 같습니다.

- `pre_hold`: 첫 hold 요청 직전까지
- `post_hold`: 첫 hold 요청 포함 이후

#### 2) 요청 간격의 CV 계산

timestamp 목록을 오름차순 정렬한 뒤, 양수인 인접 간격만 사용합니다.

```text
dt_i = ts_i - ts_(i-1),  where dt_i > 0
mean_dt = average(dt_i)
std_dt = sqrt( average((dt_i - mean_dt)^2) )
cv = std_dt / mean_dt
```

따라서,

```text
req_interval_cv_pre_hold = cv(pre_hold timestamps)
req_interval_cv_post_hold = cv(post_hold timestamps)
req_interval_cv_hold_gap = abs(req_interval_cv_post_hold - req_interval_cv_pre_hold)
```

#### 3) 결제 준비 구간 길이

event 로그에서 `tsServer` 의 최솟값과 최댓값 차이를 사용합니다.

```text
ts_payment_ready = max(evt.tsServer) - min(evt.tsServer)
```

단, 유효 timestamp가 2개 미만이면 빈 값으로 남깁니다.

#### 4) 전체 세션 길이

request 로그에서 아래 두 지점을 사용합니다.

- 첫 번째 `/api/auth/login`
- 마지막 `/api/payments/confirm`

수식은 다음과 같습니다.

```text
ts_whole_session = ts(last /api/payments/confirm) - ts(first /api/auth/login)
```

둘 중 하나라도 없으면 빈 값으로 남깁니다.

### 5.5 BE 모델이 실제 사용하는 입력 변수

`scripts/model_be.py` 에서 실제 학습에 사용하는 변수는 아래 4개입니다.

- `ts_payment_ready`
- `ts_whole_session`
- `req_interval_cv_pre_hold`
- `req_interval_cv_hold_gap`

사용하지 않는 컬럼은 아래와 같습니다.

- `req_source_file`
- `evt_source_file`

즉, BE 모델 입력 벡터는 아래처럼 정리할 수 있습니다.

```text
X_BE = [
  ts_payment_ready,
  ts_whole_session,
  req_interval_cv_pre_hold,
  req_interval_cv_hold_gap
]
```

## 6. 학습 파이프라인

FE와 BE 모두 모델 학습 전 숫자형 컬럼에 동일한 전처리를 적용합니다.

```text
1) SimpleImputer(strategy="median")
2) StandardScaler()
3) Classifier fit
```

현재 코드에서 후보 모델은 아래와 같습니다.

- Random Forest
- Decision Tree
- XGBoost
- CatBoost

`xgboost`, `catboost` 는 라이브러리가 로드될 때만 학습 후보에 포함됩니다.

## 7. 실행 순서

아래 순서대로 실행하면 raw 로그부터 모델 학습까지 재현할 수 있습니다.

```bash
python3 scripts/preprocess_fe.py
python3 scripts/preprocess_be.py
python3 scripts/build_dataset.py
python3 scripts/model_fe.py
python3 scripts/model_be.py
```

## 8. 발표용 핵심 포인트

- FE 모델은 사용자의 마우스 움직임과 페이지 체류 패턴을 숫자로 요약해 분류합니다.
- BE 모델은 요청 간격의 규칙성, hold 전후 패턴 변화, 결제 구간 시간을 이용해 분류합니다.
- FE는 행동 기반 탐지, BE는 서버 요청 기반 탐지라는 점에서 상호보완적입니다.
- 두 모델 모두 원본 로그를 바로 쓰지 않고, 해석 가능한 feature로 요약한 뒤 분류합니다.

## 9. GitHub 업로드 명령

현재 저장소 기준 remote는 `origin`, 브랜치는 `main` 입니다.

```bash
git status
git add README.md scripts/preprocess_fe.py scripts/preprocess_be.py
git commit -m "docs: add 발표용 README and align preprocessing comments"
git push origin main
```

이미 모든 변경 파일을 한 번에 올리고 싶다면 아래처럼 해도 됩니다.

```bash
git add .
git commit -m "docs: update project README"
git push origin main
```
