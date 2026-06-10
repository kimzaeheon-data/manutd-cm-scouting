# ⚽ AI 기반 축구 스카우팅 분석 시스템
### Manchester United CM 앵커 영입 후보 평가 프로젝트

> **"영상에서 뽑은 공간 데이터 + 이벤트 데이터를 결합해, 전술 요구에 부합하는 선수를 정량 평가한다"**

---

## 🎯 프로젝트 개요

단순 통계 나열이 아닌, **전술 요구 → 측정 가능한 지표 → 후보 평가**로 이어지는
실무형 스카우팅 의사결정 파이프라인을 구현했습니다.

| 항목 | 내용 |
|---|---|
| 대상 구단 | Manchester United |
| 감독 | Michael Carrick (2026년 5월 정식 선임) |
| 영입 포지션 | 중앙 미드필더 — 카세미루 이적 후 앵커 공백 |
| 분석 후보 | Mateus Fernandes, Sandro Tonali, Mason Mount(대조군) |
| 핵심 차별점 | **영상 위치 데이터(Computer Vision) + 이벤트 데이터 결합** |

---

## 🏗️ 시스템 구조

```
영상 입력
    ↓
[YOLO 선수 검출] → [ByteTrack 추적] → [팀 분류] → [호모그래피 좌표 변환]
    ↓
피치 좌표 CSV (frame, tracker_id, role, team, x, y)
    ↓
[위치 지표 계산]          [이벤트 데이터 (FBref)]
평균위치·커버범위·존점유율    인터셉트·태클·파울·출전시간
    ↓                          ↓
         [0~1 정규화 + 가중합]
                ↓
         후보별 적합도 점수 & 순위
```

---

## 📐 스카우팅 브리프 (설계 철학)

캐릭 체제 맨유의 전술 분석을 바탕으로,
**마이누·브루누 주전 고정 → 신입은 카세미루 역할(앵커)에 특화**라는 브리프를 도출했습니다.

### 평가 자질 및 가중치

| 순위 | 자질 | 가중치 | 측정 방식 |
|---|---|---|---|
| 1 | 수비 안정성 | **30%** | 인터셉트·태클(이벤트) + 수비 위치(CV) |
| 2 | 압박 저항·템포 | **25%** | 파울(역산)·패스 성공률(이벤트) |
| 3 | 피지컬·활동량 | **20%** | 총 이동거리·스프린트(CV) + 출전시간(이벤트) |
| 4 | 볼 운반 | **15%** | Progressive carries(이벤트) |
| 5 | 볼 전진 | **10%** | Progressive passes(이벤트) |

> 상세 설계 문서 → [`brief/scouting_brief_manutd_cm.md`](brief/scouting_brief_manutd_cm.md)

---

## 📊 분석 결과 (1차 모델)

> 1차 모델: 수비·압박저항·체력(가중치 합 75%) 기준. 운반·전진은 FBref 데이터 확보 후 보강 예정.

| 순위 | 선수 | 소속 | 적합도 점수 | 비고 |
|---|---|---|---|---|
| 🥇 | Mateus Fernandes | West Ham Utd | **0.800** | 육각형 완벽한 미드필더, 우선영입 대상 |
| 🥈 | Sandro Tonali | Newcastle Utd | **0.552** | 클린한 수비, 균형 잡힌 프로파일 |
| — | Mason Mount | Man Utd | 0.161 | 음성 대조군 — 앵커 역할 부적합 확인 |

**Mason Mount가 최하위로 나온 것은 모델의 변별력을 검증하는 음성 대조군 역할입니다.**
내부 자원(마운트)으로 앵커 공백을 메울 수 없음 → 외부 영입 필요의 근거.

### 레이더 차트 — 자질별 프로파일 비교
> 같은 적합도라도 수비 스타일이 다름:
> Fernandes = 적극적 인터셉트·태클형 / Tonali = 위치선정 기반 클린 수비형

---

## 🛠️ 기술 스택

| 영역 | 기술 |
|---|---|
| 선수 검출 | YOLOv8 (Roboflow sports) |
| 다중 객체 추적 | ByteTrack (supervision) |
| 팀 분류 | SigLIP + UMAP + KMeans |
| 피치 캘리브레이션 | 호모그래피 (ViewTransformer) |
| 이벤트 데이터 | soccerdata + WhoScored (수동 수집) |
| 분석·시각화 | pandas, numpy, matplotlib |
| 개발 환경 | Python 3.11, Apple Silicon MPS |

---

## 📁 파일 구조

```
manutd-cm-scouting/
├── README.md
├── brief/
│   └── scouting_brief_manutd_cm.md   # 스카우팅 설계 문서 v1.1
├── notebooks/
│   ├── 01_position_analysis.ipynb    # 위치 데이터 분석 (히트맵·산점도)
│   └── 02_scoring_model.ipynb        # 적합도 점수 모델 (레이더·바 차트)
│   └── 03_whoscored_scoring_model.ipynb        # whoscored 데이터 기반 완성판 점수 모델
├── scripts/
│   └── extract_coordinates.py        # CV 좌표 추출 스크립트
└── requirements.txt
```

---

## 🚀 실행 방법

```bash
# 1. 환경 설정
conda activate dl_study
pip install -r requirements.txt

# 2. 좌표 추출 (영상 → CSV)
python scripts/extract_coordinates.py \
  --source_video_path <영상경로> \
  --output_csv_path data/coordinates.csv \
  --device mps

# 3. 노트북 실행
jupyter lab
# notebooks/01_position_analysis.ipynb → 02_scoring_model.ipynb 순서로 실행
```

---

## ⚠️ 한계 및 향후 계획

**현재 한계**
1. CV 데이터는 분데스리가 샘플 기준 — 후보 선수 영상 확보 후 재측정 필요
2. WhoScored 데이터로 대체, 향후 FBref 전환 예정
3. 3명 기준 MinMax 정규화 — 리그 전체 분포 기준으로 보강 시 더 견고해짐

**향후 계획**
- [ ] 후보 선수 영상 확보 → CV 지표와 이벤트 지표 결합
- [ ] 타깃 B(마이누 백업) 버전 — 가중치 전환으로 두 번째 순위표 생성
- [ ] K리그 버전 파이프라인 일반화 데모

---

## 👤 About

비CS 전공(경영·마케팅·호텔경영) → AI/ML 전환 중.
축구부 출신(고등학교)의 스포츠 도메인 지식과 AI 기술을 결합해
**데이터와 현장 눈을 동시에 읽는 스카우트**를 목표로 합니다.

> 관련 문의: GitHub Issues 또는 PR 환영합니다.
