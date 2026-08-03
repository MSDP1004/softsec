# US Stock Value-Chain AI

미국주식 장기투자를 위한, **AI/기술 밸류체인 내 저평가 구간 추적 + 우량주 스크리닝** 시스템.

## 설계 배경

투자성향 3가지를 반영해 설계했다.

1. **장기투자 선호** → 단기 가격 예측(Transformer/LSTM 트레이딩 모델)이 아니라, 낮은 빈도(예: 분기)로
   재평가하는 밸류에이션 추적 구조를 사용한다.
2. **AI/기술 섹터 집중** → 반도체 장비부터 응용 소프트웨어까지 이어지는 밸류체인을 노드 단위로
   나누고, 밸류체인 내에서 노드 간 상대 밸류에이션을 비교한다.
3. **매출/이익이 실재하는 기업 선호** → 밸류에이션이 매력적인 노드를 찾은 뒤에도, 매출이 없거나
   적자인 종목은 최종 후보에서 제외한다.

파이프라인은 "지금 밸류체인 어디가 상대적으로 싸게 거래되는가"를 찾고, 그 안에서 재무적으로
건전한 종목만 후보로 남기는 2단계 구조다. 목표가·매수/매도 타이밍을 제시하지 않는다 —
장기투자자에게는 오히려 "계속 추적할 종목군"을 좁혀주는 스크리너 역할이 맞다고 판단했다.

## 구조

```
stock_ai/
├── config/
│   ├── taxonomy.yaml            # AI/기술 밸류체인 노드 정의 (고정 목록 아님, 계속 갱신 대상)
│   ├── model_weights.json       # horizon별(1d/1w/1m/1y) 기대수익률 모델 가중치
│   └── candidate_features.yaml  # 섀도우 피처 후보 상태 (proposed/shadow/승격/기각)
├── src/
│   ├── data_fetcher.py      # yfinance로 가격/재무 스냅샷 수집
│   ├── quality_filter.py    # 매출/이익 없는 기업 제외
│   ├── valuation.py         # 노드별 밸류에이션 집계 + 밸류체인 내 상대 저평가 스코어링
│   ├── taxonomy_manager.py  # taxonomy 로드/저장 + LLM 기반 동적 갱신(추가/제거 제안)
│   ├── pipeline.py          # 전체 파이프라인 오케스트레이션 (report용)
│   ├── report.py            # 마크다운 리포트 생성
│   ├── predictor.py         # 팩터 기반 기대수익률 선형모델 (가중치 로드/저장)
│   ├── calibration.py       # 실현수익률에 대한 Elastic Net(자동 피처 선택) 가중치 재적합
│   ├── backtest.py          # 예측 기록 / 만기 판정 / 실현수익률 평가
│   ├── universe.py          # 노드별 PEG 최저 상위 N종목 추적 유니버스 선정
│   ├── shadow_features.py   # 섀도우 피처 계산 레지스트리 (실거래 모델엔 미반영)
│   ├── shadow_backtest.py   # 섀도우 피처 값 기록
│   ├── feature_research.py  # LLM 기반 신규 후보 피처 리서치 (월 1회 권장)
│   ├── shadow_evaluation.py # 섀도우 피처가 실제로 도움되는지 통계적으로 검증
│   └── daily.py             # 일일 오케스트레이션: 예측→평가→재적합→요약(+섀도우 기록)
├── data/
│   ├── history/               # 매 `run` 실행 결과가 누적되는 valuation_history.csv
│   ├── predictions.csv        # 매일 기록되는 종목별·horizon별 예측
│   ├── evaluations.csv        # 만기 도달해 실제 수익률과 비교 완료된 예측
│   ├── shadow_predictions.csv # 섀도우 피처 값 (실거래 예측식에는 영향 없음)
│   └── daily_summary.md       # 가장 최근 `daily` 실행 요약
├── tests/                     # 네트워크 없이 도는 오프라인 단위/통합 테스트
└── main.py                    # CLI 진입점
```

## 밸류체인 taxonomy는 왜 고정되지 않는가

`config/taxonomy.yaml`은 반도체 장비 → 파운드리/메모리 → AI칩 → 서버/하드웨어 → 클라우드
→ AI 소프트웨어 → SaaS → 보안 → 네트워킹 → 로보틱스까지 이어지는 넓은 밸류체인을 seed로 담고
있다. "유망한 섹터/밸류체인을 계속 추가·제거하고 싶다"는 요청에 맞춰, 이 파일은 실행할 때마다
LLM이 최신 동향을 조사해 갱신 제안을 하는 대상으로 설계했다.

- `python main.py update-taxonomy` — Claude가 현재 taxonomy를 보고 새로 부상하는 노드/종목,
  더 이상 유효하지 않은 항목을 제안한다. **즉시 반영되지 않고** `taxonomy.yaml`의
  `review.pending_additions` / `pending_removals`에 대기 상태로만 기록된다.
- 제안 내용을 사람이 직접 읽고 검토한다.
- `python main.py apply-taxonomy` — 검토를 마친 뒤에만 실제 `nodes`에 반영한다.

자동으로 종목을 사고파는 게 아니라 "무엇을 추적할지"를 갱신하는 단계이므로, 반드시 사람 검토를
거치도록 자동 반영을 막아뒀다.

## 설치

```bash
cd stock_ai
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`update-taxonomy` 명령을 쓰려면 `.env.example`을 참고해 `ANTHROPIC_API_KEY`를 설정한다.
(`run`, `apply-taxonomy` 명령은 API 키 없이 동작한다.)

## 사용법

```bash
# 1. 파이프라인 실행 — 밸류체인 노드별 저평가 스코어 + 퀄리티 통과 후보 리포트 출력
python main.py run

# 2. (선택) LLM으로 최신 유망 섹터/종목 추가·제거안 조사
python main.py update-taxonomy

# 3. (선택) 검토를 마친 taxonomy 변경안을 실제 반영
python main.py apply-taxonomy

# 4. 일일 예측/평가/재적합 (GitHub Actions가 평일마다 자동 실행)
python main.py daily

# 5. (선택) LLM으로 새 후보 피처 리서치 — 월 1회 권장, ANTHROPIC_API_KEY 필요
python main.py research-features

# 6. 섀도우 피처가 실제로 도움되는지 통계 검증 — API 키 불필요, GitHub Actions가 월 1회 자동 실행
python main.py evaluate-shadow-features
```

`run`을 실행할 때마다 `data/history/valuation_history.csv`에 그날의 노드별 밸류에이션이
누적된다. 반복 실행할수록 "이 노드가 자기 역사적 밸류에이션 대비 지금 싼가"를 비교할 수 있는
데이터가 쌓인다 — 정기적으로 (예: 분기마다) 실행하는 것을 전제로 만들었다.

## 밸류에이션 스코어링 방법

노드별로 forward P/E, P/S, PEG의 중앙값을 구한 뒤, **같은 시점의 다른 노드들과 비교한 percentile**
평균을 "저평가 스코어"로 쓴다 (0에 가까울수록 밸류체인 내에서 상대적으로 저평가). 절대적인
"싸다/비싸다" 판단이 아니라 "지금 이 밸류체인 안에서 상대적으로 어디가 싼가"를 보는 방식이다.

## 일일 예측 / 백테스트 / 가중치 재적합 (`python main.py daily`)

밸류에이션·펀더멘털 팩터로 실제 향후 수익률을 얼마나 잘 설명하는지 **매일 기록하고 검증**하는
루프. `.github/workflows/daily-predict.yml`이 평일마다 자동 실행하고 결과를 커밋한다.

**추적 유니버스**: 각 밸류체인 노드에서 퀄리티 필터를 통과한 종목 중 PEG(성장 대비 가격)가
낮은 상위 2개. 저평가 상위 노드로 한정하지 않고 **모든 노드**에서 뽑는다 — 가중치 재적합에 쓸
표본을 여러 노드에 걸쳐 최대한 빨리 확보하기 위해서다. 지금은 10개 노드 × 2종목 = 20종목이고,
`update-taxonomy`로 노드가 늘어날수록 추적 종목 수도 자연히 늘어난다.

**기대수익률 모델**: 6개 피처의 선형 결합으로 1일·1주·1개월·1년 각각의 기대수익률을
예측한다. **가중치는 전부 0에서 시작한다** — 아직 아무 근거가 없으니 "초과수익 0%"가 가장
정직한 출발점이라고 판단했다.

| 피처 | 내용 | 갱신 빈도 |
|---|---|---|
| x1 | 1 − 노드 저평가스코어 | 가격이 매일 반영되므로 매일 조금씩 변함 |
| x2 | 매출성장률 | 분기 실적 발표 때만 변함 |
| x3 | 순이익률 | 분기 실적 발표 때만 변함 |
| x4 | 1 / PEG | 가격이 매일 반영되므로 매일 조금씩 변함 |
| x5 | 최근 5거래일(약 1주) 수익률 | 매일 변함 |
| x6 | 최근 21거래일(약 1개월) 수익률 | 매일 변함 |

x2·x3는 재무지표라 대부분 고정돼 있고, x1·x4는 가격이 분자/분모에 들어가 매일 소폭 변한다.
x5·x6(가격 모멘텀)은 매일 실질적으로 바뀌는 값이라 넣었다 — 짧은 horizon(1일/1주)일수록 가격
흐름 자체가 펀더멘털보다 설명력이 클 가능성이 높기 때문이다.

**피처를 계속 늘리고 싶다면**: "예측력 좋은 피처는 남기고 나쁜 건 뺀다"를 사람이 매번 판단해
수동으로 골라내는 방식은 쓰지 않는다 — 피처를 많이 만들어놓고 그때그때 상관관계가 높아 보이는
것만 골라 쓰면, 우연히 상관관계가 높게 나온 잡음 피처를 진짜 신호로 착각하는 다중검정
문제(multiple testing)에 빠지기 쉽다. 대신 회귀 자체를 **Elastic Net**(L1+L2 정규화,
coordinate descent로 직접 구현, `calibration.py`)으로 돌려서, 예측력이 없는 피처의 가중치를
알고리즘이 통계적 원칙에 따라 자동으로 정확히 0으로 수렴시키게 했다. 정규화 강도(alpha)도
고정값이 아니라 매번 k-fold 교차검증으로 데이터에서 직접 고른다. 재적합이 일어난 horizon은
`daily_summary.md`에 "이번에 실제로 살아남은(0이 아닌) 피처" 목록이 함께 기록된다. 피처 개수
자체는 표본이 쌓이는 속도에 맞춰 점진적으로만 늘릴 계획이다 — 지금 표본 규모(하루 수십 건)에서
한 번에 수십~수백 개를 추가하면 규제를 걸어도 사실상 노이즈에 맞춰 적합될 뿐이다.

**매일 일어나는 일**:
1. 오늘자 추적 종목에 대해 4개 horizon 예측을 `data/predictions.csv`에 기록
2. target_date가 오늘 이하로 도래한(=만기된) 과거 예측을 오늘 가격으로 평가해
   `data/evaluations.csv`에 실현수익률·오차 기록
3. horizon별 누적 평가 표본이 최소 기준을 넘으면 **Elastic Net(coordinate descent)**으로
   가중치 재적합 (표본 기준: 1일 40개, 1주 30개, 1개월 25개, 1년 15개 — 짧은 horizon일수록
   노이즈가 커서 더 많은 표본을 요구한다)
4. `data/daily_summary.md`에 그날 요약(표본 수, MAE, 방향 적중률, 재적합 여부, **재적합된
   horizon에서 살아남은 피처 목록**) 기록

**왜 1일/1주 표본 기준이 더 높은가**: 펀더멘털 팩터(밸류에이션, 매출성장률 등)는 분기 단위로
바뀌는데, 하루·일주일 단위 주가는 그런 팩터보다 시장 노이즈의 영향이 압도적으로 크다. 표본이
적은 상태에서 매일 재적합하면 노이즈에 과적합될 위험이 있어, 짧은 horizon일수록 재적합을 더
보수적으로(표본이 충분히 쌓였을 때만) 하도록 설계했다. **1일/1주 예측은 통계적으로 유의미한
신호가 거의 없을 가능성이 높다** — 이 시스템은 "맞는 예측"을 보장하는 게 아니라, 그 사실 자체를
정직하게 데이터로 보여주기 위해 만들었다. 1년 예측은 신호가 있을 가능성이 더 높지만, 표본이
쌓이려면 실제로 여러 해가 걸린다.

## 섀도우 피처 파이프라인 — 새 피처를 안전하게 추가하는 방법

"매일 인터넷/논문을 찾아서 피처를 추가·제외하자"는 아이디어에서 출발했지만, 매일 피처
구성 자체를 바꾸면 모델이 계속 리셋되고, "논문에 나온 팩터"를 검증 없이 바로 실거래 모델에
넣으면 다중검정(multiple testing) 문제에 빠지기 쉽다. 그래서 리서치와 실제 모델 반영 사이에
**섀도우 단계**를 뒀다.

```
LLM 리서치 (월 1회)          사람이 구현            매일 값만 기록           통계 검증 (월 1회)
candidate_features.yaml  →  shadow_features.py  →  shadow_predictions.csv  →  evaluate-shadow-features
     status: proposed          status: shadow                                  → 승격 or 기각
```

1. **`python main.py research-features`** (월 1회 권장, `ANTHROPIC_API_KEY` 필요) — Claude가
   이미 쓰는 6개 피처와 겹치지 않고, 무료 데이터로 계산 가능하고, 학술적 근거가 있는 새 후보를
   2~4개 제안한다. `config/candidate_features.yaml`에 `status: proposed`로만 기록되고 아무것도
   자동 반영되지 않는다.
2. 사람(또는 나에게 요청)이 제안을 검토하고, 계산 로직을 `src/shadow_features.py`의
   `SHADOW_FEATURE_REGISTRY`에 구현한 뒤 status를 `shadow`로 바꾼다.
3. `shadow` 상태인 피처는 `daily` 실행 때마다 값만 `data/shadow_predictions.csv`에 기록된다 —
   `predictor.py`의 실제 예측식(FEATURE_NAMES)에는 전혀 관여하지 않는다.
4. **`python main.py evaluate-shadow-features`** (월 1회, API 키 불필요) — 섀도우 피처 값을
   `predictions.csv`(기존 6개 피처), `evaluations.csv`(실현수익률)와 join해서, "기존 피처만 쓴
   Elastic Net"과 "섀도우 피처를 추가한 Elastic Net"의 **교차검증 MSE를 직접 비교**한다.
   - 어느 horizon에서든 유의미하게(2% 이상) 개선되면 → `recommended_for_promotion`
   - 표본이 충분한 horizon이 2개 이상인데 전부 개선이 없으면 → `rejected`
   - 아직 표본이 부족하면 → `shadow` 상태 유지, 계속 데이터만 쌓음
5. `recommended_for_promotion`이 뜨면, 그때 사람이 `predictor.FEATURE_NAMES`에 정식으로
   추가한다 (스키마가 바뀌므로 가중치 리셋이 필요 — momentum 피처 추가 때와 동일한 절차).

지금 시드로 넣어둔 섀도우 피처 2개:
- `analyst_upside` — 애널리스트 목표주가 컨센서스 대비 상승여력
- `week52_position` — 52주 최저~최고 구간 내 현재가 위치 (George & Hwang 2004, 52-week high momentum)

`python main.py research-features`를 GitHub Actions에서 자동 실행하려면 저장소 Settings →
Secrets and variables → Actions에 `ANTHROPIC_API_KEY`를 추가해야 한다. 없어도
`evaluate-shadow-features`(통계 검증)는 매달 자동으로 계속 돈다.

## 한계 및 주의사항

- yfinance 무료 데이터는 정확도·최신성이 제한적이다. 중요한 투자 판단 전에는 원 공시 자료로
  교차 검증해야 한다.
- 노드 간 상대 밸류에이션 비교는 서로 다른 사업 구조(마진, 성장률, 자본집약도)를 가진 기업들을
  단순 배수로 비교하는 한계가 있다.
- `update-taxonomy`가 제안하는 내용은 LLM의 추정이며 사실 오류(hallucination) 가능성이 있다.
  반드시 검토 후 `apply-taxonomy`로 반영해야 한다.
- 이 리포지토리는 투자 자문이 아니라 리서치/스크리닝 보조 도구이며, 실제 투자 결정의 책임은
  전적으로 사용자에게 있다.
- 일일 예측 시스템의 1일/1주 horizon은 통계적으로 유의미한 예측력을 가질 가능성이 낮다.
  가중치가 재적합됐다고 해서 "예측이 맞다"는 뜻이 아니라 "표본이 충분히 쌓여 회귀가 돌았다"는
  뜻일 뿐이며, `data/daily_summary.md`의 방향 적중률·MAE를 함께 봐야 한다.
