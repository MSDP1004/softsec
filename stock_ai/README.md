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
│   └── taxonomy.yaml        # AI/기술 밸류체인 노드 정의 (고정 목록 아님, 계속 갱신 대상)
├── src/
│   ├── data_fetcher.py      # yfinance로 가격/재무 스냅샷 수집
│   ├── quality_filter.py    # 매출/이익 없는 기업 제외
│   ├── valuation.py         # 노드별 밸류에이션 집계 + 밸류체인 내 상대 저평가 스코어링
│   ├── taxonomy_manager.py  # taxonomy 로드/저장 + LLM 기반 동적 갱신(추가/제거 제안)
│   ├── pipeline.py          # 전체 파이프라인 오케스트레이션
│   └── report.py            # 마크다운 리포트 생성
├── data/history/             # 매 실행 결과가 누적되는 valuation_history.csv
├── tests/                     # 네트워크 없이 도는 오프라인 단위 테스트
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
```

`run`을 실행할 때마다 `data/history/valuation_history.csv`에 그날의 노드별 밸류에이션이
누적된다. 반복 실행할수록 "이 노드가 자기 역사적 밸류에이션 대비 지금 싼가"를 비교할 수 있는
데이터가 쌓인다 — 정기적으로 (예: 분기마다) 실행하는 것을 전제로 만들었다.

## 밸류에이션 스코어링 방법

노드별로 forward P/E, P/S, PEG의 중앙값을 구한 뒤, **같은 시점의 다른 노드들과 비교한 percentile**
평균을 "저평가 스코어"로 쓴다 (0에 가까울수록 밸류체인 내에서 상대적으로 저평가). 절대적인
"싸다/비싸다" 판단이 아니라 "지금 이 밸류체인 안에서 상대적으로 어디가 싼가"를 보는 방식이다.

## 한계 및 주의사항

- yfinance 무료 데이터는 정확도·최신성이 제한적이다. 중요한 투자 판단 전에는 원 공시 자료로
  교차 검증해야 한다.
- 노드 간 상대 밸류에이션 비교는 서로 다른 사업 구조(마진, 성장률, 자본집약도)를 가진 기업들을
  단순 배수로 비교하는 한계가 있다.
- `update-taxonomy`가 제안하는 내용은 LLM의 추정이며 사실 오류(hallucination) 가능성이 있다.
  반드시 검토 후 `apply-taxonomy`로 반영해야 한다.
- 이 리포지토리는 투자 자문이 아니라 리서치/스크리닝 보조 도구이며, 실제 투자 결정의 책임은
  전적으로 사용자에게 있다.
