"""LLM 기반 후보 피처 리서치 (월 1회 정도의 느린 주기로 실행하도록 설계).

taxonomy_manager.py의 discover_updates()와 같은 원칙: LLM은 후보를
"제안"만 하고, candidate_features.yaml에 status="proposed"로 기록될 뿐
즉시 계산/사용되지 않는다. 실제로 값을 기록하려면 사람이
`shadow_features.py`에 계산 함수를 구현하고 status를 "shadow"로 바꿔야 한다.

매일이 아니라 월 단위로 도는 이유는 README 참고 — 팩터 논문/리서치가 매일
새로 나오지 않고, 새 후보를 넣을 때마다 검증용 표본이 다시 쌓여야 하므로
너무 자주 후보를 바꾸면 오히려 검증이 불가능해진다.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml


@dataclass
class CandidateFeatures:
    candidates: list[dict] = field(default_factory=list)
    research_history: list[dict] = field(default_factory=list)


def load_candidates(path: Path) -> CandidateFeatures:
    if not path.exists():
        return CandidateFeatures()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return CandidateFeatures(
        candidates=raw.get("candidates", []),
        research_history=raw.get("research_history", []),
    )


def save_candidates(doc: CandidateFeatures, path: Path) -> None:
    data = {"candidates": doc.candidates, "research_history": doc.research_history}
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


RESEARCH_PROMPT_TEMPLATE = """\
너는 퀀트/팩터 투자 리서치 어시스턴트다. 오늘 날짜는 {today}다.

우리는 미국 상장 AI/기술 밸류체인 종목의 1일/1주/1개월/1년 기대수익률을
소수의 팩터를 선형결합해 예측하는 모델을 쓰고 있다.

이미 쓰고 있는 피처: {existing}
이미 제안됐거나 검토 중인 후보: {already_proposed}

퀀트 투자 논문·실무 문헌에서 검증된 팩터 중, 위 목록과 겹치지 않고
다음 조건을 만족하는 새 후보를 2~4개 제안해줘:
1. yfinance 같은 무료 데이터 소스로 계산 가능할 것 (별도 유료 API 불필요)
2. 널리 알려진 학술적 근거(논문/저자/개념명)가 있을 것
3. 계산식이 명확하고 간단할 것 (복잡한 시계열 모델 아님)

응답 형식 (다른 텍스트 없이 이 JSON 배열만 출력):
[
  {{"id": "snake_case_id", "name": "...", "description": "계산식",
    "rationale": "왜 수익률과 관련 있다고 알려져 있는지 + 출처(저자/개념명)"}}
]
"""


def discover_candidate_features(
    existing_feature_names: list[str],
    already_proposed_ids: list[str],
    model: str | None = None,
) -> list[dict]:
    """Claude API로 새 후보 피처를 제안받는다. ANTHROPIC_API_KEY 환경변수 필요."""
    import anthropic

    client = anthropic.Anthropic()
    model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

    prompt = RESEARCH_PROMPT_TEMPLATE.format(
        today=date.today().isoformat(),
        existing=", ".join(existing_feature_names),
        already_proposed=", ".join(already_proposed_ids) or "(없음)",
    )

    response = client.messages.create(
        model=model,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return json.loads(text)


def record_proposals(doc: CandidateFeatures, proposals: list[dict]) -> int:
    """중복 id는 건너뛰고 새 후보만 status=proposed로 추가. 추가된 개수 반환."""
    existing_ids = {c["id"] for c in doc.candidates}
    added = 0
    for p in proposals:
        if p["id"] in existing_ids:
            continue
        doc.candidates.append(
            {
                "id": p["id"],
                "name": p.get("name", p["id"]),
                "description": p.get("description", ""),
                "rationale": p.get("rationale", ""),
                "status": "proposed",
                "source": "llm_research",
                "proposed_date": date.today().isoformat(),
            }
        )
        existing_ids.add(p["id"])
        added += 1

    doc.research_history.append(
        {"date": date.today().isoformat(), "n_proposed": len(proposals), "n_added": added}
    )
    return added
