"""taxonomy.yaml 로드/저장 및 LLM 기반 밸류체인 동적 갱신.

`discover_updates()`는 Claude에게 현재 taxonomy를 보여주고, 새로 부상하는
유망 섹터/기업이나 더 이상 유효하지 않은 항목을 제안받는다. 제안은 즉시
반영되지 않고 taxonomy.yaml의 review.pending_additions / pending_removals에
기록되며, `apply_pending_reviews()`를 명시적으로 호출해야 실제 nodes에 반영된다.
(자동 매매/자동 편입이 아니라 "사람이 검토 후 승인"하는 구조로 설계했다.)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml


@dataclass
class ValueChainNode:
    id: str
    name: str
    stage: str
    tickers: list[str]


@dataclass
class Taxonomy:
    version: int
    last_updated: str
    nodes: list[ValueChainNode]
    review: dict = field(default_factory=dict)


def load_taxonomy(path: Path) -> Taxonomy:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    nodes = [ValueChainNode(**n) for n in raw["nodes"]]
    return Taxonomy(
        version=raw.get("version", 1),
        last_updated=raw.get("last_updated", ""),
        nodes=nodes,
        review=raw.get(
            "review", {"pending_additions": [], "pending_removals": [], "history": []}
        ),
    )


def save_taxonomy(taxonomy: Taxonomy, path: Path) -> None:
    data = {
        "version": taxonomy.version,
        "last_updated": taxonomy.last_updated,
        "nodes": [
            {"id": n.id, "name": n.name, "stage": n.stage, "tickers": n.tickers}
            for n in taxonomy.nodes
        ],
        "review": taxonomy.review,
    }
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


DISCOVERY_PROMPT_TEMPLATE = """\
너는 미국 상장 AI/기술 밸류체인을 계속 추적하는 리서치 어시스턴트다.
오늘 날짜는 {today}다.

현재 추적 중인 밸류체인 노드 목록(JSON):
{current_nodes}

다음을 조사해서 JSON으로만 답하라:
1. additions: 새로 부상했거나 아직 목록에 없는 유망한 노드(신규 노드) 또는
   기존 노드에 추가할 만한 미국 상장 종목. 반드시 실제로 매출이 발생하는
   상장 기업만 포함할 것 (매출/이익이 없는 초기 단계 기업 제외).
2. removals: 더 이상 유망하지 않거나(구조적 쇠퇴, 상장폐지, 인수합병 등)
   제거를 고려해야 할 노드/티커와 그 이유.

응답 형식 (다른 텍스트 없이 이 JSON만 출력):
{{
  "additions": [
    {{"node_id": "...", "node_name": "...", "stage": "upstream|midstream|downstream",
      "is_new_node": true, "tickers": ["..."], "reason": "..."}}
  ],
  "removals": [
    {{"node_id": "...", "ticker": "... 또는 null(null이면 노드 전체)", "reason": "..."}}
  ]
}}
"""


def discover_updates(taxonomy: Taxonomy, model: str | None = None) -> dict:
    """Claude API로 taxonomy 갱신안을 제안받는다. ANTHROPIC_API_KEY 환경변수 필요."""
    import anthropic

    client = anthropic.Anthropic()
    model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

    current_nodes = json.dumps(
        [
            {"id": n.id, "name": n.name, "stage": n.stage, "tickers": n.tickers}
            for n in taxonomy.nodes
        ],
        ensure_ascii=False,
    )
    prompt = DISCOVERY_PROMPT_TEMPLATE.format(
        today=date.today().isoformat(), current_nodes=current_nodes
    )

    response = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return json.loads(text)


def record_pending_review(taxonomy: Taxonomy, proposal: dict) -> None:
    taxonomy.review.setdefault("pending_additions", [])
    taxonomy.review.setdefault("pending_removals", [])
    taxonomy.review.setdefault("history", [])

    taxonomy.review["pending_additions"].extend(proposal.get("additions", []))
    taxonomy.review["pending_removals"].extend(proposal.get("removals", []))
    taxonomy.review["history"].append(
        {
            "date": date.today().isoformat(),
            "n_additions": len(proposal.get("additions", [])),
            "n_removals": len(proposal.get("removals", [])),
        }
    )


def apply_pending_reviews(taxonomy: Taxonomy) -> Taxonomy:
    """사람이 검토를 마친 뒤 pending 항목을 실제 nodes에 반영한다."""
    nodes_by_id = {n.id: n for n in taxonomy.nodes}

    for addition in taxonomy.review.get("pending_additions", []):
        node_id = addition["node_id"]
        if addition.get("is_new_node") or node_id not in nodes_by_id:
            node = ValueChainNode(
                id=node_id,
                name=addition.get("node_name", node_id),
                stage=addition.get("stage", "midstream"),
                tickers=list(dict.fromkeys(addition.get("tickers", []))),
            )
            nodes_by_id[node_id] = node
            taxonomy.nodes.append(node)
        else:
            existing = nodes_by_id[node_id]
            for ticker in addition.get("tickers", []):
                if ticker not in existing.tickers:
                    existing.tickers.append(ticker)

    for removal in taxonomy.review.get("pending_removals", []):
        node_id = removal["node_id"]
        ticker = removal.get("ticker")
        if node_id not in nodes_by_id:
            continue
        if ticker is None:
            taxonomy.nodes = [n for n in taxonomy.nodes if n.id != node_id]
        else:
            nodes_by_id[node_id].tickers = [
                t for t in nodes_by_id[node_id].tickers if t != ticker
            ]

    taxonomy.review["pending_additions"] = []
    taxonomy.review["pending_removals"] = []
    taxonomy.last_updated = date.today().isoformat()
    return taxonomy
