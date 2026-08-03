"""CLI 진입점.

사용법:
  python main.py run                 # 파이프라인 1회 실행, 리포트 출력
  python main.py daily                # 예측 기록 + 만기 평가 + 가중치 재적합 (매일 크론용)
  python main.py update-taxonomy     # LLM으로 taxonomy 갱신안 제안 (검토 대기 상태로 저장)
  python main.py apply-taxonomy      # 검토 대기 중인 taxonomy 변경안을 실제 반영
"""

from __future__ import annotations

import argparse

from src.daily import run_daily
from src.pipeline import DEFAULT_TAXONOMY_PATH, run_pipeline
from src.report import render_markdown
from src.taxonomy_manager import (
    apply_pending_reviews,
    discover_updates,
    load_taxonomy,
    record_pending_review,
    save_taxonomy,
)


def cmd_run(args: argparse.Namespace) -> None:
    result = run_pipeline(top_k_nodes=args.top_k)
    print(render_markdown(result))


def cmd_daily(args: argparse.Namespace) -> None:
    result = run_daily(top_n_per_node=args.top_n)
    print(
        f"추적종목={result.n_tracked} 신규예측={result.n_predictions_logged} "
        f"평가건수={result.n_evaluated} 재적합={result.calibrated_horizons or '없음'}"
    )


def cmd_update_taxonomy(args: argparse.Namespace) -> None:
    taxonomy = load_taxonomy(DEFAULT_TAXONOMY_PATH)
    proposal = discover_updates(taxonomy)
    record_pending_review(taxonomy, proposal)
    save_taxonomy(taxonomy, DEFAULT_TAXONOMY_PATH)
    print(
        f"추가 제안 {len(proposal.get('additions', []))}건, "
        f"제거 제안 {len(proposal.get('removals', []))}건이 "
        f"{DEFAULT_TAXONOMY_PATH}의 review.pending_* 항목에 저장됐습니다."
    )
    print("내용을 검토한 뒤 `python main.py apply-taxonomy`로 반영하세요.")


def cmd_apply_taxonomy(args: argparse.Namespace) -> None:
    taxonomy = load_taxonomy(DEFAULT_TAXONOMY_PATH)
    taxonomy = apply_pending_reviews(taxonomy)
    save_taxonomy(taxonomy, DEFAULT_TAXONOMY_PATH)
    print("검토 대기 중이던 변경안을 taxonomy.yaml에 반영했습니다.")


def main() -> None:
    parser = argparse.ArgumentParser(description="밸류체인 기반 저평가 섹터 추적 및 우량주 선별")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="파이프라인 1회 실행")
    run_parser.add_argument("--top-k", type=int, default=3, help="저평가 상위 노드 개수")
    run_parser.set_defaults(func=cmd_run)

    daily_parser = sub.add_parser("daily", help="예측 기록 + 만기 평가 + 가중치 재적합")
    daily_parser.add_argument("--top-n", type=int, default=2, help="노드당 추적할 종목 수")
    daily_parser.set_defaults(func=cmd_daily)

    update_parser = sub.add_parser("update-taxonomy", help="LLM으로 taxonomy 갱신안 제안")
    update_parser.set_defaults(func=cmd_update_taxonomy)

    apply_parser = sub.add_parser("apply-taxonomy", help="검토 대기 중인 taxonomy 변경안 반영")
    apply_parser.set_defaults(func=cmd_apply_taxonomy)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
