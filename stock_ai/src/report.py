"""파이프라인 결과를 사람이 읽기 좋은 마크다운 리포트로 변환."""

from __future__ import annotations

from datetime import date

from .pipeline import PipelineResult


def render_markdown(result: PipelineResult, run_date: date | None = None) -> str:
    run_date = run_date or date.today()
    lines = [f"# 밸류체인 저평가 섹터 리포트 ({run_date.isoformat()})", ""]

    lines.append("## 노드별 상대 밸류에이션 (스코어가 낮을수록 저평가)")
    lines.append("")
    lines.append("| 노드 | 저평가 스코어 | Forward P/E | P/S | PEG | 종목 수 |")
    lines.append("|---|---|---|---|---|---|")
    for nv in sorted(
        result.node_valuations,
        key=lambda x: (x.undervaluation_score is None, x.undervaluation_score),
    ):
        score = f"{nv.undervaluation_score:.2f}" if nv.undervaluation_score is not None else "N/A"
        pe = f"{nv.median_forward_pe:.1f}" if nv.median_forward_pe else "N/A"
        ps = f"{nv.median_price_to_sales:.1f}" if nv.median_price_to_sales else "N/A"
        peg = f"{nv.median_peg:.2f}" if nv.median_peg else "N/A"
        lines.append(f"| {nv.node_name} | {score} | {pe} | {ps} | {peg} | {nv.n_tickers} |")

    lines.append("")
    lines.append("## 저평가 노드 내 퀄리티 통과 후보 (매출·이익 존재 종목만)")
    lines.append("")
    if not result.candidates:
        lines.append("_조건을 만족하는 후보가 없습니다._")
    else:
        lines.append("| 티커 | 노드 | Forward P/E | P/S | PEG | 매출성장률 | 순이익률 |")
        lines.append("|---|---|---|---|---|---|---|")
        for c in sorted(result.candidates, key=lambda x: (x.node_name, x.ticker)):
            pe = f"{c.forward_pe:.1f}" if c.forward_pe else "N/A"
            ps = f"{c.price_to_sales:.1f}" if c.price_to_sales else "N/A"
            peg = f"{c.peg_ratio:.2f}" if c.peg_ratio else "N/A"
            rg = f"{c.revenue_growth * 100:.1f}%" if c.revenue_growth is not None else "N/A"
            pm = f"{c.profit_margin * 100:.1f}%" if c.profit_margin is not None else "N/A"
            lines.append(f"| {c.ticker} | {c.node_name} | {pe} | {ps} | {peg} | {rg} | {pm} |")

    lines.append("")
    lines.append(
        "> 본 리포트는 투자 조언이 아니며, 상대 밸류에이션 비교와 재무 스크리닝 결과일 뿐입니다. "
        "최종 투자 판단은 반드시 직접 검증한 뒤 내리세요."
    )
    return "\n".join(lines)
