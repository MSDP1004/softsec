#!/usr/bin/env python3
"""
서울 마포구/용산구/성동구 아파트 단지별 전세가율 분석
- 국토교통부 아파트 매매 실거래가 API (getRTMSDataSvcAptTradeDev)
- 국토교통부 아파트 전세 실거래가 API (getRTMSDataSvcAptRentDev)
- 기간: 2025년 1월 ~ 12월
"""

import requests
import xml.etree.ElementTree as ET
import pandas as pd
from collections import defaultdict
import sys
import time

# ── 설정 ──────────────────────────────────────────────────────────────────────
API_KEY = input("공공데이터포털 API 키를 입력하세요: ").strip()

DISTRICTS = {
    "11140": "용산구",
    "11200": "성동구",
    "11440": "마포구",
}

MONTHS = [f"2025{m:02d}" for m in range(1, 13)]

TARGET_AREA_MIN = 75.0   # 84㎡ 전후 ±9㎡
TARGET_AREA_MAX = 93.0

MIN_TRADE_COUNT = 5      # 최소 거래건수

TRADE_URL = "http://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev/getRTMSDataSvcAptTradeDev"
RENT_URL  = "http://apis.data.go.kr/1613000/RTMSDataSvcAptRentDev/getRTMSDataSvcAptRentDev"

# ── 유틸 ──────────────────────────────────────────────────────────────────────

def fetch_all_pages(url: str, base_params: dict) -> list[dict]:
    """페이지네이션을 처리하며 전체 항목 반환."""
    items = []
    page = 1
    while True:
        params = {**base_params, "pageNo": page, "numOfRows": 1000}
        for attempt in range(4):
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == 3:
                    print(f"  요청 실패 (page {page}): {e}", file=sys.stderr)
                    return items
                time.sleep(2 ** attempt)

        root = ET.fromstring(resp.text)

        # API 오류 확인
        result_code = root.findtext(".//resultCode", "")
        if result_code and result_code != "00":
            msg = root.findtext(".//resultMsg", "알 수 없는 오류")
            print(f"  API 오류: {result_code} - {msg}", file=sys.stderr)
            return items

        page_items = root.findall(".//item")
        if not page_items:
            break
        items.extend({child.tag: (child.text or "").strip() for child in item} for item in page_items)
        if len(page_items) < 1000:
            break
        page += 1
    return items


def price_to_int(price_str: str) -> int:
    """'55,000' 형태의 만원 단위 문자열을 정수로 변환."""
    return int(price_str.replace(",", "").strip())


# ── 데이터 수집 ───────────────────────────────────────────────────────────────

def collect_trade_data() -> list[dict]:
    """매매 실거래가 전체 수집."""
    print("\n[1/2] 아파트 매매 실거래가 수집 중...")
    all_rows = []
    for code, name in DISTRICTS.items():
        for ym in MONTHS:
            print(f"  {name} {ym[:4]}-{ym[4:]} 매매...", end="", flush=True)
            params = {
                "serviceKey": API_KEY,
                "LAWD_CD": code,
                "DEAL_YMD": ym,
            }
            rows = fetch_all_pages(TRADE_URL, params)
            print(f" {len(rows)}건")
            for r in rows:
                r["district_code"] = code
                r["district_name"] = name
            all_rows.extend(rows)
            time.sleep(0.1)
    return all_rows


def collect_rent_data() -> list[dict]:
    """전세 실거래가 전체 수집."""
    print("\n[2/2] 아파트 전세 실거래가 수집 중...")
    all_rows = []
    for code, name in DISTRICTS.items():
        for ym in MONTHS:
            print(f"  {name} {ym[:4]}-{ym[4:]} 전세...", end="", flush=True)
            params = {
                "serviceKey": API_KEY,
                "LAWD_CD": code,
                "DEAL_YMD": ym,
            }
            rows = fetch_all_pages(RENT_URL, params)
            print(f" {len(rows)}건")
            for r in rows:
                r["district_code"] = code
                r["district_name"] = name
            all_rows.extend(rows)
            time.sleep(0.1)
    return all_rows


# ── 분석 ──────────────────────────────────────────────────────────────────────

def parse_area(row: dict, area_key: str) -> float:
    try:
        return float(row.get(area_key, "0") or "0")
    except ValueError:
        return 0.0


def build_trade_summary(trade_rows: list[dict]) -> dict:
    """단지 키 → {총액, 건수} 집계 (매매)."""
    summary = defaultdict(lambda: {"total": 0, "count": 0, "dong": "", "district_name": ""})
    for r in trade_rows:
        area = parse_area(r, "excluUseAr")
        if not (TARGET_AREA_MIN <= area <= TARGET_AREA_MAX):
            continue
        try:
            price = price_to_int(r.get("dealAmount", "0"))
        except (ValueError, AttributeError):
            continue
        if price <= 0:
            continue
        key = (r["district_name"], r.get("umdNm", ""), r.get("aptNm", ""))
        summary[key]["total"] += price
        summary[key]["count"] += 1
        summary[key]["dong"] = r.get("umdNm", "")
        summary[key]["district_name"] = r["district_name"]
    return summary


def build_rent_summary(rent_rows: list[dict]) -> dict:
    """단지 키 → {총액, 건수} 집계 (전세, 보증금만)."""
    summary = defaultdict(lambda: {"total": 0, "count": 0})
    for r in rent_rows:
        # 전세 = 월세 0원인 경우
        monthly = r.get("monthlyRent", "0") or "0"
        try:
            if int(monthly.replace(",", "").strip()) != 0:
                continue  # 월세 건 제외
        except ValueError:
            continue
        area = parse_area(r, "excluUseAr")
        if not (TARGET_AREA_MIN <= area <= TARGET_AREA_MAX):
            continue
        try:
            deposit = price_to_int(r.get("deposit", "0"))
        except (ValueError, AttributeError):
            continue
        if deposit <= 0:
            continue
        key = (r["district_name"], r.get("umdNm", ""), r.get("aptNm", ""))
        summary[key]["total"] += deposit
        summary[key]["count"] += 1
    return summary


def analyze(trade_rows: list[dict], rent_rows: list[dict]) -> pd.DataFrame:
    print("\n[분석] 단지별 전세가율 계산 중...")
    trade_summary = build_trade_summary(trade_rows)
    rent_summary  = build_rent_summary(rent_rows)

    results = []
    for key, t in trade_summary.items():
        if t["count"] < MIN_TRADE_COUNT:
            continue
        r = rent_summary.get(key)
        if not r or r["count"] < MIN_TRADE_COUNT:
            continue

        avg_trade  = t["total"] / t["count"]
        avg_rent   = r["total"] / r["count"]
        lease_rate = avg_rent / avg_trade * 100

        district_name, dong, apt_name = key
        results.append({
            "단지명":       apt_name,
            "구":           district_name,
            "동":           dong,
            "평형(㎡)":     "84",
            "매매평균가(만원)": round(avg_trade),
            "전세평균가(만원)": round(avg_rent),
            "전세가율(%)":  round(lease_rate, 2),
            "매매거래건수":  t["count"],
            "전세거래건수":  r["count"],
        })

    df = pd.DataFrame(results)
    if df.empty:
        print("분석 가능한 데이터가 없습니다.")
        return df

    df = df.sort_values("전세가율(%)", ascending=False).reset_index(drop=True)
    df.index += 1
    return df


# ── 출력 ──────────────────────────────────────────────────────────────────────

def display(df: pd.DataFrame):
    top20 = df.head(20)
    pd.set_option("display.max_rows", 25)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.2f}".format)

    print("\n" + "=" * 90)
    print(" 서울 마포구/용산구/성동구 아파트 전세가율 상위 20개 단지 (84㎡, 2025년)")
    print("=" * 90)
    print(top20.to_string())
    print("=" * 90)
    print(f"총 분석 단지 수: {len(df)}개  (매매·전세 각 {MIN_TRADE_COUNT}건 이상)")


def save_csv(df: pd.DataFrame, path: str = "seoul_apt_lease_rate.csv"):
    df.to_csv(path, index=True, index_label="순위", encoding="utf-8-sig")
    print(f"\nCSV 저장 완료: {path}")


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    trade_rows = collect_trade_data()
    rent_rows  = collect_rent_data()

    print(f"\n수집 완료 — 매매 {len(trade_rows):,}건 / 전세 {len(rent_rows):,}건")

    df = analyze(trade_rows, rent_rows)
    if df.empty:
        sys.exit(1)

    display(df)
    save_csv(df)


if __name__ == "__main__":
    main()
