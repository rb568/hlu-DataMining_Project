import requests
import pandas as pd
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO
import time
import os


# =====================================================
# 1. OpenDART API 인증키 입력
# =====================================================
API_KEY = ""


# =====================================================
# 2. CSV 저장 폴더 생성
# =====================================================
output_dir = "dart_data"
os.makedirs(output_dir, exist_ok=True)


# =====================================================
# 3. 수집할 국내 반도체 관련 종목
# =====================================================
stocks = {
    "삼성전자": "005930",
    "SK하이닉스": "000660",
    "DB하이텍": "000990",
    "한미반도체": "042700",
    "리노공업": "058470",
    "ISC": "095340",
    "이오테크닉스": "039030",
    "원익IPS": "240810",
    "주성엔지니어링": "036930"
}


# =====================================================
# 4. OpenDART 기업 고유번호 목록 다운로드
# =====================================================
def download_corp_code(api_key):
    """
    OpenDART corpCode.xml을 다운로드해서
    종목코드, 회사명, 고유번호 목록 DataFrame으로 반환
    """

    url = "https://opendart.fss.or.kr/api/corpCode.xml"

    params = {
        "crtfc_key": api_key
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print("기업 고유번호 다운로드 실패")
        print(response.text)
        return pd.DataFrame()

    try:
        zip_file = zipfile.ZipFile(BytesIO(response.content))
        xml_file_name = zip_file.namelist()[0]

        xml_data = zip_file.read(xml_file_name)
        root = ET.fromstring(xml_data)

        rows = []

        for item in root.findall("list"):
            rows.append({
                "corp_code": item.findtext("corp_code"),
                "corp_name": item.findtext("corp_name"),
                "stock_code": item.findtext("stock_code"),
                "modify_date": item.findtext("modify_date")
            })

        df = pd.DataFrame(rows)

        # 종목코드가 없는 비상장사는 제외
        df = df[df["stock_code"].notna()]
        df = df[df["stock_code"] != ""]

        return df

    except Exception as e:
        print("기업 고유번호 XML 처리 실패")
        print(e)
        print(response.text[:500])
        return pd.DataFrame()


# =====================================================
# 5. 반도체 종목 corp_code 찾기
# =====================================================
def find_target_companies(corp_df, stocks):
    result = []

    for name, stock_code in stocks.items():
        matched = corp_df[corp_df["stock_code"] == stock_code]

        if matched.empty:
            print(f"[실패] {name}({stock_code}) corp_code 찾지 못함")
            continue

        row = matched.iloc[0]

        result.append({
            "종목명": name,
            "종목코드": stock_code,
            "corp_code": row["corp_code"],
            "DART회사명": row["corp_name"],
            "최종변경일자": row["modify_date"]
        })

        print(f"[성공] {name}: corp_code={row['corp_code']}")

    return pd.DataFrame(result)


# =====================================================
# 6. 단일회사 주요계정 재무제표 조회
# =====================================================
def get_financial_statement(api_key, corp_code, year, reprt_code="11011"):
    """
    OpenDART 단일회사 주요계정 조회

    reprt_code:
    11013 = 1분기보고서
    11012 = 반기보고서
    11014 = 3분기보고서
    11011 = 사업보고서

    fs_div:
    CFS = 연결재무제표
    OFS = 별도재무제표
    """

    url = "https://opendart.fss.or.kr/api/fnlttSinglAcnt.json"

    params = {
        "crtfc_key": api_key,
        "corp_code": corp_code,
        "bsns_year": str(year),
        "reprt_code": reprt_code,
        "fs_div": "CFS"
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data.get("status") != "000":
        return pd.DataFrame()

    rows = data.get("list", [])

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


# =====================================================
# 7. 실행
# =====================================================
print("OpenDART 기업 고유번호 목록 다운로드 중...")
corp_df = download_corp_code(API_KEY)

if corp_df.empty:
    print("기업 고유번호 목록을 가져오지 못했습니다. API 키를 확인하세요.")

else:
    # 전체 기업 코드 목록 CSV 저장
    corp_code_path = os.path.join(output_dir, "dart_corp_code_list.csv")
    corp_df.to_csv(corp_code_path, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {corp_code_path}")

    # 반도체 기업 코드 목록 CSV 저장
    target_df = find_target_companies(corp_df, stocks)

    target_code_path = os.path.join(output_dir, "dart_semiconductor_corp_codes.csv")
    target_df.to_csv(target_code_path, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {target_code_path}")

    all_financials = []

    # 사업보고서 기준 2020~2024년 수집
    years = range(2020, 2025)

    for _, company in target_df.iterrows():
        name = company["종목명"]
        stock_code = company["종목코드"]
        corp_code = company["corp_code"]

        company_financials = []

        for year in years:
            print(f"{name} {year}년 사업보고서 재무제표 가져오는 중...")

            df = get_financial_statement(
                api_key=API_KEY,
                corp_code=corp_code,
                year=year,
                reprt_code="11011"
            )

            if df.empty:
                print(f"[없음] {name} {year}")

            else:
                df["종목명"] = name
                df["종목코드"] = stock_code
                df["조회연도"] = year

                all_financials.append(df)
                company_financials.append(df)

                print(f"[성공] {name} {year}: {len(df)}개 계정")

            time.sleep(0.3)

        # 종목별 재무제표 CSV 저장
        if company_financials:
            company_df = pd.concat(company_financials, ignore_index=True)

            company_file = os.path.join(
                output_dir,
                f"dart_{name}_{stock_code}_financials_2020_2024.csv"
            )

            company_df.to_csv(company_file, index=False, encoding="utf-8-sig")
            print(f"{name} 종목별 CSV 저장 완료: {company_file}")

    # 전체 반도체 기업 재무제표 통합 CSV 저장
    if all_financials:
        financial_df = pd.concat(all_financials, ignore_index=True)

        financial_path = os.path.join(
            output_dir,
            "dart_semiconductor_financials_2020_2024.csv"
        )

        financial_df.to_csv(financial_path, index=False, encoding="utf-8-sig")

        print(f"\n전체 재무제표 CSV 저장 완료: {financial_path}")
        print(f"전체 데이터 개수: {len(financial_df)}개 행")

    else:
        print("재무제표 데이터를 가져오지 못했습니다.")