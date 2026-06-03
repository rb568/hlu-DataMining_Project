import requests
import pandas as pd
import time
import os


# =========================
# 1. ECOS API 인증키 입력
# =========================
API_KEY = ""


# =========================
# 2. CSV 저장 폴더 생성
# =========================
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)


# =========================
# 3. ECOS 데이터 요청 함수
# =========================
def get_ecos_data(stat_code, period, start_date, end_date, item_code, data_name):
    url = (
        f"https://ecos.bok.or.kr/api/StatisticSearch/"
        f"{API_KEY}/json/kr/1/10000/"
        f"{stat_code}/{period}/{start_date}/{end_date}/{item_code}"
    )

    print("\n요청:", data_name)
    print(url.replace(API_KEY, "API_KEY"))

    response = requests.get(url)
    data = response.json()

    if "StatisticSearch" not in data:
        print(f"[실패] {data_name}")
        print(data)
        return pd.DataFrame()

    rows = data["StatisticSearch"]["row"]
    df = pd.DataFrame(rows)

    df = df[["TIME", "DATA_VALUE", "ITEM_NAME1", "UNIT_NAME"]]

    df = df.rename(columns={
        "TIME": "날짜",
        "DATA_VALUE": data_name,
        "ITEM_NAME1": "항목명",
        "UNIT_NAME": "단위"
    })

    df["날짜"] = pd.to_datetime(df["날짜"], format="%Y%m%d")
    df[data_name] = pd.to_numeric(df[data_name], errors="coerce")

    print(f"[성공] {data_name}: {len(df)}개 데이터")

    return df


# =========================
# 4. 가져올 ECOS 데이터 목록
# =========================
targets = [
    {
        "name": "원달러환율",
        "stat_code": "731Y001",
        "period": "D",
        "item_code": "0000001",
        "start": "20150101",
        "end": "20241231"
    },
    {
        "name": "CD91일금리",
        "stat_code": "817Y002",
        "period": "D",
        "item_code": "010502000",
        "start": "20150101",
        "end": "20241231"
    },
    {
        "name": "국고채3년금리",
        "stat_code": "817Y002",
        "period": "D",
        "item_code": "010200000",
        "start": "20150101",
        "end": "20241231"
    },
    {
        "name": "회사채3년AA금리",
        "stat_code": "817Y002",
        "period": "D",
        "item_code": "010300000",
        "start": "20150101",
        "end": "20241231"
    }
]


# =========================
# 5. 데이터 수집 및 개별 CSV 저장
# =========================
result_dict = {}

for target in targets:
    df = get_ecos_data(
        stat_code=target["stat_code"],
        period=target["period"],
        start_date=target["start"],
        end_date=target["end"],
        item_code=target["item_code"],
        data_name=target["name"]
    )

    if not df.empty:
        name = target["name"]
        result_dict[name] = df

        # 개별 CSV 저장
        individual_file = os.path.join(output_dir, f"ecos_{name}_10years.csv")
        df.to_csv(individual_file, index=False, encoding="utf-8-sig")

        print(f"개별 CSV 저장 완료: {individual_file}")

    time.sleep(1)


# =========================
# 6. 날짜 기준 병합 CSV 저장
# =========================
merged_df = None

for name, df in result_dict.items():
    temp_df = df[["날짜", name]].copy()

    if merged_df is None:
        merged_df = temp_df
    else:
        merged_df = pd.merge(merged_df, temp_df, on="날짜", how="outer")

if merged_df is not None:
    merged_df = merged_df.sort_values("날짜")

    merged_file = os.path.join(output_dir, "ecos_macro_data_merged_10years.csv")
    merged_df.to_csv(merged_file, index=False, encoding="utf-8-sig")

    print("\n병합 CSV 저장 완료")
    print(f"파일명: {merged_file}")
    print(f"전체 데이터 개수: {len(merged_df)}개 행")
else:
    print("병합할 데이터가 없습니다.")