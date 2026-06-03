from pykrx import stock
import pandas as pd
import time
import os

# =========================
# 1. 국내 반도체 관련 주요 종목
# =========================
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

# =========================
# 2. 수집 기간
# =========================
start_date = "20150101"
end_date = "20241231"

# =========================
# 3. CSV 저장 폴더 생성
# =========================
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# 전체 종목 데이터를 합치기 위한 리스트
all_data = []

# =========================
# 4. 종목별 주가 데이터 수집
# =========================
for name, ticker in stocks.items():
    print(f"{name}({ticker}) 데이터 가져오는 중...")

    df = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)

    # 인덱스에 있던 날짜를 일반 컬럼으로 변경
    df = df.reset_index()

    # 종목 정보 추가
    df["종목명"] = name
    df["종목코드"] = ticker

    # 컬럼 순서 정리
    df = df[
        [
            "날짜",
            "종목명",
            "종목코드",
            "시가",
            "고가",
            "저가",
            "종가",
            "거래량",
            "등락률"
        ]
    ]

    # 종목별 CSV 저장
    file_name = f"{name}_{ticker}_10years.csv"
    file_path = os.path.join(output_dir, file_name)

    df.to_csv(file_path, index=False, encoding="utf-8-sig")

    print(f"{name} CSV 저장 완료: {file_path}")
    print(f"{name} 데이터 개수: {len(df)}개 행")

    # 전체 통합용 리스트에 추가
    all_data.append(df)

    time.sleep(1)

# =========================
# 5. 전체 종목 데이터 하나로 합치기
# =========================
if all_data:
    total_df = pd.concat(all_data, ignore_index=True)

    total_output_path = os.path.join(output_dir, "semiconductor_all_10years.csv")

    total_df.to_csv(total_output_path, index=False, encoding="utf-8-sig")

    print("\n전체 통합 CSV 저장 완료")
    print(f"파일명: {total_output_path}")
    print(f"전체 데이터 개수: {len(total_df)}개 행")
else:
    print("수집된 데이터가 없습니다.")