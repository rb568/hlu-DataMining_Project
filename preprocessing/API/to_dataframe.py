import requests
import time
import pandas as pd

BASE_URL = "https://stock.naver.com/api/community/discussion/posts/by-item"

START_DATE = "2026-03-03T00:00:00"
END_DATE   = "2026-06-03T23:59:59"

STOCKS = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "042700": "한미반도체",
    "005935": "삼성전자우",
}


def fetch_posts(item_code, offset=None, retries=3):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
        "Referer": f"https://stock.naver.com/domestic/stock/{item_code}/discussion",
    }
    params = {
        "discussionType": "domesticStock",
        "itemCode": item_code,
        "isHolderOnly": "false",
        "excludesItemNews": "false",
        "isItemNewsOnly": "false",
        "isCleanbotPassedOnly": "false",
        "pageSize": 30,
        "viewerProfileId": "28660454356503733",
    }
    if offset:
        params["offset"] = offset
    for attempt in range(retries):
        try:
            resp = requests.get(BASE_URL, params=params, headers=headers)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < retries - 1:
                print(f" 오류({e}) → {2 ** attempt}초 후 재시도...")
                time.sleep(2 ** attempt)
            else:
                raise


def collect_to_dataframe(item_code):
    name = STOCKS.get(item_code, item_code)
    print(f"\n[{name} ({item_code})] 수집 시작")
    rows = []
    offset = None
    page = 1

    while True:
        print(f"페이지 {page} 수집 중...", end=" ")
        try:
            data = fetch_posts(item_code, offset)
        except Exception as e:
            print(f"\n재시도 실패({e}). 지금까지 수집한 데이터로 DataFrame 생성.")
            break
        posts = data.get("posts", [])

        if not posts:
            print("더 이상 데이터 없음. 종료.")
            break

        page_collected = 0
        stop = False

        for post in posts:
            written_at = post.get("writtenAt", "")
            if START_DATE <= written_at <= END_DATE:
                rows.append({
                    "writtenAt": written_at,
                    "title": post.get("title"),
                    "contentSwReplaced": post.get("contentSwReplaced"),
                })
                page_collected += 1
            elif written_at < START_DATE:
                stop = True
                break

        print(f"{page_collected}개 수집 (누적 {len(rows)}개)")

        if stop:
            print(f"\n수집 완료.")
            break

        offset = posts[-1].get("orderNo")
        page += 1
        time.sleep(0.1) # 서버 부담 줄이기 위해 100ms 대기

    df = pd.DataFrame(rows)
    df["writtenAt"] = pd.to_datetime(df["writtenAt"])
    return df

if __name__ == "__main__":
    df = collect_to_dataframe("005935")
    print(df)
    print(f"\n총 {len(df)}행")
    df.to_csv("output_005935_삼성전자우.csv", index=False, encoding="utf-8-sig")
    print("저장 완료: output_005935_삼성전자우.csv")