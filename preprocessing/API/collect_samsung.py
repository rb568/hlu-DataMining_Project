import requests
import json
import time

BASE_URL = "https://stock.naver.com/api/community/discussion/posts/by-item"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
    "Referer": "https://stock.naver.com/domestic/stock/005930/discussion",
}

PARAMS = {
    "discussionType": "domesticStock",
    "itemCode": "005930",
    "isHolderOnly": "false",
    "excludesItemNews": "false",
    "isItemNewsOnly": "false",
    "isCleanbotPassedOnly": "false",
    "pageSize": 30,
    "viewerProfileId": "28660454356503733",
}

TARGET_DATE = "2026-06-02"
MARKET_OPEN = "2026-06-01T15:30:00"   # 전날 장 마감 이후
MARKET_CLOSE = "2026-06-02T15:30:00"  # 당일 장 마감


def fetch_posts(offset=None, retries=3):
    params = PARAMS.copy()
    if offset:
        params["offset"] = offset
    for attempt in range(retries):
        try:
            resp = requests.get(BASE_URL, params=params, headers=HEADERS)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < retries - 1:
                print(f" 오류({e}) → {2 ** attempt}초 후 재시도...")
                time.sleep(2 ** attempt)
            else:
                raise


def collect():
    all_posts = []
    offset = None
    page = 1

    while True:
        print(f"페이지 {page} 수집 중...", end=" ")
        try:
            data = fetch_posts(offset)
        except Exception as e:
            print(f"\n재시도 실패({e}). 지금까지 수집한 데이터 저장 후 종료.")
            break
        posts = data.get("posts", [])

        if not posts:
            print("더 이상 데이터 없음. 종료.")
            break

        page_collected = 0
        stop = False

        for post in posts:
            written_at = post.get("writtenAt", "")

            if MARKET_OPEN <= written_at <= MARKET_CLOSE:
                all_posts.append({
                    "writtenAt": written_at,
                    "title": post.get("title"),
                    "contentSwReplaced": post.get("contentSwReplaced"),
                })
                page_collected += 1
            elif written_at < MARKET_OPEN:
                stop = True
                break

        print(f"{page_collected}개 수집 (누적 {len(all_posts)}개)")

        if stop:
            print(f"\n{TARGET_DATE} 이전 데이터 도달. 수집 완료.")
            break

        offset = posts[-1].get("orderNo")
        page += 1
        time.sleep(0.5)

    output_file = f"samsung_{TARGET_DATE.replace('-', '')}_market.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_posts, f, ensure_ascii=False, indent=2)

    print(f"\n총 {len(all_posts)}개 포스트 저장 완료: {output_file}")


if __name__ == "__main__":
    collect()
