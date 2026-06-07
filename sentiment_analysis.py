import pandas as pd
import re
import glob
import os
from datetime import datetime

def install_deps():
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "transformers", "torch", "--quiet"])

try:
    from transformers import pipeline
except ImportError:
    print("transformers 설치 중...")
    install_deps()
    from transformers import pipeline


CSV_PATTERN = "output_*.csv"
OUTPUT_DIR = "sentiment_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"https?://\S+", "", text)
    # 쉼표(,), 마침표(.), 퍼센트(%), 플러스(+), 마이너스(-) 정도는 금융 텍스트에서 살려둠
    text = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s.,%+-]", " ", text)
    
    return text.strip()


def load_sentiment_model():
    print("감성 분석 모델 로딩 중... (첫 실행 시 다운로드 포함, 수 분 소요)")
    classifier = pipeline(
        "text-classification",
        model="snunlp/KR-FinBert-SC",
        tokenizer="snunlp/KR-FinBert-SC",
        device=-1,          # CPU 사용 (-1); GPU 있으면 0
        truncation=True,
        max_length=128,
    )
    print("모델 로딩 완료.")
    return classifier


def classify_batch(classifier, texts: list[str], batch_size: int = 32) -> list[dict]:
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        out = classifier(batch, batch_size=batch_size)
        results.extend(out)
        print(f"  진행: {min(i + batch_size, len(texts))}/{len(texts)}")
    return results


def label_to_score(label: str) -> int:
    """KR-FinBert-SC 레이블 → 정수 (positive=1, negative=-1, neutral=0)"""
    label = label.lower()
    if "positive" in label:
        return 1
    if "negative" in label:
        return -1
    return 0


def process_csv(filepath: str, classifier) -> pd.DataFrame:
    stock_name = os.path.splitext(os.path.basename(filepath))[0]
    print(f"\n[{stock_name}] 처리 시작 ({filepath})")

    df = pd.read_csv(filepath, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    df["writtenAt"] = pd.to_datetime(df["writtenAt"], errors="coerce")
    df = df.dropna(subset=["writtenAt"])
    df["date"] = df["writtenAt"].dt.date

    # 제목 + 본문 합쳐서 분석
    df["text"] = (
        df["title"].fillna("").apply(clean_text)
        + " "
        + df["contentSwReplaced"].fillna("").apply(clean_text)
    ).str.strip()

    # 빈 텍스트 제거
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    print(f"  유효 댓글 수: {len(df)}")

    # 감성 분류
    results = classify_batch(classifier, df["text"].tolist())

    df["sentiment_label"] = [r["label"] for r in results]
    df["sentiment_conf"]  = [round(r["score"], 4) for r in results]
    df["sentiment_score"] = df["sentiment_label"].apply(label_to_score)

    # 날짜별 집계
    daily = (
        df.groupby("date")
        .agg(
            comment_count=("sentiment_score", "count"),
            pos_count=("sentiment_score", lambda x: (x == 1).sum()),
            neg_count=("sentiment_score", lambda x: (x == -1).sum()),
            neu_count=("sentiment_score", lambda x: (x == 0).sum()),
            mean_conf=("sentiment_conf", "mean"),
        )
        .reset_index()
    )

    daily["pos_ratio"] = (daily["pos_count"] / daily["comment_count"]).round(4)
    daily["neg_ratio"] = (daily["neg_count"] / daily["comment_count"]).round(4)
    # 순 감성 점수: -1 ~ +1
    daily["net_sentiment"] = (
        (daily["pos_count"] - daily["neg_count"]) / daily["comment_count"]
    ).round(4)

    out_path = os.path.join(OUTPUT_DIR, f"sentiment_{stock_name}.csv")
    daily.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  저장 완료: {out_path}")
    return daily


def main():
    csv_files = glob.glob(CSV_PATTERN)
    if not csv_files:
        print(f"CSV 파일을 찾을 수 없습니다. 패턴: {CSV_PATTERN}")
        return

    print(f"발견된 CSV 파일: {csv_files}")
    classifier = load_sentiment_model()

    all_results = []
    for filepath in csv_files:
        daily = process_csv(filepath, classifier)
        stock_id = os.path.splitext(os.path.basename(filepath))[0]
        daily.insert(0, "stock", stock_id)
        all_results.append(daily)

    # 전체 종목 합본
    combined = pd.concat(all_results, ignore_index=True)
    combined_path = os.path.join(OUTPUT_DIR, "sentiment_all_stocks.csv")
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    print(f"\n전체 합본 저장: {combined_path}")
    print("\n완료! 생성된 컬럼:")
    print("  date, comment_count, pos_count, neg_count, neu_count,")
    print("  pos_ratio, neg_ratio, net_sentiment, mean_conf")


if __name__ == "__main__":
    main()
