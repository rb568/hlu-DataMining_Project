# =============================================================================
# 한국어 금융 뉴스/댓글 감성 분석 파이프라인
# 모델: snunlp/KR-FinBert-SC (금융 도메인 특화 한국어 BERT)
# 입력: output_*.csv (종목별 댓글 파일)
# 출력: sentiment_output/ 폴더에 날짜별 집계 CSV 저장
# =============================================================================

# ── 표준 라이브러리 ──────────────────────────────────────────────────────────
import pandas as pd
import re
import glob
import os
from datetime import datetime


# ── 의존성 설치 (transformers/torch 없을 때 자동 설치) ────────────────────────
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


# ── 전역 설정 ─────────────────────────────────────────────────────────────────
CSV_PATTERN = "output_*.csv"        # 분석할 CSV 파일 패턴
OUTPUT_DIR = "sentiment_output"     # 결과 저장 폴더
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 텍스트 전처리 ─────────────────────────────────────────────────────────────
# HTML 태그, URL 제거 후 금융 텍스트에 필요한 특수문자(. , % + -)만 남김
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<br\s*/?>", " ", text)          # <br> 태그 → 공백
    text = re.sub(r"https?://\S+", "", text)         # URL 제거
    text = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s.,%+-]", " ", text)  # 불필요 특수문자 제거
    return text.strip()


# ── 감성 분석 모델 로드 ───────────────────────────────────────────────────────
# KR-FinBert-SC: 한국어 금융 텍스트에 특화된 BERT 기반 분류 모델
# positive / negative / neutral 3개 레이블로 분류
def load_sentiment_model():
    print("감성 분석 모델 로딩 중... (첫 실행 시 다운로드 포함, 수 분 소요)")
    classifier = pipeline(
        "text-classification",
        model="snunlp/KR-FinBert-SC",
        tokenizer="snunlp/KR-FinBert-SC",
        device=0,          # CPU 사용 (-1); GPU 있으면 0
        truncation=True,
        max_length=128,    # BERT 입력 최대 토큰 길이
    )
    print("모델 로딩 완료.")
    return classifier


# ── 배치 추론 ─────────────────────────────────────────────────────────────────
# 텍스트 리스트를 batch_size 단위로 나눠 모델에 입력 (메모리 초과 방지)
def classify_batch(classifier, texts: list[str], batch_size: int = 32) -> list[dict]:
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        out = classifier(batch, batch_size=batch_size)
        results.extend(out)
        print(f"  진행: {min(i + batch_size, len(texts))}/{len(texts)}")
    return results


# ── 레이블 → 정수 변환 ────────────────────────────────────────────────────────
# 모델 출력 레이블을 수치화: positive=+1, negative=-1, neutral=0
def label_to_score(label: str) -> int:
    """KR-FinBert-SC 레이블 → 정수 (positive=1, negative=-1, neutral=0)"""
    label = label.lower()
    if "positive" in label:
        return 1
    if "negative" in label:
        return -1
    return 0


# ── 단일 CSV 파일 처리 ────────────────────────────────────────────────────────
# 1) CSV 로드 → 2) 텍스트 전처리 → 3) 감성 분류 → 4) 날짜별 집계 → 5) 결과 저장
def process_csv(filepath: str, classifier) -> pd.DataFrame:
    stock_name = os.path.splitext(os.path.basename(filepath))[0]
    print(f"\n[{stock_name}] 처리 시작 ({filepath})")

    # CSV 로드 및 컬럼명 공백 제거
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # 날짜 파싱 및 날짜만 추출 (시간 제거)
    df["writtenAt"] = pd.to_datetime(df["writtenAt"], errors="coerce")
    df = df.dropna(subset=["writtenAt"])
    df["date"] = df["writtenAt"].dt.date

    # 제목 + 본문 합쳐서 분석 텍스트 생성
    df["text"] = (
        df["title"].fillna("").apply(clean_text)
        + " "
        + df["contentSwReplaced"].fillna("").apply(clean_text)
    ).str.strip()

    # 빈 텍스트 제거
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    print(f"  유효 댓글 수: {len(df)}")

    # 감성 분류 실행
    results = classify_batch(classifier, df["text"].tolist())

    # 분류 결과 컬럼 추가
    df["sentiment_label"] = [r["label"] for r in results]          # 원본 레이블 (positive/negative/neutral)
    df["sentiment_conf"]  = [round(r["score"], 4) for r in results] # 모델 확신도 (0~1)
    df["sentiment_score"] = df["sentiment_label"].apply(label_to_score)  # 정수 점수 (+1/0/-1)

    # ── 날짜별 집계 ──────────────────────────────────────────────────────────
    # 하루 단위로 긍/부/중립 댓글 수, 비율, 순 감성 점수 계산
    daily = (
        df.groupby("date")
        .agg(
            comment_count=("sentiment_score", "count"),              # 전체 댓글 수
            pos_count=("sentiment_score", lambda x: (x == 1).sum()), # 긍정 댓글 수
            neg_count=("sentiment_score", lambda x: (x == -1).sum()),# 부정 댓글 수
            neu_count=("sentiment_score", lambda x: (x == 0).sum()), # 중립 댓글 수
            mean_conf=("sentiment_conf", "mean"),                    # 평균 확신도
        )
        .reset_index()
    )

    daily["pos_ratio"] = (daily["pos_count"] / daily["comment_count"]).round(4)  # 긍정 비율
    daily["neg_ratio"] = (daily["neg_count"] / daily["comment_count"]).round(4)  # 부정 비율
    # 순 감성 점수: (긍정 - 부정) / 전체 → -1(완전 부정) ~ +1(완전 긍정)
    daily["net_sentiment"] = (
        (daily["pos_count"] - daily["neg_count"]) / daily["comment_count"]
    ).round(4)

    # 종목별 결과 CSV 저장
    out_path = os.path.join(OUTPUT_DIR, f"sentiment_{stock_name}.csv")
    daily.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  저장 완료: {out_path}")
    return daily


# ── 메인 실행 흐름 ────────────────────────────────────────────────────────────
# 패턴에 맞는 모든 CSV를 순회하며 처리 후 전체 종목 합본 CSV 생성
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
        daily.insert(0, "stock", stock_id)  # 종목 식별자 컬럼 추가
        all_results.append(daily)

    # 전체 종목 합본 저장 (XGBoost 등 다음 단계 모델 입력용)
    combined = pd.concat(all_results, ignore_index=True)
    combined_path = os.path.join(OUTPUT_DIR, "sentiment_all_stocks.csv")
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    print(f"\n전체 합본 저장: {combined_path}")
    print("\n완료! 생성된 컬럼:")
    print("  date, comment_count, pos_count, neg_count, neu_count,")
    print("  pos_ratio, neg_ratio, net_sentiment, mean_conf")


if __name__ == "__main__":
    main()
