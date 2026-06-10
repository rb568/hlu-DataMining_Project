"""
Feature Engineering Pipeline
pykrx + ecos + dart → 분류 모델용 피처 데이터셋 생성

타겟: 다음 거래일 종가 등락률 > 0 → 1, < 0 → 0 (보합=0 제거)
"""

import pandas as pd
import numpy as np
import os

BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYKRX_PATH  = os.path.join(BASE, 'data', 'pykrx_data', 'semiconductor_all_10years.csv')
ECOS_PATH   = os.path.join(BASE, 'data', 'ecos_data', 'ecos_macro_data_merged_10years.csv')
DART_PATH   = os.path.join(BASE, 'data', 'dart_data', 'dart_semiconductor_financials_2020_2024.csv')
OUTPUT_PATH = os.path.join(BASE, 'data', 'features_dataset.csv')


# ── 1. pykrx 주가 데이터 ────────────────────────────────────────────

def load_pykrx():
    df = pd.read_csv(PYKRX_PATH, encoding='utf-8-sig')
    df['날짜'] = pd.to_datetime(df['날짜'])
    df = df.sort_values(['종목코드', '날짜']).reset_index(drop=True)
    return df


def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def add_technical_features(df):
    g = df.groupby('종목코드')

    # 이동평균
    for w in [5, 10, 20, 60]:
        df[f'ma{w}'] = g['종가'].transform(lambda x: x.rolling(w).mean())

    # 단기/중기 이동평균 괴리율 (골든/데드크로스 신호)
    df['ma5_ma20_ratio'] = df['ma5'] / df['ma20'] - 1
    df['ma10_ma60_ratio'] = df['ma10'] / df['ma60'] - 1

    # 전일 등락률 lag 피처
    for lag in [1, 2, 3]:
        df[f'lag_return_{lag}'] = g['등락률'].transform(lambda x: x.shift(lag))

    # 거래량 피처
    df['vol_change']    = g['거래량'].transform(lambda x: x.pct_change())
    df['vol_ma5_ratio'] = df['거래량'] / g['거래량'].transform(lambda x: x.rolling(5).mean())

    # 일중 변동폭: (고가 - 저가) / 전일 종가
    df['high_low_range'] = (
        (df['고가'] - df['저가']) /
        g['종가'].transform(lambda x: x.shift(1))
    )

    # 5일 변동성 (등락률 표준편차)
    df['volatility_5'] = g['등락률'].transform(lambda x: x.rolling(5).std())

    # RSI 14일
    df['rsi_14'] = g['종가'].transform(calc_rsi)

    return df


# ── 2. ecos 거시경제 데이터 ─────────────────────────────────────────

def load_ecos():
    df = pd.read_csv(ECOS_PATH, encoding='utf-8-sig')
    df['날짜'] = pd.to_datetime(df['날짜'])
    df = df.sort_values('날짜').reset_index(drop=True)

    # 전일 대비 변화율
    df['usd_krw_change'] = df['원달러환율'].pct_change()
    df['cd_rate_change'] = df['CD91일금리'].diff()

    # 신용 스프레드 (회사채 - 국고채): 신용위험 지표
    df['credit_spread'] = df['회사채3년AA금리'] - df['국고채3년금리']

    return df[['날짜', '원달러환율', 'CD91일금리', '국고채3년금리', '회사채3년AA금리',
               'usd_krw_change', 'cd_rate_change', 'credit_spread']]


# ── 3. dart 재무 데이터 ─────────────────────────────────────────────

def load_dart():
    df = pd.read_csv(DART_PATH, encoding='utf-8-sig')

    # 연결재무제표(CFS)만 사용
    df = df[df['fs_div'] == 'CFS'].copy()

    # 금액 문자열 → 숫자 변환
    df['thstrm_amount'] = (
        df['thstrm_amount']
        .astype(str)
        .str.replace(',', '', regex=False)
        .str.replace('"', '', regex=False)
        .pipe(pd.to_numeric, errors='coerce')
    )

    # 핵심 계정과목만 사용
    key_accounts = ['자산총계', '부채총계', '자본총계', '매출액', '영업이익', '당기순이익(손실)']
    df_filtered = df[df['account_nm'].isin(key_accounts)].copy()

    # 중복 제거 후 pivot
    df_filtered = df_filtered.drop_duplicates(subset=['종목코드', 'bsns_year', 'account_nm'])
    pivot = df_filtered.pivot_table(
        index=['종목코드', 'bsns_year'],
        columns='account_nm',
        values='thstrm_amount',
        aggfunc='first'
    ).reset_index()
    pivot.columns.name = None

    # 재무비율 계산
    pivot['debt_ratio']       = pivot['부채총계'] / pivot['자본총계']
    pivot['roe']              = pivot['당기순이익(손실)'] / pivot['자본총계']
    pivot['operating_margin'] = pivot['영업이익'] / pivot['매출액']
    pivot['revenue_growth']   = pivot.groupby('종목코드')['매출액'].pct_change()

    fin_cols = ['종목코드', 'bsns_year',
                'debt_ratio', 'roe', 'operating_margin', 'revenue_growth']
    return pivot[fin_cols]


def merge_dart_to_daily(pykrx_df, dart_df):
    """
    연간 재무 데이터를 일별로 매핑
    - 미래 누수 방지: 날짜 기준 전년도 재무 사용
      (연간 보고서는 보통 다음 해 3~4월에 공시되므로 전년도가 안전)
    """
    pykrx_df = pykrx_df.copy()
    pykrx_df['bsns_year'] = pykrx_df['날짜'].dt.year - 1
    merged = pykrx_df.merge(dart_df, on=['종목코드', 'bsns_year'], how='left')
    return merged


# ── 4. 타겟 생성 ────────────────────────────────────────────────────

def create_target(df):
    df = df.copy()
    # 다음 거래일 등락률
    df['next_return'] = df.groupby('종목코드')['등락률'].transform(lambda x: x.shift(-1))

    # 보합(=0) 제거
    df = df[df['next_return'] != 0].copy()

    # 상승=1, 하락=0
    df['target'] = (df['next_return'] > 0).astype(int)
    return df


# ── 5. 메인 파이프라인 ──────────────────────────────────────────────

FEATURE_COLS = [
    # 기술적 지표
    'lag_return_1', 'lag_return_2', 'lag_return_3',
    'ma5_ma20_ratio', 'ma10_ma60_ratio',
    'vol_change', 'vol_ma5_ratio',
    'high_low_range', 'volatility_5', 'rsi_14',
    # 거시경제
    '원달러환율', 'CD91일금리', '국고채3년금리', '회사채3년AA금리',
    'usd_krw_change', 'cd_rate_change', 'credit_spread',
    # 재무비율
    'debt_ratio', 'roe', 'operating_margin', 'revenue_growth',
]


def main():
    print("=" * 50)
    print("피처 엔지니어링 파이프라인 시작")
    print("=" * 50)

    print("\n[1/5] pykrx 주가 데이터 로딩...")
    price = load_pykrx()
    print(f"  원본 shape: {price.shape}")
    price = add_technical_features(price)

    print("\n[2/5] ecos 거시경제 데이터 로딩...")
    macro = load_ecos()
    print(f"  shape: {macro.shape}")

    print("\n[3/5] dart 재무 데이터 로딩...")
    fin = load_dart()
    print(f"  shape: {fin.shape}")
    print(f"  보유 종목: {sorted(fin['종목코드'].unique())}")

    print("\n[4/5] 데이터 병합...")
    df = price.merge(macro, on='날짜', how='left')
    df = merge_dart_to_daily(df, fin)

    print("\n[5/5] 타겟 생성 및 결측치 제거...")
    df = create_target(df)
    df = df.dropna(subset=FEATURE_COLS + ['target'])

    # 저장 컬럼 정리
    keep_cols = ['날짜', '종목명', '종목코드'] + FEATURE_COLS + ['target']
    df = df[keep_cols].reset_index(drop=True)

    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

    # ── 결과 요약 ──
    print("\n" + "=" * 50)
    print("완료!")
    print("=" * 50)
    print(f"  저장 경로 : {OUTPUT_PATH}")
    print(f"  최종 shape: {df.shape}  (피처: {len(FEATURE_COLS)}개)")
    print(f"  기간       : {df['날짜'].min().date()} ~ {df['날짜'].max().date()}")
    print(f"  종목 수    : {df['종목코드'].nunique()}개")
    print(f"\n  타겟 분포:")
    vc = df['target'].value_counts()
    print(f"    상승(1): {vc[1]:,}행 ({vc[1]/len(df)*100:.1f}%)")
    print(f"    하락(0): {vc[0]:,}행 ({vc[0]/len(df)*100:.1f}%)")
    print(f"\n  피처 목록 ({len(FEATURE_COLS)}개):")
    for col in FEATURE_COLS:
        null_pct = df[col].isna().mean() * 100
        print(f"    {col:<25} null: {null_pct:.1f}%")


if __name__ == "__main__":
    main()
