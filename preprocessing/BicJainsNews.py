import os
import glob
import pandas as pd
from newspaper import Article
import kss
from tqdm import tqdm

# 1. 경로 자동 설정 
current_dir = os.path.dirname(os.path.abspath(__file__))

# 최상위폴더/data안에 빅카인즈 폴더들에 접근 세팅
input_folder = os.path.join(current_dir, '..', 'data', 'BicJainsNews_data_xlsx')

output_folder = os.path.join(current_dir, '..', 'data', 'bicjainsNews_csv')

# 결과물을 저장할 폴더가 없다면 자동 생성
os.makedirs(output_folder, exist_ok=True)

# 2. input_folder 안의 모든 엑셀 파일
excel_files = glob.glob(os.path.join(input_folder, '*.xls*'))

if not excel_files:
    print("경로를 다시 확인해주세요.")
    exit()

print(f" 총 {len(excel_files)}개의 엑셀 파일을 찾았습니다.\n")

# 3. 각 엑셀 파일별로 반복 작업
for file_path in excel_files:
    # 파일명 추출
    file_name = os.path.basename(file_path)
    base_name = os.path.splitext(file_name)[0]
    
    # 파일명에서 기업명만 추출 (예: DB하이텍)
    company_name = base_name.split('_')[0] 

    print(f"[작업 시작] : {base_name}")
    
    # 엑셀 읽기
    df = pd.read_excel(file_path)
    result_data = []

    # 해당 엑셀 파일 내의 기사 크롤링
    for index, row in tqdm(df.iterrows(), total=len(df), desc=company_name):
        url = row.get('URL') 
        if pd.isna(url): continue

        try:
            article = Article(url, language='ko')
            article.download()
            article.parse()
            text = article.text

            if text:
                sentences = kss.split_sentences(text)
                for sent in sentences:
                    result_data.append({
                        '일자': str(row.get('일자', '')), 
                        '기업명': company_name,          
                        '언론사': row.get('언론사', ''),
                        '제목': row.get('제목', ''),
                        'URL': url,
                        '문장': sent
                    })
        except Exception:
            # 에러 발생 시 패스(예시로 안 넣어두면 삭제된 기사 불러오다가 에러로 멈출 수 있음)
            pass

    # 4. 데이터프레임 변환 및 날짜순 정렬
    final_df = pd.DataFrame(result_data)
    
    # 데이터가 있을 경우 저장 프로세스 진행
    if not final_df.empty:
        # 일자 컬럼 기준으로 최신->과거 정렬
        if '일자' in final_df.columns:
            final_df = final_df.sort_values(by='일자', ascending=False)

        # 5. CSV 파일로 저장
        output_file_path = os.path.join(output_folder, f"{base_name}_문장단위결과.csv")
        final_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        print(f"[저장 완료] : {output_file_path}\n")
    else:
        print(f"[데이터 없음] : {base_name}\n")

print("모든 기업의 데이터 크롤링 및 문장 분리 작업 완료")