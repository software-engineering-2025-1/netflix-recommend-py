# 개선된 넷플릭스 콘텐츠 기반 추천: 고성능 TF-IDF 벡터라이저

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import re
from collections import Counter
import os

def preprocess_dataframe(df):
    """Unnamed 컬럼 제거"""
    delete_columns = [i for i in df.columns if 'Unnamed' in i]
    df.drop(columns=delete_columns, inplace=True)
    return df

def preprocess_text(text):
    """고급 텍스트 전처리"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    # 구두점 및 특수문자 제거 (하이픈은 유지)
    text = re.sub(r'[^\w\s-]', ' ', text)
    # 여러 공백을 하나로 압축
    text = re.sub(r'\s+', ' ', text)
    # 숫자만 있는 단어 제거
    text = re.sub(r'\b\d+\b', '', text)
    # 너무 짧은 단어 제거 (2글자 이하)
    words = [word for word in text.split() if len(word) > 2]
    
    return ' '.join(words).strip()

def create_enhanced_tags(row):
    """SBERT와 동일한 가중치 기반 태그 생성 (공정한 비교를 위해)"""
    tags = []
    
    # 장르 (가중치 3배) - 추천에서 가장 중요한 특성
    if pd.notnull(row.get('listed_in')):
        genres = [genre.strip().lower().replace(' ', '_') for genre in row['listed_in'].split(',')]
        # 장르를 3번 반복해서 가중치 부여
        tags.extend(genres * 3)
    
    # 제목 (가중치 2배) - 제목의 키워드도 중요
    if pd.notnull(row.get('title')):
        title_words = preprocess_text(row['title']).split()
        tags.extend(title_words * 2)
    
    # 설명 (가중치 1배) - 전처리된 설명
    if pd.notnull(row.get('description')):
        desc_processed = preprocess_text(row['description'])
        # 설명에서 중요한 단어들만 추출 (긴 설명은 노이즈가 될 수 있음)
        desc_words = desc_processed.split()[:50]  # 첫 50개 단어만 사용
        tags.extend(desc_words)
    
    # 감독 (가중치 2배) - 감독 스타일이 중요
    if pd.notnull(row.get('director')):
        director_name = preprocess_text(row['director']).replace(' ', '_')
        if director_name:
            tags.extend([director_name] * 2)
    
    # 주요 배우 (가중치 2배) - 상위 3명
    if pd.notnull(row.get('cast')):
        actors = [actor.strip() for actor in str(row['cast']).split(',')[:3]]
        for actor in actors:
            actor_processed = preprocess_text(actor).replace(' ', '_')
            if actor_processed:
                tags.extend([actor_processed, actor_processed])  # 2번 반복으로 가중치 효과
    
    # 국가 (가중치 1배)
    if pd.notnull(row.get('country')):
        countries = [country.strip().lower().replace(' ', '_') for country in str(row['country']).split(',')]
        tags.extend(countries)
    
    # 타입 (Movie/TV Show) 추가
    if pd.notnull(row.get('type')):
        content_type = row['type'].lower().replace(' ', '_')
        tags.extend([content_type] * 2)
    
    # 연도대 정보 추가 (10년 단위)
    if pd.notnull(row.get('release_year')):
        decade = f"decade_{int(row['release_year']) // 10 * 10}s"
        tags.append(decade)
    
    # 모든 태그를 문자열로 변환하고 정리
    clean_tags = [str(tag) for tag in tags if tag and str(tag).strip()]
    
    return ' '.join(clean_tags)

def get_enhanced_stopwords():
    """영화 도메인 특화 불용어 리스트"""
    # 기본 영어 불용어 + 영화 도메인 특화 불용어
    movie_stopwords = [
        'movie', 'film', 'story', 'character', 'plot', 'scene', 'series',
        'show', 'episode', 'season', 'cast', 'starring', 'directed',
        'production', 'cinema', 'watch', 'see', 'view', 'screen',
        'featuring', 'based', 'follows', 'tells', 'depicts', 'portrays',
        'chronicles', 'explores', 'reveals', 'discovers', 'finds',
        'becomes', 'gets', 'goes', 'comes', 'takes', 'makes', 'gives',
        'min', 'minutes', 'hour', 'hours', 'time', 'year', 'years',
        'new', 'old', 'young', 'big', 'small', 'great', 'good', 'bad',
        'best', 'worst', 'first', 'last', 'next', 'previous'
    ]
    return movie_stopwords

# 데이터 로드 및 전처리
print("데이터 로딩 중...")
if not os.path.exists('../movies_with_enhanced_tags.csv'):
    df = pd.read_csv('../video.csv')
    df = preprocess_dataframe(df)

    print(f"전체 데이터 수: {len(df)}")
    print("샘플 데이터:")
    print(df.head())

    # NaN 값들을 빈 문자열로 처리
    df = df.fillna('')

    print("\n향상된 태그 생성 중...")
    df['enhanced_tags'] = df.apply(create_enhanced_tags, axis=1)

else:
    df = pd.read_csv('../movies_with_enhanced_tags.csv')

# 생성된 태그 확인
print("\n생성된 태그 샘플:")
for i in range(3):
    print(f"\n제목: {df.iloc[i]['title']}")
    print(f"태그: {df.iloc[i]['enhanced_tags'][:200]}...")

# 최적화된 TF-IDF 벡터라이저 설정
print("\n최적화된 TF-IDF 벡터라이저 설정 중...")

# 영화 도메인 특화 불용어
movie_stopwords = get_enhanced_stopwords()

# 최적화된 TfidfVectorizer
tfidf_optimized = TfidfVectorizer(
    max_features=15000,              # 정확성과 성능의 균형
    min_df=2,                        # 최소 2개 문서에 나타나는 단어만 (노이즈/오타 제거)
    max_df=0.85,                     # 85% 이상 문서에 나타나는 과도하게 일반적인 단어 제거
    stop_words=movie_stopwords,      # 영화 도메인 특화 불용어
    ngram_range=(1, 2),             # 단일어 + 이중어 조합
    lowercase=True,                  # 대소문자 정규화
    strip_accents='unicode',         # 국제 문자 처리
    norm='l2',                      # 코사인 유사도를 위한 L2 정규화
    use_idf=True,                   # IDF 가중치 활성화
    smooth_idf=True,                # 0으로 나누기 방지
    sublinear_tf=False,             # 영화 데이터에는 선형 TF가 더 효과적
    token_pattern=r'\b[a-zA-Z][a-zA-Z_]+\b'  # 문자로 시작하는 토큰만 인식
)

print("TF-IDF 행렬 생성 중...")
tfidf_matrix_optimized = tfidf_optimized.fit_transform(df['enhanced_tags'])

print(f"최적화된 TF-IDF 행렬 크기: {tfidf_matrix_optimized.shape}")
print(f"사용된 특성 수: {len(tfidf_optimized.get_feature_names_out())}")

# 모델 저장
print("\n모델 저장 중...")
joblib.dump(tfidf_optimized, '../tfidf_vectorizer_optimized.joblib')
np.save('../tfidf_matrix_optimized.npy', tfidf_matrix_optimized.toarray())
df[['title', 'enhanced_tags']].to_csv('../movies_with_enhanced_tags.csv', index=False)

print("최적화된 TF-IDF 벡터라이저, 행렬, 태그 데이터 저장 완료!")

# 향상된 추천 함수
def get_enhanced_recommendations(title, df, tfidf_matrix, top_k=10, diversity_threshold=0.3):
    """다양성을 고려한 향상된 추천"""
    if title not in df['title'].values:
        print(f"'{title}' 영화를 찾을 수 없습니다.")
        return []
    
    idx = df[df['title'] == title].index[0]
    
    # 코사인 유사도 계산
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    
    # 유사도 점수와 인덱스를 함께 정렬
    sim_indices_scores = [(i, score) for i, score in enumerate(sim_scores) if i != idx]
    sim_indices_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 다양성을 고려한 추천 선택
    recommendations = []
    recommended_genres = set()
    
    original_genres = set()
    if pd.notnull(df.iloc[idx]['listed_in']):
        original_genres = {genre.strip().lower() for genre in df.iloc[idx]['listed_in'].split(',')}
    
    for i, (movie_idx, score) in enumerate(sim_indices_scores):
        if len(recommendations) >= top_k:
            break
            
        movie_genres = set()
        if pd.notnull(df.iloc[movie_idx]['listed_in']):
            movie_genres = {genre.strip().lower() for genre in df.iloc[movie_idx]['listed_in'].split(',')}
        
        # 다양성 체크: 너무 비슷한 장르의 영화만 추천하지 않도록
        genre_overlap = len(recommended_genres & movie_genres) / max(len(movie_genres), 1)
        
        if i < top_k // 2 or genre_overlap < diversity_threshold:  # 상위 절반은 무조건 포함, 나머지는 다양성 고려
            recommendations.append((movie_idx, score))
            recommended_genres.update(movie_genres)
    
    return recommendations

# 개선된 추천 시스템 테스트
print("\n=== 개선된 추천 시스템 테스트 ===")

# 다양한 장르의 영화들로 테스트
test_movies = [
    'Sankofa',                        # Movie
    'The Great British Baking Show',  # TV Show  
    'The Starling',                   # Movie
    'Je Suis Karl',                   # Movie
    'Jeans',                          # Movie
    'Grown Ups',                      # Movie
    'Dark Skies',                     # Movie
    'Paranoia',                       # Movie
    'Birth of the Dragon',            # Movie
    'Jaws'                           # Movie
]

for title in test_movies:
    if title in df['title'].values:
        print(f"\n{'='*60}")
        print(f"🎬 [{title}]와 유사한 영화 TOP 5 (개선된 버전)")
        print(f"{'='*60}")
        
        # 원본 영화 정보
        idx = df[df['title'] == title].index[0]
        original_movie = df.iloc[idx]
        print(f"원본 정보:")
        print(f"  장르: {original_movie['listed_in']}")
        print(f"  설명: {original_movie['description'][:100]}...")
        if original_movie['director']:
            print(f"  감독: {original_movie['director']}")
        print()
        
        recommendations = get_enhanced_recommendations(title, df, tfidf_matrix_optimized, top_k=5)
        
        for rank, (movie_idx, score) in enumerate(recommendations, 1):
            movie = df.iloc[movie_idx]
            print(f"{rank}. {movie['title']} (유사도: {score:.3f})")
            print(f"   장르: {movie['listed_in']}")
            if movie['director']:
                print(f"   감독: {movie['director']}")
            print(f"   설명: {movie['description'][:80]}...")
            print()

# 특성 중요도 분석
print("\n=== TF-IDF 특성 분석 ===")
feature_names = tfidf_optimized.get_feature_names_out()
print(f"총 특성 수: {len(feature_names)}")

# 가장 중요한 특성들 (IDF 값이 낮은 것들 = 자주 나타나는 중요한 단어들)
idf_scores = tfidf_optimized.idf_
feature_importance = list(zip(feature_names, idf_scores))
feature_importance.sort(key=lambda x: x[1])

print("\n가장 일반적인 특성들 (IDF 낮음):")
for feature, idf in feature_importance[:20]:
    print(f"  {feature}: {idf:.3f}")

print("\n가장 특별한 특성들 (IDF 높음):")
for feature, idf in feature_importance[-20:]:
    print(f"  {feature}: {idf:.3f}")

print("\n=== 개선사항 요약 ===")
print("1. ✅ 가중치 기반 태그 생성 (장르 3배, 제목/감독 2배)")
print("2. ✅ 고급 텍스트 전처리 (구두점 제거, 정규화)")
print("3. ✅ 영화 도메인 특화 불용어 제거")
print("4. ✅ n-gram (1,2) 활용으로 문맥 포착")
print("5. ✅ min_df/max_df로 노이즈 및 과도한 일반어 제거")
print("6. ✅ 다양성 고려 추천 알고리즘")
print("7. ✅ 배우, 국가, 연대 정보 활용")
print("\n이제 훨씬 더 정확하고 의미있는 추천이 가능합니다! 🚀")