from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import requests
import os
import re

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터 및 모델 로딩
df = pd.read_csv('preprocessed_video.csv')
# df = df.fillna('')  # NaN 값을 빈 문자열로 처리

# SBERT 임베딩 로드
sbert_embeddings = np.load('sbert_embeddings.npy')

print(f"데이터 로딩 완료 - 영화/TV쇼: {len(df)}개, SBERT 임베딩: {sbert_embeddings.shape}")

def get_video_detail(video_id: int):
    """비디오 상세 정보를 API에서 가져오는 함수"""
    url = f"{os.environ.get('API_URL')}/videos/{video_id}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return data
    
def get_user_history_ids(user_id: int):
    url = f"{os.environ.get('API_URL')}/users/{user_id}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    user_history_ids = [item['id'] for item in data.get('histories', [])]
    return user_history_ids

def get_group_history_ids(group_id: int):
    url = f"{os.environ.get('API_URL')}/groups/{group_id}/histories"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    group_history_ids = [item['id'] for item in data]
    return group_history_ids

def get_sbert_recommendations(history_ids, top_k=10):
    """SBERT 임베딩을 활용한 추천 시스템"""
    # 시청기록 매칭
    watched_mask = np.isin(df['id'].values, history_ids)
    watched_idx = np.where(watched_mask)[0]
    
    if len(watched_idx) == 0:
        return []
    
    # 사용자/그룹 프로필 벡터 생성 (평균)
    profile_vector = sbert_embeddings[watched_idx].mean(axis=0)
    
    # 모든 콘텐츠와의 유사도 계산
    similarities = cosine_similarity([profile_vector], sbert_embeddings)[0]
    
    # 시청한 콘텐츠 제외
    similarities[watched_idx] = -1
    
    # 상위 k개 추천
    top_indices = similarities.argsort()[::-1][:top_k]
    
    recommendations = []
    for idx in top_indices:
        if similarities[idx] > 0:
            recommendations.append({
                'id': int(df.iloc[idx]['id']),
                'title': df.iloc[idx]['title'],
                'director': df.iloc[idx]['director'],
                'rating': df.iloc[idx]['rating'],
                'type': df.iloc[idx]['type']
            })
    
    return recommendations

@app.get("/api/users/{user_id}/recommend")
def recommendForIndividual(user_id: int):
    """SBERT 기반 개인 추천"""
    try:
        user_history_ids = get_user_history_ids(user_id)
        
        if not user_history_ids:
            raise HTTPException(status_code=400, detail="유저 시청 기록에 해당하는 콘텐츠가 없습니다.")
        
        recommendations = get_sbert_recommendations(user_history_ids, top_k=9)
        
        if not recommendations:
            raise HTTPException(status_code=400, detail="추천할 콘텐츠가 없습니다.")
            
        return recommendations
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/groups/{group_id}/recommend")
def recommendForGroup(group_id: int):
    """SBERT 기반 그룹 추천"""
    try:
        group_history_ids = get_group_history_ids(group_id)
        
        if not group_history_ids:
            raise HTTPException(status_code=400, detail="그룹 시청 기록에 해당하는 콘텐츠가 없습니다.")
        
        recommendations = get_sbert_recommendations(group_history_ids, top_k=9)
        
        if not recommendations:
            raise HTTPException(status_code=400, detail="추천할 콘텐츠가 없습니다.")
            
        return recommendations
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
