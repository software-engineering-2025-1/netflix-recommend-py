from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import requests
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df = pd.read_csv('video.csv')
genre_matrix = np.load('genre_matrix.npy')

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

@app.get("/api/users/{user_id}/recommend")
def recommendForIndividual(user_id: int):

    user_history_ids = get_user_history_ids(user_id)
    watched_mask = np.isin(df['id'].values, user_history_ids)
    watched_idx = np.where(watched_mask)[0]

    if len(watched_idx) == 0:
        raise HTTPException(status_code=400, detail="유저 시청 기록에 해당하는 콘텐츠가 없습니다.")

    user_profile_vector = genre_matrix[watched_idx].mean(axis=0)
    similarities = cosine_similarity([user_profile_vector], genre_matrix)[0]
    df['similarity'] = similarities

    recommendations = df[~df['id'].isin(user_history_ids)].sort_values(by='similarity', ascending=False).head(9)

    result = recommendations[['id', 'title', 'director', 'rating', 'type']].to_dict(orient='records')

    return result

@app.get("/api/groups/{group_id}/recommend")
def recommendForGroup(group_id: int):

    group_history_ids = get_group_history_ids(group_id)
    watched_mask = np.isin(df['id'].values, group_history_ids)
    watched_idx = np.where(watched_mask)[0]

    if len(watched_idx) == 0:
        raise HTTPException(status_code=400, detail="그룹 시청 기록에 해당하는 콘텐츠가 없습니다.")

    group_profile_vector = genre_matrix[watched_idx].mean(axis=0)
    similarities = cosine_similarity([group_profile_vector], genre_matrix)[0]
    df['similarity'] = similarities

    recommendations = df[~df['id'].isin(group_history_ids)].sort_values(by='similarity', ascending=False).head(9)

    result = recommendations[['id', 'title', 'director', 'rating', 'type']].to_dict(orient='records')

    return result
