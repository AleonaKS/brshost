# Коллаборативная фильтрация user-based и item-based

from django.db import models
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from django.db.models import Q
from books.models import BookRating, UserBookRecommendation, BookRecommendation

import logging
logger = logging.getLogger(__name__)

def build_user_book_matrix():
    ratings_qs = BookRating.objects.all().values('user_id', 'book_id', 'rating')
    df = pd.DataFrame(ratings_qs)
    if df.empty:
        return pd.DataFrame()
    user_book_matrix = df.pivot(index='user_id', columns='book_id', values='rating').fillna(0)
    return user_book_matrix

def calculate_user_similarity_matrix(user_book_matrix):
    sim_matrix = cosine_similarity(user_book_matrix)
    user_ids = list(user_book_matrix.index)
    return sim_matrix, user_ids

def generate_user_based_recommendations(user_book_matrix, user_similarity_matrix, user_ids, top_n=10):
    UserBookRecommendation.objects.all().delete()
    ratings = user_book_matrix.values
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    for user_id in user_ids:
        u_idx = user_id_to_idx[user_id]
        sim_scores = user_similarity_matrix[u_idx]
        sim_scores[u_idx] = 0
        top_k_idx = sim_scores.argsort()[::-1][:10]
        top_k_sim = sim_scores[top_k_idx]
        weights = top_k_sim / top_k_sim.sum() if top_k_sim.sum() > 0 else np.zeros_like(top_k_sim)
        top_k_ratings = ratings[top_k_idx]
        pred_ratings = np.dot(weights, top_k_ratings)
        user_rated = user_book_matrix.loc[user_id]
        unrated_mask = user_rated == 0
        candidate_indices = np.where(unrated_mask)[0]
        recommended_indices = candidate_indices[np.argsort(pred_ratings[unrated_mask])[::-1][:top_n]]
        for book_idx in recommended_indices:
            book_id = user_book_matrix.columns[book_idx]
            score = pred_ratings[book_idx]
            UserBookRecommendation.objects.create(user_id=user_id, book_id=book_id, score=score)

def find_similar_books_knn(user_book_matrix, top_n=10):
    book_user_matrix = user_book_matrix.T.values
    book_ids = list(user_book_matrix.columns)
    knn = NearestNeighbors(n_neighbors=top_n+1, metric='cosine')
    knn.fit(book_user_matrix)
    distances, indices = knn.kneighbors(book_user_matrix)
    BookRecommendation.objects.all().delete()
    for i, book_id in enumerate(book_ids):
        for j in range(1, top_n+1):
            neighbor_idx = indices[i][j]
            similarity = 1 - distances[i][j]
            recommended_book_id = book_ids[neighbor_idx]
            BookRecommendation.objects.create(
                book_id=book_id,
                recommended_book_id=recommended_book_id,
                similarity=similarity
            )

def run_collaborative_filtering():
    user_book_matrix = build_user_book_matrix()
    if user_book_matrix.empty:
        print("Нет данных для коллаборативной фильтрации")
        return
    find_similar_books_knn(user_book_matrix, top_n=10)
    user_sim_matrix, user_ids = calculate_user_similarity_matrix(user_book_matrix)
    generate_user_based_recommendations(user_book_matrix, user_sim_matrix, user_ids)
    print(f"Рекомендации обновлены: {len(user_book_matrix.columns)} книг (item-based), {len(user_ids)} пользователей (user-based)")

def get_similar_books_item_based(book_id, top_n=5):
    recs = BookRecommendation.objects.filter(book_id=book_id).order_by('-similarity')[:top_n]
    return [rec.recommended_book for rec in recs]

def get_recommendations_for_user_user_based(user_id, top_n=5):
    recs = UserBookRecommendation.objects.filter(user_id=user_id).order_by('-score')[:top_n]
    return [rec.book for rec in recs]


