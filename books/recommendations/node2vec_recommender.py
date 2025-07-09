# Рекомендации на основе графов с использованием Node2Vec эмбеддингов

import numpy as np
import networkx as nx
from node2vec import Node2Vec
from gensim.models import KeyedVectors
from books.models import User, Book, UserBookStatus
import pandas as pd

MODEL_PATH = "books/recommendations/models/node2vec_embeddings.kv"

def build_bipartite_graph():
    G = nx.Graph()
    users = User.objects.values_list('id', flat=True)
    for u in users:
        G.add_node(f'u_{u}', bipartite=0)
    books = Book.objects.values_list('id', flat=True)
    for b in books:
        G.add_node(f'b_{b}', bipartite=1)
    purchases = UserBookStatus.objects.filter(status=UserBookStatus.STATUS_PURCHASED).values_list('user_id', 'book_id')
    for u_id, b_id in purchases:
        G.add_edge(f'u_{u_id}', f'b_{b_id}')
    return G


def train_and_save_node2vec():
    G = build_bipartite_graph()
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    model.wv.save(MODEL_PATH)


def recommend_books_node2vec(user_id, top_n=10):
    model = KeyedVectors.load(MODEL_PATH, mmap='r')
    user_node = f'u_{user_id}'
    if user_node not in model:
        return pd.DataFrame(columns=['book_id', 'score'])
    books = Book.objects.values_list('id', flat=True)
    bought_books = set(UserBookStatus.objects.filter(user_id=user_id, status=UserBookStatus.STATUS_PURCHASED).values_list('book_id', flat=True))
    scores = []
    for b_id in books:
        if b_id in bought_books:
            continue
        book_node = f'b_{b_id}'
        if book_node in model:
            sim = model.similarity(user_node, book_node)
            scores.append((b_id, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    top_recs = scores[:top_n]
    return pd.DataFrame(top_recs, columns=['book_id', 'score'])


def predict_node2vec(user_id, book_ids):
    model = KeyedVectors.load(MODEL_PATH, mmap='r')
    user_node = f'u_{user_id}'
    preds = {}
    if user_node not in model:
        # Возвращаем средний рейтинг 3.0, если пользователя нет в эмбеддингах
        for b in book_ids:
            preds[b] = 3.0
        return preds
    bought_books = set(UserBookStatus.objects.filter(user_id=user_id, status=UserBookStatus.STATUS_PURCHASED).values_list('book_id', flat=True))
    for book_id in book_ids:
        if book_id in bought_books:
            preds[book_id] = np.nan  # или просто не добавлять в preds
            continue
        book_node = f'b_{book_id}'
        if book_node in model:
            sim = model.similarity(user_node, book_node)
            # Нормализация косинусного сходства в диапазон 1-5
            rating = 3 + 2 * sim
            preds[book_id] = rating
        else:
            preds[book_id] = 3.0  # среднее значение, если книга отсутствует
    return preds
