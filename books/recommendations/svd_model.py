# Коллаборативная фильтрация (с SVD), основанная на разложении матрицы взаимодействий 
# пользователей и объектов (рейтингов пользователей к книгам)

import os
import pickle
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from books.models import BookRating, Book

MODEL_PATH = "books/recommendations/models/svd_model.pkl"

def train_and_save_svd_model():
    data = []
    for r in BookRating.objects.all():
        if r.rating is None:
            continue
        data.append((str(r.user_id), str(r.book_id), float(r.rating)))
    if not data:
        raise ValueError("No valid rating data found!")
    df = pd.DataFrame(data, columns=['userID', 'itemID', 'rating'])
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(df, reader)
    trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)
    algo = SVD(n_factors=20, n_epochs=30, random_state=42)
    algo.fit(trainset)
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(algo, f)
    return rmse

_algo_cache = None   

def load_svd_model():
    global _algo_cache
    if _algo_cache is not None:
        return _algo_cache
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Модель не найдена. Запустите train_and_save_svd_model.")
    with open(MODEL_PATH, "rb") as f:
        _algo_cache = pickle.load(f)
    return _algo_cache


def get_svd_recommendations_for_user(user_id, top_n=10):
    algo = load_svd_model()
    all_books = Book.objects.all()
    rated_books_ids = set(BookRating.objects.filter(user_id=user_id).values_list('book_id', flat=True))
    predictions = []
    for book in all_books:
        if book.id in rated_books_ids:
            continue
        pred = algo.predict(str(user_id), str(book.id))
        predictions.append((book, pred.est))
    predictions.sort(key=lambda x: x[1], reverse=True)
    return [book for book, _ in predictions[:top_n]]




# для метрик
def predict_svd(user_id, book_ids):
    algo = load_svd_model()   
    preds = {}
    user_str = str(user_id)
    for book_id in book_ids:
        pred = algo.predict(user_str, str(book_id))
        preds[book_id] = pred.est
    return preds
