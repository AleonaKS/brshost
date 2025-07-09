from django.core.management.base import BaseCommand
from django.db import models
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity 

from books.models import Book, BookRating, UserBookRecommendation 
from django.core.management.base import BaseCommand

from books.recommendations.collaborative import run_collaborative_filtering
from books.recommendations.content_based import compute_and_store_tfidf_vectors_on_reviews, recommend_books_for_user_content
from books.recommendations.svd_model import train_and_save_svd_model, predict_svd 
from books.recommendations.torch_model import train_torch_model, predict_torch
from books.recommendations.word2_vec import compute_and_store_word2vec_vectors
from books.recommendations.node2vec_recommender import train_and_save_node2vec, predict_node2vec
from django.contrib.auth import get_user_model
User = get_user_model()



class Command(BaseCommand):
    help = 'Генерация модели'
    def handle(self, *args, **kwargs):
        self.stdout.write('Запуск collaborative filtering...')
        run_collaborative_filtering()
        self.stdout.write('Вычисление TF-IDF векторов...')
        compute_and_store_tfidf_vectors_on_reviews()
        self.stdout.write('Обучение и сохранение SVD модели...')
        train_and_save_svd_model()
        self.stdout.write('Обучение модели с помощью torch')
        train_torch_model()
        self.stdout.write('Вычисление и сохранение word2vec векторов...')
        compute_and_store_word2vec_vectors()
        self.stdout.write('Построение графа и обучение node2vec ')
        train_and_save_node2vec()
        self.stdout.write(self.style.SUCCESS('Все модели успешно сгенерированы.'))

 # ----------------------------------------- #   
#         run_collaborative_filtering()
#         def predict(user_id, book_ids):
#             preds = {}
#             for book_id in book_ids:
#                 preds[book_id] = predict_user_based_rating(user_id, book_id)
#             return preds
#         metrics = evaluate_model(predict)
#         self.stdout.write("Метрики оценки коллаборативной модели:")
#         for k, v in metrics.items():
#             self.stdout.write(f"{k}: {v:.4f}")

#         metrics_svd = evaluate_model(predict_svd)
#         self.stdout.write("Метрики SVD модели:")
#         for k, v in metrics_svd.items():
#             self.stdout.write(f"{k}: {v:.5f}")

#         metrics_torch = evaluate_model(predict_torch)
#         self.stdout.write("Метрики Torch модели:")
#         for k, v in metrics_torch.items():
#             self.stdout.write(f"{k}: {v:.5f}")

#         metrics_node2vec = evaluate_model(predict_node2vec)
#         self.stdout.write("Метрики Node2Vec модели:")
#         for k, v in metrics_node2vec.items():
#             self.stdout.write(f"{k}: {v:.5f}")
 


# from books.models import BookRating
# import pandas as pd
# import numpy as np
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# import math


# def predict_user_based_rating(user_id, book_id):
#     #  получаем рекомендацию для пользователя и книги
#     rec = UserBookRecommendation.objects.filter(user_id=user_id, book_id=book_id).first()
#     if rec: 
#         return rec.score
#     # Если рекомендации нет, берем средний рейтинг пользователя
#     user_ratings = BookRating.objects.filter(user_id=user_id)
#     if user_ratings.exists():
#         avg_rating = user_ratings.aggregate(avg_rating=models.Avg('rating'))['avg_rating'] 
#         return avg_rating
#     # Если нет рейтингов пользователя, используем глобальный средний рейтинг
#     global_avg = BookRating.objects.aggregate(avg_rating=models.Avg('rating'))['avg_rating'] 
#     return global_avg or 3.0



# def get_train_test_split(year_cutoff=2025):
#     # Обучающая выборка — рейтинги до year_cutoff
#     train_qs = BookRating.objects.filter(rated_at__year__lt=year_cutoff).values('user_id', 'book_id', 'rating')
#     train_df = pd.DataFrame(train_qs)

#     # Тестовая выборка — рейтинги с year_cutoff и позже
#     test_qs = BookRating.objects.filter(rated_at__year__gte=year_cutoff).values('user_id', 'book_id', 'rating')
#     test_df = pd.DataFrame(test_qs)

#     return train_df, test_df



# def evaluate_model(predict_function, year_cutoff=2025, top_k=10, min_test_ratings=3):
#     """
#     predict_function(user_id, book_ids) -> dict {book_id: predicted_rating}
#     - predict_function должна принимать user_id и список book_id, возвращать предсказания рейтингов
    
#     Возвращает словарь метрик:
#     - RMSE, MAE по всем предсказаниям в тесте
#     - Precision@K, Recall@K, HitRate@K — по top K рекомендациям
#     """
#     train_df, test_df = get_train_test_split(year_cutoff)

#     if test_df.empty:
#         return {'RMSE': np.nan, 'MAE': np.nan, 'Precision@K': np.nan, 'Recall@K': np.nan, 'HitRate@K': np.nan}

#     # Формируем словарь рейтингов для теста: {(user_id, book_id): rating}
#     true_ratings = {(row.user_id, row.book_id): row.rating for row in test_df.itertuples()}

#     users = test_df['user_id'].unique()
#     all_metrics = {
#         'rmse_list': [],
#         'mae_list': [],
#         'precision_list': [],
#         'recall_list': [],
#         'hit_list': [],
#     }

#     for user_id in users:
#         # Книги, которые пользователь оценил в тесте
#         user_test = test_df[test_df['user_id'] == user_id]
        
#         # Фильтр по количеству тестовых рейтингов
#         if len(user_test) < min_test_ratings:
#             continue  # пропускаем пользователей 
        
#         test_books = user_test['book_id'].tolist()
#         true_r = user_test.set_index('book_id')['rating'].to_dict()

#         # Получаем предсказания модели для этих книг
#         preds = predict_function(user_id, test_books)  # {book_id: predicted_rating}

#         # Сопоставляем предсказания и реальные рейтинги
#         y_true = []
#         y_pred = []
#         for b in test_books:
#             y_true.append(true_r[b])
#             y_pred.append(preds.get(b, np.nan)) 

#         # Исключаем пары с np.nan в предсказаниях
#         filtered = [(t, p) for t, p in zip(y_true, y_pred) if not np.isnan(p)]
#         if not filtered:
#             continue
#         y_true_filtered, y_pred_filtered = zip(*filtered)

#         # Регрессия
#         rmse = math.sqrt(mean_squared_error(y_true_filtered, y_pred_filtered))
#         mae = mean_absolute_error(y_true_filtered, y_pred_filtered)

#         all_metrics['rmse_list'].append(rmse)
#         all_metrics['mae_list'].append(mae)
 
#         # Предсказанные топ-K книг по рейтингу
#         sorted_preds = sorted(preds.items(), key=lambda x: x[1], reverse=True)
#         top_k_preds = [b for b, r in sorted_preds[:top_k]]
 
#         true_positives = [b for b, r in true_r.items() if r >= 4]

#         # Precision@K = |top_k_preds ∩ true_positives| / K
#         hits = len(set(top_k_preds) & set(true_positives))
#         precision = hits / top_k if top_k > 0 else 0

#         # Recall@K = |top_k_preds ∩ true_positives| / |true_positives|
#         recall = hits / len(true_positives) if true_positives else 0

#         # HitRate@K = 1 если хотя бы одна из top_k_preds в true_positives, иначе 0
#         hit_rate = 1.0 if hits > 0 else 0.0

#         all_metrics['precision_list'].append(precision)
#         all_metrics['recall_list'].append(recall)
#         all_metrics['hit_list'].append(hit_rate)

#     # Средние метрики по всем пользователям
#     result = {
#         'RMSE': np.mean(all_metrics['rmse_list']) if all_metrics['rmse_list'] else np.nan,
#         'MAE': np.mean(all_metrics['mae_list']) if all_metrics['mae_list'] else np.nan,
#         'Precision@K': np.mean(all_metrics['precision_list']) if all_metrics['precision_list'] else np.nan,
#         'Recall@K': np.mean(all_metrics['recall_list']) if all_metrics['recall_list'] else np.nan,
#         'HitRate@K': np.mean(all_metrics['hit_list']) if all_metrics['hit_list'] else np.nan,
#     }
#     return result







# =============================

# from django.core.management.base import BaseCommand
# from books.models import BookRating
# import pandas as pd
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# import numpy as np

# # Импортируйте ваши функции обучения и предсказания
# from books.recommendations.collaborative import run_collaborative_filtering, evaluate_collaborative_filtering
# from books.recommendations.content_based import compute_and_store_tfidf_vectors_on_reviews, evaluate_content_based
# from books.recommendations.svd import train_and_save_svd_model, evaluate_svd_model
# from books.recommendations.torch_model import train_torch_model, evaluate_torch_model
# from books.recommendations.word2vec import compute_and_store_word2vec_vectors, evaluate_word2vec_model
# from books.recommendations.node2vec import train_and_save_node2vec, evaluate_node2vec_model

# class Command(BaseCommand):
#     help = 'Генерация моделей и вычисление метрик'

#     def handle(self, *args, **kwargs):
#         self.stdout.write('Запуск collaborative filtering...')
#         run_collaborative_filtering()
#         self.stdout.write('Оценка collaborative filtering...')
#         cf_metrics = evaluate_collaborative_filtering()
#         self.print_metrics('Collaborative Filtering', cf_metrics)

#         self.stdout.write('Вычисление TF-IDF векторов...')
#         compute_and_store_tfidf_vectors_on_reviews()
#         self.stdout.write('Оценка content-based модели...')
#         cb_metrics = evaluate_content_based()
#         self.print_metrics('Content-Based', cb_metrics)

#         self.stdout.write('Обучение и сохранение SVD модели...')
#         train_and_save_svd_model()
#         self.stdout.write('Оценка SVD модели...')
#         svd_metrics = evaluate_svd_model()
#         self.print_metrics('SVD', svd_metrics)

#         self.stdout.write('Обучение модели с помощью torch...')
#         train_torch_model()
#         self.stdout.write('Оценка torch модели...')
#         torch_metrics = evaluate_torch_model()
#         self.print_metrics('Torch Model', torch_metrics)

#         self.stdout.write('Вычисление и сохранение word2vec векторов...')
#         compute_and_store_word2vec_vectors()
#         self.stdout.write('Оценка word2vec модели...')
#         w2v_metrics = evaluate_word2vec_model()
#         self.print_metrics('Word2Vec', w2v_metrics)

#         self.stdout.write('Построение графа и обучение node2vec...')
#         train_and_save_node2vec()
#         self.stdout.write('Оценка node2vec модели...')
#         n2v_metrics = evaluate_node2vec_model()
#         self.print_metrics('Node2Vec', n2v_metrics)

#         self.stdout.write(self.style.SUCCESS('Все модели успешно сгенерированы и оценены.'))

#     def print_metrics(self, model_name, metrics_dict):
#         self.stdout.write(f'Метрики для {model_name}:')
#         for metric, value in metrics_dict.items():
#             self.stdout.write(f'  {metric}: {value:.4f}')
#         self.stdout.write('')


# # --- Пример реализации evaluate функций (в вашем модуле recommendations) ---

# def evaluate_collaborative_filtering():
#     """
#     Пример функции оценки CF модели.
#     Возвращает словарь метрик, например RMSE, MAE.
#     """
#     # Загрузите тестовые данные, предсказания модели и вычислите метрики
#     # Здесь заглушка:
#     rmse = 0.95
#     mae = 0.75
#     return {'RMSE': rmse, 'MAE': mae}

# Аналогично реализуйте evaluate_content_based, evaluate_svd_model, evaluate_torch_model, evaluate_word2vec_model, evaluate_node2vec_model








    # def handle(self, *args, **kwargs):
    #     run_collaborative_filtering()
    #     self.stdout.write(self.style.SUCCESS('Рекомендации успешно обновлены'))
    # def handle(self, *args, **options):
    #     try:
    #         rmse = train_and_save_svd_model() # 
    #         self.stdout.write(self.style.SUCCESS(f"Model training finished. RMSE: {rmse:.4f}"))
    #     except Exception as e:
    #         self.stdout.write(self.style.ERROR(f"Error during training: {e}"))
#     def handle(self, *args, **kwargs):
#         compute_and_store_tfidf_vectors_on_reviews()
#         self.stdout.write(self.style.SUCCESS('TF-IDF векторы книг успешно сгенерированы и сохранены'))




# # from books_site.books.recommendations import run_collaborative_filtering  # где лежит ваш код

# class Command(BaseCommand):
#     help = 'Генерация рекомендаций книг (item-based и user-based)'

#     def handle(self, *args, **kwargs):
#         run_collaborative_filtering()
#         self.stdout.write(self.style.SUCCESS('Рекомендации успешно обновлены'))

    # def handle(self, *args, **options):
    #     similarity, book_ids = generate_recommendations()
    #     self.stdout.write(self.style.SUCCESS('Recommendations generated successfully'))

    

    # def handle(self, *args, **options):
    #     ratings = {}
    #     for r in BookRating.objects.exclude(rating__isnull=True).values('user_id', 'book_id', 'rating'):
    #         ratings.setdefault(r['user_id'], {})[r['book_id']] = r['rating']
    #     # Составляем матрицу book-user (книги в строках, пользователи в столбцах)
    #     book_ids = list({b for user_ratings in ratings.values() for b in user_ratings})
    #     user_ids = list(ratings.keys())

    #     book_user_matrix = np.zeros((len(book_ids), len(user_ids)))

    #     book_index = {book_id: i for i, book_id in enumerate(book_ids)}
    #     user_index = {user_id: i for i, user_id in enumerate(user_ids)}

    #     for u_id, user_ratings in ratings.items():
    #         for b_id, rating in user_ratings.items():
    #             book_user_matrix[book_index[b_id], user_index[u_id]] = rating

    #     similarity = cosine_similarity(book_user_matrix)
    #     pass
