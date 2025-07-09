import sys
import os

# Получаем абсолютный путь к корню проекта (где manage.py)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))

# Добавляем в sys.path именно корень проекта, чтобы Python видел пакет books_site
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "books_site.settings")
django.setup()

import numpy as np
from collections import defaultdict
from books.models import Review, User, Book, BookVector

from .collaborative import get_similar_books_item_based, get_recommendations_for_user_user_based
from .content_based import get_similar_books_content, recommend_books_for_user_content, get_similar_books_combined
from .svd_model import get_svd_recommendations_for_user 
from .word2_vec import get_similar_books_by_w2v
from .node2vec_recommender import recommend_books_node2vec
from .torch_model import predict_torch
from .hybrid import hybrid_recommendations_for_user, hybrid_recommendations_for_book
from django.db import models
 

import csv
import math 
from statistics import mean
from itertools import combinations

# Метрики

def f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def average_precision(relevant_items, recommended_items):
    hits = 0
    sum_precisions = 0.0
    for i, item in enumerate(recommended_items, start=1):
        if item in relevant_items:
            hits += 1
            sum_precisions += hits / i
    if hits == 0:
        return 0.0
    return sum_precisions / hits

def dcg(relevant_items, recommended_items, k=None):
    dcg_value = 0.0
    for i, item in enumerate(recommended_items[:k], start=1):
        rel = 1 if item in relevant_items else 0
        dcg_value += (2 ** rel - 1) / math.log2(i + 1)
    return dcg_value

def idcg(relevant_items, k=None):
    ideal_rels = [1] * min(len(relevant_items), k if k else len(relevant_items))
    idcg_value = 0.0
    for i, rel in enumerate(ideal_rels, start=1):
        idcg_value += (2 ** rel - 1) / math.log2(i + 1)
    return idcg_value

def ndcg(relevant_items, recommended_items, k=None):
    idcg_value = idcg(relevant_items, k)
    if idcg_value == 0:
        return 0.0
    return dcg(relevant_items, recommended_items, k) / idcg_value

def reciprocal_rank(relevant_items, recommended_items):
    for i, item in enumerate(recommended_items, start=1):
        if item in relevant_items:
            return 1 / i
    return 0.0

# Coverage и Diversity

def coverage(all_recommended_items, all_items):
    unique_recommended = set(all_recommended_items)
    return len(unique_recommended) / len(all_items) if all_items else 0



def distance(book1_id, book2_id):
    try:
        vec1 = np.array(BookVector.objects.get(book_id=book1_id).vector)
        vec2 = np.array(BookVector.objects.get(book_id=book2_id).vector)
    except BookVector.DoesNotExist:
        # Если вектора нет, считаем максимальное расстояние (например, 1)
        return 1.0

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 1.0

    cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
    # расстояние = 1 - сходство
    return 1 - cosine_sim


def diversity(recommended_items):
    if len(recommended_items) < 2:
        return 0.0
    distances = []
    for b1, b2 in combinations(recommended_items, 2):
        distances.append(distance(b1, b2))
    return sum(distances) / len(distances)

# Модификация get_recommendations_all_methods — возвращаем и set и list

def get_recommendations_all_methods(user, TOP_N=10):
    recs = {}

    svd_books = get_svd_recommendations_for_user(user.id, top_n=TOP_N)
    svd_list = [b.id for b in svd_books]
    recs['svd'] = {'set': set(svd_list), 'list': svd_list}

    node2vec_df = recommend_books_node2vec(user.id, top_n=TOP_N)
    node2vec_list = node2vec_df['book_id'].tolist()
    recs['node2vec'] = {'set': set(node2vec_list), 'list': node2vec_list}

    all_book_ids = list(Book.objects.values_list('id', flat=True))
    preds = predict_torch(user.id, all_book_ids)
    sorted_preds = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
    torch_list = [bid for bid, score in sorted_preds]
    recs['torch'] = {'set': set(torch_list), 'list': torch_list}

    hybrid_books = hybrid_recommendations_for_user(user, top_n=TOP_N)
    hybrid_list = [b.id for b in hybrid_books]
    recs['hybrid'] = {'set': set(hybrid_list), 'list': hybrid_list}

    content_books = recommend_books_for_user_content(user, top_n=TOP_N)
    content_list = [b.id for b in content_books]
    recs['content'] = {'set': set(content_list), 'list': content_list}

    collab_books = get_recommendations_for_user_user_based(user.id, top_n=TOP_N)
    collab_list = [b.id for b in collab_books]
    recs['collaborative'] = {'set': set(collab_list), 'list': collab_list}

    read_books = get_read_books_for_user(user.id)
    if read_books:
        w2v_recs = get_similar_books_by_w2v(next(iter(read_books)), top_n=TOP_N)
        w2v_list = [rec['book'].id for rec in w2v_recs]
    else:
        w2v_list = []
    recs['word2vec'] = {'set': set(w2v_list), 'list': w2v_list}

    return recs




def get_relevant_books(user_id, rating_threshold=4):
    relevant = Review.objects.filter(user_id=user_id, rating__gte=rating_threshold).values_list('book_id', flat=True)
    return set(relevant)

def recall_at_k(recommended_books, relevant_books, k):
    recommended_top = list(recommended_books)[:k]
    hits = len(set(recommended_top).intersection(relevant_books))
    if not relevant_books:
        return 0.0
    return hits / len(relevant_books)

def precision_at_k(recommended_books, relevant_books, k):
    recommended_top = list(recommended_books)[:k]
    hits = len(set(recommended_top).intersection(relevant_books))
    if k == 0:
        return 0.0
    return hits / k




def calculate_metrics_for_users_extended(TOP_N_list=[10,20,30,40], year=2025, min_reads=2, rating_threshold=4, csv_path='metrics_results.csv'):
    users_ids = (
        Review.objects.filter(review_date__year=year)
        .values('user')
        .annotate(count=models.Count('book', distinct=True))
        .filter(count__gte=min_reads)
        .values_list('user', flat=True)
    )
    print(f"Найдено пользователей с минимум {min_reads} прочитанными книгами в {year}: {len(users_ids)}")

    all_books = set(Book.objects.values_list('id', flat=True))

    # Для coverage считаем глобально по всем рекомендациям и всем TOP_N
    coverage_per_method_per_k = {k: defaultdict(set) for k in TOP_N_list}

    # Словари для хранения метрик: {TOP_N: {method: [значения по пользователям]}}
    metrics = {k: defaultdict(lambda: defaultdict(list)) for k in TOP_N_list}
    # ключи метрик: recall, precision, f1, map, ndcg, mrr, diversity

    for user_id in users_ids:
        user = User.objects.get(id=user_id)
        read_books = get_read_books_for_user(user_id, year=year)
        if not read_books:
            continue

        relevant = get_relevant_books(user_id, rating_threshold)
        if not relevant:
            continue

        recs_all = get_recommendations_all_methods(user, TOP_N=max(TOP_N_list))

        for k in TOP_N_list:
            for method, recs_dict in recs_all.items():
                rec_set = recs_dict['set']
                rec_list = recs_dict['list'][:k]

                # Обновляем coverage
                coverage_per_method_per_k[k][method].update(rec_list)

                # Метрики
                r = recall_at_k(rec_set, relevant, k)
                p = precision_at_k(rec_set, relevant, k)
                f1 = f1_score(p, r)
                ap = average_precision(relevant, rec_list)
                n = ndcg(relevant, rec_list, k)
                mrr = reciprocal_rank(relevant, rec_list)
                div = diversity(rec_list)

                metrics[k][method]['recall'].append(r)
                metrics[k][method]['precision'].append(p)
                metrics[k][method]['f1'].append(f1)
                metrics[k][method]['map'].append(ap)
                metrics[k][method]['ndcg'].append(n)
                metrics[k][method]['mrr'].append(mrr)
                metrics[k][method]['diversity'].append(div)

    # Подсчёт средних и coverage
    summary = {}
    for k in TOP_N_list:
        summary[k] = {}
        for method in metrics[k]:
            summary[k][method] = {}
            for metric_name, vals in metrics[k][method].items():
                summary[k][method][metric_name] = mean(vals) if vals else 0.0
            # coverage для метода и k
            summary[k][method]['coverage'] = coverage(coverage_per_method_per_k[k][method], all_books)

    # Запись в CSV
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['TOP_N', 'Method', 'Recall', 'Precision', 'F1', 'MAP', 'NDCG', 'MRR', 'Coverage', 'Diversity']
        writer.writerow(header)
        for k in TOP_N_list:
            for method, met_dict in summary[k].items():
                writer.writerow([
                    k,
                    method,
                    f"{met_dict.get('recall',0):.4f}",
                    f"{met_dict.get('precision',0):.4f}",
                    f"{met_dict.get('f1',0):.4f}",
                    f"{met_dict.get('map',0):.4f}",
                    f"{met_dict.get('ndcg',0):.4f}",
                    f"{met_dict.get('mrr',0):.4f}",
                    f"{met_dict.get('coverage',0):.4f}",
                    f"{met_dict.get('diversity',0):.4f}",
                ])

    print(f"Метрики посчитаны и сохранены в {csv_path}")
    return summary

def get_read_books_for_user(user_id, year=2025):
    book_ids = (
        Review.objects.filter(user_id=user_id, review_date__year=year)
        .values_list('book_id', flat=True)
        .distinct()
    )
    return set(book_ids)

# def get_recommendations_all_methods(user, TOP_N=10):
#     recs = {}

#     svd_books = get_svd_recommendations_for_user(user.id, top_n=TOP_N)
#     recs['svd'] = set(b.id for b in svd_books)

#     node2vec_df = recommend_books_node2vec(user.id, top_n=TOP_N)
#     recs['node2vec'] = set(node2vec_df['book_id'].tolist())

#     all_book_ids = list(Book.objects.values_list('id', flat=True))
#     preds = predict_torch(user.id, all_book_ids)
#     sorted_preds = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
#     recs['torch'] = set(bid for bid, score in sorted_preds)

#     hybrid_books = hybrid_recommendations_for_user(user, top_n=TOP_N)
#     recs['hybrid'] = set(b.id for b in hybrid_books)

#     content_books = recommend_books_for_user_content(user, top_n=TOP_N)
#     recs['content'] = set(b.id for b in content_books)

#     collab_books = get_recommendations_for_user_user_based(user.id, top_n=TOP_N)
#     recs['collaborative'] = set(b.id for b in collab_books)

#     read_books = get_read_books_for_user(user.id)
#     if read_books:
#         w2v_recs = get_similar_books_by_w2v(next(iter(read_books)), top_n=TOP_N)
#         recs['word2vec'] = set(rec['book'].id for rec in w2v_recs)
#     else:
#         recs['word2vec'] = set()

#     return recs

# def compare_recommendations_for_user(user, TOP_N=10):
#     recs = get_recommendations_all_methods(user, TOP_N)
#     methods = list(recs.keys())
#     print(f"Сравнение рекомендаций для пользователя {user.id}:")
#     for i in range(len(methods)):
#         for j in range(i+1, len(methods)):
#             m1, m2 = methods[i], methods[j]
#             inter = recs[m1].intersection(recs[m2])
#             union = recs[m1].union(recs[m2])
#             jaccard = len(inter) / len(union) if union else 0
#             print(f"  {m1} vs {m2}: пересечение {len(inter)}, Jaccard {jaccard:.3f}")
#     return recs

# def compare_similar_books(book, top_n=10):
#     recs = {}
#     item_based = get_similar_books_item_based(book.id, top_n=top_n)
#     recs['item_based'] = set(b.id for b in item_based)

#     content_based = get_similar_books_content(book, top_n=top_n)
#     recs['content_based'] = set(b.id for b in content_based)

#     w2v_recs = get_similar_books_by_w2v(book.id, top_n=top_n)
#     recs['word2vec'] = set(rec['book'].id for rec in w2v_recs)

#     hybrid_recs = hybrid_recommendations_for_book(book, top_n=top_n)
#     recs['hybrid'] = set(b.id for b in hybrid_recs)

#     print(f"Сравнение похожих книг для книги {book.id}:")
#     methods = list(recs.keys())
#     for i in range(len(methods)):
#         for j in range(i+1, len(methods)):
#             m1, m2 = methods[i], methods[j]
#             inter = recs[m1].intersection(recs[m2])
#             union = recs[m1].union(recs[m2])
#             jaccard = len(inter) / len(union) if union else 0
#             print(f"  {m1} vs {m2}: пересечение {len(inter)}, Jaccard {jaccard:.3f}")
#     return recs



# def calculate_metrics_for_users(TOP_N=10, year=2025, min_reads=2, rating_threshold=4):
#     users_ids = (
#         Review.objects.filter(review_date__year=year)
#         .values('user')
#         .annotate(count=models.Count('book', distinct=True))
#         .filter(count__gte=min_reads)
#         .values_list('user', flat=True)
#     )
#     print(f"Найдено пользователей с минимум {min_reads} прочитанными книгами в {year}: {len(users_ids)}")

#     results_recall = defaultdict(list)
#     results_precision = defaultdict(list)

#     for user_id in users_ids:
#         user = User.objects.get(id=user_id)
#         read_books = get_read_books_for_user(user_id, year=year)
#         if not read_books:
#             continue

#         recs = get_recommendations_all_methods(user, TOP_N)

#         relevant = get_relevant_books(user_id, rating_threshold)

#         for method, rec_books in recs.items():
#             r = recall_at_k(rec_books, relevant, TOP_N)
#             p = precision_at_k(rec_books, relevant, TOP_N)
#             results_recall[method].append(r)
#             results_precision[method].append(p)

#     avg_recall = {m: (sum(vals)/len(vals) if vals else 0) for m, vals in results_recall.items()}
#     avg_precision = {m: (sum(vals)/len(vals) if vals else 0) for m, vals in results_precision.items()}

#     print("Средние Recall@{}:".format(TOP_N))
#     for m, val in avg_recall.items():
#         print(f"  {m}: {val:.4f}")
#     print("Средние Precision@{}:".format(TOP_N))
#     for m, val in avg_precision.items():
#         print(f"  {m}: {val:.4f}")

#     return avg_recall, avg_precision




# def jaccard_similarity(set_a, set_b):
#     intersection = len(set_a.intersection(set_b))
#     union = len(set_a.union(set_b))
#     return intersection / union if union > 0 else 0

# def overlap_coefficient(set_a, set_b):
#     intersection = len(set_a.intersection(set_b))
#     min_size = min(len(set_a), len(set_b))
#     return intersection / min_size if min_size > 0 else 0

# def weighted_recall(user_true_books_with_ratings, recommended_books):
#     relevant_books = set(user_true_books_with_ratings.keys())
#     recommended_set = set(recommended_books)
#     intersection = relevant_books.intersection(recommended_set)
#     if not relevant_books:
#         return 0
#     sum_ratings_intersection = sum(user_true_books_with_ratings[b] for b in intersection)
#     sum_ratings_all = sum(user_true_books_with_ratings.values())
#     return sum_ratings_intersection / sum_ratings_all if sum_ratings_all > 0 else 0

# def get_user_ratings(user_id, rating_threshold=4):
#     # Возвращает словарь book_id -> rating для книг с рейтингом >= threshold
#     qs = Review.objects.filter(user_id=user_id, rating__gte=rating_threshold)
#     return {r.book_id: r.rating for r in qs}

# def compare_methods_for_user(user, TOP_N=10, rating_threshold=4):
#     recs = get_recommendations_all_methods(user, TOP_N)
#     user_ratings = get_user_ratings(user.id, rating_threshold)

#     methods = list(recs.keys())
#     jaccard_scores = defaultdict(list)
#     overlap_scores = defaultdict(list)
#     weighted_recalls = defaultdict(list)

#     # weighted recall для каждого метода
#     for method in methods:
#         wrec = weighted_recall(user_ratings, recs[method])
#         weighted_recalls[method].append(wrec)

#     # пары методов для Jaccard и Overlap
#     for i in range(len(methods)):
#         for j in range(i+1, len(methods)):
#             m1, m2 = methods[i], methods[j]
#             jacc = jaccard_similarity(recs[m1], recs[m2])
#             ovlp = overlap_coefficient(recs[m1], recs[m2])
#             jaccard_scores[(m1, m2)].append(jacc)
#             overlap_scores[(m1, m2)].append(ovlp)

#     return jaccard_scores, overlap_scores, weighted_recalls

# def aggregate_metrics_over_users(user_ids, TOP_N=10, rating_threshold=4):
#     all_jaccard = defaultdict(list)
#     all_overlap = defaultdict(list)
#     all_weighted_recall = defaultdict(list)

#     for user_id in user_ids:
#         user = User.objects.get(id=user_id)
#         jaccard_scores, overlap_scores, weighted_recalls = compare_methods_for_user(user, TOP_N, rating_threshold)

#         for k, v in jaccard_scores.items():
#             all_jaccard[k].extend(v)
#         for k, v in overlap_scores.items():
#             all_overlap[k].extend(v)
#         for m, vals in weighted_recalls.items():
#             all_weighted_recall[m].extend(vals)

#     def summarize(d):
#         return {
#             k: {
#                 'min': min(vals),
#                 'max': max(vals),
#                 'mean': mean(vals)
#             } for k, vals in d.items()
#         }

#     return summarize(all_jaccard), summarize(all_overlap), summarize(all_weighted_recall)

# # Пример вызова
# users_ids = list(
#     Review.objects.values('user')
#     .annotate(count=models.Count('book', distinct=True))
#     .filter(count__gte=2)
#     .values_list('user', flat=True)
# )

# jaccard_stats, overlap_stats, weighted_recall_stats = aggregate_metrics_over_users(users_ids, TOP_N=20)

# print("Jaccard similarity между методами:")
# for pair, stats in jaccard_stats.items():
#     print(f"{pair}: min={stats['min']:.3f}, max={stats['max']:.3f}, mean={stats['mean']:.3f}")

# print("\nOverlap coefficient между методами:")
# for pair, stats in overlap_stats.items():
#     print(f"{pair}: min={stats['min']:.3f}, max={stats['max']:.3f}, mean={stats['mean']:.3f}")

# print("\nWeighted Recall для каждого метода:")
# for method, stats in weighted_recall_stats.items():
#     print(f"{method}: min={stats['min']:.3f}, max={stats['max']:.3f}, mean={stats['mean']:.3f}")
    
# ======================================
# if __name__ == "__main__":
#     TOP_N = 30
#     calculate_metrics_for_users(TOP_N=TOP_N, year=2025, min_reads=2)

# # TOP_N = 20

# def get_users_with_min_reads(year=2025, min_reads=2):
#     # Находим пользователей, прочитавших минимум min_reads книг в заданном году
#     user_reads = (
#         Review.objects.filter(review_date__year=year)
#         .values('user')
#         .annotate(count=models.Count('book', distinct=True))
#         .filter(count__gte=min_reads)
#         .values_list('user', flat=True)
#     )
#     return list(user_reads)

# def get_read_books_for_user(user_id, year=2025):
#     # Получаем id книг, прочитанных пользователем в году
#     book_ids = (
#         Review.objects.filter(user_id=user_id, review_date__year=year)
#         .values_list('book_id', flat=True)
#         .distinct()
#     )
#     return set(book_ids)


# def get_recommendations_all_methods(user, TOP_N):
#     # Получаем рекомендации разными методами, возвращаем set id книг
#     recs = {}

#     # SVD
#     svd_books = get_svd_recommendations_for_user(user.id, top_n=TOP_N)
#     recs['svd'] = set(b.id for b in svd_books)

#     # Node2Vec
#     node2vec_df = recommend_books_node2vec(user.id, top_n=TOP_N)
#     recs['node2vec'] = set(node2vec_df['book_id'].tolist())

#     # Torch model
#     # Используем predict_torch для получения оценок, затем топ-N
#     all_book_ids = list(Book.objects.values_list('id', flat=True))
#     preds = predict_torch(user.id, all_book_ids)
#     sorted_preds = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
#     recs['torch'] = set(bid for bid, score in sorted_preds)

#     # Hybrid
#     hybrid_books = hybrid_recommendations_for_user(user, top_n=TOP_N)
#     recs['hybrid'] = set(b.id for b in hybrid_books)

#     # Content-based
#     content_books = recommend_books_for_user_content(user, top_n=TOP_N)
#     recs['content'] = set(b.id for b in content_books)

#     # Collaborative user-based
#     collab_books = get_recommendations_for_user_user_based(user.id, top_n=TOP_N)
#     recs['collaborative'] = set(b.id for b in collab_books)

#     # Word2Vec - для пользователя нет прямого метода, можно по прочитанным книгам брать похожие
#     # Например, берем похожие книги к первой прочитанной книге
#     read_books = get_read_books_for_user(user.id)
#     if read_books:
#         w2v_recs = get_similar_books_by_w2v(next(iter(read_books)), top_n=TOP_N)
#         recs['word2vec'] = set(rec['book'].id for rec in w2v_recs)
#     else:
#         recs['word2vec'] = set()

#     return recs

# def calculate_metrics_for_users(TOP_N, year=2025, min_reads=2):
#     users_ids = get_users_with_min_reads(year=year, min_reads=min_reads)
#     print(f"Найдено пользователей с минимум {min_reads} прочитанными книгами в {year}: {len(users_ids)}")

#     results = defaultdict(list)  # метод -> list of ratios

#     for user_id in users_ids:
#         user = User.objects.get(id=user_id)
#         read_books = get_read_books_for_user(user_id, year=year)
#         if not read_books:
#             continue

#         recs = get_recommendations_all_methods(user, TOP_N)

#         for method, rec_books in recs.items():
#             # Считаем сколько прочитанных книг входят в топ-N рекомендаций
#             hits = len(read_books.intersection(rec_books))
#             recall = hits / len(read_books)  # доля прочитанных книг, покрытая рекомендациями
#             results[method].append(recall)

#     # Средние метрики по методам
#     avg_results = {}
#     for method, recalls in results.items():
#         avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
#         avg_results[method] = avg_recall

#     return avg_results


# if __name__ == "__main__":
#     metrics = calculate_metrics_for_users(year=2025, min_reads=2)
#     # for method, avg_recall in metrics.items():
#     #     print(f"Метод: {method}, Средний Recall@{TOP_N}: {avg_recall:.4f}")
