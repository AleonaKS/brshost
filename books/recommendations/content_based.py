# Контентные рекомендации на основе жанра, авторов, тегов и TF-IDF

import numpy as np
from django.db.models import Q, Count, Case, When, Value, FloatField, F
from books.models import Book, Author, UserPreferences, BookVector
from sklearn.feature_extraction.text import TfidfVectorizer
from django.db import transaction

def compute_and_store_tfidf_vectors_on_reviews():
    books = Book.objects.prefetch_related('review_set').all()
    reviews_texts = []
    book_ids = []

    for book in books:
        reviews = book.review_set.all()
        combined_reviews = ' '.join([r.text for r in reviews if r.text])
        if combined_reviews.strip():
            reviews_texts.append(combined_reviews)
            book_ids.append(book.id)

    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(reviews_texts)

    with transaction.atomic():
        for idx, book_id in enumerate(book_ids):
            vector = tfidf_matrix[idx].tocoo()
            indices = vector.col.tolist()
            values = vector.data.tolist()
            BookVector.objects.update_or_create(
                book_id=book_id,
                defaults={'indices': indices, 'values': values}
            )

def get_similar_books_content(book: Book, top_n=10):
    genre = book.genre
    authors = book.author.all()
    tags = book.tags.all()
    qs = Book.objects.exclude(id=book.id).filter(
        Q(genre=genre) | Q(author__in=authors) | Q(tags__in=tags)
    ).distinct()
    qs = qs.annotate(
        same_genre=Count('genre', filter=Q(genre=genre)),
        same_authors=Count('author', filter=Q(author__in=authors)),
        same_tags=Count('tags', filter=Q(tags__in=tags)),
    )
    qs = qs.annotate(
        similarity_score=3 * F('same_genre') + 2 * F('same_authors') + 1 * F('same_tags')
    ).order_by('-similarity_score', '-rating_chitai_gorod', '-votes_chitai_gorod')
    return qs[:top_n]


def recommend_books_for_user_content(user, top_n=10):
    try:
        prefs = user.userpreferences
    except UserPreferences.DoesNotExist:
        return Book.objects.none()
    favorite_genres = prefs.favorite_genres.all()
    favorite_authors = Author.objects.filter(favoriteauthors__userprofile=prefs)
    favorite_tags = prefs.favorite_tags.all()
    qs = Book.objects.filter(
        Q(genre__in=favorite_genres) | Q(author__in=favorite_authors) | Q(tags__in=favorite_tags)
    ).distinct()
    qs = qs.annotate(
        genre_match=Count('genre', filter=Q(genre__in=favorite_genres)),
        author_match=Count('author', filter=Q(author__in=favorite_authors)),
        tag_match=Count('tags', filter=Q(tags__in=favorite_tags)),
    )
    qs = qs.annotate(
        score=3 * F('genre_match') + 2 * F('author_match') + 1 * F('tag_match')
    ).order_by('-score', '-rating_chitai_gorod', '-votes_chitai_gorod')
    return qs[:top_n]


def cosine_sim_sparse(indices1, values1, indices2, values2):
    dict1 = dict(zip(indices1, values1))
    dict2 = dict(zip(indices2, values2))
    keys = set(dict1.keys()).union(dict2.keys())
    v1 = np.array([dict1.get(k, 0.0) for k in keys])
    v2 = np.array([dict2.get(k, 0.0) for k in keys])
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


# noML (вынести в отдельный файл)
def get_similar_books_combined(book: Book, top_n=10,
                               weight_genre=3, weight_author=2, weight_tag=1, weight_tfidf=5):
    genre = book.genre
    authors = book.author.all()
    tags = book.tags.all()
    qs = Book.objects.exclude(id=book.id).filter(
        Q(genre=genre) | Q(author__in=authors) | Q(tags__in=tags)
    ).distinct().prefetch_related('author', 'tags')

    qs = qs.annotate(
        same_genre=Count('genre', filter=Q(genre=genre)),
        same_authors=Count('author', filter=Q(author__in=authors)),
        same_tags=Count('tags', filter=Q(tags__in=tags)),
    )

    try:
        base_vector = book.vector
        base_indices, base_values = base_vector.indices, base_vector.values
    except BookVector.DoesNotExist:
        base_indices, base_values = [], []

    tfidf_scores = {}
    vectors = BookVector.objects.filter(book__in=qs).values('book_id', 'indices', 'values')
    for v in vectors:
        score = cosine_sim_sparse(base_indices, base_values, v['indices'], v['values'])
        tfidf_scores[v['book_id']] = score

    whens = [When(id=book_id, then=Value(score)) for book_id, score in tfidf_scores.items()]
    qs = qs.annotate(
        tfidf_score=Case(*whens, default=Value(0.0), output_field=FloatField())
    )

    qs = qs.annotate(
        similarity_score=weight_genre * F('same_genre') +
                         weight_author * F('same_authors') +
                         weight_tag * F('same_tags') +
                         weight_tfidf * F('tfidf_score')
    ).order_by('-similarity_score', '-rating_chitai_gorod', '-votes_chitai_gorod')

    return qs[:top_n]
