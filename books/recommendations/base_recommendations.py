import numpy as np
from books.models import Embedding, Author, Genre, Tag, BookEmbedding, Book
from django.db.models import Q, Sum, F, FloatField, ExpressionWrapper, Case, When, Value, IntegerField, Max
from django.db.models.functions import Log
from books.models import Book, BookView, UserSearchQuery, UserPreferences, UserBookStatus, UserSubscription
from collections import defaultdict
from django.utils.timezone import now
from datetime import timedelta 
import Levenshtein

def compute_and_store_entity_embeddings():
    # Авторские эмбеддинги
    for author in Author.objects.all():
        book_embs = BookEmbedding.objects.filter(book__author=author).values_list('vector', flat=True)
        vectors = [np.array(v) for v in book_embs if v]
        if not vectors:
            continue
        mean_vec = np.mean(vectors, axis=0)
        Embedding.objects.update_or_create(
            content_type='author',
            object_id=author.id,
            defaults={'vector': mean_vec.tolist()}
        )

    # Жанровые эмбеддинги
    from django.contrib.contenttypes.models import ContentType

    genre_ct = ContentType.objects.get_for_model(Genre)

    for genre in Genre.objects.all():
        book_embs = BookEmbedding.objects.filter(book__genre=genre).values_list('vector', flat=True)
        vectors = []
        for v in book_embs:
            if v:
                vec = np.array(v)   
                vectors.append(vec)
        if not vectors:
            continue
        mean_vec = np.mean(vectors, axis=0)

        Embedding.objects.update_or_create(
            content_type=genre_ct,
            object_id=genre.id,
            defaults={'vector': mean_vec.tolist()}
        )


    # Теговые эмбеддинги
    for tag in Tag.objects.all():
        book_embs = BookEmbedding.objects.filter(book__tags=tag).values_list('vector', flat=True)
        vectors = [np.array(v) for v in book_embs if v]
        if not vectors:
            continue
        mean_vec = np.mean(vectors, axis=0)
        Embedding.objects.update_or_create(
            content_type='tag',
            object_id=tag.id,
            defaults={'vector': mean_vec.tolist()}
        )







  

HALF_LIFE_DAYS = 7  # время полураспада для экспоненциального затухания
EMBEDDING_SIZE = 100  # размер эмбеддингов


def decay_weight(viewed_at):
    delta_days = (now() - viewed_at).days
    return 2 ** (-delta_days / HALF_LIFE_DAYS)


def cosine_similarity(a, b):
    if a is None or b is None:
        return 0.0
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def get_embedding_vectors(content_type, ids):
    from books.models import Embedding  
    embs_qs = Embedding.objects.filter(content_type=content_type, object_id__in=ids).values_list('vector', flat=True)
    vectors = []
    for v in embs_qs:
        if v:
            vectors.append(np.array(v))
    return vectors


def get_user_embedding(user, favorite_genre_ids, favorite_tag_ids):
    author_ids = list(user.usersubscription_set.filter(content_type='AUTHOR').values_list('author_id', flat=True))
    vectors = []
    vectors.extend(get_embedding_vectors('author', author_ids))
    vectors.extend(get_embedding_vectors('genre', favorite_genre_ids))
    vectors.extend(get_embedding_vectors('tag', favorite_tag_ids))
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


def get_book_embedding(book):
    vectors = [] 
    author_ids = list(book.author.values_list('id', flat=True))
    vectors.extend(get_embedding_vectors('author', author_ids))
    if book.genre_id:
        vectors.extend(get_embedding_vectors('genre', [book.genre_id]))
    tag_ids = list(book.tags.values_list('id', flat=True))
    vectors.extend(get_embedding_vectors('tag', tag_ids))
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


def recommendations_split(user, favorite_genre_ids, favorite_tag_ids, top_n=20):  
    # Исключаем купленные книги (статус PURCHASED)
    purchased_book_ids = set(
        UserBookStatus.objects.filter(user=user, status=UserBookStatus.STATUS_PURCHASED)
        .values_list('book_id', flat=True)
    )
    # Получаем последние просмотры пользователя 
    recent_views = BookView.objects.filter(user=user).order_by('-viewed_at')
    weighted_views = defaultdict(float)
    for view in recent_views:
        w = decay_weight(view.viewed_at)
        scroll = getattr(view, 'scroll_depth', 1.0) or 1.0
        weighted_views[view.book_id] += view.duration_seconds * scroll * w
    # Получаем поисковые запросы пользователя за последние 30 дней
    search_period = now() - timedelta(days=30)
    user_queries = UserSearchQuery.objects.filter(user=user, created_at__gte=search_period)
    book_search_counts = defaultdict(int)
    # Для ускорения ограничиваем книги только просмотренными
    candidate_book_ids = list(weighted_views.keys())
    books = Book.objects.filter(id__in=candidate_book_ids)
    for book in books:
        count = user_queries.filter(query_text__icontains=book.title).count()
        book_search_counts[book.id] = count
    # Получаем queryset книг, исключая купленные
    qs = Book.objects.exclude(id__in=purchased_book_ids)
    # Формируем эмбеддинг пользователя
    user_emb = get_user_embedding(user, favorite_genre_ids, favorite_tag_ids)
    # Ограничиваем кандидатов для производительности
    candidate_books = qs.filter(id__in=candidate_book_ids)[:1000]
    scored_books = []
    for book in candidate_books:
        base_score = weighted_views.get(book.id, 0.0)
        search_bonus = book_search_counts.get(book.id, 0) * 10   
        book_emb = get_book_embedding(book)
        emb_sim = cosine_similarity(user_emb, book_emb)
        total_score = base_score + search_bonus + emb_sim * 50  
        scored_books.append((book, total_score))

    scored_books.sort(key=lambda x: x[1], reverse=True)
    popular_books = [b for b, s in scored_books if not b.new and not b.soon][:top_n]
    new_books = [b for b, s in scored_books if b.new][:top_n]
    soon_books = [b for b, s in scored_books if b.soon][:top_n]
        # Если new_books пустой, дополняем книгами с new=True из всей базы (исключая купленные)
    if not new_books:
        extra_new_books = list(qs.filter(new=True)[:top_n])
        new_books = extra_new_books[:top_n]
    # Если soon_books пустой, дополняем книгами с soon=True из всей базы (исключая купленные)
    if not soon_books:
        extra_soon_books = list(qs.filter(soon=True)[:top_n])
        soon_books = extra_soon_books[:top_n]
    return popular_books, new_books, soon_books










def recommendations_for_anonymous(session_key, top_n=20):
    # Получаем просмотры по сессии
    recent_views = BookView.objects.filter(user__isnull=True, session_key=session_key).order_by('-viewed_at')[:500]

    weighted_views = defaultdict(float)
    HALF_LIFE_DAYS = 7

    def decay_weight(viewed_at):
        delta_days = (now() - viewed_at).days
        return 2 ** (-delta_days / HALF_LIFE_DAYS)

    for view in recent_views:
        w = decay_weight(view.viewed_at)
        scroll = getattr(view, 'scroll_depth', 1.0) or 1.0
        weighted_views[view.book_id] += view.duration_seconds * scroll * w

    candidate_book_ids = list(weighted_views.keys())
    books = Book.objects.filter(id__in=candidate_book_ids)

    # Поисковые запросы за последние 30 дней по сессии
    search_period = now() - timedelta(days=30)
    user_queries = UserSearchQuery.objects.filter(user__isnull=True, session_key=session_key, created_at__gte=search_period)

    book_search_counts = defaultdict(int)
    for book in books:
        count = user_queries.filter(query_text__icontains=book.title).count()
        book_search_counts[book.id] = count

    scored_books = []
    for book in books:
        base_score = weighted_views.get(book.id, 0.0)
        search_bonus = book_search_counts.get(book.id, 0) * 10
        total_score = base_score + search_bonus
        scored_books.append((book, total_score))

    scored_books.sort(key=lambda x: x[1], reverse=True)
    popular_books = [b for b, s in scored_books if not b.new and not b.soon][:top_n]
    new_books = [b for b, s in scored_books if b.new][:top_n]
    soon_books = [b for b, s in scored_books if b.soon][:top_n]

    # Если мало книг из истории, можно добавить просто топовые книги
    if len(popular_books) < top_n:
        extra = Book.objects.filter(new=False, soon=False).order_by('-rating_chitai_gorod')[:top_n - len(popular_books)]
        popular_books.extend(extra)

    if len(new_books) < top_n:
        extra_new = Book.objects.filter(new=True).order_by('-rating_chitai_gorod')[:top_n - len(new_books)]
        new_books.extend(extra_new)

    if len(soon_books) < top_n:
        extra_soon = Book.objects.filter(soon=True).order_by('-rating_chitai_gorod')[:top_n - len(soon_books)]
        soon_books.extend(extra_soon)

    return popular_books[:top_n], new_books[:top_n], soon_books[:top_n]





def recommendations_for_user_without_preferences(user, top_n=30):
    # Получаем последние просмотры пользователя (ID книг)
    recent_views = BookView.objects.filter(user=user).order_by('-viewed_at')
    viewed_book_ids = set(recent_views.values_list('book_id', flat=True))

    # Собираем жанры и теги просмотренных книг
    viewed_books = Book.objects.filter(id__in=viewed_book_ids).prefetch_related('genre', 'tags')

    genre_ids = set()
    tag_ids = set()

    for book in viewed_books:
        if book.genre_id:
            genre_ids.add(book.genre_id)
        tag_ids.update(book.tags.values_list('id', flat=True))

    # Поисковые запросы за последние 30 дней пользователя
    search_period = now() - timedelta(days=30)
    user_queries = UserSearchQuery.objects.filter(user=user, created_at__gte=search_period)

    # Получаем кандидатов — книги, похожие по жанрам или тегам, исключая просмотренные
    candidates = Book.objects.filter(
        (Q(genre__id__in=genre_ids) | Q(tags__id__in=tag_ids))
    ).exclude(id__in=viewed_book_ids).distinct()

    # Оцениваем кандидатов по поисковым запросам
    book_search_counts = defaultdict(int)
    for book in candidates:
        matching_queries = user_queries.filter(query_text__icontains=book.title)
        frequency_sum = matching_queries.aggregate(total_freq=Sum('frequency'))['total_freq'] or 0
        book_search_counts[book.id] = frequency_sum

    # Считаем итоговый скор — по рейтингу + бонус за поиски
    scored_books = []
    for book in candidates:
        base_score = book.rating_chitai_gorod or 0
        search_bonus = book_search_counts.get(book.id, 0) * 10
        total_score = base_score + search_bonus
        scored_books.append((book, total_score))

    scored_books.sort(key=lambda x: x[1], reverse=True)


    import re
    import Levenshtein

    def normalize_title(title): 
        title = re.split(r'\s*\+\s*', title)[0] 
        title = re.sub(r'\s*\([^)]*\)', '', title) 
        return ' '.join(title.lower().split())


    def unique_titles(books, threshold=0.85):
        import Levenshtein
        titles = []
        unique_books = []
        for book in books:
            title = normalize_title(book.title)
            if not any(Levenshtein.ratio(title, t) >= threshold for t in titles):
                titles.append(title)
                unique_books.append(book)
        return unique_books

    def unique_ids(books):
        seen = set()
        unique_books = []
        for book in books:
            if book.id not in seen:
                seen.add(book.id)
                unique_books.append(book)
        return unique_books

 
    popular_books_all = [b for b, s in scored_books if not b.new and not b.soon]
    popular_books = unique_titles(popular_books_all)[:top_n]

    new_books_all = [b for b, s in scored_books if b.new]
    new_books = unique_titles(new_books_all)[:top_n]

    soon_books_all = [b for b, s in scored_books if b.soon]
    soon_books = unique_titles(soon_books_all)[:top_n]

    # Дополнение топа
    if len(popular_books) < top_n:
        needed = top_n - len(popular_books)
        extra = Book.objects.filter(new=False, soon=False).exclude(id__in=viewed_book_ids).order_by('-rating_chitai_gorod')[:needed * 3]
        popular_books.extend(extra)
 
    if len(new_books) < top_n:
        needed = top_n - len(new_books)
        extra = Book.objects.filter(new=False, soon=False).exclude(id__in=viewed_book_ids).order_by('-rating_chitai_gorod')[:needed * 3]
        new_books.extend(extra)

    if len(soon_books) < top_n:
        needed = top_n - len(soon_books)
        extra = Book.objects.filter(new=False, soon=False).exclude(id__in=viewed_book_ids).order_by('-rating_chitai_gorod')[:needed * 3]
        soon_books.extend(extra)

    # После расширения — убираем дубликаты по id
    popular_books = unique_ids(popular_books)[:top_n]
    new_books = unique_ids(new_books)[:top_n]
    soon_books = unique_ids(soon_books)[:top_n]
    return popular_books[:top_n], new_books[:top_n], soon_books[:top_n]



# def recommendations_split(user, top_n=20,
#                           w_popular=1.0,
#                           w_new=1.0,
#                           w_soon=1.0,
#                           w_view=0.5,
#                           w_subscription=0.5,
#                           w_cart_wishlist=0.3):
#     try:
#         prefs = user.userpreferences
#     except UserPreferences.DoesNotExist:
#         return [], [], []

#     purchased_book_ids = UserBookStatus.objects.filter(
#         user=user, status=UserBookStatus.STATUS_PURCHASED
#     ).values_list('book_id', flat=True)

#     subscribed_author_ids = UserSubscription.objects.filter(
#         user=user, content_type='AUTHOR'
#     ).values_list('author_id', flat=True)

#     subscribed_cycle_ids = UserSubscription.objects.filter(
#         user=user, content_type='CYCLE'
#     ).values_list('cycle_id', flat=True)

#     viewed_books = BookView.objects.filter(user=user).values('book').annotate(
#         total_duration=Sum('duration_seconds'),
#         max_scroll=Max('scroll_depth')
#     ).order_by('-total_duration')
#     viewed_book_ids = [v['book'] for v in viewed_books[:20]]

#     cart_wishlist_book_ids = UserBookStatus.objects.filter(
#         user=user,
#         status__in=[UserBookStatus.STATUS_CART, UserBookStatus.STATUS_WISHLIST]
#     ).values_list('book_id', flat=True)

#     cart_wishlist_books = Book.objects.filter(id__in=cart_wishlist_book_ids)
#     cart_wishlist_genre_ids = cart_wishlist_books.values_list('genre_id', flat=True)
#     cart_wishlist_author_ids = cart_wishlist_books.values_list('author_id', flat=True)
#     cart_wishlist_tag_ids = []
#     for book in cart_wishlist_books.prefetch_related('tags'):
#         cart_wishlist_tag_ids.extend([tag.id for tag in book.tags.all()])
#     cart_wishlist_tag_ids = list(set(cart_wishlist_tag_ids))

#     favorite_genre_ids = prefs.favorite_genres.values_list('id', flat=True)
#     favorite_tag_ids = prefs.favorite_tags.values_list('id', flat=True)
#     disliked_genre_ids = prefs.disliked_genres.values_list('id', flat=True)
#     disliked_tag_ids = prefs.disliked_tags.values_list('id', flat=True)

#     qs = Book.objects.exclude(id__in=purchased_book_ids)

#     interest_filter = Q()
#     if favorite_genre_ids.exists():
#         interest_filter |= Q(genre_id__in=favorite_genre_ids)
#     if favorite_tag_ids.exists():
#         interest_filter |= Q(tags__in=favorite_tag_ids)
#     if subscribed_author_ids.exists():
#         interest_filter |= Q(author__in=subscribed_author_ids)
#     if subscribed_cycle_ids.exists():
#         interest_filter |= Q(cycle_id__in=subscribed_cycle_ids)

#     if not interest_filter:
#         return [], [], []

#     qs = qs.filter(interest_filter).distinct()

#     if disliked_genre_ids.exists():
#         qs = qs.exclude(genre_id__in=disliked_genre_ids)
#     if disliked_tag_ids.exists():
#         qs = qs.exclude(tags__in=disliked_tag_ids)

#     qs = qs.annotate(
#         log_votes=Log(F('votes_chitai_gorod') + 1),
#         popularity_score=F('rating_chitai_gorod') * F('log_votes'),
#         subscription_bonus=Case(
#             When(Q(author__in=subscribed_author_ids) | Q(cycle_id__in=subscribed_cycle_ids), then=Value(1)),
#             default=Value(0),
#             output_field=IntegerField()
#         ),
#         view_bonus=Case(
#             When(id__in=viewed_book_ids, then=Value(1)),
#             default=Value(0),
#             output_field=IntegerField()
#         ),
#         cart_wishlist_bonus=Case(
#             When(
#                 Q(genre_id__in=cart_wishlist_genre_ids) |
#                 Q(author_id__in=cart_wishlist_author_ids) |
#                 Q(tags__in=cart_wishlist_tag_ids),
#                 then=Value(1)
#             ),
#             default=Value(0),
#             output_field=IntegerField()
#         )
#     )

#     qs = qs.annotate(
#         score=ExpressionWrapper(
#             w_popular * F('popularity_score') +
#             w_new * F('new') +
#             w_soon * F('soon') +
#             w_subscription * F('subscription_bonus') +
#             w_view * F('view_bonus') +
#             w_cart_wishlist * F('cart_wishlist_bonus'),
#             output_field=FloatField()
#         )
#     )

#     # Отдельные запросы для каждой категории с сортировкой по score
#     popular_books = qs.filter(new=False, soon=False).order_by('-score')[:top_n]
#     new_books = qs.filter(new=True).order_by('-score')[:top_n]
#     soon_books = qs.filter(soon=True).order_by('-score')[:top_n]

#     return popular_books, new_books, soon_books