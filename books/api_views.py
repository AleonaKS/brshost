import logging
from difflib import SequenceMatcher
from rest_framework.decorators import permission_classes, api_view, authentication_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from haystack.query import SearchQuerySet
from django.db.models import F 
from django.shortcuts import get_object_or_404 
from django.utils import timezone  
from books.models import Book, BookRating, BookView, UserBookStatus, UserSearchQuery
from .recommendations.collaborative import get_recommendations_for_user_user_based   
from .serializers import BookSerializer, UserSearchQuerySerializer

logger = logging.getLogger(__name__)
# from django.views.decorators.csrf import csrf_exempt

# @api_view(['GET'])
# def user_recommendations_api(request, user_id): 
#     recommended_books = get_recommendations_for_user_user_based(user_id, top_n=10)
#     serializer = BookSerializer(recommended_books, many=True)
#     return Response(serializer.data)


@api_view(['GET'])
def autocomplete(request):
    q = request.GET.get('q', '')
    if not q:
        return Response([])

    books = Book.objects.filter(title__icontains=q)[:10]
    serializer = BookSerializer(books, many=True)
    return Response(serializer.data)


# Автодополнение по названию и автору книги:
@api_view(['GET'])
@permission_classes([AllowAny])  
def autocomplete_books(request):
    query = request.GET.get('q', '').strip()
    if len(query) < 2:
        return Response([])  
    sqs = SearchQuerySet().autocomplete(content_auto=query)[:10]
    results = []
    for res in sqs:
        book = res.object
        if book:
            results.append({
                'id': book.id,
                'title': book.title,
                'author': ', '.join(author.name for author in book.author.all())
            })
    return Response(results)


# Оценка книги пользователем
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def book_rate(request, book_id):
    try:
        rating_value = int(request.data.get('rating', 0))
    except (TypeError, ValueError):
        return Response({'error': 'Некорректный рейтинг'}, status=status.HTTP_400_BAD_REQUEST)

    if rating_value < 1 or rating_value > 5:
        return Response({'error': 'Рейтинг должен быть от 1 до 5'}, status=status.HTTP_400_BAD_REQUEST)

    book = get_object_or_404(Book, id=book_id)

    rating_obj, created = BookRating.objects.update_or_create(
        user=request.user,
        book=book,
        defaults={'rating': rating_value}
    )
    return Response({'message': 'Оценка сохранена'})

 
# Запись просмотра книги
@api_view(['POST'])
@permission_classes([AllowAny]) 
def record_book_view(request):
    data = request.data
    book_id = data.get('book_id')
    print('book_id:', book_id)   
    scroll = data.get('scroll_depth')

    if not book_id:
        return Response({'status': 'error', 'message': 'book_id is required'}, status=400)
    try:
        book = Book.objects.get(id=book_id)
    except Book.DoesNotExist:
        return Response({'status': 'error', 'message': 'Book not found'}, status=404)

    user = request.user if request.user.is_authenticated else None

    if user is None:
        if not request.session.session_key:
            request.session.create()
        session_key = request.session.session_key 

        book_view_qs = BookView.objects.filter(user__isnull=True, session_key=session_key, book=book)
        if book_view_qs.exists():
            book_view = book_view_qs.first()
            book_view.duration_seconds = (book_view.duration_seconds or 0) + (data.get('duration_seconds', 0) or 0)
            book_view.viewed_at = timezone.now()
            scroll = data.get('scroll_depth')
            if scroll is not None:
                book_view.scroll_depth = scroll
            book_view.save()
            print(f"Обновлен BookView для сессииn {session_key} и книги {book_id}")
        else:
            BookView.objects.create(
                user=None,
                session_key=session_key,
                book=book,
                viewed_at=timezone.now(),
                duration_seconds=data.get('duration_seconds', 0),
                scroll_depth=data.get('scroll_depth')
            )
            print(f"Создан BookView для сессии {session_key} и книги {book_id}")

        user_views = BookView.objects.filter(user__isnull=True, session_key=session_key).order_by('-viewed_at')
    else:
        book_view, created = BookView.objects.get_or_create(
            user=user,
            book=book,
            defaults={
                'viewed_at': timezone.now(),
                'duration_seconds': data.get('duration_seconds', 0),
                'scroll_depth': data.get('scroll_depth')
            }
        )
        if not created:
            book_view.duration_seconds = (book_view.duration_seconds or 0) + (data.get('duration_seconds', 0) or 0)
            book_view.viewed_at = timezone.now()
            scroll = data.get('scroll_depth')
            if scroll is not None:
                book_view.scroll_depth = scroll
            book_view.save()
            print(f"Обновлен BookView для пользователя {user.id} и книги {book_id}")
        else:
            print(f"Создан BookView для пользователя {user.id} и книги {book_id}")

        user_views = BookView.objects.filter(user=user).order_by('-viewed_at')

    # Ограничиваем до 50 записей
    if user_views.count() > 50:
        to_delete = user_views[50:]
        to_delete.delete()
    last_20_books_ids = user_views.values_list('book_id', flat=True).distinct()[:20]
    # Удаляем записи вне последних 20 с duration < 30 сек
    old_views = user_views.exclude(book_id__in=last_20_books_ids)
    short_duration_views = old_views.filter(duration_seconds__lt=30)
    short_duration_views.delete()

    return Response({'status': 'ok'})



# Добавление книги в корзину и в закладки
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def add_to_cart(request):
    logger.info("Корзина вызвана")
    book_id = request.data.get('book_id')
    if not book_id:
        return Response({'error': 'book_id is required'}, status=status.HTTP_400_BAD_REQUEST)
    book = get_object_or_404(Book, id=book_id)
    user = request.user 
    obj, created = UserBookStatus.objects.update_or_create(
        user=user,
        book=book,
        defaults={'status': UserBookStatus.STATUS_CART}
    ) 
    return Response({'detail': 'Книга добавлена в корзину'}, status=status.HTTP_200_OK)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def add_to_bookmarks(request):
    logger.info("Закладки вызваны")
    print("Закладки")
    book_id = request.data.get('book_id')
    if not book_id:
        return Response({'error': 'book_id is required'}, status=status.HTTP_400_BAD_REQUEST)
    book = get_object_or_404(Book, id=book_id)
    user = request.user 
    obj, created = UserBookStatus.objects.update_or_create(
        user=user,
        book=book,
        defaults={'status': UserBookStatus.STATUS_WISHLIST}
    ) 
    return Response({'detail': 'Книга добавлена в закладки'}, status=status.HTTP_200_OK)




@api_view(['POST'])
@permission_classes([IsAuthenticated])
def remove_from_cart(request):
    book_id = request.data.get('book_id')
    if not book_id:
        return Response({'error': 'book_id is required'}, status=status.HTTP_400_BAD_REQUEST)
    book = get_object_or_404(Book, id=book_id)
    user = request.user

    try:
        obj = UserBookStatus.objects.get(user=user, book=book, status=UserBookStatus.STATUS_CART)
        obj.delete()
        return Response({'detail': 'Книга удалена из корзины'}, status=status.HTTP_200_OK)
    except UserBookStatus.DoesNotExist:
        return Response({'error': 'Книга не найдена в корзине'}, status=status.HTTP_404_NOT_FOUND)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def remove_from_bookmarks(request):
    book_id = request.data.get('book_id')
    if not book_id:
        return Response({'error': 'book_id is required'}, status=status.HTTP_400_BAD_REQUEST)
    book = get_object_or_404(Book, id=book_id)
    user = request.user

    try:
        obj = UserBookStatus.objects.get(user=user, book=book, status=UserBookStatus.STATUS_WISHLIST)
        obj.delete()
        return Response({'detail': 'Книга удалена из закладок'}, status=status.HTTP_200_OK)
    except UserBookStatus.DoesNotExist:
        return Response({'error': 'Книга не найдена в закладках'}, status=status.HTTP_404_NOT_FOUND)


# 
def log_user_search(user, query_text):
    if not user.is_authenticated:
        return  

    obj, created = UserSearchQuery.objects.get_or_create(user=user, query_text=query_text)
    if not created:
        obj.frequency += 1
        obj.last_searched = timezone.now()
        obj.save()






SIMILARITY_THRESHOLD = 0.8
from difflib import SequenceMatcher

def are_queries_similar(q1, q2, threshold=SIMILARITY_THRESHOLD):
    return SequenceMatcher(None, q1.lower(), q2.lower()).ratio() >= threshold

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def record_user_search(request):
    query = request.data.get('query_text', '').strip()
    if not query:
        return Response({'error': 'query_text is required'}, status=status.HTTP_400_BAD_REQUEST)

    obj, created = UserSearchQuery.objects.get_or_create(
        user=request.user,
        query_text=query,
        defaults={'frequency': 1}
    )
    if not created:
        obj.frequency = F('frequency') + 1
        obj.last_searched = timezone.now()
        obj.save(update_fields=['frequency', 'last_searched'])
        obj.refresh_from_db()

    serializer = UserSearchQuerySerializer(obj)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes([AllowAny])
def get_user_search_history(request):
    user = request.user if request.user.is_authenticated else None

    if user is None:
        if not request.session.session_key:
            request.session.create()
        session_key = request.session.session_key
        queries = UserSearchQuery.objects.filter(user__isnull=True, session_key=session_key)
        user_queries = queries.order_by('-last_searched')
    else:
        user_queries = UserSearchQuery.objects.filter(user=user).order_by('-last_searched')
        # Ограничиваем количество записей  
    if user_queries.count() > 50:
        to_delete = user_queries[50:]
        to_delete.delete()
    serializer = UserSearchQuerySerializer(user_queries[:10], many=True)
    return Response(serializer.data)
