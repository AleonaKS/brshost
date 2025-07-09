import logging
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth import login, authenticate
from django.contrib.auth.models import User 
from django.contrib.auth.decorators import login_required 
from django.http import JsonResponse 
from django.views.decorators.http import require_GET, require_POST
from django.middleware.csrf import get_token
from django.db import transaction
from django.db.models import OuterRef, Subquery 
from haystack.query import SearchQuerySet
from .filters import BookFilter 
from .forms import SignUpForm, UserPreferencesForm #, LoginForm, BookRatingForm
from .models import Book, Review, BookView, BookRating, UserBookStatus, UserPreferences
from .models import UserPreferences, FavoriteAuthors, FavoriteGenres, FavoriteTags, DislikedGenres, DislikedTags
from books.recommendations.base_recommendations import recommendations_split, recommendations_for_anonymous, recommendations_for_user_without_preferences
from books.recommendations.hybrid import hybrid_recommendations_for_user, hybrid_recommendations_for_book

logger = logging.getLogger(__name__)
 

def switch_keyboard_layout(text):
    layout_en = 'qwertyuiop[]asdfghjkl;\'zxcvbnm,./`QWERTYUIOP{}ASDFGHJKL:"ZXCVBNM<>?~'
    layout_ru = 'йцукенгшщзхъфывапролджэячсмитьбю.ёЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ,Ё'
    assert len(layout_en) == len(layout_ru), "Раскладки должны быть одинаковой длины"
    trans_table = str.maketrans(layout_en + layout_ru, layout_ru + layout_en)
    return text.translate(trans_table)


@require_GET
def search_books(request):
    query = request.GET.get('q', '').strip()
    alt_query = switch_keyboard_layout(query) if query else ''
    sort = request.GET.get('sort_by', '')
    filtered_books = []
    book_filter = None

    if query:
        sqs_title_main = SearchQuerySet().autocomplete(content_auto=query)
        sqs_title_alt = SearchQuerySet().autocomplete(content_auto=alt_query) if alt_query else SearchQuerySet().none()
        sqs_author_main = SearchQuerySet().autocomplete(author_auto=query)
        sqs_author_alt = SearchQuerySet().autocomplete(author_auto=alt_query) if alt_query else SearchQuerySet().none()

        sqs = (sqs_title_main | sqs_title_alt | sqs_author_main | sqs_author_alt).load_all()
        books = [res.object for res in sqs if res.object is not None][:100]
        book_ids = [book.id for book in books]

        book_filter = BookFilter(request.GET, queryset=Book.objects.filter(id__in=book_ids))
        filtered_qs = book_filter.qs

        if sort == 'popularity':
            filtered_qs = filtered_qs.order_by('-popularity_score')
        elif sort == 'newness':
            filtered_qs = filtered_qs.order_by('-year_of_publishing')

        filtered_books = list(filtered_qs)
    else:
        qs = Book.objects.all()
        book_filter = BookFilter(request.GET, queryset=qs)
        filtered_books = list(book_filter.qs)

    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        results = [{'id': b.id, 'title': b.title, 'author': ', '.join(a.name for a in b.author.all())} for b in filtered_books[:20]]
        return JsonResponse(results, safe=False)

    context = {
        'books': filtered_books,
        'filter': book_filter,
        'query': query,
        'sort': sort,
    }
    return render(request, 'search.html', context)


def home_view(request):
    popular_books, new_books, soon_books = None, None, None
    if request.user.is_authenticated:
        user_views = BookView.objects.filter(user=request.user).order_by('-viewed_at')
        user = request.user
        recommended_books = hybrid_recommendations_for_user(user, top_n=30)
        try:
            prefs = user.userpreferences
            favorite_genres = prefs.favorite_genres.all().values_list('id', flat=True)
            favorite_tags = prefs.favorite_tags.all().values_list('id', flat=True)
            recommended_books = hybrid_recommendations_for_user(user, top_n=30)
            popular_books, new_books, soon_books = recommendations_split(user, list(favorite_genres), list(favorite_tags), top_n=30)
        except UserPreferences.DoesNotExist:
            popular_books, new_books, soon_books = recommendations_for_user_without_preferences(user, top_n=30)
    else:
        session_key = request.session.session_key
        if not session_key:
            request.session.create()
            session_key = request.session.session_key
        user_views = BookView.objects.filter(user__isnull=True, session_key=session_key).order_by('-viewed_at')
        recommended_books = Book.objects.all()
        popular_books, new_books, soon_books = recommendations_for_anonymous(session_key)

    last_20_book_ids = list(user_views.values_list('book_id', flat=True).distinct()[:20])
    books = list(Book.objects.filter(id__in=last_20_book_ids))
    last_books = sorted(books, key=lambda b: last_20_book_ids.index(b.id))

    context = {
        'recommended_books': recommended_books,
        'popular_books': popular_books,
        'new_books': new_books,
        'soon_books': soon_books,
        'last_books': last_books
    }
    return render(request, 'home.html', context)




def book_detail(request, book_id):
    user = request.user
    book = get_object_or_404(Book, id=book_id)
    in_cart = False
    in_bookmarks = False
    if user.is_authenticated:
        in_cart = UserBookStatus.objects.filter(user=user, book=book, status=UserBookStatus.STATUS_CART).exists()
        in_bookmarks = UserBookStatus.objects.filter(user=user, book=book, status=UserBookStatus.STATUS_WISHLIST).exists()
    # book = get_object_or_404(Book, id=book_id)
    similar_books = hybrid_recommendations_for_book(book, top_n=10) 
    context = {
        'book': book,
        'in_cart': in_cart,
        'in_bookmarks': in_bookmarks,
        'csrf_token': get_token(request), 
        'similar_books': similar_books
    }
    return render(request, 'book_detail.html', context)






@login_required
def profile(request): 
    user_ratings_subquery = BookRating.objects.filter(
        user=request.user,
        book=OuterRef('pk')
    ).values('rating')[:1]

    books = Book.objects.annotate(
        user_rating=Subquery(user_ratings_subquery)
    ).filter(user_rating__isnull=False)
 
    user = request.user
    preferences, created = UserPreferences.objects.get_or_create(user=user)

    if request.method == 'POST':
        form = UserPreferencesForm(request.POST, instance=preferences, user=user)
        if form.is_valid():
            with transaction.atomic():
                # Обновляем UserPreferences (т.к. поля ManyToMany через промежуточные модели, надо обновлять вручную)
                FavoriteAuthors.objects.filter(userprofile=preferences).delete()
                FavoriteGenres.objects.filter(userprofile=preferences).delete()
                FavoriteTags.objects.filter(userprofile=preferences).delete()
                DislikedGenres.objects.filter(userprofile=preferences).delete()
                DislikedTags.objects.filter(userprofile=preferences).delete()

                for author in form.cleaned_data['favorite_authors']:
                    FavoriteAuthors.objects.create(userprofile=preferences, author=author)

                for genre in form.cleaned_data['favorite_genres']:
                    fg = FavoriteGenres(userprofile=preferences, genre=genre)
                    try:
                        fg.clean()
                        fg.save()
                    except Exception as e:
                        messages.error(request, f'Ошибка при добавлении любимого жанра {genre}: {e}')

                for tag in form.cleaned_data['favorite_tags']:
                    ft = FavoriteTags(userprofile=preferences, tag=tag)
                    try:
                        ft.clean()
                        ft.save()
                    except Exception as e:
                        messages.error(request, f'Ошибка при добавлении любимого тега {tag}: {e}')

                for genre in form.cleaned_data['disliked_genres']:
                    dg = DislikedGenres(userprofile=preferences, genre=genre)
                    try:
                        dg.clean()
                        dg.save()
                    except Exception as e:
                        messages.error(request, f'Ошибка при добавлении нелюбимого жанра {genre}: {e}')

                for tag in form.cleaned_data['disliked_tags']:
                    dt = DislikedTags(userprofile=preferences, tag=tag)
                    try:
                        dt.clean()
                        dt.save()
                    except Exception as e:
                        messages.error(request, f'Ошибка при добавлении нелюбимого тега {tag}: {e}')

            messages.success(request, 'Настройки сохранены')
            return redirect('profile')
    else:
        form = UserPreferencesForm(instance=preferences, user=user)


    context = {
        'books': books,
        'form': form, 
    }
    return render(request, 'profile.html', context)





@login_required
def cart(request):
    cart_items = UserBookStatus.objects.filter(user=request.user, status=UserBookStatus.STATUS_CART).select_related('book')
    books = [item.book for item in cart_items]
    return render(request, 'cart.html', {'books': books})


@login_required
def bookmarks(request):
    bookmarks_items = UserBookStatus.objects.filter(user=request.user, status=UserBookStatus.STATUS_WISHLIST).select_related('book')
    books = [item.book for item in bookmarks_items]
    return render(request, 'bookmarks.html', {'books': books})


def signup(request):
    return render(request, 'home.html', {'user': request.user})


def registration(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        username = request.POST.get('username')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')

        if User.objects.filter(username=username).exists():
            return JsonResponse({'error': 'Пользователь с таким логином уже существует!'}, status=400)
        if password1 != password2:
            return JsonResponse({'error': 'Пароли не совпадают!'}, status=400)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return JsonResponse({'success': 'Регистрация успешна!'})
        else:
            return JsonResponse({'error': form.errors.as_json()}, status=400)
    else:
        form = SignUpForm()
    return render(request, 'home.html', {'form': form})

