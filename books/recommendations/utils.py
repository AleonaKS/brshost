# Вспомогательные функции, включая фильтрацию по дате для метрик
from books.models import BookRating
import pandas as pd

def filter_ratings_by_year(year_cutoff=2025):
    """
    Возвращает DataFrame с рейтингами, отсекая данные с review_date >= year_cutoff. 
    """
    ratings_qs = BookRating.objects.exclude(review_date__year__gte=year_cutoff).values('user_id', 'book_id', 'rating')
    df = pd.DataFrame(ratings_qs)
    return df

def build_user_book_matrix_filtered(year_cutoff=2025):
    df = filter_ratings_by_year(year_cutoff)
    if df.empty:
        return pd.DataFrame()
    user_book_matrix = df.pivot(index='user_id', columns='book_id', values='rating').fillna(0)
    return user_book_matrix
 