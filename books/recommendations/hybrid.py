from django.db.models import Value, FloatField
from books.models import Book

from .collaborative import get_similar_books_item_based, get_recommendations_for_user_user_based
from .content_based import get_similar_books_content, recommend_books_for_user_content, get_similar_books_combined
from .svd_model import get_svd_recommendations_for_user
from .torch_model import recommend_books_for_user_simple
from .word2_vec import get_similar_books_by_w2v
from .node2vec_recommender import recommend_books_node2vec

# для пользователя
def hybrid_recommendations_for_user(user, top_n=20,
                                   w_user_based=1.0,
                                   w_content=1.0,
                                   w_svd=1.0,
                                   w_torch=1.0,
                                   w_node2vec=1.0):
    scores = {}

    def add_scores(recs, weight):
        for book_id, score in recs:
            scores[book_id] = scores.get(book_id, 0) + weight * score

    # Обертки, приводящие к (book_id, score)
    def wrap_user_based(user):
        books = get_recommendations_for_user_user_based(user.id, top_n=50)
        return [(b.id, 1.0) for b in books]

    def wrap_content(user):
        qs = recommend_books_for_user_content(user, top_n=50)
        return [(b.id, 1.0) for b in qs]

    def wrap_svd(user):
        books = get_svd_recommendations_for_user(user.id, top_n=50)
        return [(b.id, 1.0) for b in books]

    def wrap_torch(user):
        qs = recommend_books_for_user_simple(user.id, top_n=50)
        return [(b.id, 1.0) for b in qs]

    def wrap_node2vec(user):
        df = recommend_books_node2vec(user.id, top_n=50)
        return list(df.itertuples(index=False, name=None))  

    add_scores(wrap_user_based(user), w_user_based)
    add_scores(wrap_content(user), w_content)
    add_scores(wrap_svd(user), w_svd)
    add_scores(wrap_torch(user), w_torch)
    add_scores(wrap_node2vec(user), w_node2vec)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_ids = [bid for bid, _ in ranked[:top_n]]

    books = Book.objects.filter(id__in=top_ids)
    books_dict = {b.id: b for b in books}
    ordered_books = [books_dict[bid] for bid in top_ids if bid in books_dict]
    return ordered_books
 

# для книги
def hybrid_recommendations_for_book(book: Book, top_n=10,
                                    w_item_based=1.0,
                                    w_content=1.0,
                                    w_w2v=1.0):
    scores = {}

    def add_scores(recs, weight):
        for book_id, score in recs:
            scores[book_id] = scores.get(book_id, 0) + weight * score

    def wrap_item_based(book):
        books = get_similar_books_item_based(book.id, top_n=50)
        return [(b.id, 1.0) for b in books]

    def wrap_content(book):
        qs = get_similar_books_content(book, top_n=50)
        return [(b.id, 1.0) for b in qs]

    def wrap_w2v(book):
        recs = get_similar_books_by_w2v(book.id, top_n=50)
        return [(rec['book'].id, rec['similarity']) for rec in recs]

    add_scores(wrap_item_based(book), w_item_based)
    add_scores(wrap_content(book), w_content)
    add_scores(wrap_w2v(book), w_w2v)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_ids = [bid for bid, _ in ranked[:top_n]]

    books = Book.objects.filter(id__in=top_ids)
    books_dict = {b.id: b for b in books}
    ordered_books = [books_dict[bid] for bid in top_ids if bid in books_dict]
    return ordered_books

