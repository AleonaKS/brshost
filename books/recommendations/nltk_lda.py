import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
# в процессе разработки "Topic Modeling" для выявления скрытых тем
import pymorphy2
from nltk.corpus import stopwords
import gensim

# morph = pymorphy2.MorphAnalyzer()
# russian_stopwords = set(stopwords.words('russian'))
# =======================
# import os
# os.environ["TRANSFORMERS_NO_TF"] = "1"

# import pymorphy2
# from nltk.corpus import stopwords
# import gensim
# import numpy as np
# from scipy.spatial.distance import cosine
# from datetime import datetime

# import torch
# import torch.nn.functional as F
# from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# from books.models import (
#     Book, BookTopicVector, BookEmbedding,
#     Review, ReviewEmbedding, ReviewSentiment,
#     UserTopicVector,
# )






def get_embedding_model():
    if not hasattr(get_embedding_model, "tokenizer"):
        get_embedding_model.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        get_embedding_model.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        get_embedding_model.model.eval()
    return get_embedding_model.tokenizer, get_embedding_model.model

def embed_text(text):
    tokenizer_embed, model_embed = get_embedding_model()
    inputs = tokenizer_embed(text, return_tensors='pt', truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model_embed(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embeddings
















 

def get_sentiment_model():
    if not hasattr(get_sentiment_model, "tokenizer"):
        get_sentiment_model.tokenizer = AutoTokenizer.from_pretrained("blanchefort/rubert-base-cased-sentiment")
        get_sentiment_model.model = AutoModelForSequenceClassification.from_pretrained("blanchefort/rubert-base-cased-sentiment")
        get_sentiment_model.model.eval()
    return get_sentiment_model.tokenizer, get_sentiment_model.model

def analyze_sentiment(text):
    tokenizer_sentiment, model_sentiment = get_sentiment_model()
    inputs = tokenizer_sentiment(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model_sentiment(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs[0].tolist()











def get_morph_and_stopwords():
    if not hasattr(get_morph_and_stopwords, "morph"):
        get_morph_and_stopwords.morph = pymorphy2.MorphAnalyzer()
        get_morph_and_stopwords.russian_stopwords = set(stopwords.words('russian'))
    return get_morph_and_stopwords.morph, get_morph_and_stopwords.russian_stopwords



def preprocess_russian(text):
    morph, russian_stopwords = get_morph_and_stopwords()
    tokens = gensim.utils.simple_preprocess(text, deacc=True)
    lemmas = []
    for token in tokens:
        if token in russian_stopwords or len(token) < 3:
            continue
        p = morph.parse(token)[0]
        lemmas.append(p.normal_form)
    return lemmas

# def preprocess_russian(text):
#     tokens = gensim.utils.simple_preprocess(text, deacc=True)
#     lemmas = []
#     for token in tokens:
#         if token in russian_stopwords or len(token) < 3:
#             continue
#         p = morph.parse(token)[0]
#         lemmas.append(p.normal_form)
#     return lemmas


from datetime import datetime

def review_weight(review):
    rating_weight = (review.rating - 1) / 4  # [0,1]
    length_weight = min(len(review.text) / 1000, 1.0)
    days_passed = (datetime.now() - review.date).days if review.date else 365
    freshness_weight = 1 / (1 + days_passed / 30)
    weight = 0.5 * rating_weight + 0.3 * length_weight + 0.2 * freshness_weight
    return weight


import numpy as np
from books.models import UserTopicVector  # модель для хранения вектора пользователя

def get_user_topic_vector(user, num_topics=20):
    user_reviews = Review.objects.filter(user=user).select_related('book')
    vectors = []
    weights = []
    for review in user_reviews:
        try:
            btv = review.book.topic_vector
            vec = np.array(get_dense_topic_vector(btv, num_topics))
            w = review_weight(review)
            vectors.append(vec * w)
            weights.append(w)
        except BookTopicVector.DoesNotExist:
            continue
    if not vectors or sum(weights) == 0:
        return None
    user_vec = np.sum(vectors, axis=0) / sum(weights)
    # Кэшируем в базу
    UserTopicVector.objects.update_or_create(user=user, defaults={'vector': user_vec.tolist()})
    return user_vec



def update_review_sentiments():
    reviews = Review.objects.filter(sentiment__isnull=True)
    for review in reviews:
        if review.text:
            sentiment_probs = analyze_sentiment(review.text)
            sentiment_score = sentiment_probs[2] - sentiment_probs[0]  # pos - neg
            ReviewSentiment.objects.update_or_create(review=review, defaults={'score': sentiment_score})




import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk 
import numpy as np
from scipy.spatial.distance import cosine
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
import torch

from books.models import Book, BookTopicVector, BookEmbedding, ReviewEmbedding, ReviewSentiment, Review  # импортируйте свои модели


def preprocess(text):
    stop_words = set(stopwords.words('russian'))
    lemmatizer = WordNetLemmatizer()

    tokens = gensim.utils.simple_preprocess(text, deacc=True)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return tokens


def train_lda_and_save(num_topics=20, no_below=5, no_above=0.5, passes=10):
 

    descriptions = list(Book.objects.values_list('description', flat=True))
    texts = [preprocess_russian(desc or '') for desc in descriptions]

    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    corpus = [dictionary.doc2bow(text) for text in texts]

    lda = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes)

    lda.save('lda_model.model')
    dictionary.save('lda_dictionary.dict')

    # Сохраняем векторы тем в БД
    books = Book.objects.all()
    for book, bow in zip(books, corpus):
        topic_dist = lda.get_document_topics(bow, minimum_probability=0)
        vector = [0.0] * num_topics
        for topic_id, prob in topic_dist:
            vector[topic_id] = prob

        BookTopicVector.objects.update_or_create(
            book=book,
            defaults={
                'indices': [i for i, v in enumerate(vector) if v > 0],
                'values': [v for v in vector if v > 0]
            }
        )


def get_dense_topic_vector(book_topic_vector, num_topics=20):
    vector = [0.0] * num_topics
    for idx, val in zip(book_topic_vector.indices, book_topic_vector.values):
        vector[idx] = val
    return vector


# def get_user_topic_vector(user, num_topics=20): 
#     user_reviews = Review.objects.filter(user=user).select_related('book')
#     topic_vectors = []
#     for review in user_reviews:
#         try:
#             btv = review.book.topic_vector
#             vec = get_dense_topic_vector(btv, num_topics)
#             topic_vectors.append(np.array(vec))
#         except BookTopicVector.DoesNotExist:
#             continue
#     if not topic_vectors:
#         return None
#     user_vec = np.mean(topic_vectors, axis=0)
#     return user_vec


# def recommend_books_by_topics(user, top_n=10, num_topics=20):
#     user_vec = get_user_topic_vector(user, num_topics)
#     if user_vec is None:
#         return Book.objects.none()

#     all_btv = BookTopicVector.objects.select_related('book').all()
#     scores = []
#     for btv in all_btv:
#         vec = get_dense_topic_vector(btv, num_topics)
#         dist = cosine(user_vec, vec)
#         scores.append((dist, btv.book))
#     scores.sort(key=lambda x: x[0])
#     recommended = [book for _, book in scores[:top_n]]
#     return recommended

def recommend_books_by_topics(user, top_n=10, num_topics=20):
    # Попытка загрузить из кэша
    try:
        cached = UserTopicVector.objects.get(user=user)
        user_vec = np.array(cached.vector)
    except UserTopicVector.DoesNotExist:
        user_vec = get_user_topic_vector(user, num_topics)
        if user_vec is None:
            return Book.objects.none()

    all_btv = BookTopicVector.objects.select_related('book').all()
    scores = []
    for btv in all_btv:
        vec = np.array(get_dense_topic_vector(btv, num_topics))
        dist = cosine(user_vec, vec)
        scores.append((dist, btv.book))
    scores.sort(key=lambda x: x[0])
    recommended = [book for _, book in scores[:top_n]]
    return recommended


 
import torch
import torch.nn.functional as F

# tokenizer_sentiment = AutoTokenizer.from_pretrained("blanchefort/rubert-base-cased-sentiment")
# model_sentiment = AutoModelForSequenceClassification.from_pretrained("blanchefort/rubert-base-cased-sentiment")
# model_sentiment.eval()

# def analyze_sentiment(text):
#     inputs = tokenizer_sentiment(text, return_tensors="pt", truncation=True, max_length=512)
#     with torch.no_grad():
#         outputs = model_sentiment(**inputs)
#     probs = F.softmax(outputs.logits, dim=1)
#     # probs: [negative, neutral, positive]
#     return probs[0].tolist()


# def analyze_sentiment(text):
#     sia = SentimentIntensityAnalyzer()
#     score = sia.polarity_scores(text)
#     return score['compound']  # от -1 до 1


# def update_review_sentiments():
#     reviews = Review.objects.filter(sentiment__isnull=True)
#     sentiment_probs = analyze_sentiment(review.text)
#     sentiment_score = sentiment_probs[2] - sentiment_probs[0]  # положительный минус отрицательный
#     ReviewSentiment.objects.update_or_create(review=review, defaults={'score': sentiment_score})

#     for review in reviews:
#         if review.text:
#             sentiment_score = analyze_sentiment(review.text)
#             ReviewSentiment.objects.update_or_create(review=review, defaults={'score': sentiment_score})



# tokenizer_embed = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
# model_embed = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
# model_embed.eval()

# def embed_text(text):
#     inputs = tokenizer_embed(text, return_tensors='pt', truncation=True, max_length=128)
#     with torch.no_grad():
#         outputs = model_embed(**inputs)
#     embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
#     return embeddings



def update_book_embeddings():
    for book in Book.objects.all():
        emb = embed_text(book.description or "")
        BookEmbedding.objects.update_or_create(book=book, defaults={'vector': emb.tolist()})


def update_review_embeddings():
    reviews = Review.objects.filter(reviewembedding__isnull=True).exclude(text__isnull=True).exclude(text__exact='')
    for review in reviews:
        emb = embed_text(review.text)
        ReviewEmbedding.objects.create(review=review, vector=emb.tolist())


def get_user_embedding(user):
    reviews = ReviewEmbedding.objects.filter(review__user=user)
    if not reviews.exists():
        return None
    embs = [np.array(r.vector) for r in reviews]
    return np.mean(embs, axis=0)


def recommend_books_by_embedding(user, top_n=10):
    user_emb = get_user_embedding(user)
    if user_emb is None:
        return Book.objects.none()

    all_embeddings = BookEmbedding.objects.select_related('book').all()
    scores = []
    for be in all_embeddings:
        dist = cosine(user_emb, np.array(be.vector))
        scores.append((dist, be.book))
    scores.sort(key=lambda x: x[0])
    recommended = [book for _, book in scores[:top_n]]
    return recommended
