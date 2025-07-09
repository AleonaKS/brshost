import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from books.models import Book, Review, User
import pickle

MODEL_PATH = "books/recommendations/models/book_recommender.pth"

def load_and_preprocess_data():
    books_qs = Book.objects.all().select_related('genre').values(
        'id', 'genre_id'
    )
    books_df = pd.DataFrame(list(books_qs))
    users_qs = User.objects.all().values('id')
    users_df = pd.DataFrame(list(users_qs))
    reviews_qs = Review.objects.all().values('user_id', 'book_id', 'rating', 'review_date')
    reviews_df = pd.DataFrame(list(reviews_qs))
    user_id_map = {uid: idx for idx, uid in enumerate(users_df['id'].unique())}
    book_id_map = {bid: idx for idx, bid in enumerate(books_df['id'].unique())}
    books_df['genre_id'] = books_df['genre_id'].fillna(-1).astype(int)
    genre_list = books_df['genre_id'].unique()
    genre_map = {gid: idx for idx, gid in enumerate(genre_list)}
    books_df['genre_idx'] = books_df['genre_id'].map(genre_map)
    genre_idx_map = books_df.set_index('id')['genre_idx'].to_dict()
    reviews_df['user_idx'] = reviews_df['user_id'].map(user_id_map)
    reviews_df['book_idx'] = reviews_df['book_id'].map(book_id_map)
    reviews_df['genre_idx'] = reviews_df['book_id'].map(genre_idx_map)
    reviews_df = reviews_df.dropna(subset=['rating', 'user_idx', 'book_idx', 'genre_idx'])
    reviews_df['user_idx'] = reviews_df['user_idx'].astype(int)
    reviews_df['book_idx'] = reviews_df['book_idx'].astype(int)
    reviews_df['genre_idx'] = reviews_df['genre_idx'].astype(int)
    return books_df, users_df, reviews_df, user_id_map, book_id_map, genre_map

class BookRecommender(nn.Module):
    def __init__(self, num_users, num_books, num_genres, embedding_dim=50):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.book_embedding = nn.Embedding(num_books, embedding_dim)
        self.genre_embedding = nn.Embedding(num_genres, embedding_dim // 2)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim*2 + embedding_dim//2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    def forward(self, user_idx, book_idx, genre_idx):
        user_emb = self.user_embedding(user_idx)
        book_emb = self.book_embedding(book_idx)
        genre_emb = self.genre_embedding(genre_idx)
        x = torch.cat([user_emb, book_emb, genre_emb], dim=1)
        rating = self.fc(x)
        return rating.squeeze()

def save_all_data_pickle():
    books_df, users_df, reviews_df, user_id_map, book_id_map, genre_map = load_and_preprocess_data()
    with open('books/recommendations/models/all_data.pkl', 'wb') as f:
        pickle.dump({
            'books_df': books_df,
            'reviews_df': reviews_df,
            'user_id_map': user_id_map,
            'book_id_map': book_id_map,
            'genre_map': genre_map
        }, f)

def load_all_data_pickle():
    with open('books/recommendations/models/all_data.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['books_df'], data['reviews_df'], data['user_id_map'], data['book_id_map'], data['genre_map']

def train_model(reviews_df, num_users, num_books, num_genres, epochs=10, batch_size=64):
    user_tensor = torch.LongTensor(reviews_df['user_idx'].values)
    book_tensor = torch.LongTensor(reviews_df['book_idx'].values)
    genre_tensor = torch.LongTensor(reviews_df['genre_idx'].values)
    ratings_tensor = torch.FloatTensor(reviews_df['rating'].values)
    dataset = TensorDataset(user_tensor, book_tensor, genre_tensor, ratings_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = BookRecommender(num_users, num_books, num_genres)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for user_idx, book_idx, genre_idx, rating in loader:
            optimizer.zero_grad()
            output = model(user_idx, book_idx, genre_idx)
            loss = criterion(output, rating)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
    torch.save(model.state_dict(), MODEL_PATH)
    return model


def train_torch_model(epochs=10, batch_size=64):
    books_df, users_df, reviews_df, user_id_map, book_id_map, genre_map = load_and_preprocess_data()
    num_users = len(user_id_map)
    num_books = len(book_id_map)
    num_genres = len(genre_map)
    model = train_model(reviews_df, num_users, num_books, num_genres, epochs, batch_size)
    torch.save(model.state_dict(), MODEL_PATH)
    save_all_data_pickle()  # сохраняем все данные и маппинги вместе
    return model


def recommend_books_for_user_torch(user_id, model, books_df, user_id_map, book_id_map, genre_map, reviews_df, top_n=5):
    model.eval()
    if user_id not in user_id_map:
        print("Unknown user")
        return []
    user_idx = user_id_map[user_id] 

    user_rated_books = set(reviews_df[reviews_df['user_id'] == user_id]['book_id'].values) 
    # Фильтруем книги, исключая уже оценённые
    candidate_books = [bid for bid in book_id_map.keys() if bid not in user_rated_books]
    if not candidate_books:
        print("No new books to recommend")
        return []

    book_indices = np.array([book_id_map[bid] for bid in candidate_books])
    books_df_indexed = books_df.set_index('id')
    genre_series = books_df_indexed.loc[candidate_books]['genre_id']
    genre_indices = genre_series.map(genre_map).values

    user_idx_tensor = torch.LongTensor([user_idx] * len(book_indices))
    book_idx_tensor = torch.LongTensor(book_indices)
    genre_idx_tensor = torch.LongTensor(genre_indices)

    with torch.no_grad():
        preds = model(user_idx_tensor, book_idx_tensor, genre_idx_tensor).numpy()
    top_indices = preds.argsort()[-top_n:][::-1]
    top_book_ids = [candidate_books[i] for i in top_indices]
    return Book.objects.filter(id__in=top_book_ids)


def recommend_books_for_user_simple(user_id, top_n=10):
    books_df, reviews_df, user_id_map, book_id_map, genre_map = load_all_data_pickle()
    model = BookRecommender(len(user_id_map), len(book_id_map), len(genre_map))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    return recommend_books_for_user_torch(user_id, model, books_df, user_id_map, book_id_map, genre_map, reviews_df, top_n=top_n)




# для метрик
def predict_torch(user_id, book_ids):
    books_df, reviews_df, user_id_map, book_id_map, genre_map = load_all_data_pickle()
    model = BookRecommender(len(user_id_map), len(book_id_map), len(genre_map))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    preds = {}
    if user_id not in user_id_map: # Неизвестный пользователь — возвращаем средний рейтинг 3.0
        for b in book_ids:
            preds[b] = 3.0
        return preds

    uid = user_id_map[user_id]

    books_df_indexed = books_df.set_index('id')

    user_idx_tensor = torch.LongTensor([uid] * len(book_ids))
    book_indices = []
    genre_indices = []

    for book_id in book_ids:
        if book_id in book_id_map:
            book_indices.append(book_id_map[book_id])
            if book_id in books_df_indexed.index:
                genre_id = books_df_indexed.loc[book_id]['genre_id']
                genre_idx = genre_map.get(genre_id, 0)
                genre_indices.append(genre_idx)
            else:
                genre_indices.append(0)  # жанр неизвестен
        else:
            # Книга неизвестна — положим индекс -1, потом отфильтруем
            book_indices.append(-1)
            genre_indices.append(0)

    # Отфильтруем неизвестные книги
    valid = [i for i, idx in enumerate(book_indices) if idx >= 0]
    if not valid: 
        for b in book_ids:
            preds[b] = 3.0
        return preds

    user_idx_tensor = user_idx_tensor[valid]
    book_idx_tensor = torch.LongTensor([book_indices[i] for i in valid])
    genre_idx_tensor = torch.LongTensor([genre_indices[i] for i in valid])

    with torch.no_grad():
        ratings_pred = np.atleast_1d(model(user_idx_tensor, book_idx_tensor, genre_idx_tensor).detach().cpu().numpy())


    for idx, i in enumerate(valid):
        preds[book_ids[i]] = float(ratings_pred[idx])
 
    for i in range(len(book_ids)):
        if i not in valid:
            preds[book_ids[i]] = 3.0

    return preds

 