from django.db import models
from django.contrib.auth.models import User
from django.utils.text import slugify
from django.core.exceptions import ValidationError
from django.utils import timezone


class SluggedModel(models.Model):
    name = models.CharField(max_length=100, unique=True)
    slug = models.SlugField(unique=True, blank=True)

    class Meta:
        abstract = True

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)
    
    def __str__(self):
        return self.name

class Author(SluggedModel):
    pass

class Genre(SluggedModel):
    pass

class Tag(SluggedModel):
    pass

class Series(SluggedModel):
    pass

class Publisher(SluggedModel):
    pass

class Cycle(SluggedModel):
    pass


class Book(models.Model):
    isbn = models.CharField(max_length=20, unique=True, null=True, blank=True)
    author = models.ManyToManyField(Author) 
    tags = models.ManyToManyField(Tag)
    genre = models.ForeignKey(Genre, on_delete=models.SET_NULL, null=True, blank=True)
    series = models.ForeignKey(Series, on_delete=models.SET_NULL, null=True, blank=True)
    publisher = models.ForeignKey(Publisher, on_delete=models.SET_NULL, null=True, blank=True)
    cycle =  models.ForeignKey(Cycle, on_delete=models.SET_NULL, null=True, blank=True)
    book_number_in_cycle = models.PositiveIntegerField(null=True, blank=True)
    title = models.CharField(max_length=300)
    soon = models.BooleanField(default=False)
    new = models.BooleanField(default=False)
    year_of_publishing = models.PositiveIntegerField(null=True, blank=True) 
    number_of_pages = models.PositiveIntegerField(null=True, blank=True)     # где-то учитывать
    age_restriction = models.CharField(max_length=3, blank=True, null=True)
    cover_type = models.CharField(max_length=30, blank=True, null=True)
    description = models.TextField()
    rating_chitai_gorod = models.FloatField(blank=True, null=True)   
    votes_chitai_gorod = models.PositiveIntegerField(default=0)
    rating_livelib = models.FloatField(blank=True, null=True) 
    votes_livelib = models.PositiveIntegerField(default=0)
    price = models.PositiveIntegerField()
    image_link = models.URLField()


class Review(models.Model):
    book = models.ForeignKey(Book, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=False)
    rating = models.FloatField(blank=True, null=True)  
    review_date = models.DateTimeField(null=True, blank=True)
    text = models.TextField(blank=True, null=True)


class BookRating(models.Model):
    book = models.ForeignKey(Book, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    rating = models.FloatField(null=True, blank=False) 
    rated_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = ('book', 'user')




class UserBookStatus(models.Model):
    STATUS_PURCHASED = 'PURCHASED'
    STATUS_CART = 'CART'
    STATUS_WISHLIST = 'WISHLIST'

    STATUS_CHOICES = [
        (STATUS_PURCHASED, 'Куплена'),
        (STATUS_CART, 'В корзине'),
        (STATUS_WISHLIST, 'Отложена'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    book = models.ForeignKey(Book, on_delete=models.CASCADE)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES)
    added_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('user', 'book', 'status') 

    def __str__(self):
        return f"{self.user.username} - {self.book.title} - {self.status}"


class UserSubscription(models.Model):
    SUBSCRIPTION_TYPES = [
        ('AUTHOR', 'Автор'),
        ('CYCLE', 'Цикл'),   
    ]
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    content_type = models.CharField(max_length=10, choices=SUBSCRIPTION_TYPES) 
    author = models.ForeignKey(Author, null=True, blank=True, on_delete=models.CASCADE)
    cycle = models.ForeignKey(Cycle, null=True, blank=True, on_delete=models.CASCADE)
    subscribed_at = models.DateTimeField(auto_now_add=True)


class BookView(models.Model):
    user = models.ForeignKey(User, null=True, blank=True, on_delete=models.CASCADE)
    session_key = models.CharField(max_length=40, null=True, blank=True)
    book = models.ForeignKey(Book, on_delete=models.CASCADE)
    viewed_at = models.DateTimeField(auto_now=True)
    duration_seconds = models.IntegerField(default=0)
    scroll_depth = models.IntegerField(null=True, blank=True)


class UserSearchQuery(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='search_queries')
    session_key = models.CharField(max_length=40, null=True, blank=True, db_index=True)
    query_text = models.CharField(max_length=255)
    frequency = models.PositiveIntegerField(default=1)  # количество повторений
    last_searched = models.DateTimeField(auto_now=True)  # время последнего поиска
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'query_text')
        indexes = [
            models.Index(fields=['user', 'query_text']),
        ]


class UserPreferences(models.Model):  
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    favorite_authors = models.ManyToManyField(Author, through='FavoriteAuthors', related_name='profile_users')
    favorite_genres = models.ManyToManyField(Genre, through='FavoriteGenres', related_name='profile_users')
    favorite_tags = models.ManyToManyField(Tag, through='FavoriteTags', related_name='profile_users')
    disliked_genres = models.ManyToManyField(Genre, related_name='disliked_by_users')
    disliked_tags = models.ManyToManyField(Tag, related_name='disliked_by_users')
    updated_at = models.DateTimeField(auto_now=True)
    

class FavoriteAuthors(models.Model): 
    userprofile = models.ForeignKey(UserPreferences, on_delete=models.CASCADE)
    author = models.ForeignKey(Author, on_delete=models.CASCADE)
    score = models.FloatField(default=0) 

    class Meta:
        unique_together = ('userprofile', 'author')


class FavoriteGenres(models.Model):
    userprofile = models.ForeignKey(UserPreferences, on_delete=models.CASCADE)
    genre = models.ForeignKey(Genre, on_delete=models.CASCADE)
    score = models.FloatField(default=0) 

    class Meta:
        unique_together = ('userprofile', 'genre')

    def clean(self): 
        if DislikedGenres.objects.filter(userprofile=self.userprofile, genre=self.genre).exists():
            raise ValidationError(f"Genre '{self.genre}' уже добавлен в disliked_genres для этого пользователя.")


class FavoriteTags(models.Model):
    userprofile = models.ForeignKey(UserPreferences, on_delete=models.CASCADE)
    tag = models.ForeignKey(Tag, on_delete=models.CASCADE)
    score = models.FloatField(default=0) 

    class Meta:
        unique_together = ('userprofile', 'tag')

    def clean(self):
        if DislikedTags.objects.filter(userprofile=self.userprofile, tag=self.tag).exists():
            raise ValidationError(f"Tag '{self.tag}' уже добавлен в disliked_tags для этого пользователя.")


class DislikedGenres(models.Model):
    userprofile = models.ForeignKey(UserPreferences, on_delete=models.CASCADE)
    genre = models.ForeignKey(Genre, on_delete=models.CASCADE)
    score = models.FloatField(default=0) 
    
    class Meta:
        unique_together = ('userprofile', 'genre')

    def clean(self): 
        if FavoriteGenres.objects.filter(userprofile=self.userprofile, genre=self.genre).exists():
            raise ValidationError(f"Genre '{self.genre}' уже добавлен в favorite_genres для этого пользователя.")


class DislikedTags(models.Model):
    userprofile = models.ForeignKey(UserPreferences, on_delete=models.CASCADE)
    tag = models.ForeignKey(Tag, on_delete=models.CASCADE)
    score = models.FloatField(default=0) 

    class Meta:
        unique_together = ('userprofile', 'tag')

    def clean(self):
        if FavoriteTags.objects.filter(userprofile=self.userprofile, tag=self.tag).exists():
            raise ValidationError(f"Tag '{self.tag}' уже добавлен в favorite_tags для этого пользователя.")

 



class BookRecommendation(models.Model):
    book = models.ForeignKey(Book, on_delete=models.CASCADE, related_name='recommendations')
    recommended_book = models.ForeignKey(Book, on_delete=models.CASCADE, related_name='recommended_by', null=True)
    similarity = models.FloatField(null=True)

    class Meta:
        unique_together = ('book', 'recommended_book')


class UserBookRecommendation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='book_recommendations')
    book = models.ForeignKey(Book, on_delete=models.CASCADE)
    score = models.FloatField()

    class Meta:
        unique_together = ('user', 'book')


class BookVector(models.Model):
    book = models.OneToOneField('Book', on_delete=models.CASCADE, related_name='vector')
    indices = models.JSONField(default=list, blank=True)
    values = models.JSONField(default=list, blank=True)
    w2v_vector = models.JSONField(default=list, blank=True)   







class Embedding(models.Model):
    content_type = models.CharField(max_length=50)  # 'author', 'genre', 'tag' и т.п.
    object_id = models.PositiveIntegerField()
    vector = models.JSONField()  # список float

    class Meta:
        unique_together = ('content_type', 'object_id')


class BookEmbedding(models.Model):
    """
    Эмбеддинг книги (например, из BERT) для similarity-рекомендаций.
    """
    book = models.OneToOneField('Book', on_delete=models.CASCADE, related_name='embedding')
    vector = models.JSONField(default=list, blank=True)  


class ReviewEmbedding(models.Model):
    """
    Эмбеддинг отзыва пользователя для анализа и рекомендаций.
    """
    review = models.OneToOneField('Review', on_delete=models.CASCADE, related_name='embedding')
    vector = models.JSONField(default=list, blank=True)  


from django.db import models
from django.contrib.auth.models import User

class UserTopicVector(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='topic_vector')
    vector = models.JSONField(default=list, blank=True)  # список чисел

    def __str__(self):
        return f"UserTopicVector(user={self.user.username})"

# ----------------------------------------------------------------------

# для дальнейшей работы с NLP (модель LDA)

# class BookTopicVector(models.Model):
#     """
#     Вектор распределения тем для книги, например, из LDA.
#     Храним два поля: индексы тем и значения вероятностей, чтобы экономить место.
#     """
#     book = models.OneToOneField('Book', on_delete=models.CASCADE, related_name='topic_vector')
#     indices = models.JSONField(default=list, blank=True)  # список индексов тем с ненулевой вероятностью
#     values = models.JSONField(default=list, blank=True)   # соответствующие вероятности

#     def get_dense_vector(self, num_topics=20):
#         """
#         Восстановить полный плотный вектор из разреженного.
#         """
#         vector = [0.0] * num_topics
#         for idx, val in zip(self.indices, self.values):
#             vector[idx] = val
#         return vector


# class BookEmbedding(models.Model):
#     """
#     Эмбеддинг книги (например, из BERT) для similarity-рекомендаций.
#     """
#     book = models.OneToOneField('Book', on_delete=models.CASCADE, related_name='embedding')
#     vector = models.JSONField(default=list, blank=True)  


# class ReviewEmbedding(models.Model):
#     """
#     Эмбеддинг отзыва пользователя для анализа и рекомендаций.
#     """
#     review = models.OneToOneField('Review', on_delete=models.CASCADE, related_name='embedding')
#     vector = models.JSONField(default=list, blank=True)  


# class ReviewSentiment(models.Model):
#     """
#     Тональность (сентимент) отзыва: число от -1 до 1.
#     """
#     review = models.OneToOneField('Review', on_delete=models.CASCADE, related_name='sentiment')
#     score = models.FloatField(null=True, blank=True)
