from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from .models import BookRating, UserSubscription, BookView, UserPreferences, Author, Tag, Genre


class SignUpForm(UserCreationForm): 
    class Meta:
        model = User
        fields = ('username', 'password1', 'password2')

class LoginForm(AuthenticationForm):
    username = forms.CharField(label='Имя пользователя')
    password = forms.CharField(label='Пароль', widget=forms.PasswordInput)


class BookRatingForm(forms.ModelForm):
    class Meta:
        model = BookRating
        fields = ['book', 'rating']

class UserSubscriptionForm(forms.ModelForm):
    class Meta:
        model = UserSubscription
        fields = ['content_type', 'author', 'cycle']

class BookViewForm(forms.ModelForm):
    class Meta:
        model = BookView
        fields = ['book', 'duration_seconds', 'scroll_depth']

# class UserPreferencesForm(forms.ModelForm):
#     class Meta:
#         model = UserPreferences
#         fields = []    

from django import forms
from .models import Author, Genre, Tag, UserPreferences, FavoriteAuthors, FavoriteGenres, FavoriteTags, DislikedGenres, DislikedTags

from django import forms

class UserPreferencesForm(forms.ModelForm):
    favorite_authors = forms.ModelMultipleChoiceField(
        queryset=Author.objects.all(),
        required=False,
        widget=forms.CheckboxSelectMultiple
    )
    favorite_genres = forms.ModelMultipleChoiceField(
        queryset=Genre.objects.all(),
        required=False,
        widget=forms.CheckboxSelectMultiple
    )
    favorite_tags = forms.ModelMultipleChoiceField(
        queryset=Tag.objects.all(),
        required=False,
        widget=forms.CheckboxSelectMultiple
    )
    disliked_genres = forms.ModelMultipleChoiceField(
        queryset=Genre.objects.all(),
        required=False,
        widget=forms.CheckboxSelectMultiple
    )
    disliked_tags = forms.ModelMultipleChoiceField(
        queryset=Tag.objects.all(),
        required=False,
        widget=forms.CheckboxSelectMultiple
    ) 

    class Meta:
        model = UserPreferences
        fields = []

    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user')
        super().__init__(*args, **kwargs)

        # Получаем выбранные объекты из instance для каждого поля
        fav_authors = self.instance.favorite_authors.all()
        fav_genres = self.instance.favorite_genres.all()
        fav_tags = self.instance.favorite_tags.all()
        dis_genres = self.instance.disliked_genres.all()
        dis_tags = self.instance.disliked_tags.all()

        # Устанавливаем начальные выбранные значения
        self.fields['favorite_authors'].initial = fav_authors
        self.fields['favorite_genres'].initial = fav_genres
        self.fields['favorite_tags'].initial = fav_tags
        self.fields['disliked_genres'].initial = dis_genres
        self.fields['disliked_tags'].initial = dis_tags
