# books/serializers.py
from rest_framework import serializers
from books.models import Book, UserBookStatus, UserSearchQuery

class BookSerializer(serializers.ModelSerializer):
    class Meta:
        model = Book
        fields = ['id', 'title', 'author', 'description']   


class UserBookStatusSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserBookStatus
        fields = ['user', 'book', 'status', 'added_at', 'updated_at']
        read_only_fields = ['user', 'added_at', 'updated_at']

class BookAutocompleteSerializer(serializers.ModelSerializer):
    class Meta:
        model = Book
        fields = ['id', 'title']

class UserSearchQuerySerializer(serializers.ModelSerializer):
    class Meta:
        model = UserSearchQuery
        fields = ['query_text', 'frequency', 'last_searched']
