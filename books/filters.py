import django_filters
from django.db import models
from .models import Book

class BookFilter(django_filters.FilterSet):
    genre = django_filters.CharFilter(field_name='genre__name', lookup_expr='icontains')
    author = django_filters.CharFilter(field_name='author__name', lookup_expr='icontains')
    cycle = django_filters.CharFilter(field_name='cycle_name', lookup_expr='icontains')
    series = django_filters.CharFilter(field_name='series__name', lookup_expr='icontains')
    publisher = django_filters.CharFilter(field_name='publisher__name', lookup_expr='icontains')


    tag = django_filters.CharFilter(method='filter_tag')

    class Meta:
        model = Book
        fields = ['genre', 'author', 'cycle', 'series', 'publisher']
        filter_overrides = {
            models.JSONField: {
                'filter_class': django_filters.CharFilter,
                'extra': lambda f: {
                    'lookup_expr': 'icontains',
                },
            },
        }

    def filter_tag(self, queryset, name, value):
        return queryset.filter(tags__name__icontains=value)
      