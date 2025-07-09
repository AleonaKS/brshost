from haystack import indexes
from .models import Book
  

class BookIndex(indexes.SearchIndex, indexes.Indexable):
    text = indexes.CharField(document=True, use_template=True)
    content_auto = indexes.NgramField(model_attr='title')
    title = indexes.CharField(model_attr='title')
    author = indexes.MultiValueField()
    author_auto = indexes.NgramField()
    
    tags = indexes.MultiValueField()
    genre = indexes.CharField(model_attr='genre__name', null=True)
    series = indexes.CharField(model_attr='series__name', null=True)
    cycle = indexes.CharField(model_attr='cycle__name', null=True)
    publisher = indexes.CharField(model_attr='publisher__name', null=True)
    soon = indexes.BooleanField(model_attr='soon')
    new = indexes.BooleanField(model_attr='new')
    year_of_publishing = indexes.IntegerField(model_attr='year_of_publishing')

    def get_model(self):
        return Book

    def prepare_author(self, obj):
        return [author.name for author in obj.author.all()]

    def prepare_author_auto(self, obj): 
        return ' '.join(author.name for author in obj.author.all())

    def prepare_tags(self, obj):
        return [tag.name for tag in obj.tags.all()]

    def index_queryset(self, using=None):
        return self.get_model().objects.all()
