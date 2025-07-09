import os
import subprocess
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = 'Запускает scrapy и reviews.py'

    def handle(self, *args, **options):
        # Абсолютный путь к файлу команды parsing.py
        current_file = os.path.abspath(__file__)
        # Поднимаемся на 4 уровня вверх, чтобы попасть в корень books_site/
        base_dir = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(current_file)
                )
            )
        )
        # Путь к папке spiders
        spiders_dir = os.path.join(base_dir, 'scraper', 'parsing_books', 'spiders')

        self.stdout.write(f'Рабочая директория для запуска: {spiders_dir}')

        # Запуск scrapy crawl books
        result = subprocess.run(
            ['scrapy', 'crawl', 'books'],
            cwd=spiders_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            self.stdout.write(self.style.SUCCESS('Scrapy успешно завершился'))
        else:
            self.stderr.write(self.style.ERROR(f'Ошибка scrapy: {result.stderr}'))

        # Запуск reviews.py
        result = subprocess.run(
            ['python3', 'reviews.py'],
            cwd=spiders_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            self.stdout.write(self.style.SUCCESS('reviews.py успешно завершился'))
        else:
            self.stderr.write(self.style.ERROR(f'Ошибка reviews.py: {result.stderr}'))




        # заполнение бд с книгами
        self.stdout.write(f'Запуск import_books.py из {base_dir}')
        result = subprocess.run(
            ['python3', 'import_books.py'],
            cwd=base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            self.stdout.write(self.style.SUCCESS('Заполнение бд с книгами успешно завершилось'))
        else:
            self.stderr.write(self.style.ERROR(f'Ошибка import_books.py: {result.stderr}'))

        # заполнение бд с рецензиями
        self.stdout.write(f'Запуск import_reviews.py из {base_dir}')
        result = subprocess.run(
            ['python3', 'import_reviews.py'],
            cwd=base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            self.stdout.write(self.style.SUCCESS('Заполнение бд с рецензиями успешно завершилось'))
        else:
            self.stderr.write(self.style.ERROR(f'Ошибка import_reviews.py: {result.stderr}'))

        # заполнение пользовательских предпочтений
        self.stdout.write(f'Запуск user_preferences.py из {base_dir}')
        result = subprocess.run(
            ['python3', 'user_preferences.py'],
            cwd=base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        ) 
        if result.returncode == 0:
            self.stdout.write(self.style.SUCCESS('Заполнение пользовательских предпочтений успешно завершилось'))
        else:
            self.stderr.write(self.style.ERROR(f'Ошибка user_preferences.py: {result.stderr}'))