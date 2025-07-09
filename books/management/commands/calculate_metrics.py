from django.core.management.base import BaseCommand
from books.recommendations.calculate_metrics import calculate_metrics_for_users_extended #, aggregate_metrics_over_users, calculate_metrics_for_users


class Command(BaseCommand):
    help = 'Вычисляет метрики рекомендаций и сохраняет в CSV'

    def handle(self, *args, **options):
        TOP_N_list = [10, 20, 30, 40]
        calculate_metrics_for_users_extended(TOP_N_list=TOP_N_list, csv_path='metrics_results_2.csv')
      