FROM python:3.9-slim

# Устанавливаем системные зависимости для сборки пакетов с C/C++ расширениями
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    cython3 \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем только файл зависимостей для кеширования слоев
COPY requirements.txt /app/

# Устанавливаем Python-зависимости
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект после установки зависимостей
COPY . /app/

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]