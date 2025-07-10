FROM python:3.9-slim

RUN apt-get update && apt-get install -y build-essential python3-dev libpq-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

EXPOSE 8080

CMD ["gunicorn", "books_site.wsgi:application", "--bind", "0.0.0.0:8080"]
