FROM python:3.11-slim

# Встановлюємо системні залежності (включаючи ffmpeg для роботи з аудіо)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Встановлюємо робочу директорію
WORKDIR /app

# Копіюємо файл залежностей та встановлюємо їх
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Створюємо папку для збереження та кешування моделей Whisper
RUN mkdir -p /app/models
ENV HF_HOME=/app/models

# Копіюємо вихідний код сервера
COPY main.py .

# Зазначаємо порт, який слухає сервер
EXPOSE 8002

# Запуск сервера через uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
