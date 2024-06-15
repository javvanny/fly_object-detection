
# Используем базовый образ Python 3.11-slim
FROM python:3.11-slim

# Устанавливаем системные зависимости
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    git \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем ffmpeg и другие необходимые зависимости для обработки видео
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей в контейнер
COPY requirements.txt .

# Устанавливаем зависимости Python
RUN pip install --no-cache-dir -r requirements.txt

# Установка пакетов PyTorch с указанием пользовательского индекса
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Копируем все файлы проекта в контейнер
COPY . .

# Открываем порт, на котором будет работать FastAPI
EXPOSE 8000

# Команда для запуска приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
