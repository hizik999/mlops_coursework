# Курсовая работа по дисицплине MLOps на тему "Разработка MLOps-пайплайна для автоматической классификации новостных текстов по достоверности"
Выполнил Малышев Андрей, ШАД-311

## Описание проекта

Этот проект реализует MLOps-пайплайн для автоматической классификации новостных текстов на фейковые и реальные.  
Используется датасет Fake and Real News Dataset (Kaggle), модели Logistic Regression / RandomForest, векторизация TF-IDF и FastAPI-сервис для инференса.  
Пайплайн поддерживает DVC, Docker и CI/CD на GitHub Actions. Инференс сервер реализован на FastAPI.

---

## Структура репозитория
```
mlops_coursework/
│
├── configs/  
│   ├── train.yaml  
│   └── inference.yaml  
│
├── data/  
├── models/
│
├── src/  
│   ├── api/  
│   │   ├── app.py  
│   │   └── schemas.py  
│   ├── data/  
│   │   ├── split_dataset.py  
│   │   └── preprocess.py  
│   ├── features/  
│   │   └── build_features.py  
│   ├── models/  
│   │   ├── train.py  
│   │   └── infer.py  
│   └── utils/  
│
├── tests/  
│   └── test_health.py  
│
├── dvc.yaml  
├── Dockerfile  
├── requirements.txt  
├── README.md  
├── MODEL_CARD.md  
└── DATASET_CARD.md
```
---

## Установка окружения

```
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Обучение моделей

### 1. Разделение датасета
```
python -m src.data.split_dataset
```

### 2. Запуск DVC-пайплайна
после repro можно написать название стадии, которую необходимо запустить

```
dvc repro
dvc repro stage_name
```


### 3. Эксперименты

```
dvc exp run -S model.name=logreg -S data.version=v1
dvc exp run -S model.name=rf -S data.version=v2
```

### 4. Просмотр экспериментов
```
dvc exp show
```

### 5. Версионирование датасета и препроцессинга
```
dvc repro preprocess_v1
dvc repro preprocess_v2
```

### 6. Просмотр структуры пайплайна
```
dvc dag
```

---

## Запуск FastAPI сервиса

```
uvicorn src.api.app:app --reload
```


Доступные маршруты:
- `/health`
- `/predict`
- `/docs` (Swagger UI)

---

## Docker

### Сборка

```
docker build -t fake-news-api .
```

### Запуск

```
docker run -p 8000:8000 fake-news-api
```

Проверка

```
curl http://127.0.0.1:8000/health
```
Документация (Swagger UI)
```
http://127.0.0.1:8000/docs
```

---

## CI/CD в GitHub Actions

Пайплайн состоит из 4 стадий:

1. **build** — проверка сборки проекта  
2. **codecheck** — ruff + mypy  
3. **tests** — pytest  
4. **deploy** — сборка Docker-образа, запуск контейнера, health-check  

Файл workflow: `.github/workflows/ci.yml`.

---
