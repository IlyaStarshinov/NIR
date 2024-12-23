import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Определение классов для данных
class Business:
    def __init__(self, business_id, name, stars, categories):
        self.business_id = business_id
        self.name = name
        self.stars = stars
        self.categories = categories

class Review:
    def __init__(self, user_id, business_id, stars, text, review_id):
        self.review_id = review_id
        self.user_id = user_id
        self.business_id = business_id
        self.stars = stars
        self.text = text

class User:
    def __init__(self, user_id, average_stars, yelping_since):
        self.user_id = user_id
        self.average_stars = average_stars
        self.yelping_since = yelping_since

# Класс для загрузки данных
class Dataset:
    def __init__(self):
        self.businesses = []
        self.reviews = []
        self.users = []
        self.reviews_df = None  # Для удобства работы с отзывами

    def load_data(self, business_file, review_file, user_file, sample_size=1000):
        # Загрузка данных о бизнесах
        print("Загружаем данные о бизнесах...")
        with open(business_file, 'r', encoding='utf-8') as f:
            business_data = [json.loads(line) for _, line in zip(range(1000), f)]
        self.businesses = [Business(**{k: business[k] for k in ['business_id', 'name', 'stars', 'categories'] if k in business}) 
                           for business in business_data]

        # Загрузка данных об отзывах с ограничением
        print("Загружаем данные об отзывах...")
        with open(review_file, 'r', encoding='utf-8') as f:
            count = 0
            for line in f:
                if count >= sample_size:
                    break
                data = json.loads(line)
                self.reviews.append(Review(
                    user_id=data.get('user_id'),
                    business_id=data.get('business_id'),
                    stars=data.get('stars'),
                    text=data.get('text'),
                    review_id=data.get('review_id')  # Добавляем поддержку review_id
                ))
                count += 1
        # Загрузка данных о пользователях
        print("Загружаем данные о пользователях...")
        with open(user_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                self.users.append(User(
                    user_id=data.get('user_id'),
                    average_stars=data.get('average_stars'),
                    yelping_since=data.get('yelping_since')
                ))

        # Создаем DataFrame для отзывов
        self.reviews_df = pd.DataFrame([{
            'user_id': review.user_id,
            'business_id': review.business_id,
            'stars': review.stars,
            'text': review.text
        } for review in self.reviews])

    def print_summary(self):
        print("\nСводка по загруженным данным:")
        print(f"Количество бизнесов: {len(self.businesses)}")
        print(f"Количество отзывов: {len(self.reviews)}")
        print(f"Количество пользователей: {len(self.users)}")

        print("\nПример бизнеса:")
        print(vars(self.businesses[0]) if self.businesses else "Нет данных")

        print("\nПример отзыва:")
        print(vars(self.reviews[0]) if self.reviews else "Нет данных")

        print("\nПример пользователя:")
        print(vars(self.users[0]) if self.users else "Нет данных")

# Класс для системы рекомендаций
class RecommenderSystem:
    def __init__(self, reviews_df):
        self.reviews_df = reviews_df
        self.user_item_matrix = None
        self.similarity_matrix = None

    def prepare_data(self):
        print("Готовим матрицу пользователь-бизнес...")
        self.user_item_matrix = self.reviews_df.pivot_table(
            index='user_id',
            columns='business_id',
            values='stars',
            fill_value=0
        )

    def calculate_similarity(self, method='user'):
        print(f"Вычисляем {method}-based схожесть...")
        if method == 'user':
            self.similarity_matrix = cosine_similarity(self.user_item_matrix)
        elif method == 'item':
            self.similarity_matrix = cosine_similarity(self.user_item_matrix.T)

    def recommend(self, target_id, method='user', top_k=5):
        print(f"Генерируем рекомендации ({method}-based) для пользователя {target_id}...")
        self.calculate_similarity(method)

        if method == 'user':
            idx = self.user_item_matrix.index.get_loc(target_id)
            similar_users = np.argsort(-self.similarity_matrix[idx])[:top_k+1]
            similar_scores = self.user_item_matrix.iloc[similar_users].mean().sort_values(ascending=False)
        else:  # Item-based
            user_ratings = self.user_item_matrix.loc[target_id]
            scores = user_ratings.dot(self.similarity_matrix)
            similar_scores = pd.Series(scores, index=self.user_item_matrix.columns).sort_values(ascending=False)
        
        return similar_scores.head(top_k)

# Функция для оценки рекомендаций
def calculate_metrics(recommended, ground_truth, k=5):
    recommended = recommended[:k]  # Обрезаем до топ-k
    ground_truth = set(ground_truth)

    # Precision и Recall
    hits = len(set(recommended) & ground_truth)
    precision = hits / len(recommended) if recommended else 0
    recall = hits / len(ground_truth) if ground_truth else 0

    # NDCG
    dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(recommended) if item in ground_truth)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    ndcg = dcg / idcg if idcg > 0 else 0

    return precision, recall, ndcg

# Основной запуск кода
if __name__ == "__main__":
    dataset = Dataset()
    try:
        # Загружаем данные
        dataset.load_data(
            'yelp_academic_dataset_business.json',
            'yelp_academic_dataset_review.json',
            'yelp_academic_dataset_user.json',
            sample_size=1000
        )
        dataset.print_summary()

        # Инициализируем систему рекомендаций
        recommender = RecommenderSystem(dataset.reviews_df)
        recommender.prepare_data()

        # Получаем target_user (первый пользователь)
        target_user = recommender.user_item_matrix.index[0]
        print(f"\nЦелевой пользователь: {target_user}")

        # User-based рекомендации
        user_recommendations = recommender.recommend(target_user, method='user')
        print(f"\nUser-based рекомендации для пользователя {target_user}:")
        print(user_recommendations)

        # Item-based рекомендации
        item_recommendations = recommender.recommend(target_user, method='item')
        print(f"\nItem-based рекомендации для пользователя {target_user}:")
        print(item_recommendations)

        # Истинные значения (ground truth)
        y_true = dataset.reviews_df[dataset.reviews_df['user_id'] == target_user]['business_id'].tolist()

        # Метрики для User-based
        user_pred = user_recommendations.index.tolist()
        precision_user, recall_user, ndcg_user = calculate_metrics(user_pred, y_true, k=5)
        print("\nUser-based Метрики:")
        print(f"Precision: {precision_user:.4f}, Recall: {recall_user:.4f}, NDCG: {ndcg_user:.4f}")

        # Метрики для Item-based
        item_pred = item_recommendations.index.tolist()
        precision_item, recall_item, ndcg_item = calculate_metrics(item_pred, y_true, k=5)
        print("\nItem-based Метрики:")
        print(f"Precision: {precision_item:.4f}, Recall: {recall_item:.4f}, NDCG: {ndcg_item:.4f}")

    except Exception as e:
        print(f"Произошла ошибка: {e}")
        

import matplotlib.pyplot as plt
import seaborn as sns

# График рекомендаций для пользователя
def plot_recommendations(recommendations, method, target_user):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=recommendations.index, y=recommendations.values)
    plt.title(f'Top {len(recommendations)} Recommendations for User {target_user} ({method}-based)')
    plt.xlabel('Business ID')
    plt.ylabel('Average Rating')
    plt.xticks(rotation=45)
    plt.show()

# График метрик
def plot_metrics(precision_user, recall_user, ndcg_user, precision_item, recall_item, ndcg_item):
    metrics = ['Precision', 'Recall', 'NDCG']
    user_values = [precision_user, recall_user, ndcg_user]
    item_values = [precision_item, recall_item, ndcg_item]

    df = pd.DataFrame({
        'Metric': metrics,
        'User-based': user_values,
        'Item-based': item_values
    })

    df.set_index('Metric', inplace=True)
    df.plot(kind='bar', figsize=(10, 6))
    plt.title('Metrics for User-based vs Item-based Recommendations')
    plt.ylabel('Score')
    plt.show()

# График распределения оценок пользователей
def plot_user_rating_distribution(reviews_df):
    plt.figure(figsize=(8, 6))
    sns.histplot(reviews_df['stars'], bins=20, kde=True)
    plt.title('Distribution of User Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.show()

# График распределения оценок бизнесов
def plot_business_rating_distribution(businesses):
    business_ratings = [business.stars for business in businesses]
    plt.figure(figsize=(8, 6))
    sns.histplot(business_ratings, bins=20, kde=True)
    plt.title('Distribution of Business Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.show()

# Визуализация
if __name__ == "__main__":
    # Визуализируем рекомендации
    plot_recommendations(user_recommendations, 'User', target_user)
    plot_recommendations(item_recommendations, 'Item', target_user)

    # Визуализируем метрики
    plot_metrics(precision_user, recall_user, ndcg_user, precision_item, recall_item, ndcg_item)

    # Визуализируем распределение рейтингов
    plot_user_rating_distribution(dataset.reviews_df)
    plot_business_rating_distribution(dataset.businesses)
