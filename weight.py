import pandas as pd

# Загружаем данные
data = pd.read_csv("C:/Users/ivox/PycharmProjects/pythonProject/turism.csv")


def normalize_column(column):
    """Нормализует значения колонки в диапазоне от 0 до 1."""
    return (column - column.min()) / (column.max() - column.min()) if column.max() != column.min() else 0


def normalize_all_columns(data):
    """Нормализует все числовые столбцы в датафрейме."""
    normalized_data = data.copy()
    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        normalized_data[col] = normalize_column(data[col])
    return normalized_data


def calculate_weights(preferences, data):
    """Автоматически рассчитывает веса для каждого критерия на основе выбора пользователя."""
    total_categories = len(data.columns)
    weights = {}

    for key in data.columns:
        if key in preferences:
            user_choices = preferences[key]
            if isinstance(user_choices, list):
                weights[key] = len(user_choices)
            else:
                weights[key] = 1
        else:
            # Равномерное распределение веса для неуказанных категорий
            weights[key] = 1 / total_categories

    total_weight = sum(weights.values())
    return {k: v / total_weight for k, v in weights.items()}


def calculate_score_dynamic(data, preferences):
    """Рассчитывает общий и пользовательский Score для мест."""
    # Нормализуем все числовые столбцы
    normalized_data = normalize_all_columns(data)

    # Расчет весов
    weights = calculate_weights(preferences, data)

    # Универсальная функция для расчета категории
    def calculate_category_score(row, preferred_values, column_prefix=None):
        if not preferred_values:
            return 0

        if column_prefix:
            # Для категориальных данных (например, terrains или activity_types)
            score = 0
            for value in preferred_values:
                category_column = f"{column_prefix}_{value.lower()}"
                if category_column in row and row[category_column] == 1:
                    score += 1
            return score / len(preferred_values)
        else:
            # Для числовых данных (например, цена, связь)
            return row if isinstance(row, (float, int)) else 0

    # Рассчитываем пользовательский и общий Score
    def calculate_row_score(row, use_preferences=True):
        score = 0
        for col in normalized_data.columns:
            if col not in weights:
                continue

            if use_preferences and col in weights:
                weight = weights[col]
            else:
                weight = 1 / len(normalized_data.columns)

            if col in preferences:
                if isinstance(preferences[col], list):
                    score += calculate_category_score(row, preferences[col], col) * weight
                elif isinstance(row[col], (float, int)):
                    score += row[col] * weight
            elif isinstance(row[col], (float, int)):
                score += row[col] * weight
        return score

    # Применяем расчет
    normalized_data['custom_score'] = normalized_data.apply(lambda row: calculate_row_score(row, use_preferences=True),
                                                            axis=1)
    normalized_data['overall_score'] = normalized_data.apply(
        lambda row: calculate_row_score(row, use_preferences=False), axis=1)

    return normalized_data.sort_values(by='custom_score', ascending=False)


# Пример предпочтений пользователя
user_preferences = {
    "approximate_cost": "дорого",  # Цена
    "network_availability": "хорошая связь",  # Связь
    "terrains": ["пляж", "лес", "пещера"],  # Предпочтительные местности
    "activity_types": ["активный", "расслабленный"]  # Предпочтительные типы отдыха
}

# Рассчитываем Score
ranked_data = calculate_score_dynamic(data, user_preferences)

# Выводим топ-5 мест по пользовательскому Score
print(ranked_data[['name', 'approximate_cost', 'location']].head())

