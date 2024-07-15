import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Для моделей машинного обучения
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def detect_and_treat_outliers(df, column, method='iqr', treatment='remove'):
    """
    Обнаруживает и обрабатывает выбросы в заданном числовом столбце.

    :param df: DataFrame, содержащий данные
    :param column: str, название числового столбца
    :param method: str, метод обнаружения выбросов ('iqr' или 'z_score' или 'iso_forest' или 'lof')
        'iqr' - IQR (межквартильный размах). Идентификация точек данных, которые находятся за переделами диапазона 1.5*IQR
        'z_score' - Z-скор для идентификации точек данных, которые находятся на опреденное количество стандартных отклонений от среднего
        'iso_forest' - Isolation Forest идентифицирует аномалии через рекурсивное разбиение.
        'lof' - Локальный фактор выброса. Этот метод учитывает плотность соседних точек для идентификации выбросов.
    :param treatment: str, метод обработки выбросов ('remove', 'median', 'clip')
        'remove' - Удаление выбросов с помощью метода обнаружения выбросов
        'median' - Импутация медианой/средним. Этот метод сохраняет все точки данных, но заменяет выбросы на статистическую меру
        'clip' - Ограничение значений. Замена экстремальных значений на верхние или нижние пределы.
    :return: DataFrame с обработанными данными
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    elif method == 'z_score':
        z_scores = np.abs(stats.zscore(df[column]))
        threshold = 3  # Common threshold is 3
        outliers = df[z_scores > threshold]
    elif method == 'iso_forest':
        iso_forest = IsolationForest(contamination=0.01)
        df['anomaly'] = iso_forest.fit_predict(df[[column]])
        outliers = df[df['anomaly'] == -1]
    elif method == 'lof':
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
        df['anomaly'] = lof.fit_predict(df[[column]])
        outliers = df[df['anomaly'] == -1]
    else:
        raise ValueError("Invalid method parameter. Use 'iqr' or 'z_score'")

    print(f"Detected outliers: {len(outliers)}")

    if treatment == 'remove':
        df = df.drop(outliers.index)
    elif treatment == 'median':
        median = df[column].median()
        df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound), median, df[column])
    elif treatment == 'clip':
        df[column] = np.clip(df[column], lower_bound, upper_bound)
    else:
        raise ValueError("Invalid treatment parameter. Use 'remove', 'median', or 'clip'")

    return df


# Пример использования функции:
if __name__ == "__main__":
    # Добавление тестового DataFrame
    #df = pd.read_csv('data.csv')
    df = pd.DataFrame()
    
    print("Данные до обработки выбросов:")
    print(df.describe())

    
    # Обработка выбросов
    df_cleaned = df
    for column in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Boxplot для column1
        sns.boxplot(x=df[column], ax=axes[0])
        axes[0].set_title(f'Boxplot of {column}')
        
        # Distplot с KDE для column1
        sns.histplot(df[column], kde=True, ax=axes[1])  # Используйте histplot вместо displot
        axes[1].set_title(f'Distribution with KDE of {column}')

        plt.tight_layout()
        plt.show()
        
        df_cleaned = detect_and_treat_outliers(df_cleaned, column, method='iqr', treatment='remove')

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Boxplot для column1
        sns.boxplot(x=df_cleaned[column], ax=axes[0])
        axes[0].set_title(f'Boxplot of {column}')
        
        # Distplot с KDE для column1
        sns.histplot(df_cleaned[column], kde=True, ax=axes[1])  # Используйте histplot вместо displot
        axes[1].set_title(f'Distribution with KDE of {column}')

        plt.tight_layout()
        plt.show()
        

    print("Данные после обработки выбросов:")
    print(df_cleaned.describe())