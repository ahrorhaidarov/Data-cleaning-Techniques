import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

# Определим функцию для масштабирования признаков
def scale_features(df, columns, method='minmax'):
    """
    Масштабирование признаков в датафрейме.

    Args:
    df (pd.DataFrame): Исходный датафрейм.
    columns (list): Список колонок для масштабирования.
    method (str): Метод масштабирования ('minmax', 'standard', 'robust', 'maxabs').

    Returns:
    pd.DataFrame: Датафрейм с масштабированными признаками.

    Resume:
    - Масштабирование по мин.-макс.: Подходит для алгоритмов, которые не предполагают какое-либо распределение данных.
    - Стандартизация: Полезно, когда данные следуют нормальному распределению.
    - Устойчивое масштабирование: Подходит для данных с выбросами.
    - MaxAbs масштабирование: Полезно для разреженных данных.
    """
    
    scaler = None
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'maxabs':
        scaler = MaxAbsScaler()
    else:
        raise ValueError("Unknown method. Available methods: 'minmax', 'standard', 'robust', 'maxabs'.")

    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df[columns])
    return df_scaled


if __name__ == "__main__":
    # Добавление тестового DataFrame
    # Пример использования функции
    data = {
        'Feature1': [10, 20, 30, 40, 50],
        'Feature2': [100, 200, 300, 400, 500],
        'NonNumeric': ['A', 'B', 'C', 'D', 'E']
    }

    df = pd.DataFrame(data)

    scaled_df = scale_features(df, columns=['Feature1', 'Feature2'], method='standard')
    print(scaled_df)