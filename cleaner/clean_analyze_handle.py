import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import CategoricalDtype


def infer_and_convert_dtypes(row_data: pd.DataFrame, n_category: int = 10) -> pd.DataFrame:
    """
    Infers the best data type for each column in a DataFrame and converts it,
    ensuring that missing values are properly handled.

    :param row_data: input DataFrame
    :param n_category: threshold number of unique values for categorical data
    :return: DataFrame with converted data types

    :example:

    data = {
        'col1': ['yes', 'no', 'yes', None],
        'col2': ['a', 'b', 'c', 'd'],
        'col3': [1, 2, 3, None]
    }
    df = pd.DataFrame(data)
    converted_df = infer_and_convert_dtypes(df)
    print(converted_df.dtypes)
    col1    category
    col2    category
    col3    float64
    dtype: object
    """

    row_data = row_data.copy()
    for column in row_data.columns:
        col_data = row_data[column]
        unique_values = col_data.dropna().unique()
        unique_count = len(unique_values)



                # Check for datetime
        if pd.api.types.is_string_dtype(col_data):
            try:
                row_data[column] = pd.to_datetime(col_data)
            except (ValueError, TypeError):
                row_data[column] = col_data.astype('object')

                # Check for numeric data
        elif pd.api.types.is_numeric_dtype(col_data):
            if pd.api.types.is_integer_dtype(col_data) and col_data.isnull().any():
                row_data[column] = col_data.astype('float')
            else:
                row_data[column] = col_data.astype('float' if col_data.dtype.kind in 'fc' else 'int')

        else:
            row_data[column] = row_data[column].astype(str)

                # Check for categorical data
        if unique_count <= n_category and pd.api.types.is_object_dtype(col_data):
            if pd.api.types.is_string_dtype(col_data):
                ordered_values = sorted(unique_values)
                row_data[column] = col_data.astype(CategoricalDtype(categories=ordered_values, ordered=True))
            else:
                row_data[column] = col_data.astype('category')

    return row_data


def describe(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Дает полное описание ваших данных.

    :param raw_data: DataFrame pandas
    :return: DataFrame с описанием данных

    :example:

    data = {
        'col1': [1, 2, 3, None],
        'col2': ['a', 'b', 'a', 'b'],
        'col3': [1.1, 1.2, 1.3, None]
    }
    df = pd.DataFrame(data)
    description = describe(df)
    print(description)
    """

    int_data = raw_data.select_dtypes(['int', 'float'])

    data = int_data.describe().transpose()
    median = pd.DataFrame(int_data.median()).rename(columns={0: 'median'})
    data = data.join(median, how='outer')
    data = data.join(pd.DataFrame(index=raw_data.select_dtypes(exclude='int').columns), how='outer')

    data['dtype'] = data.index.map(lambda col: raw_data[col].dtype)
    data['count'] = data.index.map(lambda col: raw_data[col].shape[0])
    data['true values count'] = data.index.map(lambda col: raw_data[col].count())

    data['missing rate %'] = round((data['count'] - data['true values count']) / data['count'] * 100, 1)
    data['unique count'] = data.index.map(lambda col: len(raw_data[col].unique()))
    data['identical rate %'] = data.index.map(lambda col: round(raw_data[col].duplicated().sum() /
                                                                raw_data[col].shape[0] * 100, 1))

    data['mode'] = data.index.map(
        lambda col: raw_data[col].mode().values[0] if not raw_data[col].mode().empty else None)

    return data


def analize_plot(raw_data: pd.DataFrame, columns: list) -> sns.axisgrid:
    """
    Строит распределение каждого столбца в данных.

    :param raw_data: исходные данные
    :param columns: столбцы для построения графиков
    :return: объект Seaborn с построенными распределениями

    :example:

    data = {
        'col1': [1, 2, 3, 4, 5],
        'col2': [2, 3, 4, 5, 6],
        'col3': [5, 4, 3, 2, 1]
    }
    df = pd.DataFrame(data)
    columns_to_plot = ['col1', 'col2']
    plot = analize_plot(df, columns_to_plot)
    plt.show()
    """
    for column in columns:
        sns.displot(raw_data[column].astype(str), kde=True)
        plt.show()


def check_missing_rate(data: pd.DataFrame, column: str, threshold: int) -> bool:
    """
    Проверяет процент пропусков в столбце данных.

    :param data: DataFrame
    :param column: название столбца
    :param threshold: пороговое значение для пропусков
    :return: True, если процент пропусков меньше порогового значения, иначе False

    :example:

    data = {
        'col1': [1, 2, None],
        'col2': [2, None, None]
    }
    df = pd.DataFrame(data)
    check = check_missing_rate(df, 'col1', 0.5)
    print(check)
    True
    """
    rate = (data.shape[0] - data[column].count()) / data.shape[0]
    if rate >= threshold:
        return True
    return False


def has_outliers(df, column):
    """
    Проверяет, есть ли выбросы в указанном столбце DataFrame с использованием метода IQR.

    :param df: DataFrame для проверки
    :param column: название столбца для проверки выбросов
    :return: True, если имеются выбросы, иначе False

    :example:

    data = {
        'col1': [1, 2, 100],
        'col2': [2, 3, 4]
    }
    df = pd.DataFrame(data)
    outliers = has_outliers(df, 'col1')
    print(outliers)
    True
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return not outliers.empty


def handle_missing_values(raw_data: pd.DataFrame, break_list: list = [], threshold: int = 0.5) -> pd.DataFrame:
    """
    Обрабатывает пропущенные значения в DataFrame в зависимости от типа данных и наличия выбросов.

    :param raw_data: исходный DataFrame
    :param break_list: список столбцов, которые не нужно обрабатывать
    :param threshold: пороговое значение для пропусков
    :return: DataFrame с обработанными пропущенными значениями

    :example:

    data = {
        'col1': [1, 2, None],
        'col2': [None, None, None],
        'col3': [2, 3, 4]
    }
    df = pd.DataFrame(data)
    break_list = ['col2']
    handled_df = handle_missing_values(df, break_list)
    print(handled_df)
    """

    raw_data = raw_data.copy()
    for column in raw_data.drop(break_list, axis=1).columns:
        col_data = raw_data[column]

        if check_missing_rate(raw_data, column, threshold):
            raw_data.drop(column, axis=1, inplace=True)
            print(column, 'column deleted!!!')
            continue

        if pd.api.types.is_categorical_dtype(col_data):
            mode = raw_data[column].mode().values[0]
            print(column, 'mode', mode)
            raw_data[column].fillna(mode, inplace=True)

        elif pd.api.types.is_numeric_dtype(col_data) and has_outliers(raw_data, column):
            median = raw_data[column].median()
            print(column, 'median', median)
            raw_data[column].fillna(median, inplace=True)

        elif pd.api.types.is_numeric_dtype(col_data) and not has_outliers(raw_data, column):
            mean = raw_data[column].mean()
            print(column, 'mean', mean)
            raw_data[column].fillna(mean, inplace=True)
        else:
            print('Can not handle!!!')


    return raw_data


if __name__ == '__main__':
    data = {
        'col1': ['yes', 'no', None, None, 'yes', 'yes', 'no', 'no', None, 'no'],
        'col2': ['a', None, 'c', 'a', 'b', 'c', 'a', 'b', None, 'a'],
        'col3': [1, 2, None, 2, 3, 1, 2, 3, None, 2],
        'col4': [True, None, True, True, False, None, True, True, False, None],
        'col5': [2.5, None, 2.9, 3.5, 2.7, 3.2, 2.9, 3.4, None, 3.3],
        'col6': ['2021-07-01', '2021-07-02', None, '2021-07-04', '2021-07-05', '2021-07-06', None, None, '2021-07-09',
                 '2021-07-10'],
        'col7': [None, None, None, None, None, None, None, None, None, None]
    }
    df = pd.DataFrame(data)
    df.info()

    # Step 1
    # Describe the row data
    df_after_description = describe(df)
    print(df_after_description)

    # Step 2
    # Analyze dtypes of row data and convert it to correct dtype if it's possible
    df_after_conversion = infer_and_convert_dtypes(df)
    print('Before conversion:')
    df.info()
    print('After conversion:')
    df_after_conversion.info()

    # Step 3
    # Analyze of row data with plots
    analize_plot(df_after_conversion, df_after_conversion.columns[1:])

    # Step 4
    # Analyze missing values and handle it if possible
    df_after_conversion.isna().sum()
    df_after_handling = handle_missing_values(df_after_conversion)
    df_after_handling.info()