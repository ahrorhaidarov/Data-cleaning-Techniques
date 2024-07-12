import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import CategoricalDtype




def convert_to_bool(value):
    """Converts various string/numeric representations to boolean."""
    true_values = {'true', 'yes', '1'}
    false_values = {'false', 'no', '0'}

    value_str = str(value).strip().lower()
    if value_str in true_values:
        return True
    elif value_str in false_values:
        return False
    else:
        raise ValueError(f"Cannot convert {value} to boolean")


def infer_and_convert_dtypes(row_data: pd.DataFrame, n_category: int = 10) ->pd.DataFrame:
    """
    Infers the best data type for each column in the DataFrame and converts it accordingly.
    """
    row_data = row_data.copy()
    for column in row_data.columns:
        col_data = row_data[column]

        # Check for booleans and convert
        if (col_data.dropna().astype(str).str.lower().isin(['true', 'false', 'yes', 'no']).all() or
            col_data.dropna().isin([0, 1]).all()):
            row_data[column] = col_data.apply(lambda x: convert_to_bool(x))

        # check for categorical
        elif col_data.dropna().nunique() <= n_category:
            row_data[column] = col_data.astype('category')

        # check for ordered categories (ordinal)
        elif pd.api.types.is_string_dtype(col_data) and len(col_data.unique()) <= n_category:
            ordered_values = sorted(col_data.unique())
            row_data[column] = col_data.astype(CategoricalDtype(categories=ordered_values, ordered=True))

        # check for datetime
        elif pd.api.types.is_string_dtype(col_data):
            try:
                row_data[column] = pd.to_datetime(col_data)
            except (ValueError, TypeError):
                pass

        # check for numeric data
        elif pd.api.types.is_numeric_dtype(col_data):
            if pd.api.types.is_integer_dtype(col_data) and col_data.isnull().any():
                row_data[column] = col_data.astype('int')
            else:
                row_data[column] = col_data.astype('float')

        else:
            row_data[column] = col_data.astype('object')

    return row_data


def describe(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Gives full description of your data
    :param raw_data: pandas DataFrame
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
    Plots distribution of each column in the raw data

    :param raw_data: source data
    :param columns: columns to plot

    :return: Seaborn object with the plotted distributions.

    :example:

    # >>> import pandas as pd
    # >>> import seaborn as sns
    # >>> import matplotlib.pyplot as plt
    >>> # Creating a sample DataFrame
    # >>> data = {
    # >>>     'col1': [1, 2, 3, 4, 5],
    # >>>     'col2': [2, 3, 4, 5, 6],
    # >>>     'col3': [5, 4, 3, 2, 1]
    # >>> }
    # >>> df = pd.DataFrame(data)
    # >>> columns_to_plot = ['col1', 'col2']
    # >>> plot = analize_plot(df, columns_to_plot)
    # >>> plt.show()"""

    for column in columns:
        sns.displot(raw_data[column], kde=True)
        plt.show()




def check_missing_rate(data: pd.DataFrame, column: str, thresold: int) -> bool:
    if data.shape[0] - data[column].count() / data.shape[0] >= thresold:
        return False
    return True


def has_outliers(df, column):
    """
    Check if the specified column in the DataFrame has outliers using the IQR method.

    Parameters:
    df (pandas.DataFrame): The DataFrame to check.
    column (str): The column name to check for outliers.

    Returns:
    bool: True if there are outliers, False otherwise.
    """
    # Calculate Q1 (25th quantile) and Q3 (75th quantile)
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    # Calculate IQR
    IQR = Q3 - Q1

    # Determine the lower and upper bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

def handle_missing_values(raw_data: pd.DataFrame, break_list: list, threshold: int = 0.5) -> pd.DataFrame:
    raw_data = raw_data.copy()
    for column in raw_data.drop(break_list, axis=1).columns:
        col_data = raw_data[column]
        if check_missing_rate(raw_data, column, threshold):
            raw_data.drop(column, axis=1, inplace=True)
            continue

        if pd.api.types.is_categorical_dtype(col_data) or pd.api.types.is_bool_dtype(col_data):
            mode = raw_data[column].mode()[0]
            print('mode', mode)
            raw_data[column].fillna(mode, inplace=True)

        elif pd.api.types.is_float_dtype(col_data)
            median = raw_data[column].median()
            print('median', median)
            raw_data[column].fillna(median, inplace=True)
        elif key == 'mode':
            mean = raw_data[column].mean()
            print('mean', mean)
            raw_data[column].fillna(mean, inplace=True)
        else:
            print('Key not recognized!!!', key)
            break

return raw_data



