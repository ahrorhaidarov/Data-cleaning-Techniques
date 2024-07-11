import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_excel('cleaning/data.xlsx')


def describe(raw_data: pd.DataFrame) -> pd.DataFrame:
    """"Gives full description of your data
    :raw_data: pandas DataFrame
    """
    int_data = raw_data.select_dtypes('int')

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

df.describe(include='all').T.to_excel('bdhbfdyhbfshdfb.xlsx')
describtion = describe(df)

describtion.to_excel('describtion.xlsx')


def analize_plot(raw_data: pd.DataFrame, columns: list):
    for column in columns:
        sns.displot(raw_data[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.show()

analize_plot(df, ['Customer ID', 'Account ID'])
df.select_dtypes('number')