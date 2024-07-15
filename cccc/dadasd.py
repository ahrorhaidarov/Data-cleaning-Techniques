# import necessary functions from main.py file
from cccc.main import infer_and_convert_dtypes, describe, analize_plot, handle_missing_values
import pandas as pd



data = {
    'col1': ['yes', 'no', None, None, 'yes', 'yes', 'no', 'no', None, 'no'],
    'col2': ['a', None, 'c', 'a', 'b', 'c', 'a', 'b', None, 'a'],
    'col3': [1, 2, None, 2, 3, 1, 2, 3, None, 2],
    'col4': [True, None, True, True, False, None, True, True, False, None],
    'col5': [2.5, None, 2.9, 3.5, 2.7, 3.2, 2.9, 3.4, None, 3.3],
    'col6': ['2021-07-01', '2021-07-02', None, '2021-07-04', '2021-07-05', '2021-07-06', None, None, '2021-07-09', '2021-07-10'],
    'col7': [None, None, None, None, None, None, None, None, None, None]
}
df = pd.DataFrame(data)
df.info()

# Step 1
# Analizing

# Sub step 1
# Describe the row data
df_after_description = describe(df)
print(df_after_description)


# Sub step 2
# Analyze dtypes of row data and convert it to correct dtype if it's possible
df_after_conversion = infer_and_convert_dtypes(df)
print('Before conversion:')
df.info()
print('After conversion:')
df_after_conversion.info()


# Sub step 3
# Analyze of row data with plots
analize_plot(df_after_conversion, df_after_conversion.columns[1:])

# Sub step 4
# Analyze missing values and handle it if possible
df_after_conversion.isna().sum()
df_after_handling = handle_missing_values(df_after_conversion)
df_after_handling.info()


# Sub step 5
# Analyze dtypes of row data and convert it to correct dtype if it's possible again
df_after_conversion = infer_and_convert_dtypes(df_after_handling)
df_after_conversion.info()

# Step 2
# Analyze after process


# Step 1
# Analizing

# Sub step 1
# Describe the row data
df_after_description = describe(df_after_handling)
print(df_after_description)
df_after_description.to_excel('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.xlsx')

# Sub step 2
# Analyze dtypes of row data and convert it to correct dtype if it's possible
df_after_conversion = infer_and_convert_dtypes(df)
print('Before conversion:')
df.info()
print('After conversion:')
df_after_conversion.info()


# Sub step 3
# Analyze of row data with plots
analize_plot(df_after_conversion, df_after_conversion.columns[1:])

# Sub step 4
# Analyze missing values and handle it if possible
df_after_conversion.isna().sum()
df.isna().sum()
df_after_handling = handle_missing_values(df_after_conversion)



