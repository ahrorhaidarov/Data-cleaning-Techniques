# import necessary functions from main.py file
from cccc.main import infer_and_convert_dtypes, describe, analize_plot, handle_missing_values
import pandas as pd



df = pd.read_csv('cccc/smalldata.csv')

# Step 1
# Describe the row data
df_after_description = describe(df)
print(df_after_description)

df_after_description.to_excel('described.xlsx')

# Step 2
# Analyze dtypes of row data and convert it to correct dtype if it's possible
df_after_conversion = infer_and_convert_dtypes(df)
print('Before conversion:', df.info())
print('After conversion:', df_after_conversion.info())


# Step 3
# Analyze of row data with plots
analize_plot(df_after_conversion, df_after_conversion.columns[1:])

# step 4
# Analyze missing values and handle it if possible
df_after_conversion.isna().sum()
df_after_conversion.info()



filter = {
    'mean': ['Value1', 'Value2', 'Value3', 'Value4', 'Value5'],
    'mode': ['IsActive', 'Score', 'Category2', 'Category']
    }

df_after_handling_missing = handle_missing_values(df_after_conversion, filter, threshold=0.61)
df_after_handling_missing.isna().sum()


for i in df.columns:
    print(df[i].dtype)


pd.api.types.is_categorical_dtype(df_after_conversion['Category'])