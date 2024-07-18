import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import category_encoders as ce

# Sample Data for Demonstration
data = {
    'Category': ['Low', 'Medium', 'High', 'Medium', 'Low'],
    'Color': ['Red', 'Green', 'Blue', 'Green', 'Red'],
    'Neighborhood': ['A', 'B', 'C', 'A', 'B'],
    'HousePrice': [100, 200, 300, 150, 250],
    'Fruit': ['Apple', 'Banana', 'Apple', 'Cherry', 'Banana'],
    'AgeGroup': ['Young', 'Middle-aged', 'Old', 'Young', 'Old']
}
df = pd.DataFrame(data)

print("Original DataFrame:\n", df, "\n")

# 1. Label Encoding
# Use Case:
# - Ordinal Data: Categories have a meaningful order or ranking (e.g., Low, Medium, High).
# - Tree-based Models: Decision trees, random forests.
le = LabelEncoder()
df['Category_Label_Encoded'] = le.fit_transform(df['Category'])
print("Label Encoding:\n", df[['Category', 'Category_Label_Encoded']], "\n")

# 2. One-Hot Encoding
# Use Case:
# - Nominal Data: Categories do not have a meaningful order (e.g., colors, types of fruits).
# - Linear Models: Linear regression, logistic regression.
df_one_hot_encoded = pd.get_dummies(df, columns=['Color'])
print("One-Hot Encoding:\n", df_one_hot_encoded[['Color_Red', 'Color_Green', 'Color_Blue']], "\n")

# 3. Binary Encoding
# Use Case:
# - High Cardinality: Categorical features with a large number of unique categories.
be = ce.BinaryEncoder(cols=['Color'])
df_binary_encoded = be.fit_transform(df[['Color']])
print("Binary Encoding:\n", df_binary_encoded, "\n")

# 4. Target Encoding
# Use Case:
# - High Cardinality & Large Datasets: High cardinality features, be cautious of data leakage.
# - K-Fold Cross-Validation: Validate using K-fold cross-validation to avoid overfitting.
te = ce.TargetEncoder(cols=['Neighborhood'])
df['Neighborhood_Target_Encoded'] = te.fit_transform(df['Neighborhood'], df['HousePrice'])
print("Target Encoding:\n", df[['Neighborhood', 'Neighborhood_Target_Encoded']], "\n")

# 5. Hash Encoding
# Use Case:
# - Memory Constraints: Very high cardinality features and memory constraints are a concern.
he = ce.HashingEncoder(cols=['Color'], n_components=3)
df_hash_encoded = he.fit_transform(df[['Color']])
print("Hash Encoding:\n", df_hash_encoded, "\n")

# 6. Frequency Encoding
# Use Case:
# - Importance of Frequency: Frequency of occurrence is more important than actual categories.
frequency_encoding = df['Fruit'].value_counts().to_dict()
df['Fruit_Frequency_Encoded'] = df['Fruit'].map(frequency_encoding)
print("Frequency Encoding:\n", df[['Fruit', 'Fruit_Frequency_Encoded']], "\n")

# 7. Leave-One-Out Encoding
# Use Case:
# - Reducing Bias: Reduce bias in scenarios similar to target encoding, useful in cross-validation.
loo = ce.LeaveOneOutEncoder(cols=['Neighborhood'])
df['Neighborhood_LOO_Encoded'] = loo.fit_transform(df['Neighborhood'], df['HousePrice'])
print("Leave-One-Out Encoding:\n", df[['Neighborhood', 'Neighborhood_LOO_Encoded']], "\n")

# 8. Ordinal Encoding
# Use Case:
# - Ordered Categories: Categorical feature has a natural ordinal relationship (e.g., age groups, ratings).
oe = OrdinalEncoder(categories=[['Young', 'Middle-aged', 'Old']])
df['AgeGroup_Ordinal_Encoded'] = oe.fit_transform(df[['AgeGroup']])
print("Ordinal Encoding:\n", df[['AgeGroup', 'AgeGroup_Ordinal_Encoded']], "\n")
