import pandas as pd
from helper_fun import generate_plot
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('data/raw/MathE.csv', sep=';', encoding='latin-1')
categorical_single = ["Student Country", "Question Level", "Topic", "Subtopic"]

#MultiLabelBinarizer for Keywords
dataset_mlb = dataset.copy()
dataset_mlb['Keywords'] = dataset_mlb['Keywords'].str.split(',')

# Apply MultiLabelBinarizer on Keywords
mlb = MultiLabelBinarizer()
keywords_encoded = mlb.fit_transform(dataset_mlb["Keywords"])
df_keywords = pd.DataFrame(keywords_encoded, columns=mlb.classes_, index=dataset_mlb.index)

# Apply OneHotEncoder on other categorical columns
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ohe_encoded = ohe.fit_transform(dataset_mlb[categorical_single])
df_ohe = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out(categorical_single), index=dataset_mlb.index)

cat_data_mlb = pd.concat([df_ohe, df_keywords], axis=1)
num_data = dataset.select_dtypes(include=['float', 'int'])
if 'Student ID' in num_data.columns:
    num_data = num_data.drop('Student ID', axis=1)
X_mlb = pd.concat([num_data, cat_data_mlb], axis=1)


target = X_mlb['Type of Answer']
X_mlb = X_mlb.drop('Type of Answer', axis=1)
X_train_mlb, X_test_mlb, y_train, y_test = train_test_split(X_mlb, target, random_state=42, test_size=0.2)

X_train_mlb.to_csv('data/processed/X_train_MLB.csv', index=False)
X_test_mlb.to_csv('data/processed/X_test_MLB.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

# OneHotEncoder for all categorical columns (including Keywords)
dataset_ohe = dataset.copy()
cat_dataset_ohe = dataset_ohe.select_dtypes(include=['object'])

# Apply OneHotEncoder on all categorical columns
ohe_all = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
cat_encoded_all = ohe_all.fit_transform(cat_dataset_ohe)
df_cat_ohe = pd.DataFrame(cat_encoded_all, columns=ohe_all.get_feature_names_out(cat_dataset_ohe.columns), index=dataset_ohe.index)

X_ohe = pd.concat([num_data, df_cat_ohe], axis=1)
target_ohe = X_ohe['Type of Answer']
X_ohe = X_ohe.drop('Type of Answer', axis=1)

X_train_ohe, X_test_ohe, y_train_ohe, y_test_ohe = train_test_split(X_ohe, target_ohe, random_state=42, test_size=0.2)

X_train_ohe.to_csv('data/processed/X_train_OHE.csv', index=False)
X_test_ohe.to_csv('data/processed/X_test_OHE.csv', index=False)
y_train_ohe.to_csv('data/processed/y_train_OHE.csv', index=False)
y_test_ohe.to_csv('data/processed/y_test_OHE.csv', index=False)