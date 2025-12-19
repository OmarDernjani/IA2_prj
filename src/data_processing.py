import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit

# --- Load dataset ---
dataset = pd.read_csv(
    'data/raw/MathE.csv',
    sep=';',
    encoding='latin-1'
)

# Fill missing keywords (non useremo più le keywords)
dataset['Keywords'] = dataset['Keywords'].fillna('')

# Sort by student and time proxy
dataset = dataset.sort_values(
    ['Student ID', 'Question ID']
).reset_index(drop=True)

# Encode target
le = LabelEncoder()
dataset['target'] = le.fit_transform(dataset['Type of Answer'])

# --- Student success rate features ---
student_blocks = []
for student_id, df_student in dataset.groupby('Student ID'):
    df_student = df_student.copy()
    cumsum = df_student['target'].expanding().sum().shift(1)
    cnt = df_student['target'].expanding().count().shift(1)
    df_student['student_success_rate'] = (cumsum / cnt).fillna(0.5)
    df_student['student_num_attempts'] = cnt.fillna(0).astype(int)
    student_blocks.append(df_student)

dataset = pd.concat(student_blocks).sort_index()

# --- Train/test split by student ---
gss = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(dataset, groups=dataset['Student ID']))
train_df = dataset.iloc[train_idx].copy()
test_df = dataset.iloc[test_idx].copy()

# --- Target encoding for categorical features ---
def add_target_encoding(train, test, col, target_col='target', prior=0.5):
    stats = train.groupby(col)[target_col].mean()
    train[f'{col}_success_rate'] = train[col].map(stats).fillna(prior)
    test[f'{col}_success_rate'] = test[col].map(stats).fillna(prior)

for col in ['Topic', 'Subtopic', 'Question ID', 'Student Country']:
    add_target_encoding(train_df, test_df, col)

train_df = train_df.rename(columns={'Question ID_success_rate': 'question_success_rate'})
test_df = test_df.rename(columns={'Question ID_success_rate': 'question_success_rate'})

# --- Question level numeric ---
train_df['question_level_numeric'] = (train_df['Question Level'] == 'advanced').astype(int)
test_df['question_level_numeric'] = (test_df['Question Level'] == 'advanced').astype(int)

# --- Student-topic success rate ---
topic_blocks = []
for student_id, df_student in train_df.groupby('Student ID'):
    df_student = df_student.copy()
    for topic, df_topic in df_student.groupby('Topic'):
        cumsum = df_topic['target'].expanding().sum().shift(1)
        cnt = df_topic['target'].expanding().count().shift(1)
        df_student.loc[df_topic.index, 'student_topic_success_rate'] = (cumsum / cnt).fillna(0.5)
    topic_blocks.append(df_student)

train_df = pd.concat(topic_blocks).sort_index()
train_df['student_topic_success_rate'] = train_df['student_topic_success_rate'].fillna(0.5)
test_df['student_topic_success_rate'] = 0.5

# --- Derived features ---
train_df['topic_familiarity'] = train_df['student_topic_success_rate'] - train_df['Topic_success_rate']
test_df['topic_familiarity'] = 0.0
train_df['ability_difficulty_gap'] = train_df['student_success_rate'] - train_df['question_success_rate']
test_df['ability_difficulty_gap'] = 0.0

# --- Categorical encoding ---
cat_cols = ['Topic', 'Subtopic', 'Student Country']
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_cat_train = ohe.fit_transform(train_df[cat_cols])
X_cat_test = ohe.transform(test_df[cat_cols])

df_cat_train = pd.DataFrame(X_cat_train, columns=ohe.get_feature_names_out(cat_cols), index=train_df.index)
df_cat_test = pd.DataFrame(X_cat_test, columns=ohe.get_feature_names_out(cat_cols), index=test_df.index)

# --- Numeric features ---
num_cols = [
    'student_success_rate',
    'student_num_attempts',
    'Topic_success_rate',
    'Subtopic_success_rate',
    'question_success_rate',
    'question_level_numeric',
    'student_topic_success_rate',
    'topic_familiarity',
    'ability_difficulty_gap',
    'Student Country_success_rate'
]

X_num_train = train_df[num_cols]
X_num_test = test_df[num_cols]

# --- Final datasets (senza keywords) ---
X_train = pd.concat([X_num_train, df_cat_train], axis=1)
X_test = pd.concat([X_num_test, df_cat_test], axis=1)
y_train = train_df['target']
y_test = test_df['target']

# --- Save processed data ---
X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)
