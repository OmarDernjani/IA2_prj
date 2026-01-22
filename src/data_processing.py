import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


# Global variable
DATA_PATH = 'data/raw/MathE.csv'
OUTPUT_PATH = 'data/processed/'
RANDOM_STATE = 42
TEST_SIZE = 0.2


#Load dataset
def load_data(file_path):
    """Load and sort dataset by student and question to preserve temporal order."""
    df = pd.read_csv(file_path, sep=';', encoding='latin-1')
    df = df.sort_values(['Student ID', 'Question ID']).reset_index(drop=True)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


#Feature engineering 

def compute_student_features(df):
    """
    Compute cumulative student performance metrics.
    For each student, calculate their success rate and number of attempts
    up to (but not including) the current question.
    """
    student_blocks = []
    
    for student_id, df_student in df.groupby('Student ID'):
        df_student = df_student.copy()
        
        # Cumulative success rate (shifted to avoid data leakage)
        cumsum = df_student['Type of Answer'].expanding().sum().shift(1)
        cnt = df_student['Type of Answer'].expanding().count().shift(1)
        df_student['student_success_rate'] = (cumsum / cnt).fillna(0.5)
        
        # Cumulative number of attempts
        df_student['student_num_attempts'] = cnt.fillna(0).astype(int)
        
        student_blocks.append(df_student)
    
    return pd.concat(student_blocks).sort_index()


def compute_student_topic_features(df):
    """
    Compute cumulative success rate for each student on each topic.
    This captures how well a student performs on specific topics over time.
    """
    student_topic_blocks = []
    
    for student_id, df_student in df.groupby('Student ID'):
        df_student = df_student.copy()
        df_student['student_topic_success_rate'] = 0.5
        
        for topic, df_topic in df_student.groupby('Topic'):
            cumsum = df_topic['Type of Answer'].expanding().sum().shift(1)
            cnt = df_topic['Type of Answer'].expanding().count().shift(1)
            topic_success = (cumsum / cnt).fillna(0.5)
            df_student.loc[df_topic.index, 'student_topic_success_rate'] = topic_success
        
        student_topic_blocks.append(df_student)
    
    return pd.concat(student_topic_blocks).sort_index()


def compute_question_features(train_df, test_df):
    """
    Compute question difficulty based on historical success rates.
    Uses only training data to avoid data leakage.
    """
    global_avg = train_df['Type of Answer'].mean()
    
    # Question-level difficulty
    question_stats = train_df.groupby('Question ID')['Type of Answer'].mean()
    train_df['question_difficulty'] = train_df['Question ID'].map(question_stats).fillna(global_avg)
    test_df['question_difficulty'] = test_df['Question ID'].map(question_stats).fillna(global_avg)
    
    # Question level as binary
    train_df['question_level_advanced'] = (train_df['Question Level'] == 'advanced').astype(int)
    test_df['question_level_advanced'] = (test_df['Question Level'] == 'advanced').astype(int)
    
    return train_df, test_df


def compute_topic_features(train_df, test_df):
    """
    Compute topic and subtopic difficulty based on historical success rates.
    Uses only training data to avoid data leakage.
    """
    global_avg = train_df['Type of Answer'].mean()
    
    # Topic-level difficulty
    topic_stats = train_df.groupby('Topic')['Type of Answer'].mean()
    train_df['topic_difficulty'] = train_df['Topic'].map(topic_stats)
    test_df['topic_difficulty'] = test_df['Topic'].map(topic_stats).fillna(global_avg)
    
    # Subtopic-level difficulty
    subtopic_stats = train_df.groupby('Subtopic')['Type of Answer'].mean()
    train_df['subtopic_difficulty'] = train_df['Subtopic'].map(subtopic_stats)
    test_df['subtopic_difficulty'] = test_df['Subtopic'].map(subtopic_stats).fillna(global_avg)
    
    return train_df, test_df


def compute_country_features(train_df, test_df):
    """
    Compute country-level performance statistics.
    Uses only training data to avoid data leakage.
    """
    global_avg = train_df['Type of Answer'].mean()
    country_stats = train_df.groupby('Student Country')['Type of Answer'].mean()
    
    train_df['country_avg_performance'] = train_df['Student Country'].map(country_stats)
    test_df['country_avg_performance'] = test_df['Student Country'].map(country_stats).fillna(global_avg)
    
    return train_df, test_df


def compute_interaction_features(df):
    """
    Create interaction features that capture relationships between
    student ability and question/topic difficulty.
    """
    df['topic_familiarity'] = df['student_topic_success_rate'] - df['topic_difficulty']
    df['ability_difficulty_gap'] = df['student_success_rate'] - df['question_difficulty']
    return df


#Encoding function
def encode_categorical_features(train_df, test_df, categorical_columns):
    """
    Apply one-hot encoding to categorical features.
    Fit encoder on training data only to avoid data leakage.
    """
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    X_cat_train = ohe.fit_transform(train_df[categorical_columns])
    X_cat_test = ohe.transform(test_df[categorical_columns])
    
    cat_feature_names = ohe.get_feature_names_out(categorical_columns)
    df_cat_train = pd.DataFrame(X_cat_train, columns=cat_feature_names, index=train_df.index)
    df_cat_test = pd.DataFrame(X_cat_test, columns=cat_feature_names, index=test_df.index)
    
    return df_cat_train, df_cat_test


def preprocess_data(data_path, output_path, test_size=0.2, random_state=42):
    """
    Complete preprocessing pipeline:
    1. Load data
    2. Split into train/test
    3. Apply feature engineering
    4. Encode categorical features
    5. Save processed datasets
    """
    
    # Load data
    dataset = load_data(data_path)
    
    
    # Train-test split
    train_data, test_data = train_test_split(
        dataset, 
        test_size=test_size, 
        random_state=random_state
    )
    print(f"Train size: {len(train_data)} ({len(train_data)/len(dataset)*100:.1f}%)")
    print(f"Test size:  {len(test_data)} ({len(test_data)/len(dataset)*100:.1f}%)")
    
    
    # Feature engineering
    train_data = compute_student_features(train_data)
    test_data = compute_student_features(test_data)

    train_data = compute_student_topic_features(train_data)
    test_data = compute_student_topic_features(test_data)

    train_data, test_data = compute_question_features(train_data, test_data)
    
    train_data, test_data = compute_topic_features(train_data, test_data)

    train_data, test_data = compute_country_features(train_data, test_data)

    train_data = compute_interaction_features(train_data)
    test_data = compute_interaction_features(test_data)
    
    
    # Define feature sets
    numeric_features = [
        'student_success_rate',
        'student_num_attempts',
        'student_topic_success_rate',
        'question_difficulty',
        'question_level_advanced',
        'topic_difficulty',
        'subtopic_difficulty',
        'country_avg_performance',
        'topic_familiarity',
        'ability_difficulty_gap'
    ]
    
    categorical_columns = ['Topic', 'Subtopic', 'Student Country']
    
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical columns: {len(categorical_columns)}")
    
    
    # Encode categorical features
    df_cat_train, df_cat_test = encode_categorical_features(
        train_data, test_data, categorical_columns
    )
    
    
    # Combine features
    
    X_num_train = train_data[numeric_features]
    X_num_test = test_data[numeric_features]
    
    X_train = pd.concat([X_num_train, df_cat_train], axis=1)
    X_test = pd.concat([X_num_test, df_cat_test], axis=1)
    
    y_train = train_data['Type of Answer']
    y_test = test_data['Type of Answer']
    
    print(f"Final training set: {X_train.shape}")
    print(f"Final test set: {X_test.shape}")
    
    
    # Save processed data
    
    X_train.to_csv(output_path + 'X_train.csv', index=False)
    X_test.to_csv(output_path + 'X_test.csv', index=False)
    y_train.to_csv(output_path + 'y_train.csv', index=False)
    y_test.to_csv(output_path + 'y_test.csv', index=False)
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data(
        data_path=DATA_PATH,
        output_path=OUTPUT_PATH,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )