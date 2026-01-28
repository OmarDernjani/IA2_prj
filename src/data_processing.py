import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split


# Global variable
DATA_PATH = 'data/raw/MathE.csv'
OUTPUT_PATH = 'data/processed/'
RANDOM_STATE = 42
TEST_SIZE = 0.2


#Load dataset
def load_data(file_path):
    """Load dataset."""
    df = pd.read_csv(file_path, sep=';', encoding='latin-1')
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()}")
    return df


#Feature engineering basato sulle domande

def compute_question_statistics(df):
    """
    Compute statistics for each question based on student responses.
    These features help predict question difficulty.
    """
    # Group by Question ID and compute aggregates
    question_stats = df.groupby('Question ID').agg({
        'Type of Answer': ['mean', 'std', 'count'],  # Success rate, variability, attempts
        'Student ID': 'nunique'  # Number of unique students who attempted
    }).reset_index()
    
    # Flatten column names
    question_stats.columns = [
        'Question ID',
        'question_success_rate',  # % of correct answers
        'question_success_std',   # Variability in success
        'question_attempt_count', # Total attempts
        'question_unique_students' # Number of students
    ]
    
    # Fill NaN std with 0 (happens when only 1 attempt)
    question_stats['question_success_std'] = question_stats['question_success_std'].fillna(0)
    
    return question_stats


def compute_topic_statistics(df):
    """
    Compute statistics for each topic and subtopic.
    """
    # Topic statistics
    topic_stats = df.groupby('Topic').agg({
        'Type of Answer': ['mean', 'count']
    }).reset_index()
    topic_stats.columns = ['Topic', 'topic_avg_success', 'topic_question_count']
    
    # Subtopic statistics
    subtopic_stats = df.groupby('Subtopic').agg({
        'Type of Answer': ['mean', 'count']
    }).reset_index()
    subtopic_stats.columns = ['Subtopic', 'subtopic_avg_success', 'subtopic_question_count']
    
    return topic_stats, subtopic_stats


def compute_country_statistics(df):
    """
    Compute how students from different countries perform on each question.
    This can indicate cultural/linguistic difficulty.
    """
    country_question_stats = df.groupby(['Question ID', 'Student Country']).agg({
        'Type of Answer': 'mean'
    }).reset_index()
    
    # Compute variance across countries for each question
    country_variance = country_question_stats.groupby('Question ID').agg({
        'Type of Answer': 'std'
    }).reset_index()
    country_variance.columns = ['Question ID', 'question_country_variance']
    country_variance['question_country_variance'] = country_variance['question_country_variance'].fillna(0)
    
    return country_variance


def compute_keyword_features(df):
    """
    Extract features from the Keywords column.
    These can indicate question complexity.
    """
    # Complex mathematical terms that might indicate difficulty
    complex_terms = [
        'derivative', 'integral', 'limit', 'matrix', 'theorem',
        'probability', 'distribution', 'variance', 'covariance',
        'eigenvalue', 'eigenvector', 'polynomial', 'optimization',
        'differential', 'logarithm', 'exponential', 'series',
        'convergence', 'divergence', 'transformation', 'complex'
    ]
    
    keyword_features = []
    
    for qid, group in df.groupby('Question ID'):
        keywords_str = group['Keywords'].iloc[0]
        
        # Handle missing or empty keywords
        if pd.isna(keywords_str) or keywords_str.strip() == '':
            keyword_features.append({
                'Question ID': qid,
                'num_keywords': 0,
                'avg_keyword_length': 0,
                'has_complex_term': 0
            })
            continue
        
        # Split keywords by comma
        keywords = [k.strip() for k in str(keywords_str).split(',')]
        
        # Number of keywords
        num_kw = len(keywords)
        
        # Average keyword length
        avg_len = np.mean([len(k) for k in keywords]) if keywords else 0
        
        # Check for complex mathematical terms
        has_complex = int(any(
            term.lower() in keywords_str.lower() 
            for term in complex_terms
        ))
        
        keyword_features.append({
            'Question ID': qid,
            'num_keywords': num_kw,
            'avg_keyword_length': avg_len,
            'has_complex_term': has_complex
        })
    
    return pd.DataFrame(keyword_features)


def compute_derived_features(question_df):
    """
    Compute derived features from existing ones.
    These can capture non-linear relationships.
    """
    # Success to attempts ratio (normalized success rate)
    question_df['success_attempts_ratio'] = (
        question_df['question_success_rate'] / 
        (question_df['question_attempt_count'] + 1)  # +1 to avoid division by zero
    )
    
    # Interaction: topic success vs question success (difficulty gap)
    question_df['topic_difficulty_gap'] = (
        question_df['topic_avg_success'] - question_df['question_success_rate']
    )
    
    # Interaction: subtopic success vs question success
    question_df['subtopic_difficulty_gap'] = (
        question_df['subtopic_avg_success'] - question_df['question_success_rate']
    )
    
    # Student engagement: unique students / total attempts
    question_df['student_engagement'] = (
        question_df['question_unique_students'] / 
        (question_df['question_attempt_count'] + 1)
    )
    
    # Popularity score: log-transformed attempt count
    question_df['question_popularity'] = np.log1p(question_df['question_attempt_count'])
    
    return question_df


def aggregate_to_question_level(df):
    """
    Aggregate the dataset to question level.
    Each question appears only once with its features and target label.
    """
    # Get unique questions with their metadata
    question_df = df.groupby('Question ID').first()[['Question Level', 'Topic', 'Subtopic', 'Keywords']].reset_index()
    
    # Compute question statistics
    question_stats = compute_question_statistics(df)
    
    # Merge statistics
    question_df = question_df.merge(question_stats, on='Question ID', how='left')
    
    # Add country variance
    country_variance = compute_country_statistics(df)
    question_df = question_df.merge(country_variance, on='Question ID', how='left')
    
    # Compute topic and subtopic statistics
    topic_stats, subtopic_stats = compute_topic_statistics(df)
    question_df = question_df.merge(topic_stats, on='Topic', how='left')
    question_df = question_df.merge(subtopic_stats, on='Subtopic', how='left')
    
    # Add keyword features
    keyword_features = compute_keyword_features(df)
    question_df = question_df.merge(keyword_features, on='Question ID', how='left')
    
    # Add derived features
    question_df = compute_derived_features(question_df)
    
    print(f"Aggregated to {question_df.shape[0]} unique questions")
    return question_df


def encode_target(train_df, test_df, target_column='Question Level'):
    """
    Encode target labels: Basic -> 0, Advanced -> 1
    """
    le = LabelEncoder()
    
    # Fit on combined data to ensure same encoding
    all_labels = pd.concat([train_df[target_column], test_df[target_column]])
    le.fit(all_labels)
    
    train_df['target'] = le.transform(train_df[target_column])
    test_df['target'] = le.transform(test_df[target_column])
    
    print(f"Target encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    return train_df, test_df, le


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
    2. Aggregate to question level
    3. Split into train/test
    4. Encode target
    5. Encode categorical features
    6. Save processed datasets
    """
    
    # Load data
    dataset = load_data(data_path)
    
    # Aggregate to question level
    question_df = aggregate_to_question_level(dataset)
    
    # Split into train/test at question level
    train_data, test_data = train_test_split(
        question_df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=question_df['Question Level']  # Stratified split
    )
    
    print(f"Train set: {train_data.shape[0]} questions")
    print(f"Test set: {test_data.shape[0]} questions")
    print(f"\nTrain distribution:\n{train_data['Question Level'].value_counts()}")
    print(f"\nTest distribution:\n{test_data['Question Level'].value_counts()}")
    
    # Encode target
    train_data, test_data, label_encoder = encode_target(train_data, test_data)
    
    # Define feature sets
    numeric_features = [
        # Base question statistics
        'question_success_rate',
        'question_success_std',
        'question_attempt_count',
        'question_unique_students',
        'question_country_variance',
        # Topic/Subtopic statistics
        'topic_avg_success',
        'topic_question_count',
        'subtopic_avg_success',
        'subtopic_question_count',
        # Keyword features
        'num_keywords',
        'avg_keyword_length',
        'has_complex_term',
        # Derived features
        'success_attempts_ratio',
        'topic_difficulty_gap',
        'subtopic_difficulty_gap',
        'student_engagement',
        'question_popularity'
    ]
    
    categorical_columns = ['Topic', 'Subtopic']
    
    print(f"\nNumeric features: {len(numeric_features)}")
    print(f"Categorical columns: {len(categorical_columns)}")
    
    # Encode categorical features
    df_cat_train, df_cat_test = encode_categorical_features(
        train_data, test_data, categorical_columns
    )
    
    # Combine features
    X_num_train = train_data[numeric_features]
    X_num_test = test_data[numeric_features]
    
    X_train = pd.concat([X_num_train.reset_index(drop=True), df_cat_train.reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_num_test.reset_index(drop=True), df_cat_test.reset_index(drop=True)], axis=1)
    
    y_train = train_data['target'].reset_index(drop=True)
    y_test = test_data['target'].reset_index(drop=True)
    
    print(f"\nFinal training set: {X_train.shape}")
    print(f"Final test set: {X_test.shape}")
    print(f"Feature names: {X_train.columns.tolist()[:10]}... (showing first 10)")
    
    # Save processed data
    X_train.to_csv(output_path + 'X_train.csv', index=False)
    X_test.to_csv(output_path + 'X_test.csv', index=False)
    y_train.to_csv(output_path + 'y_train.csv', index=False)
    y_test.to_csv(output_path + 'y_test.csv', index=False)
    
    print(f"\nData saved to {output_path}")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data(
        data_path=DATA_PATH,
        output_path=OUTPUT_PATH,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )