import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_answer_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot
    answer_counts = df['Type of Answer'].value_counts()
    axes[0].bar(answer_counts.index, answer_counts.values, color=['#2ecc71', '#e74c3c'])
    axes[0].set_xlabel('Type of Answer (0=Incorrect, 1=Correct)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Answers')
    axes[0].grid(axis='y', alpha=0.3)

    # Pie chart
    labels = ['Incorrect', 'Correct']
    colors = ['#e74c3c', '#2ecc71']
    axes[1].pie(answer_counts.values, labels=labels, autopct='%1.1f%%',
                colors=colors, startangle=90)
    axes[1].set_title('Answer Distribution Percentage')

    plt.tight_layout()
    plt.savefig(os.path.join("results", "output1.jpg"))
    plt.show()

def plot_performance_by_country(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Calcola percentuale di risposte corrette per paese
    country_perf = df.groupby('Student Country')['Type of Answer'].agg(['sum', 'count'])
    country_perf['accuracy'] = (country_perf['sum'] / country_perf['count']) * 100
    country_perf = country_perf.sort_values('accuracy', ascending=False)

    # Bar plot
    axes[0].barh(country_perf.index, country_perf['accuracy'], color='steelblue')
    axes[0].set_xlabel('Accuracy (%)')
    axes[0].set_ylabel('Country')
    axes[0].set_title('Performance by Country')
    axes[0].grid(axis='x', alpha=0.3)

    # Count per paese
    country_counts = df['Student Country'].value_counts()
    axes[1].bar(country_counts.index, country_counts.values, color='coral')
    axes[1].set_xlabel('Country')
    axes[1].set_ylabel('Number of Answers')
    axes[1].set_title('Number of Answers by Country')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join("results", "output2.jpg"))
    plt.show()

def plot_country_heatmap(df):
    # Pivot table per heatmap
    pivot_data = df.groupby(['Student Country', 'Question Level'])['Type of Answer'].mean()
    pivot_data = pivot_data.unstack(fill_value=0)

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_data, annot=True, fmt='.2%', cmap='RdYlGn',
                cbar_kws={'label': 'Accuracy'}, linewidths=0.5)
    plt.title('Accuracy Heatmap: Country vs Question Level')
    plt.xlabel('Question Level')
    plt.ylabel('Country')
    plt.tight_layout()
    plt.savefig(os.path.join("results", "output3.jpg"))
    plt.show()

def plot_question_difficulty(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Count plot
    level_counts = df['Question Level'].value_counts()
    axes[0].bar(level_counts.index, level_counts.values, color=['#3498db', '#e67e22'])
    axes[0].set_xlabel('Question Level')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Question Levels')
    axes[0].grid(axis='y', alpha=0.3)

    # Box plot accuracy per livello
    level_accuracy = df.groupby(['Question Level', 'Question ID'])['Type of Answer'].mean().reset_index()
    sns.boxplot(data=level_accuracy, x='Question Level', y='Type of Answer', ax=axes[1])
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Distribution by Question Level')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join("results", "output4.jpg"))
    plt.show()

def plot_topic_analysis(df):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Performance per topic
    topic_perf = df.groupby('Topic')['Type of Answer'].agg(['sum', 'count'])
    topic_perf['accuracy'] = (topic_perf['sum'] / topic_perf['count']) * 100
    topic_perf = topic_perf.sort_values('accuracy', ascending=True)

    axes[0].barh(topic_perf.index, topic_perf['accuracy'], color='teal')
    axes[0].set_xlabel('Accuracy (%)')
    axes[0].set_ylabel('Topic')
    axes[0].set_title('Performance by Topic')
    axes[0].grid(axis='x', alpha=0.3)

    # Distribuzione domande per topic
    topic_counts = df['Topic'].value_counts().sort_values(ascending=True)
    axes[1].barh(topic_counts.index, topic_counts.values, color='salmon')
    axes[1].set_xlabel('Number of Answers')
    axes[1].set_ylabel('Topic')
    axes[1].set_title('Distribution of Answers by Topic')
    axes[1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join("results", "output5.jpg"))
    plt.show()

def plot_topic_subtopic_heatmap(df):
    # Pivot table per heatmap
    pivot_data = df.groupby(['Topic', 'Subtopic'])['Type of Answer'].mean().unstack(fill_value=0)

    plt.figure(figsize=(16, 10))
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='coolwarm',
                cbar_kws={'label': 'Accuracy'}, linewidths=0.5)
    plt.title('Accuracy Heatmap: Topic vs Subtopic')
    plt.xlabel('Subtopic')
    plt.ylabel('Topic')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join("results", "output6.jpg"))
    plt.show()

def plot_student_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Numero di studenti per paese
    student_country = df.groupby('Student Country')['Student ID'].nunique().sort_values(ascending=False)
    axes[0].bar(student_country.index, student_country.values, color='purple')
    axes[0].set_xlabel('Country')
    axes[0].set_ylabel('Number of Students')
    axes[0].set_title('Number of Students by Country')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)

    # Histogram domande per studente
    questions_per_student = df.groupby('Student ID').size()
    axes[1].hist(questions_per_student, bins=30, color='olive', edgecolor='black')
    axes[1].set_xlabel('Number of Questions Answered')
    axes[1].set_ylabel('Number of Students')
    axes[1].set_title('Distribution of Questions per Student')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join("results", "output7.jpg"))
    plt.show()

def plot_question_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribuzione tentativi per domanda
    attempts_per_question = df.groupby('Question ID').size()
    axes[0].hist(attempts_per_question, bins=30, color='darkblue', edgecolor='black')
    axes[0].set_xlabel('Number of Attempts per Question')
    axes[0].set_ylabel('Number of Questions')
    axes[0].set_title('Distribution of Attempts per Question')
    axes[0].grid(axis='y', alpha=0.3)

    # Box plot difficoltà effettiva (% errori)
    question_difficulty = df.groupby('Question ID')['Type of Answer'].agg(['mean', 'count'])
    question_difficulty = question_difficulty[question_difficulty['count'] >= 5]
    axes[1].hist(question_difficulty['mean'], bins=20, color='darkred', edgecolor='black')
    axes[1].set_xlabel('Question Accuracy')
    axes[1].set_ylabel('Number of Questions')
    axes[1].set_title('Distribution of Question Difficulty (min 5 attempts)')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join("results", "output8.jpg"))
    plt.show()

def plot_student_performance_scatter(df):
    student_stats = df.groupby('Student ID').agg({
        'Type of Answer': ['mean', 'count']
    }).reset_index()
    student_stats.columns = ['ID', 'accuracy', 'num_questions']

    plt.figure(figsize=(12, 6))
    plt.scatter(student_stats['num_questions'], student_stats['accuracy'] * 100,
                alpha=0.6, s=50, c=student_stats['accuracy'], cmap='viridis')
    plt.colorbar(label='Accuracy')
    plt.xlabel('Number of Questions Answered')
    plt.ylabel('Accuracy (%)')
    plt.title('Student Performance: Questions Answered vs Accuracy')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "output9.jpg"))
    plt.show()

def plot_grouped_performance(df):
    pivot_data = df.groupby(['Student Country', 'Question Level'])['Type of Answer'].mean().unstack()

    pivot_data.plot(kind='bar', figsize=(14, 6), color=['#3498db', '#e67e22'])
    plt.xlabel('Country')
    plt.ylabel('Accuracy')
    plt.title('Performance by Level and Country')
    plt.legend(title='Question Level')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "output10.jpg"))
    plt.show()

def plot_difficult_keywords(df, top_n=15):
    # Trova domande difficili (accuracy < 50%)
    difficult_questions = df.groupby('Question ID')['Type of Answer'].mean()
    difficult_questions = difficult_questions[difficult_questions < 0.5].index

    difficult_df = df[df['Question ID'].isin(difficult_questions)]

    # Estrai keywords più frequenti nelle domande difficili
    all_keywords = ' '.join(difficult_df['Keywords'].dropna().astype(str))
    keywords_list = all_keywords.strip().split(',')
    print(keywords_list)
    keyword_counts = pd.Series(keywords_list).value_counts().head(top_n)

    plt.figure(figsize=(12, 6))
    plt.barh(keyword_counts.index, keyword_counts.values, color='crimson')
    plt.xlabel('Frequency')
    plt.ylabel('Keyword')
    plt.title(f'Top {top_n} Keywords in Difficult Questions (Accuracy < 50%)')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "output11.jpg"))
    plt.show()

def plot_correlation_matrix(df):
    # Prepara dati numerici per correlazione
    numeric_data = df.copy()
    numeric_data['Country_encoded'] = pd.Categorical(numeric_data['Student Country']).codes
    numeric_data['Topic_encoded'] = pd.Categorical(numeric_data['Topic']).codes
    numeric_data['Level_encoded'] = pd.Categorical(numeric_data['Question Level']).codes

    corr_cols = ['Type of Answer', 'Country_encoded', 'Topic_encoded', 'Level_encoded']
    correlation = numeric_data[corr_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, linewidths=1)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join("results", "output12.jpg"))
    plt.show()

def plot_topic_subtopic_stats(df):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Accuracy per subtopic (top 20)
    subtopic_perf = df.groupby('Subtopic')['Type of Answer'].agg(['mean', 'count'])
    subtopic_perf = subtopic_perf[subtopic_perf['count'] >= 10]
    subtopic_perf = subtopic_perf.sort_values('mean', ascending=True).tail(20)

    axes[0].barh(subtopic_perf.index, subtopic_perf['mean'] * 100, color='seagreen')
    axes[0].set_xlabel('Accuracy (%)')
    axes[0].set_ylabel('Subtopic')
    axes[0].set_title('Top 20 Easiest Subtopics (min 10 answers)')
    axes[0].grid(axis='x', alpha=0.3)

    # Accuracy per subtopic (bottom 20)
    subtopic_perf_hard = df.groupby('Subtopic')['Type of Answer'].agg(['mean', 'count'])
    subtopic_perf_hard = subtopic_perf_hard[subtopic_perf_hard['count'] >= 10]
    subtopic_perf_hard = subtopic_perf_hard.sort_values('mean', ascending=True).head(20)

    axes[1].barh(subtopic_perf_hard.index, subtopic_perf_hard['mean'] * 100, color='firebrick')
    axes[1].set_xlabel('Accuracy (%)')
    axes[1].set_ylabel('Subtopic')
    axes[1].set_title('Top 20 Hardest Subtopics (min 10 answers)')
    axes[1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join("results", "output13.jpg"))
    plt.show()

def generate_plot(df):
    plot_answer_distribution(df)
    plot_performance_by_country(df)
    plot_country_heatmap(df)
    plot_question_difficulty(df)
    plot_topic_analysis(df)
    plot_topic_subtopic_heatmap(df)
    plot_student_distribution(df)
    plot_question_analysis(df)
    plot_student_performance_scatter(df)
    plot_grouped_performance(df)
    plot_difficult_keywords(df, top_n=15)
    plot_correlation_matrix(df)
    plot_topic_subtopic_stats(df)