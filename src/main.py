import pandas as pd
from EDA_helper_fun import generate_plot
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer

if __name__ == '__main__':
    dataset = pd.read_csv('data/MathE.csv', sep = ';', encoding = 'latin-1')
    #generate_plot(dataset)

    #we have to transform the single string 'keywords' into a list of words
    dataset['Keywords'] = dataset['Keywords'].str.split(',')
    categorical_single = ["Student Country", "Question Level", "Topic", "Subtopic"]

    #MultiLabelBinarizer on the keywords column, ohe on the others
    mlb = MultiLabelBinarizer()
    keywords_encoded = mlb.fit_transform(dataset["Keywords"])
    df_keywords = pd.DataFrame(keywords_encoded, columns=mlb.classes_, index=dataset.index)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe_encoded = ohe.fit_transform(dataset[categorical_single])
    df_ohe = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out(categorical_single), index=dataset.index)
    cat_data = pd.concat([df_ohe, df_keywords], axis=1)

    #numeric dataset without Student ID, it isn't a strong predictive feature
    num_data = dataset.select_dtypes(include = ['float','int'])
    if 'Student ID' in num_data:
        num_data = num_data.drop(labels = ['Student ID'], axis = 1)
    
    X = pd.concat((num_data, cat_data), axis = 1)
    target = X['Type of Answer']
    if 'Type of Answer' in X:
        X = X.drop(labels = ['Type of Answer'], axis = 1)
    