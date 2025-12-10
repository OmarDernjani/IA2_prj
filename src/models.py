import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from helper_fun import evaluate_models

PATH = 'data/processed/'

X_train = pd.read_csv(PATH + 'X_train_MLB.csv')
X_test = pd.read_csv(PATH + 'X_test_MLB.csv')
y_train = pd.read_csv(PATH + 'y_train.csv')
y_test = pd.read_csv(PATH + 'y_test.csv')

y_train = y_train['Type of Answer']
y_test = y_test['Type of Answer']

#istanciate the model class (vedere dopo se aggiungerli tutti o meno)
logistic_regression = LogisticRegression()
multinomial_nb = MultinomialNB()
bernoulli_nb = BernoulliNB()
knn_classifier = KNeighborsClassifier()
dt_classifier = DecisionTreeClassifier()
lda_classifier = LinearDiscriminantAnalysis()
qda_classifier = QuadraticDiscriminantAnalysis()


logistic_regression.fit(X_train, y_train)
multinomial_nb.fit(X_train, y_train)
bernoulli_nb.fit(X_train, y_train)
knn_classifier.fit(X_train, y_train)
dt_classifier.fit(X_train, y_train)
lda_classifier.fit(X_train, y_train)
qda_classifier.fit(X_train, y_train)

models = {
    "Logistic Regression": logistic_regression,
    "Multinomial NB": multinomial_nb,
    "Bernoulli NB": bernoulli_nb,
    "KNN": knn_classifier,
    "Decision Tree": dt_classifier,
    "LDA": lda_classifier,
    "QDA": qda_classifier
}

results_df = evaluate_models(models, X_test, y_test)
print(results_df)