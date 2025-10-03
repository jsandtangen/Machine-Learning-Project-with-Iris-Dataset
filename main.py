import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target

print("Første radene i datasettet:")
print(iris_df.head())

sns.pairplot(iris_df, hue='species')
plt.title("Iris Dataset Pairplot")
plt.show()

scaler = StandardScaler()
iris_scaled = scaler.fit_transform(iris_df.iloc[:, :-1])

X_train, X_test, y_train, y_test = train_test_split(iris_scaled, iris_df['species'], test_size=0.2, random_state=42)

models = {
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

results = {}

for model_name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5)
    results[model_name] = scores
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"\n{model_name} Resultater:")
    print("Forvirringsmatrise:")
    print(confusion_matrix(y_test, predictions))
    print("\nKlassifikasjonsrapport:")
    print(classification_report(y_test, predictions))
    plt.figure(figsize=(10, 6))
    plt.subplot(211)
    plt.boxplot(scores, labels=[model_name])
    plt.title(f'{model_name} Kryssvalideringsresultater')
    plt.ylabel('Accuracy')

knn_params = {'n_neighbors': np.arange(1, 31)}
grid_search = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5)
grid_search.fit(X_train, y_train)

print("\nBeste hyperparametre for KNN:")
print(grid_search.best_params_)

best_knn_model = grid_search.best_estimator_
best_knn_predictions = best_knn_model.predict(X_test)

print("\nKNN med beste hyperparametere Resultater:")
print("Forvirringsmatrise:")
print(confusion_matrix(y_test, best_knn_predictions))
print("\nKlassifikasjonsrapport:")
print(classification_report(y_test, best_knn_predictions))

plt.subplot(212)
plt.boxplot(results['KNN'], labels=['KNN'])
plt.title('Sammenligning av modeller')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()
