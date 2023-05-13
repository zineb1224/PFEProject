from sklearn.model_selection import train_test_split
from models.SVMModelMaladieCardiaque import SVMModelMaladieCardiaque , import_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc ,confusion_matrix


#model maladie cardiaque
svmmodelMaladieCardiaque = SVMModelMaladieCardiaque()
# Chargement des données
maladie_data = import_data("datasets/dataset_maladie.csv")

# Séparation des données et de target
y_maladie = maladie_data['target']
X_maladie = maladie_data.drop(['target'], axis = 1)

# Diviser les données en ensembles d'entraînement et de test
featuresMaladie_train, featuresMaladie_test, targetMaladie_train, targetMaladie_test = train_test_split(X_maladie, y_maladie, test_size = 0.2, random_state = 0)

# Prédiction sur l'ensemble de test
svmmodelMaladieCardiaque.fit(featuresMaladie_train, targetMaladie_train)

maladie_pred = svmmodelMaladieCardiaque.predict(featuresMaladie_test)
# Évaluer les performances du modèle
accuracy = accuracy_score(targetMaladie_test, maladie_pred)
precision = precision_score(targetMaladie_test, maladie_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)