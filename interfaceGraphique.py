# import les biblio
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter.ttk import Combobox

from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, f1_score, precision_score
from sklearn.model_selection import train_test_split

from models.SVMModelIris import SVMModelIris, import_dataIris
from models.SVMModelPenguin import SVMModelPenguin, import_dataPenguin
from models.SVMModelMaladieCardiaque import SVMModelMaladieCardiaque, import_dataMaladie
from models.SVMModelDiabets import SVMModelDiabets, import_dataDiabets
from random import sample

# Définition de la palette de couleurs
BG_COLOR = "#071119"
FG_COLOR = "#121212"
LABEL_BG_COLOR = "#3a3a3a"
ENTRY_BG_COLOR = "#071119"
bg_color_frame = "#1B272C"


# définition des fonctions


# fct pour recuperer la valeur de size du test
def getValeurTestSize():
    val = testSize.get()
    return val


# fct pour recuperer la valeur de parametre kernel
def getValeurParamKernel():
    val = paramKernel.get()
    return val


# fct pour recuperer la valeur de parametre C
def getValeurParamC():
    val = paramC.get()
    return val


def getValeurGamma():
    val = paramGamma.get()
    return val


def getValeurXlabelTrain():
    val = paraXtrain.get()
    return val


def getValeurYlabelTrain():
    val = paraYtrain.get()
    return val


def getValeurXlabelTest():
    val = paraXtest.get()
    return val


def getValeurYlabelTest():
    val = paraYtest.get()
    return val


# fct pour afficher la description des differentes datasets (treeview pour affihcer 10 lignes aleatoire de dataset et
# un treeview pour afficher les statistiques )
def afficher_description():
    # Obtenir l'option sélectionnée dans le combobox
    option = combo_box.get()

    # Vider le Treeview
    donnees_treeview.delete(*donnees_treeview.get_children())

    # Afficher les données correspondantes dans l'onglet 2
    if option == "Dataset Iris":
        # Chargement du jeu de données iris
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        iris_data = iris.data.tolist()  # Convertir les données en liste
        feature_names = iris.feature_names
        # Obtenir 10 lignes aléatoires
        iris_data_subset = sample(iris_data, 10)
        for data in iris_data_subset:
            donnees_treeview.insert("", "end", values=data, tags=("Custom.Treeview",))

            # Définir le titre de l'onglet 2
        notebook.tab(ongletDescription, text="Iris Dataset")
        column_headings = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
        description_text = "Description de l'ensemble de données Iris"
        # Définir les en-têtes des colonnes dans le Treeview
        donnees_treeview["columns"] = column_headings
        for i, heading in enumerate(column_headings):
            donnees_treeview.heading(i, text=heading)
            donnees_treeview.column(i, width=220)  # Définir la largeur de la colonne

        # Calculer les statistiques descriptives
        stats = df.describe()

        # Rename the columns to match the feature names
        stats.columns = feature_names

        # Create a Treeview widget for displaying the table
        tree["columns"] = ["Statistic"] + list(stats.columns)
        # Modifier l'arrière-plan du TreeView
        donnees_treeview.configure(style='Custom.Treeview')
        # Appliquer le style personnalisé au TreeView
        donnees_treeview.tag_configure("Custom.Treeview", background=bg_color_frame, foreground="#FFFFFF",
                                       font=("Arial", 12))

        # Add column headings to the Treeview
        tree.heading("Statistic", text="Statistic")
        for column in stats.columns:
            tree.heading(column, text=column)

        # Add rows to the Treeview
        for index, row in stats.iterrows():
            values = [index.capitalize()] + list(row)
            tree.insert("", "end", values=values, tags=("Custom.Treeview",))

    elif option == "Dataset Maladies Cardiaques":

        # Charger les données depuis le fichier CSV
        maladie_data = pd.read_csv('datasets/dataset_maladie.csv')

        # Obtenir 10 lignes aléatoires de l'ensemble de données Maladie
        maladie_data_subset = maladie_data.sample(20)
        for _, row in maladie_data_subset.iterrows():
            donnees_treeview.insert("", "end", values=row.tolist(), tags=("Custom.Treeview",))

        # Définir les en-têtes des colonnes pour l'ensemble de données Maladie
        column_headings = maladie_data.columns.tolist()

        # Définir le titre de l'onglet 2 pour l'ensemble de données Maladie
        notebook.tab(ongletDescription, text="Maladie Dataset")
        description_text = "Description de l'ensemble de données Maladie"

        # Définir les en-têtes des colonnes dans le Treeview
        donnees_treeview["columns"] = column_headings
        for i, heading in enumerate(column_headings):
            donnees_treeview.heading(i, text=heading)
            donnees_treeview.column(i, width=120)  # Définir la largeur de la colonne

        # Calculer les statistiques descriptives
        statsMaladie = maladie_data.describe()
        # Afficher les statistiques dans un seul graphique

        # Create a Treeview widget for displaying the table
        tree["columns"] = ["Statistic"] + column_headings

        # Add column headings to the Treeview
        tree.heading("Statistic", text="Statistic")
        for column in column_headings:
            tree.heading(column, text=column)

        # Add rows to the Treeview
        for index, row in statsMaladie.iterrows():
            values = [index.capitalize()] + list(row)
            tree.insert("", "end", values=values, tags=("Custom.Treeview",))

    elif option == "Dataset Penguin":
        # Charger les données depuis le fichier CSV
        penguin_data = pd.read_csv('datasets/penguins.csv')

        # Obtenir 10 lignes aléatoires de l'ensemble de données penguins
        maladie_data_subset = penguin_data.sample(20)
        for _, row in maladie_data_subset.iterrows():
            donnees_treeview.insert("", "end", values=row.tolist(), tags=("Custom.Treeview",))
        # Définir les en-têtes des colonnes pour l'ensemble de données Maladie
        column_headings = penguin_data.columns.tolist()
        # Définir le titre de l'onglet 2 pour l'ensemble de données diabetes
        notebook.tab(ongletDescription, text="Penguin Dataset")
        description_text = "Description de l'ensemble de données Penguins"

        # Définir les en-têtes des colonnes dans le Treeview
        donnees_treeview["columns"] = column_headings
        for i, heading in enumerate(column_headings):
            donnees_treeview.heading(i, text=heading)
            donnees_treeview.column(i, width=180)  # Définir la largeur de la colonne

        # Calculer les statistiques descriptives
        statsPenguin = penguin_data.describe()
        # Afficher les statistiques dans un seul graphique

        # Create a Treeview widget for displaying the table
        tree["columns"] = ["Statistic"] + column_headings

        # Add column headings to the Treeview
        tree.heading("Statistic", text="Statistic")
        for column in column_headings:
            tree.heading(column, text=column)

        # Add rows to the Treeview
        for index, row in statsPenguin.iterrows():
            values = [index.capitalize()] + list(row)
            tree.insert("", "end", values=values, tags=("Custom.Treeview",))

    elif option == "Dataset Diabets":
        # Charger les données depuis le fichier CSV
        diabet_data = pd.read_csv('datasets/diabetes.csv')

        # Obtenir 10 lignes aléatoires de l'ensemble de données diabetes
        maladie_data_subset = diabet_data.sample(20)
        for _, row in maladie_data_subset.iterrows():
            donnees_treeview.insert("", "end", values=row.tolist(), tags=("Custom.Treeview",))
        # Définir les en-têtes des colonnes pour l'ensemble de données Maladie
        column_headings = diabet_data.columns.tolist()
        # Définir le titre de l'onglet 2 pour l'ensemble de données Diabetes
        notebook.tab(ongletDescription, text="Diabetes Dataset")
        description_text = "Description de l'ensemble de données Diabetes"

        # Définir les en-têtes des colonnes dans le Treeview
        donnees_treeview["columns"] = column_headings
        for i, heading in enumerate(column_headings):
            donnees_treeview.heading(i, text=heading)
            donnees_treeview.column(i, width=180)  # Définir la largeur de la colonne

        # Calculer les statistiques descriptives
        statsDiabet = diabet_data.describe()
        # Afficher les statistiques dans un seul graphique

        # Create a Treeview widget for displaying the table
        tree["columns"] = ["Statistic"] + column_headings

        # Add column headings to the Treeview
        tree.heading("Statistic", text="Statistic")
        for column in column_headings:
            tree.heading(column, text=column)

        # Add rows to the Treeview
        for index, row in statsDiabet.iterrows():
            values = [index.capitalize()] + list(row)
            tree.insert("", "end", values=values, tags=("Custom.Treeview",))

    # Sélectionner l'onglet 2
    notebook.select(ongletDescription)
    # Modifier le titre de l'onglet 2 avec le texte de description correspondant
    titre_onglet2.config(text=description_text)


def show_button():
    bouton_onglet1.pack(padx=5, pady=5)


def hide_button():
    bouton_onglet1.pack_forget()


def show_entryGamma():
    paraGamma.pack(padx=50, pady=10)
    paramGamma.pack(padx=20, ipady=5, pady=5)


def hide_entryGamma():
    paramGamma.pack_forget()
    paraGamma.pack_forget()


# fct pour verifier que les inputs sont bien remplis et rendre le boutton de train et de test normal
def check_fields():
    if len(getValeurTestSize()) > 0 and len(getValeurParamC()) > 0 and len(getValeurParamKernel()) > 0 and len(combo_box.get()) > 0:
        btnTraining.config(state="normal")
        btnTesting.config(state="normal")
    else:
        btnTraining.config(state="disabled")
        btnTesting.config(state="disabled")


# Fonction appelée lors de la sélection d'une option dans la ComboBox
def update_label(event):
    selected_value = combo_box.get()
    # Effacer les anciennes valeurs de combobox2
    paraXtrain['values'] = ()
    paraYtrain['values'] = ()
    paraXtest['values'] = ()
    paraYtest['values'] = ()

    if selected_value == "Dataset Maladies Cardiaques":
        descriptiontxt.configure(text="Le fichier csv contient 303 lignes, chaque ligne pour chaque personne. "
                                      " Il y a 14 colonnes. La dernière colonne contient les libellés de prédiction : "
                                      " 1 pour malade, 0 pour non malade. il est preféré d'utiliser le kernel rbf pour "
                                      "entrainer cette dataset."
                                 )
        show_button()
        paraXtrain['values'] = ('age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal')
        paraYtrain['values'] = ('age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal')
        paraXtest['values'] = ('age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal')
        paraYtest['values'] = ('age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal')
    elif selected_value == "Dataset Penguin":
        descriptiontxt.configure(text="Le fichier csv contient 344 lignes, chaque ligne pour chaque penguin. "
                                      " Il y a 9 colonnes. La colonne species contient les types de punguins de prédiction : "
                                      " Adelie, Gentoo , Chinstrap. il est preféré d'utiliser le kernel linear pour "
                                      "entrainer cette dataset.")
        show_button()
        paraXtrain['values'] = ('island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex', 'year')
        paraYtrain['values'] = ('island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex', 'year')
        paraXtest['values'] = ('island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex', 'year')
        paraYtest['values'] = ('island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex', 'year')

    elif selected_value == "Dataset Iris":
        descriptiontxt.configure(text="Le fichier csv contient 150 lignes, chaque ligne pour chaque fleur Iris. "
                                      " Il y a 4 colonnes. La colonne target contient les types de fleurs de prédiction : "
                                      " setosa, versicolor , virginica. il est preféré d'utiliser le kernel linear pour"
                                      " entrainer cette dataset.")
        show_button()
        paraXtrain['values'] = ('sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)')
        paraYtrain['values'] = ('sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)')
        paraXtest['values'] = ('sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)')
        paraYtest['values'] = ('sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)')
    elif selected_value == "Dataset Diabets":
        descriptiontxt.configure(text="Le fichier csv contient 768 lignes, chaque ligne pour chaque personne diabet. "
                                      " Il y a 9 colonnes. La colonne outcome contient les personnes diabets : "
                                      " 1 pour malade, 0 pour non malade. il est preféré d'utiliser le kernel linear"
                                      " pour entrainer cette dataset.")
        show_button()
        paraXtrain['values'] = ('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age')
        paraYtrain['values'] = ('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age')
        paraXtest['values'] = ('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age')
        paraYtest['values'] = ('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age')
    else:
        hide_button()
    check_fields()


def showGamma(event):
    if getValeurParamKernel() == "rbf":
        show_entryGamma()
    else:
        hide_entryGamma()
    check_fields()


# fct pour entrainer les differents models de svm et afficher les graphe d'entrainement
def fitModel():
    sizetest = getValeurTestSize()
    kernel = getValeurParamKernel()
    C = getValeurParamC()
    selected_value = combo_box.get()
    if selected_value == "Dataset Maladies Cardiaques":
        trainModelSVMMaladie(kernel, float(sizetest), float(C))
        tracer_grapheMaladie_train(kernel, float(sizetest), float(C))
    elif selected_value == "Dataset Penguin":
        trainModelSVMPenguin(kernel, float(sizetest), float(C))
        tracer_graphePenguin_train(kernel, float(sizetest), float(C))
    elif selected_value == "Dataset Iris":
        trainModelSVMIris(kernel, float(sizetest), float(C))
        tracer_grapheIris_train(kernel, float(sizetest), float(C))
    elif selected_value == "Dataset Diabets":
        trainModelSVMDiabets(kernel, float(sizetest), float(C))
        tracer_grapheDiabets_train(kernel, float(sizetest), float(C))


# fct pour tester les models et afficher les graphes du test et la matrice de confusion
def tracerGraphe():
    sizetest = getValeurTestSize()
    kernel = getValeurParamKernel()
    C = getValeurParamC()
    selected_value = combo_box.get()
    if selected_value == "Dataset Maladies Cardiaques":
        testModelSvmMaladie(kernel, float(sizetest), float(C))
        tracer_grapheMaladie_test(kernel, float(sizetest), float(C))
        tracer_matriceConfusionMaladie(kernel, float(sizetest), float(C))
    elif selected_value == "Dataset Penguin":
        testModelSvmPenguin(kernel, float(sizetest), float(C))
        tracer_graphePenguin_test(kernel, float(sizetest), float(C))
        tracer_matriceConfusionPenguin(kernel, float(sizetest), float(C))
    elif selected_value == "Dataset Iris":
        testModelSvmIris(kernel, float(sizetest), float(C))
        tracer_matriceConfusionIris(kernel, float(sizetest), float(C))
        tracer_grapheIris_test(kernel, float(sizetest), float(C))
    elif selected_value == "Dataset Diabets":
        testModelSvmDiabets(kernel, float(sizetest), float(C))
        tracer_matriceConfusionDiabets(kernel, float(sizetest), float(C))
        tracer_grapheDiabets_test(kernel, float(sizetest), float(C))


# model iris
# fct pour entrainer le model des maladies cardiaques
def trainModelSVMIris(kernel, testsize, c, gamma=0):
    if getValeurParamKernel() == "rbf":
        gamma = float(getValeurGamma())
    # model maladie cardiaque
    svmModelIris = SVMModelIris(kernel, c, gamma)
    # Chargement des données
    irisData = import_dataIris()
    feature_index_x = getFeatureIndex(getValeurXlabelTest())
    feature_index_y = getFeatureIndex(getValeurYlabelTest())
    # Séparation des données et de target
    X_iris = irisData.data[:, :2]  # Utiliser seulement les deux premières caractéristiques
    y_iris = irisData.target
    # Diviser les données en ensembles d'entraînement et de test
    featuresIris_train, featuresIris_test, targetIris_train, targetIris_test = train_test_split(X_iris, y_iris, test_size=testsize, random_state=0)
    # faire le train
    svmModelIris.fit(featuresIris_train, targetIris_train)
    iris_pred = svmModelIris.predict(featuresIris_train)
    # Évaluer les performances du modèle
    accuracy = accuracy_score(targetIris_train, iris_pred)
    f1 = f1_score(targetIris_train, iris_pred, average='weighted')
    precision = precision_score(targetIris_train, iris_pred, average='weighted')
    accuracyLabeltrain.configure(text=str("{:.3f}".format(accuracy)))
    scoreLabeltrain.configure(text=str("{:.3f}".format(f1)))
    precisionLabeltrain.configure(text=str("{:.3f}".format(precision)))
    return featuresIris_train, featuresIris_test, svmModelIris, targetIris_train, targetIris_test


# fct pour faire le test du model des maladies cardiaques
def testModelSvmIris(kernel, testsize, c, gamma=0):
    if getValeurParamKernel() == "rbf":
        gamma = float(getValeurGamma())
    featuresIris_train, featuresIris_test, svmModelIris, targetIris_train, targetIris_test = trainModelSVMIris(kernel, testsize, c, gamma)
    iris_pred = svmModelIris.predict(featuresIris_test)
    # Évaluer les performances du modèle
    accuracy = accuracy_score(targetIris_test, iris_pred)
    # Calcul du score F1
    f1 = f1_score(targetIris_test, iris_pred, average='weighted')
    precision = precision_score(targetIris_test, iris_pred, average='weighted')
    # Affichage du score F1 et accuracy dans les labels
    accuracyLabel.configure(text=str("{:.3f}".format(accuracy)))
    scoreLabel.configure(text=str("{:.3f}".format(f1)))
    precisionLabel.configure(text=str("{:.3f}".format(precision)))


# fct pour tracer la matrice de confusion de model des maladies cardiaques
canvas_metrics = None


def tracer_matriceConfusionIris(kernel, testSize, C, gamma=0):
    global canvas_metrics
    # Détruire le canvas s'il existe déjà
    if canvas_metrics:
        canvas_metrics.get_tk_widget().destroy()

    if getValeurParamKernel() == "rbf":
        gamma = float(getValeurGamma())
    # Entraîner le modèle SVM et extraire l'objet de modèle SVM
    model_tuple = trainModelSVMIris(kernel, testSize, C, gamma)
    svmmodelIris = model_tuple[2]
    cm = ConfusionMatrixDisplay.from_predictions(model_tuple[4], svmmodelIris.predict(model_tuple[1]))
    # Obtenir la figure de la matrice de confusion
    fig, ax = plt.subplots(figsize=(4, 4))
    cm.plot(ax=ax)
    # Créer un widget Tkinter pour afficher la figure
    canvas_metrics = FigureCanvasTkAgg(fig, master=f_graphetest)
    canvas_metrics.draw()
    canvas_metrics.get_tk_widget().pack(side=tk.RIGHT)


# Fonction pour tracer le graphe avec les données d'entraînement du diabet
canvas_test = None


def getFeatureIndex(feature):
    feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    try:
        return feature_names.index(feature)
    except ValueError:
        print("La caractéristique spécifiée n'existe pas dans le jeu de données.")
        return None


def tracer_grapheIris_test(kernel, testSize, C, gamma=0):
    global canvas_test
    # Détruire le canvas s'il existe déjà
    if canvas_test:
        canvas_test.get_tk_widget().destroy()

    if getValeurParamKernel() == "rbf":
        gamma = float(getValeurGamma())

    # Entraîner le modèle SVM et extraire l'objet de modèle SVM
    model_tuple = trainModelSVMIris(kernel, testSize, C, gamma)
    svmmodelIris = model_tuple[2]
    # Création du graphe avec la marge et les vecteurs de support
    fig = plt.figure(figsize=(4, 4))
    # Afficher les données en fonction de la caractéristique sélectionnée
    feature_index_x = getFeatureIndex(getValeurXlabelTest())
    feature_index_y = getFeatureIndex(getValeurYlabelTest())
    # Extraction des indices des vecteurs de support
    support_indices = svmmodelIris.support_()
    # Limiter le nombre de vecteurs de support à afficher

    num_support_vectors = 3  # Nombre souhaité de vecteurs de support à afficher par classe
    selected_support_vectors = []

    for class_label in np.unique(model_tuple[3]):
        class_support_indices = support_indices[np.where(model_tuple[3][support_indices] == class_label)]
        num_vectors = min(num_support_vectors, len(class_support_indices))
        random_indices = np.random.choice(class_support_indices, num_vectors, replace=False)
        selected_support_vectors.extend(model_tuple[0][random_indices])

    selected_support_vectors = np.array(selected_support_vectors)

    plt.scatter(model_tuple[1][:, 0], model_tuple[1][:, 1], c=model_tuple[4], cmap='viridis')
    # Marquer les vecteurs de support d'une croix
    plt.scatter(selected_support_vectors[:, 0], selected_support_vectors[:, 1], s=100, linewidth=1, facecolors='#FFAAAA',  edgecolors='k')
    # Calcul des coordonnées de l'hyperplan
    x_min, x_max = model_tuple[1][:, 0].min() - 1, model_tuple[1][:, 0].max() + 1
    y_min, y_max = model_tuple[1][:, 1].min() - 1, model_tuple[1][:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = svmmodelIris.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Tracé du graphe avec les vecteurs de support et les marges
    # Plot de l'hyperplan et des marges
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
    plt.scatter(selected_support_vectors[:, 0], selected_support_vectors[:, 1], s=100, facecolors='none', edgecolors='k')
    plt.xlabel(getValeurXlabelTest())
    plt.ylabel(getValeurYlabelTest())
    # Créer le canvas pour afficher le graphe
    canvas_test = FigureCanvasTkAgg(fig, master=f_graphetest)
    canvas_test.draw()
    canvas_test.get_tk_widget().pack(side=tk.LEFT)


# Fonction pour tracer le graphe avec les données d'entraînement du iris
canvas_train = None


def tracer_grapheIris_train(kernel, testSize, C, gamma=0):
    global canvas_train
    # Détruire le canvas s'il existe déjà
    if canvas_train:
        canvas_train.get_tk_widget().destroy()

    if getValeurParamKernel() == "rbf":
        gamma = float(getValeurGamma())
    # Entraîner le modèle SVM et extraire l'objet de modèle SVM
    model_tuple = trainModelSVMIris(kernel, testSize, C, gamma)
    svmmodelIris = model_tuple[2]
    # Création du graphe avec la marge et les vecteurs de support
    fig = plt.figure(figsize=(4, 4))
    # Extraction des indices des vecteurs de support
    support_indices = svmmodelIris.support_()
    # Limiter le nombre de vecteurs de support à afficher

    num_support_vectors = 4  # Nombre souhaité de vecteurs de support à afficher par classe
    selected_support_vectors = []

    for class_label in np.unique(model_tuple[3]):
        class_support_indices = support_indices[np.where(model_tuple[3][support_indices] == class_label)]
        num_vectors = min(num_support_vectors, len(class_support_indices))
        random_indices = np.random.choice(class_support_indices, num_vectors, replace=False)
        selected_support_vectors.extend(model_tuple[0][random_indices])

    selected_support_vectors = np.array(selected_support_vectors)
    # afficher les données
    plt.scatter(model_tuple[0][:, 0], model_tuple[0][:, 1], c=model_tuple[3], cmap='viridis')
    # Marquer les vecteurs de support d'une croix
    plt.scatter(selected_support_vectors[:, 0], selected_support_vectors[:, 1], s=100, linewidth=1, facecolors='#FFAAAA',
                edgecolors='k')
    # Calcul des coordonnées de l'hyperplan
    x_min, x_max = model_tuple[0][:, 0].min() - 1, model_tuple[0][:, 0].max() + 1
    y_min, y_max = model_tuple[0][:, 1].min() - 1, model_tuple[0][:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = svmmodelIris.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Tracé du graphe avec les vecteurs de support et les marges
    # Plot de l'hyperplan et des marges
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
    plt.scatter(selected_support_vectors[:, 0], selected_support_vectors[:, 1], s=100, facecolors='none', edgecolors='k')
    plt.xlabel(getValeurXlabelTrain())
    plt.ylabel(getValeurYlabelTrain())
    # Créer le canvas pour afficher le graphe
    canvas_train = FigureCanvasTkAgg(fig, master=f_graphetrain)
    canvas_train.draw()
    canvas_train.get_tk_widget().pack(side=tk.LEFT)


# model diabet
# fct pour entrainer le model des personnes diabetes
def trainModelSVMDiabets(kernel, testsize, c, gamma=0):
    if getValeurParamKernel() == "rbf":
        gamma = float(getValeurGamma())
    # model maladie cardiaque
    svmModelDiabets = SVMModelDiabets(kernel, c, gamma)
    # Chargement des données
    diabetData = import_dataDiabets("datasets/diabetes.csv")
    # Séparation des données et de target
    X_diabet = diabetData[["Glucose", "BloodPressure", "Insulin", "DiabetesPedigreeFunction"]].to_numpy()
    y_diabets = diabetData["Outcome"]
    X = X_diabet[:, :2]
    # Diviser les données en ensembles d'entraînement et de test
    featuresDiabets_train, featuresDiabets_test, targetDiabets_train, targetDiabets_test = train_test_split(X, y_diabets, test_size=testsize, random_state=0)
    # faire le train
    svmModelDiabets.fit(featuresDiabets_train, targetDiabets_train)
    diabet_pred = svmModelDiabets.predict(featuresDiabets_test)
    # Évaluer les performances du modèle
    accuracy = accuracy_score(targetDiabets_test, diabet_pred)
    # Calcul du score F1
    f1 = f1_score(targetDiabets_test, diabet_pred)
    # calcul de precision
    precision = precision_score(targetDiabets_test, diabet_pred)
    # Affichage du score F1 et accuracy dans les labels
    accuracyLabeltrain.configure(text=str("{:.3f}".format(accuracy)))
    scoreLabeltrain.configure(text=str("{:.3f}".format(f1)))
    precisionLabeltrain.configure(text=str("{:.3f}".format(precision)))
    return featuresDiabets_train, featuresDiabets_test, svmModelDiabets, targetDiabets_train, targetDiabets_test


# fct pour faire le test du model des maladies cardiaques
def testModelSvmDiabets(kernel, testsize, c, gamma=0):
    if getValeurParamKernel() == "rbf":
        gamma = float(getValeurGamma())
    featuresDiabets_train, featuresDiabets_test, svmModelDiabets, targetDiabets_train, targetDiabets_test = trainModelSVMDiabets(kernel, testsize, c, gamma)
    diabet_pred = svmModelDiabets.predict(featuresDiabets_test)
    # Évaluer les performances du modèle
    accuracy = accuracy_score(targetDiabets_test, diabet_pred)
    # Calcul du score F1
    f1 = f1_score(targetDiabets_test, diabet_pred)
    precision = precision_score(targetDiabets_test, diabet_pred)
    # Affichage du score F1 et accuracy dans les labels
    accuracyLabel.configure(text=str("{:.3f}".format(accuracy)))
    scoreLabel.configure(text=str("{:.3f}".format(f1)))
    precisionLabel.configure(text=str("{:.3f}".format(precision)))


# fct pour tracer la matrice de confusion de model des diabets


def tracer_matriceConfusionDiabets(kernel, testSize, C, gamma=0):
    global canvas_metrics
    # Détruire le canvas s'il existe déjà
    if canvas_metrics:
        canvas_metrics.get_tk_widget().destroy()

    if getValeurParamKernel() == "rbf":
        gamma = float(getValeurGamma())
    # Entraîner le modèle SVM et extraire l'objet de modèle SVM
    model_tuple = trainModelSVMDiabets(kernel, testSize, C, gamma)
    svmmodelDiabet = model_tuple[2]
    cm = ConfusionMatrixDisplay.from_predictions(model_tuple[4], svmmodelDiabet.predict(model_tuple[1]))
    # Obtenir la figure de la matrice de confusion
    fig, ax = plt.subplots(figsize=(4, 4))
    cm.plot(ax=ax)
    # Créer un widget Tkinter pour afficher la figure
    canvas_metrics = FigureCanvasTkAgg(fig, master=f_graphetest)
    canvas_metrics.draw()
    canvas_metrics.get_tk_widget().pack(side=tk.RIGHT)


# Fonction pour tracer le graphe avec les données d'entraînement du diabet


def tracer_grapheDiabets_test(kernel, testSize, C, gamma=0):
    global canvas_test
    # Détruire le canvas s'il existe déjà
    if canvas_test:
        canvas_test.get_tk_widget().destroy()

    if getValeurParamKernel() == "rbf":
        gamma = float(getValeurGamma())
    # Entraîner le modèle SVM et extraire l'objet de modèle SVM
    model_tuple = trainModelSVMDiabets(kernel, testSize, C, gamma)
    svmmodelDiabets = model_tuple[2]
    # Création du graphe avec la marge et les vecteurs de support
    fig = plt.figure(figsize=(4, 4))
    # afficher les données
    plt.plot(model_tuple[1][:, 0][model_tuple[4] == 0], model_tuple[1][:, 1][model_tuple[4] == 0], "yo")
    # afficher les données
    plt.plot(model_tuple[1][:, 0][model_tuple[4] == 1], model_tuple[1][:, 1][model_tuple[4] == 1], "bo")
    # Limites du cadre
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    support_vectors_ = svmmodelDiabets.support_vectors_()
    # Marquer les vecteurs de support d'une croix
    ax.scatter(support_vectors_[:, 0], support_vectors_[:, 1], linewidth=1, facecolors='#FFAAAA', s=180)
    # Grille de points sur lesquels appliquer le modèle
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    # Prédire pour les points de la grille
    Z = svmmodelDiabets.decision_function(xy).reshape(XX.shape)
    # Afficher la frontière de décision et la marge
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    # Tracé du graphe avec les vecteurs de support et les marges
    plt.scatter(support_vectors_[:, 0], support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k')
    plt.xlabel(getValeurXlabelTest())
    plt.ylabel(getValeurYlabelTest())
    # Créer le canvas pour afficher le graphe
    canvas_test = FigureCanvasTkAgg(fig, master=f_graphetest)
    canvas_test.draw()
    canvas_test.get_tk_widget().pack(side=tk.LEFT)


# Fonction pour tracer le graphe avec les données d'entraînement du diabets
canvas_train = None


def tracer_grapheDiabets_train(kernel, testSize, C, gamma=0):
    global canvas_train
    # Détruire le canvas s'il existe déjà
    if canvas_train:
        canvas_train.get_tk_widget().destroy()

    if getValeurParamKernel() == "rbf":
        gamma = float(getValeurGamma())
    # Entraîner le modèle SVM et extraire l'objet de modèle SVM
    model_tuple = trainModelSVMDiabets(kernel, testSize, C, gamma)
    svmmodelDiabets = model_tuple[2]
    # Création du graphe avec la marge et les vecteurs de support
    fig = plt.figure(figsize=(4, 4))
    # afficher les données
    plt.plot(model_tuple[0][:, 0][model_tuple[3] == 0], model_tuple[0][:, 1][model_tuple[3] == 0], "yo")
    # afficher les données
    plt.plot(model_tuple[0][:, 0][model_tuple[3] == 1], model_tuple[0][:, 1][model_tuple[3] == 1], "bo")
    # Limites du cadre
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    support_vectors_ = svmmodelDiabets.support_vectors_()
    # Marquer les vecteurs de support d'une croix
    ax.scatter(support_vectors_[:, 0], support_vectors_[:, 1], linewidth=1, facecolors='#FFAAAA', s=180)
    # Grille de points sur lesquels appliquer le modèle
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    # Prédire pour les points de la grille
    Z = svmmodelDiabets.decision_function(xy).reshape(XX.shape)
    # Afficher la frontière de décision et la marge
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    # Tracé du graphe avec les vecteurs de support et les marges
    plt.scatter(support_vectors_[:, 0], support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k')
    plt.xlabel(getValeurXlabelTrain())
    plt.ylabel(getValeurYlabelTrain())
    # Créer le canvas pour afficher le graphe
    canvas_train = FigureCanvasTkAgg(fig, master=f_graphetrain)
    canvas_train.draw()
    canvas_train.get_tk_widget().pack(side=tk.LEFT)


# model maladies cardiaques
# fct pour entrainer le model des maladies cardiaques
def trainModelSVMMaladie(kernel, testsize, c, gamma=0):
    if getValeurParamKernel() == "rbf":
        gamma = float(getValeurGamma())
    # model maladie cardiaque
    svmmodelMaladieCardiaque = SVMModelMaladieCardiaque(kernel, c, gamma)
    # Chargement des données
    maladie_data = import_dataMaladie("datasets/dataset_maladie.csv")
    # Séparation des données et de target
    X_maladie = maladie_data[["age", "thalach"]].to_numpy()
    y_maladie = maladie_data["target"].values
    # Diviser les données en ensembles d'entraînement et de test
    featuresMaladie_train, featuresMaladie_test, targetMaladie_train, targetMaladie_test = train_test_split(X_maladie, y_maladie, test_size=testsize, random_state=0)
    # faire le train
    svmmodelMaladieCardiaque.fit(featuresMaladie_train, targetMaladie_train)
    maladie_pred = svmmodelMaladieCardiaque.predict(featuresMaladie_test)
    # Évaluer les performances du modèle
    accuracy = accuracy_score(targetMaladie_test, maladie_pred)
    # Calcul du score F1
    f1 = f1_score(targetMaladie_test, maladie_pred)
    precision = precision_score(targetMaladie_test, maladie_pred)
    # Affichage du score F1 et accuracy dans les labels
    accuracyLabeltrain.configure(text=str("{:.3f}".format(accuracy)))
    scoreLabeltrain.configure(text=str("{:.3f}".format(f1)))
    precisionLabeltrain.configure(text=str("{:.3f}".format(precision)))
    return featuresMaladie_train, featuresMaladie_test, svmmodelMaladieCardiaque, targetMaladie_train, targetMaladie_test


# fct pour faire le test du model des maladies cardiaques
def testModelSvmMaladie(kernel, testsize, c, gamma=0):
    if getValeurParamKernel() == "rbf":
        gamma = float(getValeurGamma())
    featuresMaladie_train, featuresMaladie_test, svmmodelMaladieCardiaque, targetMaladie_train, targetMaladie_test = trainModelSVMMaladie(kernel, testsize, c, gamma)
    maladie_pred = svmmodelMaladieCardiaque.predict(featuresMaladie_test)
    # Évaluer les performances du modèle
    accuracy = accuracy_score(targetMaladie_test, maladie_pred)
    # Calcul du score F1
    f1 = f1_score(targetMaladie_test, maladie_pred)
    precision = precision_score(targetMaladie_test, maladie_pred)
    # Affichage du score F1 et accuracy dans les labels
    accuracyLabel.configure(text=str("{:.3f}".format(accuracy)))
    scoreLabel.configure(text=str("{:.3f}".format(f1)))
    precisionLabel.configure(text=str("{:.3f}".format(precision)))


# fct pour tracer la matrice de confusion de model des maladies cardiaques


def tracer_matriceConfusionMaladie(kernel, testSize, C, gamma=0):
    global canvas_metrics
    # Détruire le canvas s'il existe déjà
    if canvas_metrics:
        canvas_metrics.get_tk_widget().destroy()

    if getValeurParamKernel() == "rbf":
        gamma = float(getValeurGamma())

    # Entraîner le modèle SVM et extraire l'objet de modèle SVM
    model_tuple = trainModelSVMMaladie(kernel, testSize, C, gamma)
    svmmodelMaladieCardiaque = model_tuple[2]
    cm = ConfusionMatrixDisplay.from_predictions(model_tuple[4], svmmodelMaladieCardiaque.predict(model_tuple[1]))
    # Obtenir la figure de la matrice de confusion
    fig, ax = plt.subplots(figsize=(4, 4))
    cm.plot(ax=ax)
    # Créer un widget Tkinter pour afficher la figure
    canvas_metrics = FigureCanvasTkAgg(fig, master=f_graphetest)
    canvas_metrics.draw()
    canvas_metrics.get_tk_widget().pack(side=tk.RIGHT)


# Fonction pour tracer le graphe avec les données d'entraînement du maladie


def tracer_grapheMaladie_test(kernel, testSize, C, gamma=0):
    global canvas_test
    # Détruire le canvas s'il existe déjà
    if canvas_test:
        canvas_test.get_tk_widget().destroy()

    if getValeurParamKernel() == "rbf":
        gamma = float(getValeurGamma())
    # Entraîner le modèle SVM et extraire l'objet de modèle SVM
    model_tuple = trainModelSVMMaladie(kernel, testSize, C, gamma)
    svmmodelMaladieCardiaque = model_tuple[2]
    # Création du graphe avec la marge et les vecteurs de support
    fig = plt.figure(figsize=(4, 4))
    # afficher les données
    plt.plot(model_tuple[1][:, 0][model_tuple[4] == 0], model_tuple[1][:, 1][model_tuple[4] == 0], "yo", label="0:non malade")
    # afficher les données
    plt.plot(model_tuple[1][:, 0][model_tuple[4] == 1], model_tuple[1][:, 1][model_tuple[4] == 1], "bo", label="1:malade")
    # Limites du cadre
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    support_vectors_ = svmmodelMaladieCardiaque.support_vectors_()
    # Marquer les vecteurs de support d'une croix
    ax.scatter(support_vectors_[:, 0], support_vectors_[:, 1], linewidth=1, facecolors='#FFAAAA', s=180)
    # Grille de points sur lesquels appliquer le modèle
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    # Prédire pour les points de la grille
    Z = svmmodelMaladieCardiaque.decision_function(xy).reshape(XX.shape)
    # Afficher la frontière de décision et la marge
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    # Tracé du graphe avec les vecteurs de support et les marges
    plt.scatter(support_vectors_[:, 0], support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k')
    plt.xlabel(getValeurXlabelTest())
    plt.ylabel(getValeurYlabelTest())
    # Créer le canvas pour afficher le graphe
    canvas_test = FigureCanvasTkAgg(fig, master=f_graphetest)
    canvas_test.draw()
    canvas_test.get_tk_widget().pack(side=tk.LEFT)


# Fonction pour tracer le graphe avec les données d'entraînement du maladie


def tracer_grapheMaladie_train(kernel, testSize, C, gamma=0):
    global canvas_train
    # Détruire le canvas s'il existe déjà
    if canvas_train:
        canvas_train.get_tk_widget().destroy()

    if getValeurParamKernel() == "rbf":
        gamma = float(getValeurGamma())
    # Entraîner le modèle SVM et extraire l'objet de modèle SVM
    model_tuple = trainModelSVMMaladie(kernel, testSize, C, gamma)
    svmmodelMaladieCardiaque = model_tuple[2]
    # Création du graphe avec la marge et les vecteurs de support
    fig = plt.figure(figsize=(4, 4))
    # afficher les données
    plt.plot(model_tuple[0][:, 0][model_tuple[3] == 0], model_tuple[0][:, 1][model_tuple[3] == 0], "yo", label="0:non malade")
    # afficher les données
    plt.plot(model_tuple[0][:, 0][model_tuple[3] == 1], model_tuple[0][:, 1][model_tuple[3] == 1], "bo", label="1:malade")
    # Limites du cadre
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    support_vectors_ = svmmodelMaladieCardiaque.support_vectors_()
    # Marquer les vecteurs de support d'une croix
    ax.scatter(support_vectors_[:, 0], support_vectors_[:, 1], linewidth=1, facecolors='#FFAAAA', s=180)
    # Grille de points sur lesquels appliquer le modèle
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    # Prédire pour les points de la grille
    Z = svmmodelMaladieCardiaque.decision_function(xy).reshape(XX.shape)
    # Afficher la frontière de décision et la marge
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    # Tracé du graphe avec les vecteurs de support et les marges
    plt.scatter(support_vectors_[:, 0], support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k')
    plt.xlabel(getValeurXlabelTrain())
    plt.ylabel(getValeurYlabelTrain())
    # Créer le canvas pour afficher le graphe
    canvas_train = FigureCanvasTkAgg(fig, master=f_graphetrain)
    canvas_train.draw()
    canvas_train.get_tk_widget().pack(side=tk.LEFT)


# model penguin
# fct pour entrainer le model des penguins
def trainModelSVMPenguin(kernel, testsize, c, gamma=0):
    if getValeurParamKernel() == "rbf":
        gamma = float(getValeurGamma())
    # Chargement des données
    svmmodelPenguin = SVMModelPenguin(kernel, c, gamma)
    penguins = import_dataPenguin("datasets/penguins.csv")
    data = penguins[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]].to_numpy()
    labels = pd.Categorical(penguins["species"]).astype('category').codes
    # Nous allons ici nous se limiter à deux espèces :
    # Adelie (0) et Gentoo (2)
    # et deux variables : bill_length_mm et bill_depth_mm.
    y_penguin_data = labels
    Adelie_or_Gentoo = (y_penguin_data == 0) | (y_penguin_data == 2)
    X_penguin_data = data[:, :2][Adelie_or_Gentoo]
    y_penguin = y_penguin_data[Adelie_or_Gentoo]
    # Instantiate the imputer
    imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent' etc.
    # Impute missing values in X_penguin
    X_penguin = imputer.fit_transform(X_penguin_data)
    # Diviser les données en ensembles d'entraînement et de test
    featuresPenguin_train, featuresPenguin_test, targetPenguin_train, targetPenguin_test = train_test_split(X_penguin, y_penguin, test_size=testsize, random_state=0)
    # Prédiction sur l'ensemble de test
    svmmodelPenguin.fit(featuresPenguin_train, targetPenguin_train)
    penguin_pred = svmmodelPenguin.predict(featuresPenguin_test)
    # Évaluer les performances du modèle
    accuracy = accuracy_score(targetPenguin_test, penguin_pred)
    # Calcul du score F1
    f1 = f1_score(targetPenguin_test, penguin_pred, pos_label=0)
    precision = precision_score(targetPenguin_test, penguin_pred, pos_label=0)
    # Affichage du score F1 et accuracy
    accuracyLabeltrain.configure(text=str("{:.3f}".format(accuracy)))
    scoreLabeltrain.configure(text=str("{:.3f}".format(f1)))
    precisionLabeltrain.configure(text=str("{:.3f}".format(precision)))
    return featuresPenguin_train, featuresPenguin_test, svmmodelPenguin, targetPenguin_train, targetPenguin_test


# fct pour faire le test du model des maladies cardiaques
def testModelSvmPenguin(kernel, testsize, c, gamma=0):
    if getValeurParamKernel() == "rbf":
        gamma = float(getValeurGamma())
    featuresPenguin_train, featuresPenguin_test, svmmodelPenguin, targetPenguin_train, targetPenguin_test = trainModelSVMPenguin(kernel, testsize, c, gamma)
    penguin_pred = svmmodelPenguin.predict(featuresPenguin_test)
    # Évaluer les performances du modèle
    accuracy = accuracy_score(targetPenguin_test, penguin_pred)
    # Calcul du score F1
    f1 = f1_score(targetPenguin_test, penguin_pred, pos_label=0)
    precision = precision_score(targetPenguin_test, penguin_pred, pos_label=0)
    # Affichage du score F1 et accuracy
    accuracyLabel.configure(text=str("{:.3f}".format(accuracy)))
    scoreLabel.configure(text=str("{:.3f}".format(f1)))
    precisionLabel.configure(text=str("{:.3f}".format(precision)))


# fct pour afficher la matrice de confusion de model penguin


def tracer_matriceConfusionPenguin(kernel, testSize, C, gamma=0):
    global canvas_metrics
    # Détruire le canvas s'il existe déjà
    if canvas_metrics:
        canvas_metrics.get_tk_widget().destroy()

    if getValeurParamKernel() == "rbf":
        gamma = float(getValeurGamma())

    # Entraîner le modèle SVM et extraire l'objet de modèle SVM
    model_tuple = trainModelSVMPenguin(kernel, testSize, C, gamma)
    svmmodelPenguin = model_tuple[2]
    cm = ConfusionMatrixDisplay.from_predictions(model_tuple[4], svmmodelPenguin.predict(model_tuple[1]))
    # Obtenir la figure de la matrice de confusion
    fig, ax = plt.subplots(figsize=(4, 4))
    cm.plot(ax=ax)
    # Créer un widget Tkinter pour afficher la figure
    canvas_metrics = FigureCanvasTkAgg(fig, master=f_graphetest)
    canvas_metrics.draw()
    canvas_metrics.get_tk_widget().pack(side=tk.RIGHT)


# Fonction pour tracer le graphe avec les données d'entraînement du penguin


def tracer_graphePenguin_test(kernel, testSize, C, gamma=0):
    global canvas_test
    # Détruire le canvas s'il existe déjà
    if canvas_test:
        canvas_test.get_tk_widget().destroy()

    if getValeurParamKernel() == "rbf":
        gamma = float(getValeurGamma())
    # Entraîner le modèle SVM et extraire l'objet de modèle SVM
    model_tuple = trainModelSVMPenguin(kernel, testSize, C, gamma)
    svmmodelPenguin = model_tuple[2]
    # Création du graphe avec la marge et les vecteurs de support
    fig = plt.figure(figsize=(4, 4))
    # afficher les données
    plt.plot(model_tuple[1][:, 0][model_tuple[4] == 0], model_tuple[1][:, 1][model_tuple[4] == 0], "yo", label="Adelie")
    # afficher les données
    plt.plot(model_tuple[1][:, 0][model_tuple[4] == 2], model_tuple[1][:, 1][model_tuple[4] == 2], "bo", label="Gentoo")
    # Limites du cadre
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    support_vectors_ = svmmodelPenguin.support_vectors_()
    # Marquer les vecteurs de support d'une croix
    ax.scatter(support_vectors_[:, 0], support_vectors_[:, 1], linewidth=1, facecolors='#FFAAAA', s=180)
    # Grille de points sur lesquels appliquer le modèle
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    # Prédire pour les points de la grille
    Z = svmmodelPenguin.decision_function(xy).reshape(XX.shape)
    # Afficher la frontière de décision et la marge
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    # Tracé du graphe avec les vecteurs de support et les marges
    plt.scatter(support_vectors_[:, 0], support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k')
    plt.xlabel(getValeurXlabelTest())
    plt.ylabel(getValeurYlabelTest())
    # Créer le canvas pour afficher le graphe
    canvas_test = FigureCanvasTkAgg(fig, master=f_graphetest)
    canvas_test.draw()
    canvas_test.get_tk_widget().pack(side=tk.LEFT)


# Fonction pour tracer le graphe avec les données d'entraînement du penguin


def tracer_graphePenguin_train(kernel, testSize, C, gamma=0):
    global canvas_train
    # Détruire le canvas s'il existe déjà
    if canvas_train:
        canvas_train.get_tk_widget().destroy()

    if getValeurParamKernel() == "rbf":
        gamma = float(getValeurGamma())
    # Entraîner le modèle SVM et extraire l'objet de modèle SVM
    model_tuple = trainModelSVMPenguin(kernel, testSize, C, gamma)
    svmmodelPenguin = model_tuple[2]
    # Création du graphe avec la marge et les vecteurs de support
    fig = plt.figure(figsize=(4, 4))
    # afficher les données
    plt.plot(model_tuple[0][:, 0][model_tuple[3] == 0], model_tuple[0][:, 1][model_tuple[3] == 0], "yo", label="Adelie")
    # afficher les données
    plt.plot(model_tuple[0][:, 0][model_tuple[3] == 2], model_tuple[0][:, 1][model_tuple[3] == 2], "bo", label="Gentoo")
    # Limites du cadre
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    support_vectors_ = svmmodelPenguin.support_vectors_()
    # Marquer les vecteurs de support d'une croix
    ax.scatter(support_vectors_[:, 0], support_vectors_[:, 1], linewidth=1, facecolors='#FFAAAA', s=180)
    # Grille de points sur lesquels appliquer le modèle
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    # Prédire pour les points de la grille
    Z = svmmodelPenguin.decision_function(xy).reshape(XX.shape)
    # Afficher la frontière de décision et la marge
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    # Tracé du graphe avec les vecteurs de support et les marges
    plt.scatter(support_vectors_[:, 0], support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k')
    plt.xlabel(getValeurXlabelTrain())
    plt.ylabel(getValeurYlabelTrain())
    # Créer le canvas pour afficher le graphe
    canvas_train = FigureCanvasTkAgg(fig, master=f_graphetrain)
    canvas_train.draw()
    canvas_train.get_tk_widget().pack(side=tk.LEFT)


# creation de la fenetre de splash screen
splash_root = tk.Tk()
# Centrer la fenêtre au milieu de l'écran
screen_width = splash_root.winfo_screenwidth()
screen_height = splash_root.winfo_screenheight()
x = int((screen_width / 2) - (1750 / 2))
y = int((screen_height / 2) - (980 / 2))
splash_root.geometry(f"1750x980+{x}+{y}")
splash_root.config(bg=BG_COLOR)
# Rendre la fenêtre non-redimensionnable
splash_root.resizable(width=False, height=False)
# Créez un canvas pour ajouter une image
splash_canvas = Canvas(splash_root, width=1350, height=950, bg=BG_COLOR, highlightthickness=0)
splash_canvas.pack()

# Charger l'image et la convertir pour Tkinter
image = Image.open("imgs/logo.png")
largeur = 1000
hauteur = 800
image_redimensionnee = image.resize((largeur, hauteur))
photo = ImageTk.PhotoImage(image_redimensionnee)
splash_canvas.create_image(760, 460, anchor=CENTER, image=photo)
# Définissez le temps d'affichage du splash screen en millisecondes
splash_time = 3000
# Fermez la fenêtre du splash screen après le temps d'affichage
splash_root.after(splash_time, splash_root.destroy)


# Création de la principale fenetre
appSVM = tk.Tk()  # nouvelle instance de Tk
appSVM.title("Interface Home Machine")
# Centrer la fenêtre au milieu de l'écran
screen_width = appSVM.winfo_screenwidth()
screen_height = appSVM.winfo_screenheight()
x = int((screen_width / 2) - (1750 / 2))
y = int((screen_height / 2) - (980 / 2))
appSVM.geometry(f"1750x980+{x}+{y}")
appSVM.config(bg=BG_COLOR)
# Rendre la fenêtre non-redimensionnable
appSVM.resizable(width=False, height=False)


# Création du Notebook (les onglets)
notebook = ttk.Notebook(appSVM)


# Création du premier onglet
ongletPrincipale = tk.LabelFrame(notebook, background=BG_COLOR)
notebook.add(ongletPrincipale, text='Onglet Principale')


# creation des frames
f_description = tk.LabelFrame(ongletPrincipale, bd=0, text="", bg=bg_color_frame, relief="groove", width=1200, height=60)
f_parametre = tk.LabelFrame(ongletPrincipale, bd=0, text="", bg=bg_color_frame, relief="groove", width=430, height=790)
f_model = tk.LabelFrame(ongletPrincipale, bd=0, text="", bg=bg_color_frame, relief="groove", width=1270, height=798)
f_desc = tk.LabelFrame(f_parametre, bd=0, text="", bg=bg_color_frame, relief="groove", width=210, height=200)
f_descriptionDataset = tk.LabelFrame(f_desc, bd=0, text="", bg=bg_color_frame, highlightthickness=0)
f3_btn_train = tk.LabelFrame(f_model, bd=0, text="", bg="#26333A", highlightthickness=0, width=430, height=770)
f3_btn_test = tk.LabelFrame(f_model, bd=0, text="", bg="#26333A", highlightthickness=0, width=830, height=770)
f4_grp = tk.LabelFrame(f3_btn_test, bd=0, text="", bg="#26333A", highlightthickness=0)
f_graphetrain = tk.LabelFrame(f3_btn_train, bd=0, text="", bg="#26333A", highlightthickness=0)
f_graphetest = tk.LabelFrame(f4_grp, bd=0, text="", bg="#26333A", highlightthickness=0)
f_labelsTest = tk.LabelFrame(f3_btn_test, bd=0, text="", bg="#26333A", highlightthickness=0)
f_accuracy = tk.LabelFrame(f_labelsTest, bd=0, text="", bg="#26333A", highlightthickness=0)
f_score = tk.LabelFrame(f_labelsTest, bd=0, text="", bg="#26333A", highlightthickness=0)
f_precision = tk.LabelFrame(f_labelsTest, bd=0, text="", bg="#26333A", highlightthickness=0)
f_labelsTrain = tk.LabelFrame(f3_btn_train, bd=0, text="", bg="#26333A", highlightthickness=0)
f_accuracyTrain = tk.LabelFrame(f_labelsTrain, bd=0, text="", bg="#26333A", highlightthickness=0)
f_scoreTrain = tk.LabelFrame(f_labelsTrain, bd=0, text="", bg="#26333A", highlightthickness=0)
f_precisionTrain = tk.LabelFrame(f_labelsTrain, bd=0, text="", bg="#26333A", highlightthickness=0)
f_comboboxTrain = tk.LabelFrame(f3_btn_train, bd=0, text="", bg="#26333A", highlightthickness=0)
f_comboboxTest = tk.LabelFrame(f3_btn_test, bd=0, text="", bg="#26333A", highlightthickness=0)
f_comboboxTrainL1 = tk.LabelFrame(f_comboboxTrain, bd=0, text="", bg="#26333A", highlightthickness=0)
f_comboboxTrainL2 = tk.LabelFrame(f_comboboxTrain, bd=0, text="", bg="#26333A", highlightthickness=0)
f_comboboxTestL1 = tk.LabelFrame(f_comboboxTest, bd=0, text="", bg="#26333A", highlightthickness=0)
f_comboboxTestL2 = tk.LabelFrame(f_comboboxTest, bd=0, text="", bg="#26333A", highlightthickness=0)

f_description.pack(side=tk.TOP, padx=20, pady=20, ipady=10)
f_parametre.pack(side=tk.LEFT, padx=10, pady=15)
f_model.pack(side=tk.RIGHT, padx=5, pady=15)
f_desc.pack(padx=10, pady=20)
f_descriptionDataset.pack(side=tk.BOTTOM, padx=10, pady=20)
f3_btn_train.pack(side=tk.LEFT, padx=5)
f3_btn_test.pack(side=tk.RIGHT, padx=5)
f4_grp.pack(side=tk.BOTTOM)
f_graphetrain.pack(side=tk.BOTTOM)
f_graphetest.pack(side=tk.BOTTOM)
f_comboboxTrain.place(x=-2, y=110)
f_comboboxTest.place(x=98, y=110)
f_comboboxTrainL1.pack(side=tk.TOP)
f_comboboxTrainL2.pack(side=tk.BOTTOM)
f_comboboxTestL1.pack(side=tk.TOP)
f_comboboxTestL2.pack(side=tk.BOTTOM)
f_labelsTest.place(x=260, y=210)
f_accuracy.pack(side=tk.TOP)
f_score.pack(side=tk.BOTTOM)
f_precision.pack(side=tk.BOTTOM)
f_labelsTrain.place(x=110, y=205)
f_accuracyTrain.pack(side=tk.TOP)
f_scoreTrain.pack(side=tk.BOTTOM, pady=5)
f_precisionTrain.pack(side=tk.BOTTOM)

# Fixer la taille du cadre
f_description.pack_propagate(0)
f_parametre.pack_propagate(0)
f_model.pack_propagate(0)
f3_btn_train.pack_propagate(0)
f3_btn_test.pack_propagate(0)

# Création du deuxième onglet
ongletDescription = tk.LabelFrame(notebook, background=BG_COLOR)
notebook.add(ongletDescription, text='onglet de Description')


# Définir le titre de l'onglet 2 avec un Label
titre_onglet2 = ttk.Label(ongletDescription, foreground="#FFFFFF", font=("Arial", 16, "bold"), background="#74B0FF", padding=25)
titre_onglet2.pack(pady=20)

# Création du Treeview pour afficher les données dans l'onglet 2
donnees_treeview = ttk.Treeview(ongletDescription, show="headings", height=10)
# Modifier l'arrière-plan du TreeView
donnees_treeview.configure(style='Custom.Treeview')
# Appliquer le style personnalisé au TreeView
donnees_treeview.tag_configure("Custom.Treeview", background=bg_color_frame, foreground="#FFFFFF", font=("Arial", 13))
# Ajouter le Treeview dans l'onglet 2
donnees_treeview.pack(padx=20, pady=20)

frame_statistique = tk.LabelFrame(ongletDescription, bd=0, text="", bg=BG_COLOR, highlightthickness=0, width=1600, height=760)
frame_statistique.pack(padx=10, pady=10)
frame_statistique.pack_propagate(0)

# Create a Treeview widget for displaying the table
tree = ttk.Treeview(frame_statistique, show="headings", height=8)
# Modifier l'arrière-plan du TreeView
tree.configure(style='Custom.Treeview')
# Appliquer le style personnalisé au TreeView
tree.tag_configure("Custom.Treeview", background=bg_color_frame, foreground="#FFFFFF", font=("Arial", 13))
tree.pack()


# L'onglet 2 est initialisé masqué
notebook.hide(ongletDescription)

# Affichage du Notebook
notebook.pack(expand=True, fill="both")


# creations des composants de frame des parametres
descrProjetlabel = tk.Label(f_description, text="Notre interface graphique consiste à implémenter des models SVM pour des differentes datasets, "
                                                "ainsi elle fournit des divers fonctionnalités à l'utilisitateur pour faciliter l'implémentation "
                                                "des modèles et la visualisation claire des données", fg="#d9d9d9", bg=bg_color_frame, wraplength=1100, font=("Helvetica", 14, "bold"))
descrProjetlabel.pack(padx=50, pady=10)

style = ttk.Style()
style.map("Custom.TCombobox", fieldbackground=[('readonly', 'red')])

# Créer une liste déroulante hh
datasets = ["Dataset Maladies Cardiaques", "Dataset Penguin", "Dataset Iris", "Dataset Diabets"]
combo_box = ttk.Combobox(f_desc, values=datasets, font=("Helvetica", 13), width=35, style="Custom.TCombobox")
# Appliquer le style personnalisé au Combobox
combo_box.state(["readonly"])
combo_box.pack(padx=10, pady=5, ipady=2)

description = tk.Label(f_desc, text="Description : ", fg="#d9d9d9", bg=bg_color_frame, font=("Helvetica", 14, "bold"))
description.pack(padx=10, pady=5)

# Création du Label pour afficher la valeur sélectionnée
descriptiontxt = tk.Label(f_descriptionDataset, text=" ", fg="#d9d9d9", bg=bg_color_frame, font=("Helvetica", 13), wraplength=360, justify="left")
descriptiontxt.pack(side=tk.TOP, padx=2, pady=2)

# Création du bouton pour afficher plus de details
bouton_onglet1 = Button(f_descriptionDataset, height=2, width=20, font=('Helvetica', 13, "bold"), fg='#FFFFFF', bg='#9ED8FA', text="Voir plus de description", bd=0, command=afficher_description)

tstsize = tk.Label(f_parametre, text="La taille du test: ", fg="#d9d9d9", bg=bg_color_frame, font=("Helvetica", 14, "bold"))
tstsize.pack(padx=50, pady=10)

testSize = tk.Entry(f_parametre, width=40, font=("Helvetica", 13), background=ENTRY_BG_COLOR, bd=0, foreground="#FFFFFF")
testSize.pack(padx=2, pady=8, ipady=5)

parac = tk.Label(f_parametre, text="Paramètre C: ", fg="#d9d9d9", bg=bg_color_frame, font=("Helvetica", 14, "bold"))
parac.pack(padx=50, pady=10)

paramC = tk.Entry(f_parametre, width=40, font=("Helvetica", 13), background=ENTRY_BG_COLOR, bd=0, foreground="#FFFFFF")
paramC.pack(padx=2, pady=8, ipady=5)

paramk = tk.Label(f_parametre, text="Paramètre Kernel: ", fg="#d9d9d9", bg=bg_color_frame, font=("Helvetica", 14, "bold"))
paramk.pack(padx=50, pady=10)

# Créer une liste déroulante
kernels = ["linear", "rbf", "poly"]
paramKernel = Combobox(f_parametre, values=kernels, font=("Helvetica", 13), width=38)
paramKernel.pack(padx=5, pady=5, ipady=2)
paramKernel.state(["readonly"])

paraGamma = tk.Label(f_parametre, text="Paramètre Gamma: ", fg="#d9d9d9", bg=bg_color_frame, font=("Helvetica", 14, "bold"))

paramGamma = tk.Entry(f_parametre, width=40, font=("Helvetica", 13), background=ENTRY_BG_COLOR, bd=0, foreground="#FFFFFF")

# creation de boutton pour entrainer le modele
btnTraining = Button(f3_btn_train, height=2, width=20, text="Entrainer", font=('Helvetica', 17, "bold"), fg='#FFFFFF', bg='#9ED8FA', bd=0, command=fitModel, state="disabled")
btnTraining.pack(padx=20, pady=12, side=tk.TOP)

# creation de boutton pour tester le modele   E87F39
btnTesting = tk.Button(f3_btn_test, height=2, width=20, text="Tester", font=('Helvetica', 17, "bold"), fg='#FFFFFF', bg='#FA98D4', bd=0, command=tracerGraphe, state="disabled")
btnTesting.pack(padx=20, pady=12, side=tk.TOP)

accuracyLbl = tk.Label(f_accuracy, text="Accuracy : ", fg="#9ED8FA", bg="#26333A", font=("Helvetica", 14, "bold"))
accuracyLbl.pack(padx=30, pady=10, side=tk.LEFT)
accuracyLabel = tk.Label(f_accuracy, text="", fg="#FFFFFF", bg="#26333A", font=("Helvetica", 14, "bold"))
accuracyLabel.pack(padx=50, pady=10, side=tk.RIGHT)

scoref1Lbl = tk.Label(f_score, text="F1_Score : ", fg="#9ED8FA", bg="#26333A", font=("Helvetica", 14, "bold"))
scoref1Lbl.pack(padx=30, pady=10, side=tk.LEFT)
scoreLabel = tk.Label(f_score, text="", fg="#FFFFFF", bg="#26333A", font=("Helvetica", 14, "bold"))
scoreLabel.pack(padx=50, pady=10, side=tk.RIGHT)

precisionLbl = tk.Label(f_precision, text="Precision : ", fg="#9ED8FA", bg="#26333A", font=("Helvetica", 14, "bold"))
precisionLbl.pack(padx=30, pady=10, side=tk.LEFT)
precisionLabel = tk.Label(f_precision, text="", fg="#FFFFFF", bg="#26333A", font=("Helvetica", 14, "bold"))
precisionLabel.pack(padx=50, pady=10, side=tk.RIGHT)

accuracyLbltrain = tk.Label(f_accuracyTrain, text="Accuracy : ", fg="#9ED8FA", bg="#26333A", font=("Helvetica", 14, "bold"))
accuracyLbltrain.pack(padx=10, pady=10, side=tk.LEFT)
accuracyLabeltrain = tk.Label(f_accuracyTrain, text="", fg="#FFFFFF", bg="#26333A", font=("Helvetica", 14, "bold"))
accuracyLabeltrain.pack(padx=10, pady=10, side=tk.RIGHT)

scoref1Lbltrain = tk.Label(f_scoreTrain, text="F1_Score : ", fg="#9ED8FA", bg="#26333A", font=("Helvetica", 14, "bold"))
scoref1Lbltrain.pack(padx=10, pady=10, side=tk.LEFT)
scoreLabeltrain = tk.Label(f_scoreTrain, text="", fg="#FFFFFF", bg="#26333A", font=("Helvetica", 14, "bold"))
scoreLabeltrain.pack(padx=10, pady=10, side=tk.RIGHT)

precisionLbltrain = tk.Label(f_precisionTrain, text="Precision : ", fg="#9ED8FA", bg="#26333A", font=("Helvetica", 14, "bold"))
precisionLbltrain.pack(padx=10, pady=10, side=tk.LEFT)
precisionLabeltrain = tk.Label(f_precisionTrain, text="", fg="#FFFFFF", bg="#26333A", font=("Helvetica", 14, "bold"))
precisionLabeltrain.pack(padx=10, pady=10, side=tk.RIGHT)

paramXlabelTrain = tk.Label(f_comboboxTrainL1, text="X label : ", fg="#d9d9d9", bg="#26333A", font=("Helvetica", 14, "bold"))
paramXlabelTrain.pack(padx=20, pady=10, side=tk.LEFT)

# Créer une liste déroulante
paraXtrain = Combobox(f_comboboxTrainL1, font=("Helvetica", 13), width=28)
paraXtrain.pack(padx=2, pady=5, ipady=2, side=tk.RIGHT)
paraXtrain.state(["readonly"])

paramYlabelTrain = tk.Label(f_comboboxTrainL2, text="Y label : ", fg="#d9d9d9", bg="#26333A", font=("Helvetica", 14, "bold"))
paramYlabelTrain.pack(padx=20, pady=10, side=tk.LEFT)

# Créer une liste déroulante
paraYtrain = Combobox(f_comboboxTrainL2, font=("Helvetica", 13), width=28)
paraYtrain.pack(padx=2, pady=5, ipady=2, side=tk.RIGHT)
paraYtrain.state(["readonly"])

paramXlabelTest = tk.Label(f_comboboxTestL1, text="X label : ", fg="#d9d9d9", bg="#26333A", font=("Helvetica", 14, "bold"))
paramXlabelTest.pack(padx=30, pady=10, side=tk.LEFT)

# Créer une liste déroulante
paraXtest = Combobox(f_comboboxTestL1, font=("Helvetica", 13), width=35)
paraXtest.pack(padx=2, pady=5, ipady=2, side=tk.RIGHT)
paraXtest.state(["readonly"])

paramYlabelTest = tk.Label(f_comboboxTestL2, text="Y label : ", fg="#d9d9d9", bg="#26333A", font=("Helvetica", 14, "bold"))
paramYlabelTest.pack(padx=30, pady=10, side=tk.LEFT)

# Créer une liste déroulante
paraYtest = Combobox(f_comboboxTestL2, font=("Helvetica", 13), width=35)
paraYtest.pack(padx=2, pady=5, ipady=2, side=tk.RIGHT)
paraYtest.state(["readonly"])


# liason des evenement avec les composants pour checker
testSize.bind("<KeyRelease>", lambda event: check_fields())
paramC.bind("<KeyRelease>", lambda event: check_fields())
paramKernel.bind("<<ComboboxSelected>>", showGamma)
combo_box.bind("<<ComboboxSelected>>", update_label)

appSVM.mainloop()
