# import les biblio
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import train_test_split

from models.SVMModelPenguin import SVMModelPenguin, import_dataPenguin
from models.SVMModelSpam import SVMModelSpam, import_dataSpam
from models.SVMModelMaladieCardiaque import SVMModelMaladieCardiaque, import_dataMaladie


# Définition de la palette de couleurs
BG_COLOR = "#1c1c1c"
FG_COLOR = "#d9d9d9"
LABEL_BG_COLOR = "#3a3a3a"
ENTRY_BG_COLOR = "#3a3a3a"
bg_color_frame = "#3a3a3a"


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


# fct pour verifier que les inputs sont bien remplis et rendre le boutton de train et de test normal
def check_fields():
    if len(getValeurTestSize()) > 0 and len(getValeurParamC()) > 0 and len(getValeurParamC()) > 0 and len(combo_box.get()) > 0:
        btnTraining.config(state="normal")
        btnTesting.config(state="normal")
    else:
        btnTraining.config(state="disabled")
        btnTesting.config(state="disabled")


# Fonction appelée lors de la sélection d'une option dans la ComboBox
def update_label(event):
    selected_value = combo_box.get()
    if selected_value == "Dataset Spam Email" :
        descriptiontxt.configure(text="Le fichier csv contient 5172 lignes, chaque ligne"
                                      " pour chaque e-mail. Il y a 3002 colonnes. La première colonne indique le nom de l'e-mail."
                                      " Le nom a été défini avec des chiffres et non avec le nom des destinataires pour protéger la confidentialité. "
                                      "La dernière colonne contient les libellés de prédiction : 1 pour spam, 0 pour non spam."
                                      " Les 3000 colonnes restantes sont les 3000 mots les plus courants dans tous les e-mails,"
                                      " après exclusion des caractères/mots non alphabétiques. Pour chaque ligne, "
                                      "le nombre de chaque mot (colonne) dans cet e-mail (ligne) est stocké dans les cellules respectives. "
                                       )
    elif selected_value == "Dataset Maladies Cardiaques":
        descriptiontxt.configure(text="description de maladie cardiaque ")
    elif selected_value == "Dataset Penguin":
        descriptiontxt.configure(text="description de penguin ")
    check_fields()


# fct pour entrainer les differents models de svm et afficher les graphe d'entrainement
def fitModel():
    sizetest = getValeurTestSize()
    kernel = getValeurParamKernel()
    C = getValeurParamC()
    selected_value = combo_box.get()
    if selected_value == "Dataset Spam Email":
        trainModelSvmSpam(kernel, float(sizetest), float(C))
        # tracer_grapheSpam_train(kernel, sizetest)
    elif selected_value == "Dataset Maladies Cardiaques":
        trainModelSVMMaladie(kernel, float(sizetest), float(C))
        tracer_grapheMaladie_train(kernel, float(sizetest), float(C))
    elif selected_value == "Dataset Penguin":
        trainModelSVMPenguin(kernel, float(sizetest), float(C))
        tracer_graphePenguin_train(kernel, float(sizetest), float(C))


# fct pour tester les models et afficher les graphes du test et la matrice de confusion
def tracerGraphe():
    sizetest = getValeurTestSize()
    kernel = getValeurParamKernel()
    C = getValeurParamC()
    selected_value = combo_box.get()
    if selected_value == "Dataset Spam Email":
        testModelSvmSpam(kernel, float(sizetest), float(C))
        # tracer_grapheSpam_test(kernel, float(sizetest))
    elif selected_value == "Dataset Maladies Cardiaques":
        testModelSvmMaladie(kernel, float(sizetest), float(C))
        tracer_grapheMaladie_test(kernel, float(sizetest), float(C))
        tracer_matriceConfusionMaladie(kernel, float(sizetest), float(C))
    elif selected_value == "Dataset Penguin":
        testModelSvmPenguin(kernel, float(sizetest), float(C))
        tracer_graphePenguin_test(kernel, float(sizetest), float(C))
        tracer_matriceConfusionPenguin(kernel, float(sizetest), float(C))


# model spam
# fct pour entrainer le model du Spam
def trainModelSvmSpam(kernel, testsize, c):
    # model spam email
    svmmodelSpam = SVMModelSpam(kernel, c)
    # Chargement des données
    emails_data = import_dataSpam('datasets/labeled_emails.csv')
    # Séparation des données en ensembles d'entraînement et de test
    emails = emails_data['email']
    labels = np.where(emails_data['label'] == 'spam', 1, 0)  # Encoder les étiquettes en 0 et 1
    # Vectoriser les courriers électroniques en utilisant la transformation TF-IDF
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(emails)
    # Diviser les données en ensembles d'entraînement et de test
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=testsize, random_state=42)
    # faire le train
    svmmodelSpam.fit(features_train, labels_train)
    return features_train, labels_train, svmmodelSpam, labels_test, features_test


# fct pour faire le test du model de Spam
def testModelSvmSpam(kernel, testsize, c):
    # récuperer les caracteristiques et target retournés par la fct d'entrainement
    features_train, labelstrain, svmmodelSpam, labels_test, features_test = trainModelSvmSpam(kernel, testsize, c)
    mail_pred = svmmodelSpam.predict(features_test)
    # Évaluer les performances du modèle
    accuracy = accuracy_score(labels_test, mail_pred)
    # afficher l'accuracy dans un label
    accuracyLabel.configure(text=str(accuracy))


# model maladies cardiaques
# fct pour entrainer le model des maladies cardiaques
def trainModelSVMMaladie(kernel, testsize, c):
    # model maladie cardiaque
    svmmodelMaladieCardiaque = SVMModelMaladieCardiaque(kernel, c)
    # Chargement des données
    maladie_data = import_dataMaladie("datasets/dataset_maladie.csv")
    # Séparation des données et de target
    X_maladie = maladie_data[["age", "thalach"]].to_numpy()
    y_maladie = maladie_data["target"].values
    # Diviser les données en ensembles d'entraînement et de test
    featuresMaladie_train, featuresMaladie_test, targetMaladie_train, targetMaladie_test = train_test_split(X_maladie, y_maladie, test_size=testsize, random_state=0)
    # faire le train
    svmmodelMaladieCardiaque.fit(featuresMaladie_train, targetMaladie_train)
    return featuresMaladie_train, featuresMaladie_test, svmmodelMaladieCardiaque, targetMaladie_train, targetMaladie_test


# fct pour faire le test du model des maladies cardiaques
def testModelSvmMaladie(kernel, testsize, c):
    featuresMaladie_train, featuresMaladie_test, svmmodelMaladieCardiaque, targetMaladie_train, targetMaladie_test = trainModelSVMMaladie(kernel, testsize, c)
    mail_pred = svmmodelMaladieCardiaque.predict(featuresMaladie_test)
    # Évaluer les performances du modèle
    accuracy = accuracy_score(targetMaladie_test, mail_pred)
    # Calcul du score F1
    f1 = f1_score(targetMaladie_test, mail_pred)
    # Affichage du score F1 et accuracy dans les labels
    accuracyLabel.configure(text=str(accuracy))
    scoreLabel.configure(text=str(f1))


# fct pour tracer la matrice de confusion de model des maladies cardiaques
canvas_metricsMaladie = None


def tracer_matriceConfusionMaladie(kernel, testSize, C):
    global canvas_metricsMaladie
    # Détruire le canvas s'il existe déjà
    if canvas_metricsMaladie:
        canvas_metricsMaladie.get_tk_widget().destroy()

    # Entraîner le modèle SVM et extraire l'objet de modèle SVM
    model_tuple = trainModelSVMMaladie(kernel, float(testSize), C)
    svmmodelMaladieCardiaque = model_tuple[2]
    cm = ConfusionMatrixDisplay.from_predictions(model_tuple[4], svmmodelMaladieCardiaque.predict(model_tuple[1]))
    # Obtenir la figure de la matrice de confusion
    fig, ax = plt.subplots(figsize=(4, 4))
    cm.plot(ax=ax)
    # Créer un widget Tkinter pour afficher la figure
    canvas_metricsMaladie = FigureCanvasTkAgg(fig, master=f_matriceC)
    canvas_metricsMaladie.draw()
    canvas_metricsMaladie.get_tk_widget().pack(side=tk.RIGHT)


# Fonction pour tracer le graphe avec les données d'entraînement du maladie
canvas_testMaladie = None


def tracer_grapheMaladie_test(kernel, testSize ,C):
    global canvas_testMaladie
    # Détruire le canvas s'il existe déjà
    if canvas_testMaladie:
        canvas_testMaladie.get_tk_widget().destroy()
    # Entraîner le modèle SVM et extraire l'objet de modèle SVM
    model_tuple = trainModelSVMMaladie(kernel, float(testSize), C)
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
    plt.xlabel('Âge')
    plt.ylabel('Thalach')
    # Créer le canvas pour afficher le graphe
    canvas_testMaladie = FigureCanvasTkAgg(fig, master=f_graphe)
    canvas_testMaladie.draw()
    canvas_testMaladie.get_tk_widget().pack(side=tk.LEFT)


# Fonction pour tracer le graphe avec les données d'entraînement du maladie
canvas_trainMaladie = None


def tracer_grapheMaladie_train(kernel, testSize, C):
    global canvas_trainMaladie
    # Détruire le canvas s'il existe déjà
    if canvas_trainMaladie:
        canvas_trainMaladie.get_tk_widget().destroy()
    # Entraîner le modèle SVM et extraire l'objet de modèle SVM
    model_tuple = trainModelSVMMaladie(kernel, float(testSize), C)
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
    plt.xlabel('Âge')
    plt.ylabel('Thalach')
    # Créer le canvas pour afficher le graphe
    canvas_trainMaladie = FigureCanvasTkAgg(fig, master=f_graphe)
    canvas_trainMaladie.draw()
    canvas_trainMaladie.get_tk_widget().pack(side=tk.LEFT)


# model penguin
# fct pour entrainer le model des penguins
def trainModelSVMPenguin(kernel, testsize, c):
    # model penguin
    # Chargement des données
    svmmodelPenguin = SVMModelPenguin(kernel, c)
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
    return featuresPenguin_train, featuresPenguin_test, svmmodelPenguin, targetPenguin_train, targetPenguin_test


# fct pour faire le test du model des maladies cardiaques
def testModelSvmPenguin(kernel, testsize, c):
    featuresPenguin_train, featuresPenguin_test, svmmodelPenguin, targetPenguin_train, targetPenguin_test = trainModelSVMPenguin(kernel, testsize, c)
    penguin_pred = svmmodelPenguin.predict(featuresPenguin_test)
    # Évaluer les performances du modèle
    accuracy = accuracy_score(targetPenguin_test, svmmodelPenguin.predict(featuresPenguin_test))
    # Calcul du score F1
    f1 = f1_score(targetPenguin_test, svmmodelPenguin.predict(featuresPenguin_test), pos_label=0)
    # Affichage du score F1 et accuracy
    accuracyLabel.configure(text=str(accuracy))
    scoreLabel.configure(text=str(f1))


# fct pour afficher la matrice de confusion de model penguin
canvas_metricsPenguin = None


def tracer_matriceConfusionPenguin(kernel, testSize, C):
    global canvas_metricsPenguin
    # Détruire le canvas s'il existe déjà
    if canvas_metricsPenguin:
        canvas_metricsPenguin.get_tk_widget().destroy()

    # Entraîner le modèle SVM et extraire l'objet de modèle SVM
    model_tuple = trainModelSVMPenguin(kernel, float(testSize), C)
    svmmodelPenguin = model_tuple[2]
    cm = ConfusionMatrixDisplay.from_predictions(model_tuple[4],svmmodelPenguin.predict(model_tuple[1]))
    # Obtenir la figure de la matrice de confusion
    fig, ax = plt.subplots(figsize=(4, 4))
    cm.plot(ax=ax)
    # Créer un widget Tkinter pour afficher la figure
    canvas_metricsPenguin = FigureCanvasTkAgg(fig, master=f_matriceC)
    canvas_metricsPenguin.draw()
    canvas_metricsPenguin.get_tk_widget().pack(side=tk.RIGHT)


# Fonction pour tracer le graphe avec les données d'entraînement du penguin
canvas_testPenguin = None


def tracer_graphePenguin_test(kernel, testSize ,C):
    global canvas_testPenguin
    # Détruire le canvas s'il existe déjà
    if canvas_testPenguin:
        canvas_testPenguin.get_tk_widget().destroy()
    # Entraîner le modèle SVM et extraire l'objet de modèle SVM
    model_tuple = trainModelSVMPenguin(kernel, float(testSize), C)
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
    plt.xlabel('bill_length_mm')
    plt.ylabel('bill_depth_mm')
    # Créer le canvas pour afficher le graphe
    canvas_testPenguin = FigureCanvasTkAgg(fig, master=f_graphe)
    canvas_testPenguin.draw()
    canvas_testPenguin.get_tk_widget().pack(side=tk.LEFT)


# Fonction pour tracer le graphe avec les données d'entraînement du penguin
canvas_trainPenguin = None


def tracer_graphePenguin_train(kernel, testSize, C):
    global canvas_trainPenguin
    # Détruire le canvas s'il existe déjà
    if canvas_trainPenguin:
        canvas_trainPenguin.get_tk_widget().destroy()
    # Entraîner le modèle SVM et extraire l'objet de modèle SVM
    model_tuple = trainModelSVMPenguin(kernel, float(testSize), C)
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
    plt.xlabel('bill_length_mm')
    plt.ylabel('bill_depth_mm')
    # Créer le canvas pour afficher le graphe
    canvas_trainPenguin = FigureCanvasTkAgg(fig, master=f_graphe)
    canvas_trainPenguin.draw()
    canvas_trainPenguin.get_tk_widget().pack(side=tk.LEFT)


# creation de la fenetre de splash screen
splash_root = tk.Tk()
# Centrer la fenêtre au milieu de l'écran
screen_width = splash_root.winfo_screenwidth()
screen_height = splash_root.winfo_screenheight()
x = int((screen_width / 2) - (1600 / 2))
y = int((screen_height / 2) - (950 / 2))
splash_root.geometry(f"1600x950+{x}+{y}")
splash_root.config(bg=BG_COLOR)

# Rendre la fenêtre non-redimensionnable
splash_root.resizable(width=False, height=False)
# Créez un canvas pour ajouter une image
splash_canvas = Canvas(splash_root, width=1600, height=950, bg=BG_COLOR, highlightthickness=0)
splash_canvas.pack()

# Charger l'image et la convertir pour Tkinter
image = Image.open("imgs/splashScreen.png")
largeur = 700
hauteur = 500
image_redimensionnee = image.resize((largeur, hauteur))
photo = ImageTk.PhotoImage(image_redimensionnee)
splash_canvas.create_image(650, 375, anchor=CENTER, image=photo)
# Définissez le temps d'affichage du splash screen en millisecondes
splash_time = 3000
# Fermez la fenêtre du splash screen après le temps d'affichage
splash_root.after(splash_time, splash_root.destroy)


# Création de la principale fenetre
appSVM = tk.Tk()  # nouvelle instance de Tk
appSVM.title("Interface Home Machine")
# logo_img = tk.PhotoImage(file="imgs/16x16.png")
# Définition du logo pour la fenêtre
# splash_root.iconphoto(True, logo_img)

# Affichage de votre logo dans la barre de titre
# splash_root.tk.call('wm', 'iconphoto', splash_root._w, logo_img)
# Centrer la fenêtre au milieu de l'écran
screen_width = appSVM.winfo_screenwidth()
screen_height = appSVM.winfo_screenheight()
x = int((screen_width / 2) - (1600 / 2))
y = int((screen_height / 2) - (950 / 2))
appSVM.geometry(f"1600x950+{x}+{y}")
appSVM.config(bg=BG_COLOR)
# Rendre la fenêtre non-redimensionnable
appSVM.resizable(width=False, height=False)

# creation des frames
f_parametre = tk.LabelFrame(appSVM, bd=2, text="", bg=bg_color_frame, relief="groove", width=200, height=100)
f_model = tk.LabelFrame(appSVM, bd=2, text="", bg=bg_color_frame, relief="groove", width=200, height=200)
f3_btn = tk.LabelFrame(f_model, bd=2, text="", bg=bg_color_frame, highlightthickness=0)
f4_grp = tk.LabelFrame(f_model, bd=2, text="", bg=bg_color_frame, highlightthickness=0)
f_matriceC = tk.LabelFrame(f4_grp, bd=2, text="", bg=bg_color_frame, highlightthickness=0)
f_graphe = tk.LabelFrame(f4_grp, bd=2, text="", bg=bg_color_frame, highlightthickness=0)
f_parametre.pack(side=tk.LEFT, padx=20, pady=20)
f_model.pack(side=tk.RIGHT, padx=20, pady=20)
f3_btn.pack(side=tk.TOP)
f4_grp.pack(side=tk.BOTTOM)
f_matriceC.pack(side=tk.RIGHT)
f_graphe.pack(side=tk.LEFT)


# creations des composants de frame des parametres
datalabel = tk.Label(f_parametre, text="choisir le dataset : ", fg="#d9d9d9", bg=bg_color_frame, font=("Helvetica", 13))
datalabel.pack(padx=50, pady=10)

# Créer une liste déroulante
datasets = ["Dataset Spam Email", "Dataset Maladies Cardiaques", "Dataset Penguin"]
combo_box = ttk.Combobox(f_parametre, values=datasets, font=("Helvetica", 12), width=35)

# Configuration des couleurs de fond et de texte pour la ComboBox
combo_box.configure(background=LABEL_BG_COLOR, foreground="black")
combo_box.pack(padx=50, pady=5)

description = tk.Label(f_parametre, text="description : ", fg="#d9d9d9", bg=bg_color_frame, font=("Helvetica", 13))
description.pack(padx=20, pady=5)

# Création du Label pour afficher la valeur sélectionnée
descriptiontxt = tk.Label(f_parametre, text=" ", fg="#d9d9d9", bg=bg_color_frame, font=("Helvetica", 11), wraplength=360, justify="left")
descriptiontxt.pack(padx=5, pady=5)

# Configuration de la ComboBox pour appeler la fonction update_label lors de la sélection d'une option
combo_box.bind("<<ComboboxSelected>>", update_label)

# Création d'un style personnalisé pour Entry
style = ttk.Style()
style.theme_use("clam")
# Configuration de la bordure et du relief pour l'Entry
style.configure("Custom.TEntry", fieldbackground=ENTRY_BG_COLOR, foreground=FG_COLOR)

tstsize = tk.Label(f_parametre, text="test size: ", fg="#d9d9d9", bg=bg_color_frame, font=("Helvetica", 13))
tstsize.pack(padx=50, pady=10)

testSize = ttk.Entry(f_parametre, style="Custom.TEntry", width=40, font=("Helvetica", 11))
testSize.pack(pady=8,ipady=5)

parac = tk.Label(f_parametre, text="parametre C: ", fg="#d9d9d9", bg=bg_color_frame, font=("Helvetica", 13))
parac.pack(padx=50, pady=10)

paramC = ttk.Entry(f_parametre, style="Custom.TEntry", width=40, font=("Helvetica", 11))
paramC.pack(pady=8,ipady=5)

paramk = tk.Label(f_parametre, text="parametre Kernel: ", fg="#d9d9d9", bg=bg_color_frame, font=("Helvetica", 13))
paramk.pack(padx=50, pady=10)

paramKernel = ttk.Entry(f_parametre, style="Custom.TEntry", width=40, font=("Helvetica", 11))
paramKernel.pack(pady=8, ipady=5)

# Charger l'image et la convertir pour Tkinter
# icon_training = PhotoImage(file="imgs/training_80px.gif")

# creation de boutton pour entrainer le modele
btnTraining = tk.Button(f3_btn, height=4, width=26, text="Training", font=('Helvetica', 15), fg='#FFFFFF', bg='#9AC8EB', bd=0, command=fitModel, state="disabled")
btnTraining.pack(padx=20, pady=5, side=tk.LEFT)

# creation de boutton pour tester le modele
btnTesting = tk.Button(f3_btn, height=4, width=26, text="Testing", font=('Helvetica', 15), fg='#FFFFFF', bg='#9AC8EB', bd=0, command=tracerGraphe, state="disabled")
btnTesting.pack(padx=20, pady=5, side=tk.RIGHT)

accuracyLabel = tk.Label(f_model, text="", fg="#d9d9d9", bg=bg_color_frame, font=("Helvetica", 12))
accuracyLabel.pack(padx=50, pady=10)
scoreLabel = tk.Label(f_model, text="", fg="#d9d9d9", bg=bg_color_frame, font=("Helvetica", 12))
scoreLabel.pack(padx=50, pady=10)
# liason des evenement avec les composants pour checker
testSize.bind("<KeyRelease>", lambda event: check_fields())
paramC.bind("<KeyRelease>", lambda event: check_fields())
paramKernel.bind("<KeyRelease>", lambda event: check_fields())
combo_box.bind("<<ComboboxSelected>>", update_label)

appSVM.mainloop()
