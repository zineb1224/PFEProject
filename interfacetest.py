# objet : fenêtre avec deux frames
#import les biblio
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split

from models.SVMModelSpam import SVMModelSpam, import_data
from models.SVMModelMaladieCardiaque import SVMModelMaladieCardiaque , import_data


#fct pour recuperer la valeur de size du test
def getValeurTestSize() :
    val = testSize.get()
    return val
#fct pour recuperer la valeur de parametre kernel
def getValeurParamKernel() :
    val = paramKernel.get()
    return val
#fct pour recuperer la valeur de parametre C
def getValeurParamC() :
    val = paramC.get()
    return val

#fct pour entrainer le modele de svm et afficher le graphe d'entrainement
def fitModel() :
    sizetest = getValeurTestSize()
    kernel = getValeurParamKernel()
    C = getValeurParamC()
    selected_value = combo_box.get()
    if selected_value == "Dataset Spam Email":
        trainModelSvmSpam(kernel, float(sizetest), float(C))
        #tracer_grapheSpam_train(kernel, sizetest)
    elif selected_value == "Dataset Maladies Cardiaques":
        trainModelSVMMaladie(kernel, float(sizetest), float(C))

#fct pour tester le model et afficher le graphe du test
def tracerGraphe() :
    sizetest = getValeurTestSize()
    kernel = getValeurParamKernel()
    C = getValeurParamC()
    selected_value = combo_box.get()
    if selected_value == "Dataset Spam Email":
        testModelSvmSpam(kernel, float(sizetest), float(C))
        #tracer_grapheSpam_test(kernel, float(sizetest))
    elif selected_value == "Dataset Maladies Cardiaques":
        testModelSvmMaladie(kernel, float(sizetest), float(C))

#fct pour entrainer le model du Spam
def trainModelSvmSpam(kernel, testsize ,c) :
    # model spam email
    svmmodelSpam = SVMModelSpam(kernel, c)

    # Chargement des données
    emails_data = import_data('datasets/labeled_emails.csv')

    # Séparation des données en ensembles d'entraînement et de test
    emails = emails_data['email']
    labels = np.where(emails_data['label'] == 'spam', 1, 0)  # Encoder les étiquettes en 0 et 1

    # Vectoriser les courriers électroniques en utilisant la transformation TF-IDF
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(emails)

    # Diviser les données en ensembles d'entraînement et de test
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=testsize, random_state=42)

    # Prédiction sur l'ensemble de test
    svmmodelSpam.fit(features_train, labels_train)
    return  features_train, labels_train, svmmodelSpam, labels_test ,features_test

#fct pour faire le test du model de Spam
def testModelSvmSpam(kernel, testsize, c) :
    features_train,labelstrain, svmmodelSpam, labels_test,features_test = trainModelSvmSpam(kernel, testsize, c)
    mail_pred = svmmodelSpam.predict(features_test)
    # Évaluer les performances du modèle
    accuracy = accuracy_score(labels_test, mail_pred)
    precision = precision_score(labels_test, mail_pred)
    accuracyLabel.configure(text=str(accuracy))
    print("Accuracy:", accuracy)
    print("Precision:", precision)


#fct pour entrainer le model des maladies cardiaques
def trainModelSVMMaladie(kernel, testsize, c):
    # model maladie cardiaque
    svmmodelMaladieCardiaque = SVMModelMaladieCardiaque(kernel , c)
    # Chargement des données
    maladie_data = import_data("datasets/dataset_maladie.csv")
    # Séparation des données et de target
    y_maladie = maladie_data['target']
    X_maladie = maladie_data.drop(['target'], axis=1)

    # Diviser les données en ensembles d'entraînement et de test
    featuresMaladie_train, featuresMaladie_test, targetMaladie_train, targetMaladie_test = train_test_split(X_maladie, y_maladie, test_size=testsize, random_state=0)
    # Prédiction sur l'ensemble de test
    svmmodelMaladieCardiaque.fit(featuresMaladie_train, targetMaladie_train)

    return featuresMaladie_train, featuresMaladie_test, svmmodelMaladieCardiaque, featuresMaladie_test, targetMaladie_test

#fct pour faire le test du model des maladies cardiaques
def testModelSvmMaladie(kernel, testsize, c) :
    featuresMaladie_train, featuresMaladie_test , svmmodelMaladieCardiaque, featuresMaladie_test, targetMaladie_test = trainModelSVMMaladie(kernel, testsize, c)
    mail_pred = svmmodelMaladieCardiaque.predict(featuresMaladie_test)
    # Évaluer les performances du modèle
    accuracy = accuracy_score(targetMaladie_test, mail_pred)
    precision = precision_score(targetMaladie_test, mail_pred)
    accuracyLabel.configure(text=str(accuracy))
    print("Accuracy:", accuracy)
    print("Precision:", precision)

#fct pour verifier que les inputs sont bien remplis et rendre le boutton de train et de test normal
def check_fields():
    if len(testSize.get()) > 0 and len(paramC.get()) > 0 and len(paramKernel.get()) > 0 and len(combo_box.get()) > 0:
        btnTraining.config(state="normal")
        btnTesting.config(state="normal")
    else:
        btnTraining.config(state="disabled")
        btnTesting.config(state="disabled")

# Fonction pour tracer le graphe avec les données d'entraînement du spam
canvas_train = None
def tracer_grapheSpam_train(kernel, testSize):
    global canvas_train
    # Détruire le canvas s'il existe déjà
    if canvas_train:
        canvas_train.get_tk_widget().destroy()
    # Entraîner le modèle SVM et extraire l'objet de modèle SVM
    model_tuple = trainModelSvmSpam(kernel, float(testSize))
    model = model_tuple[2]

    # Créer la grille de points pour l'affichage du graphe
    xx = np.linspace(-1, 5)
    yy = np.linspace(-1, 5)
    xx, yy = np.meshgrid(xx, yy)
    xy = np.vstack([xx.ravel(), yy.ravel()]).T

    # Vectoriser les courriers électroniques en utilisant la transformation TF-IDF
    vectorizer = TfidfVectorizer()
    features_train = vectorizer.fit_transform(model_tuple[0]).toarray()

    # Calculer les prédictions du modèle SVM sur la grille de points
    Z = model.decision_function(vectorizer.transform(xy)).reshape(xx.shape)

    # Créer le graphe
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    # Afficher les données de classification binaire
    ax.scatter(features_train[:, 0], features_train[:, 1], c=model_tuple[1], cmap=plt.cm.Paired, edgecolors='k')
    # Afficher la frontière de décision
    ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('SVM Spam - Training Data')

    # Créer le canvas pour afficher le graphe
    canvas_train = FigureCanvasTkAgg(fig, master=f2)
    canvas_train.draw()
    canvas_train.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


#fct pour tracer le graphe avec les donnees de test du spam
canvas_test = None
def tracer_grapheSpam_test(kernel , testSize):
    global canvas_test
    # Détruire le canvas s'il existe déjà
    if canvas_test:
        canvas_test.get_tk_widget().destroy()
    # Entraîner le modèle SVM et extraire l'objet de modèle SVM
    model_tuple = trainModelSvmSpam(kernel, float(testSize))
    model = model_tuple[3]

    # Vectoriser les courriers électroniques de test en utilisant le même transformateur de caractéristiques
    vectorizer = TfidfVectorizer()
    features_train = vectorizer.fit_transform(model_tuple[0]).toarray()
    emails_test = vectorizer.transform(model_tuple[1])

    # Créer la grille de points pour l'affichage du graphe
    xx = np.linspace(-1, 5)
    yy = np.linspace(-1, 5)
    xx, yy = np.meshgrid(xx, yy)
    xy = np.vstack([xx.ravel(), yy.ravel()]).T

    # Calculer les prédictions du modèle SVM sur la grille de points
    Z = model.decision_function(xy).reshape(xx.shape)
    # Créer le graphe
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    # Afficher les données de classification binaire
    ax.scatter(emails_test[:, 0], emails_test[:, 1], c=model_tuple[3].predict(emails_test), cmap='coolwarm')
    # Afficher les frontières de décision du modèle SVM
    ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('SVM Classification')
    # Afficher le graphe dans la frame
    canvas_test = FigureCanvasTkAgg(fig, master=f2)
    canvas_test.get_tk_widget().pack()


# Fonction appelée lors de la sélection d'une option dans la ComboBox
def update_label(event):
    selected_value = combo_box.get()
    if selected_value=="Dataset Spam Email" :
        descriptiontxt.configure(text="Le fichier csv contient 5172 lignes, chaque ligne"
                                      " pour chaque e-mail. Il y a 3002 colonnes. La première colonne indique le nom de l'e-mail."
                                      " Le nom a été défini avec des chiffres et non avec le nom des destinataires pour protéger la confidentialité. "
                                      "La dernière colonne contient les libellés de prédiction : 1 pour spam, 0 pour non spam."
                                      " Les 3000 colonnes restantes sont les 3000 mots les plus courants dans tous les e-mails,"
                                      " après exclusion des caractères/mots non alphabétiques. Pour chaque ligne, "
                                      "le nombre de chaque mot (colonne) dans cet e-mail (ligne) est stocké dans les cellules respectives. "
                                       )
    elif selected_value=="Dataset Maladies Cardiaques" :
        descriptiontxt.configure(text="description de maladie cardiaque ")
    check_fields()

#creation de splash screen
splash_root = tk.Tk()
# Centrer la fenêtre au milieu de l'écran
screen_width = splash_root.winfo_screenwidth()
screen_height = splash_root.winfo_screenheight()
x = int((screen_width / 2) - (1300 / 2))
y = int((screen_height / 2) - (750 / 2))
splash_root.geometry(f"1300x750+{x}+{y}")

# Rendre la fenêtre non-redimensionnable
splash_root.resizable(width=False, height=False)

# Créez un canevas pour ajouter une image
splash_canvas = Canvas(splash_root, width=1300, height=750)
splash_canvas.pack()

# Charger l'image et la convertir pour Tkinter
image = Image.open("imgs/AI.gif")
photo = ImageTk.PhotoImage(image)

splash_canvas.create_image(650, 375, anchor=CENTER, image=photo)

# Ajoutez un message de chargement
splash_label = Label(splash_root, text="Chargement en cours...")
splash_label.pack()

# Définissez le temps d'affichage du splash screen en millisecondes
splash_time = 3000

# Fermez la fenêtre du splash screen après le temps d'affichage
splash_root.after(splash_time, splash_root.destroy)

# Définition de la palette de couleurs
BG_COLOR = "#1c1c1c"
FG_COLOR = "#d9d9d9"
LABEL_BG_COLOR = "#3a3a3a"
ENTRY_BG_COLOR = "#3a3a3a"
bg_color_frame = "#3a3a3a"

# Création d'un objet "fenêtre"
appSVM = tk.Tk()  # nouvelle instance de Tk
appSVM.title("Interface Home Machine")

# Centrer la fenêtre au milieu de l'écran
screen_width = appSVM.winfo_screenwidth()
screen_height = appSVM.winfo_screenheight()
x = int((screen_width / 2) - (1300 / 2))
y = int((screen_height / 2) - (750 / 2))
appSVM.geometry(f"1300x750+{x}+{y}")
appSVM.config(bg=BG_COLOR)

# Rendre la fenêtre non-redimensionnable
appSVM.resizable(width=False, height=False)

f1 = tk.LabelFrame(appSVM, bd=2, text="", bg=bg_color_frame, relief="groove")
f2 = tk.LabelFrame(appSVM, bd=2, text="", bg=bg_color_frame, relief="groove")
f1.pack(side=tk.LEFT, padx=20, pady=20)
f2.pack(side=tk.RIGHT, padx=20, pady=20)

datalabel = tk.Label(f1, text="choisir le dataset : ", fg="#d9d9d9", bg=bg_color_frame, font=("Helvetica", 13))
datalabel.pack(padx=50, pady=10)

# Créer une liste déroulante
datasets = ["Dataset Spam Email" , "Dataset Maladies Cardiaques" ]
combo_box = ttk.Combobox(f1, values=datasets, font=("Helvetica", 12), width=35)

# Configuration des couleurs de fond et de texte pour la ComboBox
combo_box.configure(background=LABEL_BG_COLOR, foreground="black")
combo_box.pack(padx=50, pady=5)

description = tk.Label(f1, text="description : ", fg="#d9d9d9", bg=bg_color_frame, font=("Helvetica", 13))
description.pack(padx=20, pady=5)

# Création du Label pour afficher la valeur sélectionnée
descriptiontxt = tk.Label(f1, text=" ", fg="#d9d9d9", bg=bg_color_frame, font=("Helvetica", 11), wraplength=360, justify="left")
descriptiontxt.pack(padx=5, pady=5)

# Configuration de la ComboBox pour appeler la fonction update_label lors de la sélection d'une option
combo_box.bind("<<ComboboxSelected>>", update_label)

# Création d'un style personnalisé pour Entry
style = ttk.Style()
style.theme_use("clam")
# Configuration de la bordure et du relief pour l'Entry
style.configure("Custom.TEntry", fieldbackground=ENTRY_BG_COLOR, foreground=FG_COLOR)

tstsize = tk.Label(f1, text="test size: ", fg="#d9d9d9", bg=bg_color_frame, font=("Helvetica", 13))
tstsize.pack(padx=50, pady=10)

testSize = ttk.Entry(f1, style="Custom.TEntry", width=40, font=("Helvetica", 11))
testSize.pack(pady=8,ipady=5)

parac = tk.Label(f1, text="parametre C: ", fg="#d9d9d9", bg=bg_color_frame, font=("Helvetica", 13))
parac.pack(padx=50, pady=10)

paramC = ttk.Entry(f1, style="Custom.TEntry", width=40, font=("Helvetica", 11))
paramC.pack(pady=8,ipady=5)

paramk = tk.Label(f1, text="parametre Kernel: ", fg="#d9d9d9", bg=bg_color_frame, font=("Helvetica", 13))
paramk.pack(padx=50, pady=10)

paramKernel = ttk.Entry(f1, style="Custom.TEntry", width=40,font=("Helvetica", 11))
paramKernel.pack(pady=8,ipady=5)

# Charger l'image et la convertir pour Tkinter
#icon_training = PhotoImage(file="imgs/training_80px.gif")

#creation de boutton pour entrainer le modele
btnTraining = tk.Button(f2 , height=4, width=26, text="Training", font=('Helvetica', 15), fg='#FFFFFF', bg='#9AC8EB', bd=0, command=fitModel, state="disabled")
btnTraining.pack(padx=20,pady=5)

# Création d'un cadre dans la fenêtre Tkinter pour y afficher le graphe
frame_graphe = tk.LabelFrame(f2, bd=0, bg="#f3f3f3", relief="groove")
frame_graphe.pack()

#creation de boutton pour tester le modele
btnTesting = tk.Button(f2 , height=4, width=26, text="Testing", font=('Helvetica', 15), fg='#FFFFFF', bg='#9AC8EB', bd=0, command=tracerGraphe, state="disabled")
btnTesting.pack(padx=20,pady=5)

accuracyLabel = tk.Label(f2, text="", fg="#d9d9d9", bg=bg_color_frame, font=("Helvetica", 12))
accuracyLabel.pack(padx=50, pady=10)

testSize.bind("<KeyRelease>", lambda event: check_fields())
paramC.bind("<KeyRelease>", lambda event: check_fields())
paramKernel.bind("<KeyRelease>", lambda event: check_fields())
combo_box.bind("<<ComboboxSelected>>", update_label)

appSVM.mainloop()