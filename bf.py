def tracer_grapheIris_test(selected_feature, testSize, C, gamma=0):
    global canvas_testIris

    # Détruire le canvas s'il existe déjà
    if canvas_testIris:
        canvas_testIris.get_tk_widget().destroy()

    if getValeurParamKernel() == "rbf":
        gamma = float(getValeurGamma())

    # Entraîner le modèle SVM et extraire l'objet de modèle SVM
    model_tuple = trainModelSVMIris(selected_feature, testSize, C, gamma)
    svmmodelIris = model_tuple[2]

    # Création du graphe avec la marge et les vecteurs de support
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)

    # Afficher les données en fonction de la caractéristique sélectionnée
    feature_index_x = getFeatureIndex(selected_feature[0])
    feature_index_y = getFeatureIndex(selected_feature[1])
    ax.plot(model_tuple[1][:, feature_index_x][model_tuple[4] == 0], model_tuple[1][:, feature_index_y][model_tuple[4] == 0], "yo", label="0: non malade")
    ax.plot(model_tuple[1][:, feature_index_x][model_tuple[4] == 1], model_tuple[1][:, feature_index_y][model_tuple[4] == 1], "bo", label="1: malade")

    # Limites du cadre
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    support_vectors_ = svmmodelIris.support_vectors_()

    # Marquer les vecteurs de support d'une croix
    ax.scatter(support_vectors_[:, feature_index_x], support_vectors_[:, feature_index_y], linewidth=1, facecolors='#FFAAAA', s=180)

    # Grille de points sur lesquels appliquer le modèle
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    # Prédire pour les points de la grille
    Z = svmmodelIris.predict(xy)
    Z = Z.reshape(XX.shape)

    # Afficher la frontière de décision et la marge
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    # Tracé du graphe avec les vecteurs de support et les marges
    ax.scatter(support_vectors_[:, feature_index_x], support_vectors_[:, feature_index_y], s=100, facecolors='none', edgecolors='k')

    ax.set_xlabel(selected_feature[0])
    ax.set_ylabel(selected_feature[1])

    ax.legend()

    # Créer le canvas pour afficher le graphe
    canvas_testIris = FigureCanvasTkAgg(fig, master=f_graphetest)
    canvas_testIris.draw()
    canvas_testIris.get_tk_widget().pack(side=tk.LEFT)
