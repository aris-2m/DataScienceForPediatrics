import numpy
import matplotlib.pyplot as plt

import random
import math
import time


def SCA(fonction_objectif, model_, lb, ub, dim_individu, nombreIndividus, Max_iter):

    # Le_best : Le meilleur des Xi
    Le_best = numpy.zeros(dim_individu) #Initialisation
    best_score = float("inf")

    #Les individus doivent etre dans une marge donnée déterminée par lb et ub
    if not isinstance(lb, list):
        lb = [lb] * dim_individu
    if not isinstance(ub, list):
        ub = [ub] * dim_individu

    # Initialisation de la matrice population
    Population = numpy.zeros((nombreIndividus, dim_individu))
    
    #Generation de la première population
    for i in range(dim_individu):
        #On s'assure de garder les individus dans les bornes précisées
        Population[:, i] = (numpy.random.uniform(0, 1, nombreIndividus) * (ub[i] - lb[i]) + lb[i])

    #Population[i, :] sera donc un individu

    #Initialisation d'un vecteur qui contiendra toutes les meilleures fitness à chaque itération:
    Tous_les_best_score = numpy.zeros(Max_iter)
    
    #Initialisation d'une matrice qui contiendra tous les best à chaque itération:
    #Tous_les_best_pts = numpy.zeros((nombreIndividus, dim_individu, Max_iter))

    #s = solution()

    # Loop counter
    print('SCA optimise  "' + fonction_objectif.__name__ + '"')

    timerStart = time.time()

    # Main loop
    for l in range(0, Max_iter):

        for i in range(0, nombreIndividus):

            # Calibrage des composantes de la population entre les marges
            for j in range(dim_individu):
                Population[i, j] = numpy.clip(Population[i, j], lb[j], ub[j])

            # Calcul de la fitness de chaque individu
            fitness = fonction_objectif(Population[i, :], model_)

            for chaque in fitness:
                # Mise a jour du best - Ici on veut minimiser
                if chaque < best_score:
                    best_score = chaque
                    Le_best = Population[i, :].copy()

        # On fixe a à 2
        a = 2
        Max_iteration = Max_iter

        #Mise à jour de r1
        r1 = a - l * ((a) / Max_iteration)  # r1 decreases linearly from a to 0

        # Mise à mise de la populatiom
        for i in range(0, nombreIndividus):
            for j in range(0, dim_individu):

                # Mise à jour de r2, r3, and r4
                r2 = (2 * numpy.pi) * random.random()
                r3 = 2 * random.random()
                r4 = random.random()

                #Choix de la fonction avec cos ou avec sin
                if r4 < (0.5):
                    #Population[i, j] eest chaque composante du vecteur Population[i, :]
                    Population[i, j] = Population[i, j] + (r1 * numpy.sin(r2) * abs(r3 * Le_best[j] - Population[i, j]))
                else:
                    #Population[i, j] eest chaque composante du vecteur Population[i, :]
                    Population[i, j] = Population[i, j] + (r1 * numpy.cos(r2) * abs(r3 * Le_best[j] - Population[i, j]))

        Tous_les_best_score[l] = best_score
        #Tous_les_best_pts[l] = Le_best

        if l % 1 == 0:
            print(
                ["A l'iteration " + str(l) + " la meilleure fitness est: " + str(best_score)]
            )
            print(
                ["Le meilleur associé est: " + str(Le_best)]
            )

            print("\n")

    timerEnd = time.time()
    temps_exec=timerEnd - timerStart

    retour=[Le_best,temps_exec,Tous_les_best_score]

    return retour