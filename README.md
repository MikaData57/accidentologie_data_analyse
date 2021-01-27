# Analyse de l'accidentologie sur les routes Françaises

Projet personnel à caractère social cloturant la **formation Data Analyst OpenClassrooms** 2020 - 2021 *(projet 8)*.
Description et nettoyage des données, analyses statistiques, classification non supervisée, machine learning et modèles prédictifs des timeseries et sinistralité.

## Contexte
En France, le nombre d’accidents de la route n’a cessé de baisser depuis les années 1980. Mise en place d’infrastructures plus sécuritaires, évolution dans le domaine de la sécurité active et passive, évolution du code de la route … autant de mesures qui ont été misent en place et visent à réduire significativement le nombre de tués sur le réseau routier.

Mais cette réussite apparente ne cache-t-elle pas une réalité plus contrastée ? N’existe-t-il pas une disparité nationale de la répartition des accidents ? Les véhicules impliqués sont-ils statistiquement toujours les mêmes ? Quels sont les facteurs aggravants de ces accidents ? 

Autant de questions auxquelles nous allons tenter de répondre avant de pouvoir définir s’il est possible de prédire ces accidents, d’anticiper leur évolution voir même de prévoir leur gravité en fonction de divers facteurs connus.

## Les fichiers Python disponibles
- **P8_1_Preparation_Donnees.ipynb** : Nettoyage et description des données Open Data du fichier des accidents corporels. Création d'une base de données relationnelle.
- **P8_2_Analyses_Correlations.ipynb** : Preprocessing *(encodage, standardisation ...)*, analyse des corrélations, classification non supervisée par algorithme Kmeans++.
- **P8_3_prediction_evolution_accidents.ipynb** : Analyse de timeseries, modélisation FB Prophet, XGBoost et SARIMA. Fonction de forcast pour les prédictions.
- **P8_4_V2_modèle_sinistrabilité.ipynb** : Modélisation pour la prédiction de la gravité *(risque d'hospitalisation ou de mort)* en fonction des features connues avant l'accident.
- **preprocessing.py** : Fonction complète de preprocessing pour les données utilisées.
- **features_glossary.py** : Dictionnaire des variables utilisé pour mapping des features.
