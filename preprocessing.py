import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import features_glossary as fct
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


def single_pt_haversine(lat, lng, degrees=True):  
    r = 6371 # Rayon de la terre (km).
    # Convertir les degrés décimaux en radians
    if degrees:
        lat, lng = map(radians, [lat, lng])
    # Formule 'Single-point' Haversine
    a = sin(lat/2)**2 + cos(lat) * sin(lng/2)**2
    d = 2 * r * asin(sqrt(a)) 
    return d

def accPreprocessing(df):
    #Calcul des points de coordonnées
    df['coords'] = [single_pt_haversine(lat, lon) for lat, lon in zip(df.latitude, df.longitude)]
    df = df.drop(["latitude","longitude"], axis=1)
    
    #Recatégorisation des catégories de véhicules
    df["gpe_catv"] = np.where(df["catv"].isin(["Autobus","Autocar","Référence inutilisée depuis 2006 (tramway)","Référence inutilisée depuis 2006 (transport en commun)","Tramway"]),"Bus et tramway",
                                            np.where(df["catv"].isin(["Scooter < 50 cm3","Cyclomoteur <50cm3","Quad léger <= 50 cm3 (Quadricycle à moteur non carrossé)"]),"Cyclomoteur et Scooter < 50 cm3",
                                                     np.where(df["catv"].isin(["Motocyclette > 125 cm3","Motocyclette > 50 cm3 et <= 125 cm3","Quad lourd > 50 cm3 (Quadricycle à moteur non carrossé)","Référence inutilisée depuis 2006 (motocyclette)","Référence inutilisée depuis 2006 (scooter immatriculé)","Référence inutilisée depuis 2006 (side-car)","Scooter > 125 cm3","Scooter > 50 cm3 et <= 125 cm3"]),"Motocyclette et Scooter > 50 cm3",
                                                             np.where(df["catv"].isin(["PL > 3,5T + remorque","PL seul > 7,5T","PL seul 3,5T <PTCA <= 7,5T","Tracteur routier + semi-remorque","Tracteur routier seul"]),"Poid Lourd > 3,5T",
                                                                     np.where(df["catv"].isin(["Référence inutilisée depuis 2006 (VL + caravane)","Référence inutilisée depuis 2006 (VL + remorque)","VL seul","Voiturette (Quadricycle à moteur carrossé) (anciennement voiturette ou tricycle à moteur)"]),"Véhicule léger",
                                                                             np.where(df["catv"].isin(["Référence inutilisée depuis 2006 (VU (10) + caravane)","Référence inutilisée depuis 2006 (VU (10) + remorque)","VU seul 1,5T <= PTAC <= 3,5T avec ou sans remorque (anciennement VU seul 1,5T <= PTAC <= 3,5T)"]),"Véhicule utilitaire",
                                                                                     np.where(df["catv"].isin(["Bicyclette"]),"Bicyclette","Autres")))))))
    df = df.drop(["catv"],axis=1)
    print("Calcul des coordonnées ........... OK")
    
    #Séparation par type de variables
    y = df[["sinitrabilite"]]
    df = df.drop("sinitrabilite", axis=1)
    numerical_features = df.select_dtypes(include=["int64","float64"])
    categorical_features = df.select_dtypes(exclude=["int64","float64"])
    
    #Création du dictionnaire de features inversé pour mapping
    reverse_features_indices = {}
    for key, values in fct.features_indices.items():
        sousdict = {}
        for index, value in values.items():
            sousdict[value] = index
            reverse_features_indices[key] = sousdict
    
    #Affectation du mapping inversé pour catégoriser les categorical features
    errorlist = []
    for item in categorical_features.columns :
        try:
            categorical_features[item] = categorical_features[item].map(reverse_features_indices[item])
        except:
            errorlist.append(item)
            continue
    print("Catégorisation des features ........... OK")
    
    # On remplace les valeurs nulles
    numerical_features = numerical_features.fillna(numerical_features.mean())
    categorical_features = categorical_features.fillna(categorical_features.mode().iloc[0])
    
    #Normalisation min/max des variables numériques
    scaler = MinMaxScaler()
    numerical_sc = scaler.fit_transform(numerical_features)
    numerical_sc = pd.DataFrame(numerical_sc, index=numerical_features.index, columns=numerical_features.columns)
    print("Normalisation MinMax ........... OK")
    
    #Encodage des variables catégorielles
    encoder = OneHotEncoder()
    categorical_encode = pd.DataFrame(encoder.fit_transform(categorical_features).toarray(), 
                                      index=categorical_features.index, 
                                      columns=encoder.get_feature_names(list(categorical_features.columns)))
    print("OneHotEncoder ........... OK")
    
    #Création du DataFrame final
    df = pd.concat([numerical_sc, categorical_encode, y], axis=1)
    print("Preprocessing terminé.")
    return df, errorlist
