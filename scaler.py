import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from ydata_profiling import ProfileReport
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc,  accuracy_score, f1_score, roc_auc_score, confusion_matrix,roc_auc_score
from sklearn.metrics import f1_score,make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import gdown
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV




# Mode interactif ON
plt.ion()

###1. Chargement et aperçu des données


output = "Financial_inclusion_dataset.csv"

# Télécharger le fichier depuis Google Drive
###gdown.download(url, output, quiet=False)

# Charger le fichier CSV dans un DataFrame
df = pd.read_csv(output)

print(df.head())



#Affichez des informations générales sur l'ensemble de données
print(f"Head :")
print(df.head())
print(f"describe :")
print(df.describe())
print(f"Info :")
print(df.info())
print(f"Shape :")
print(df.shape)

print(df.columns)


#Créez un rapport de profilage pandas pour obtenir des informations sur l'ensemble de données
profile = ProfileReport(df, title="Pandas Profiling Report")
profile.to_notebook_iframe()




#Le rapport profiling nous a indiqué  
#- Pas des données manquantes
#- Pas de données en doubles
#- La variable country est fortement corrélée avec la variable year
#- La variable gender_of_respondent est fortement corrélée avec relationship_with_head


## nombre de données NaN
print(f"valeur null : {df.isna().sum()}")

## nombre des données en double
print(f"Doublons : {df.duplicated().sum()}")

# On exclut 'uniqueid' donnée non utile pour l'entrainement
df = df.drop(columns=['uniqueid'])


# Sélectionner des variables
X = df.drop(["bank_account"], axis=1)
y = df["bank_account"]


# Coder les variables categorielles
df_encoded = pd.get_dummies(X)
df_encoded.head()

## scaler les donnes
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)
X_scaled = pd.DataFrame(X_scaled, columns=df_encoded.columns)
X_scaled.head()



# Split données train et test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Conversion y_test en 0/1
y_test_num = y_test.map({'No': 0, 'Yes': 1})


import joblib



# Sauvegarde des trois modèles
joblib.dump(scaler, "scaler.pkl")


columns_train = df_encoded.columns.tolist()


joblib.dump(columns_train, "columns.pkl")



