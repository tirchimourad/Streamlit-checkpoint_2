<<<<<<< HEAD
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


# Modèle 1 : Random SVM 

param_grid_svm = {
    'C': [0.01, 0.1, 1],
    'gamma': [0.01, 0.1, 1],
    'kernel': ['rbf', 'poly']
}

random_svm = RandomizedSearchCV(
    SVC(probability=True, random_state=42),
    param_grid_svm,
    n_iter=10,  # 10 tests au lieu de toutes les combinaisons
    scoring='f1',
    cv=3,
    n_jobs=-1,
    random_state=42,
    verbose=2
)

random_svm.fit(X_train, y_train)
best_svm = random_svm.best_estimator_

y_pred_svm = best_svm.predict(X_test)
y_proba_svm = best_svm.predict_proba(X_test)[:, 1]

acc_svm = accuracy_score(y_test_num, (y_pred_svm == 'Yes').astype(int))
f1_svm = f1_score(y_test_num, (y_pred_svm == 'Yes').astype(int))
auc_svm = roc_auc_score(y_test_num, y_proba_svm)

#  Modèle 2 : Random Forest

param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid_rf, 
    cv=3, 
    scoring='f1', 
    n_jobs=-1, 
    verbose=2
    )
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_

y_pred_rf = best_rf.predict(X_test)
y_proba_rf = best_rf.predict_proba(X_test)[:, 1]

acc_rf = accuracy_score(y_test_num, (y_pred_rf == 'Yes').astype(int))
f1_rf = f1_score(y_test_num, (y_pred_rf == 'Yes').astype(int))
auc_rf = roc_auc_score(y_test_num, y_proba_rf)


# Modèle 3 : Logistic Regression

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

y_pred_lr = log_reg.predict(X_test)
y_proba_lr = log_reg.predict_proba(X_test)[:, 1]

acc_lr = accuracy_score(y_test_num, (y_pred_lr == 'Yes').astype(int))
f1_lr = f1_score(y_test_num, (y_pred_lr == 'Yes').astype(int))
auc_lr = roc_auc_score(y_test_num, y_proba_lr)


# Comparaison
results = pd.DataFrame({
    'Modèle': ['SVM', 'Random Forest', 'Logistic Regression'],
    'Accuracy': [acc_svm, acc_rf, acc_lr],
    'F1-score': [f1_svm, f1_rf, f1_lr],
    'AUC': [auc_svm, auc_rf, auc_lr]
})

print("\nRésultats comparatifs :")
print(results)

 #Matrice de confusion pour le meilleur modèle

best_model_name = results.sort_values(by="F1-score", ascending=False).iloc[0]['Modèle']
if best_model_name == 'SVM':
    cm = confusion_matrix(y_test, y_pred_svm)
elif best_model_name == 'Random Forest':
    cm = confusion_matrix(y_test, y_pred_rf)
else:
    cm = confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.title(f"Matrice de confusion - {best_model_name}")
plt.show()





import joblib



# Sauvegarde des trois modèles
joblib.dump(best_svm, "svm_model.pkl")
joblib.dump(best_rf, "random_forest_model.pkl")
joblib.dump(log_reg, "regression_logistic.pkl")
joblib.dump(scaler, "scaler.pkl")
columns_train = df_encoded.columns.tolist()
joblib.dump(columns_train, "columns.pkl")





=======
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


# Modèle 1 : Random SVM 

param_grid_svm = {
    'C': [0.01, 0.1, 1],
    'gamma': [0.01, 0.1, 1],
    'kernel': ['rbf', 'poly']
}

random_svm = RandomizedSearchCV(
    SVC(probability=True, random_state=42),
    param_grid_svm,
    n_iter=10,  # 10 tests au lieu de toutes les combinaisons
    scoring='f1',
    cv=3,
    n_jobs=-1,
    random_state=42,
    verbose=2
)

random_svm.fit(X_train, y_train)
best_svm = random_svm.best_estimator_

y_pred_svm = best_svm.predict(X_test)
y_proba_svm = best_svm.predict_proba(X_test)[:, 1]

acc_svm = accuracy_score(y_test_num, (y_pred_svm == 'Yes').astype(int))
f1_svm = f1_score(y_test_num, (y_pred_svm == 'Yes').astype(int))
auc_svm = roc_auc_score(y_test_num, y_proba_svm)

#  Modèle 2 : Random Forest

param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid_rf, 
    cv=3, 
    scoring='f1', 
    n_jobs=-1, 
    verbose=2
    )
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_

y_pred_rf = best_rf.predict(X_test)
y_proba_rf = best_rf.predict_proba(X_test)[:, 1]

acc_rf = accuracy_score(y_test_num, (y_pred_rf == 'Yes').astype(int))
f1_rf = f1_score(y_test_num, (y_pred_rf == 'Yes').astype(int))
auc_rf = roc_auc_score(y_test_num, y_proba_rf)


# Modèle 3 : Logistic Regression

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

y_pred_lr = log_reg.predict(X_test)
y_proba_lr = log_reg.predict_proba(X_test)[:, 1]

acc_lr = accuracy_score(y_test_num, (y_pred_lr == 'Yes').astype(int))
f1_lr = f1_score(y_test_num, (y_pred_lr == 'Yes').astype(int))
auc_lr = roc_auc_score(y_test_num, y_proba_lr)


# Comparaison
results = pd.DataFrame({
    'Modèle': ['SVM', 'Random Forest', 'Logistic Regression'],
    'Accuracy': [acc_svm, acc_rf, acc_lr],
    'F1-score': [f1_svm, f1_rf, f1_lr],
    'AUC': [auc_svm, auc_rf, auc_lr]
})

print("\nRésultats comparatifs :")
print(results)

 #Matrice de confusion pour le meilleur modèle

best_model_name = results.sort_values(by="F1-score", ascending=False).iloc[0]['Modèle']
if best_model_name == 'SVM':
    cm = confusion_matrix(y_test, y_pred_svm)
elif best_model_name == 'Random Forest':
    cm = confusion_matrix(y_test, y_pred_rf)
else:
    cm = confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.title(f"Matrice de confusion - {best_model_name}")
plt.show()





import joblib



# Sauvegarde des trois modèles
joblib.dump(best_svm, "svm_model.pkl")
joblib.dump(best_rf, "random_forest_model.pkl")
joblib.dump(log_reg, "regression_logistic.pkl")
joblib.dump(scaler, "scaler.pkl")
columns_train = df_encoded.columns.tolist()
joblib.dump(columns_train, "columns.pkl")





>>>>>>> e924131 (Sauvegarde avant merge)
