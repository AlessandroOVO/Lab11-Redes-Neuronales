import pandas as pd
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Función para Hold Out 70/30
def hold_out_70_30(X, y, classifier):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return acc, cm

# Función para 10-Fold Cross-Validation
def k_fold_cv(X, y, classifier, k=10):
    kf = KFold(n_splits=k, random_state=42, shuffle=True)
    scores = cross_val_score(classifier, X, y, cv=kf)
    return scores.mean()

# Función para Leave-One-Out
def loo_cv(X, y, classifier):
    loo = LeaveOneOut()
    scores = cross_val_score(classifier, X, y, cv=loo)
    return scores.mean()


# Cargar datasets
wine = load_wine(as_frame=True)
cancer = load_breast_cancer(as_frame=True)

# Leer y procesar bezdekIris.data
data = pd.read_csv('bezdekIris.data', header=None)
data.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
X_iris = data.iloc[:, :-1]
y_iris = data.iloc[:, -1].factorize()[0]

# Estandarizar datos
scaler = StandardScaler()
X_iris_scaled = scaler.fit_transform(X_iris)

# Crear diccionario de datasets
datasets = {
    "Wine": (wine['data'], wine['target']),
    "Cancer": (cancer['data'], cancer['target']),
    "Iris": (X_iris_scaled, y_iris),
}


#Configuramos el clasificador MLP
mlp_clasificator = MLPClassifier(hidden_layer_sizes=(50,25), max_iter=1000, random_state=42, solver='adam', learning_rate_init=0.001, early_stopping=True)

# Estandarizar los datos
for name, (X, y) in datasets.items():
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    datasets[name] = (X_scaled, y)

# Evaluar cada dataset
for name, (X,y) in datasets.items():
    print(f'Dataset: {name}')
    
    #Hold Out 70/30
    acc_hold_out, cm_hold_out = hold_out_70_30(X, y, mlp_clasificator)
    print(f'Hold Out 70/30 Accuracy: {acc_hold_out:.4f}')
    print(f'Hold Out 70/30 Matriz de Confusión:\n{cm_hold_out}')
    
    # 10-Fold Cross-Validation
    acc_k_fold = k_fold_cv(X, y, mlp_clasificator)
    print(f'10-Fold Cross-Validation Accuracy: {acc_k_fold:.4f}')
    
    #Leave-One-Out
    acc_loo = loo_cv(X, y, mlp_clasificator)
    print(f'Leave-One-Out Accuracy: {acc_loo:.4f}')
    print('\n')
    