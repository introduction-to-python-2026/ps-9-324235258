import pandas as pd
import seaborn as sns
dt = pd.read_csv('/content/parkinsons.csv')
dt = dt.dropna()

X = dt[['spread1', 'PPE','spread2', 'MDVP:Fo(Hz)', 'MDVP:Flo(Hz)']]  
y = dt['status'] 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test) 

rom sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train_scaled, y_train)

from sklearn.metrics import accuracy_score

best_k = None
best_accuracy = 0

for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    y_pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_k = k

print("Best k:", best_k)
print("Test accuracy:", best_accuracy)

import joblib
selected_features = ['spread1', 'spread2', 'PPE', 'MDVP:Fo(Hz)', 'MDVP:Flo(Hz)']
path = "knn.joblib" 
joblib.dump(selected_features,path)

