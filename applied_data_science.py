import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================
# Adım 1: Veri Setinin Yüklenmesi
# ==========================================
cancer_data = load_breast_cancer()
X = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
y = cancer_data.target

print("--- Dataset Loaded Successfully ---")
print(f"Features shape: {X.shape}")
print(f"Target classes: {cancer_data.target_names}\n")

# ==========================================
# Adım 2: Veriyi Bölme (Train/Test Split) ve Temizleme
# ==========================================
# Eksik veri kontrolü (Hazır veri setinde eksik veri yoktur ama kontrolü göstermek iyi kod pratiğidir)
if X.isnull().sum().sum() == 0:
    print("Data Integrity Check: No missing values found.")

# Veriyi %80 Eğitim, %20 Test olarak bölüyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# Adım 3: Özellik Ölçeklendirme (Feature Scaling)
# ==========================================
# k-NN gibi mesafe tabanlı algoritmaların sütunlardaki büyük sayısal farklardan 
# etkilenmemesi için veriyi standartlaştırıyoruz (ortalama=0, standart sapma=1).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling successfully applied.\n")

# ==========================================
# Adım 4: Algoritmaların Tanımlanması ve Eğitilmesi
# ==========================================
models = {
    "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors (k-NN)": KNeighborsClassifier(n_neighbors=5)
}

# ==========================================
# Adım 5: Modellerin Test Edilmesi ve Metriklerin Basılması
# ==========================================
for name, model in models.items():
    # Modeli eğitiyoruz
    model.fit(X_train_scaled, y_train)
    
    # Modelin hiç görmediği test verisi üzerinde tahmin yapmasını sağlıyoruz
    y_pred = model.predict(X_test_scaled)
    
    # Doğruluk skoru ve Karmaşıklık Matrisini hesaplıyoruz
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"=== {name} Performance ===")
    print(f"Accuracy Score: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("-" * 40)

# ==========================================
# Adım 6: Kritik Değerlendirme ve Sonuç (Conclusion)
# ==========================================
"""
🔍 CONCLUSION & CRITICAL EVALUATION:
In this pipeline, both models achieve high accuracy, but Random Forest Classifier typically 
outperforms K-Nearest Neighbors on this specific dataset. 

1. Random Forest is an ensemble method that builds multiple decision trees and merges their 
   predictions. It handles non-linear relationships and high-dimensional features robustly 
   without worrying about spatial distance.
2. K-Nearest Neighbors (k-NN) relies heavily on the geometric distance between points in the 
   feature space. Even though we applied StandardScaler to normalize the data scales, k-NN is still 
   susceptible to the 'curse of dimensionality' due to the large number of features (30), 
   which can introduce noise and slightly degrade its predictive performance compared to tree-based voting.
"""