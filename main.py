from sklearn.datasets import load_iris
import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris() # Veri Seti -> Data Frame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

print("Veri hazırlandı, satır sayısı:", df.shape[0])
print(df.head())

client = MongoClient("mongodb://localhost:27017/") # MongoDB bağlantısı
db = client["ml_simple"]
collection = db["iris_dataset"]

if collection.count_documents({}) == 0:
    records = df.to_dict(orient="records")
    collection.insert_many(records)
    print("Veri MongoDB'ye yüklendi.")
else:
    print("Veri zaten MongoDB'de mevcut.")

cursor = collection.find()  # MongoDB'den veri çekme
data = pd.DataFrame(list(cursor)).drop(columns=["_id"])

X = data[iris.feature_names]
y = data["target"]

print("MongoDB'den çekilen veri, X boyutu:", X.shape, "y boyutu:", y.shape)

# Eğitim ve Test
X_train, X_test, y_train, y_test = train_test_split( 
    X, y, test_size=0.2, random_state=42
)
print("Eğitim örnek sayısı:", X_train.shape[0], "Test örnek sayısı:", X_test.shape[0])

# Model Oluşturma ve Eğitme
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print("Model başarıyla eğitildi.")

# Test ve doğruluk 
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test doğruluğu: %{accuracy * 100:.2f}")

# Örnek tahmin
ornek = [[5.1, 3.5, 1.4, 0.2]]
tahmin_idx = model.predict(ornek)[0]
tahmin_isim = iris.target_names[tahmin_idx]
print("Örnek veri tahmini:", tahmin_isim)

