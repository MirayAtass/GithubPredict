import requests
import numpy as np
import pandas as pd
import re
from langdetect import detect, LangDetectException
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras import regularizers


# GitHub API token tanımlaması yapma
GITHUB_TOKEN = "your_personal_access_token"
headers = {
    "Authorization": f"token {GITHUB_TOKEN}"
}

# GitHub depolarını çekme
def fetch_github_repos(query, per_page=100, pages=10):  
    all_repos = []
    for page in range(1, pages + 1):
        url = f"https://api.github.com/search/repositories"
        params = {"q": query, "per_page": per_page, "page": page}
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            all_repos.extend(response.json()["items"])
        else:
            raise Exception(f"{response.status_code} - {response.text}")
    return all_repos

queries = ["machine learning", "deep learning", "artificial intelligence", "computer vision"]
all_repos = []

emoji_pattern = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]")

def is_english(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

# Çekilen dataların bazı alanlarının filtrelenmesi
for query in queries:
    try:
        repos = fetch_github_repos(query, per_page=100, pages=10)
        data = [
            {
                "Name": repo["name"],
                "Full Name": repo["full_name"],
                "Stars": repo["stargazers_count"],
                "Forks": repo["forks_count"],
                "Language": repo.get("language", "N/A"),
                "Description": repo.get("description", "N/A"),
                "Created At": repo["created_at"],
                "Updated At": repo["updated_at"]
            }
            for repo in repos
        ]

        # Satırda emoji, ingilizce dışında saçma sapan bir dil veya boş ifade varsa o satırı veri setinden
        filtered_data = []
        for entry in data:
            description = entry["Description"]
            language = entry["Language"]
            
            if description is None or language in [None, "N/A"]:
                continue

            if is_english(description) and not emoji_pattern.search(description):
                # Açıklama uzunluğunun veri setinde yeni bir sütun olarak eklenmesi
                entry["Description Length"] = len(description)
                filtered_data.append(entry)

        all_repos.extend(filtered_data)

    except Exception as e:
        print(f"Hata: {e}")

df = pd.DataFrame(all_repos)

df['Created At'] = pd.to_datetime(df['Created At']).dt.tz_localize(None)
df['Age'] = (pd.to_datetime('today').normalize() - df['Created At']).dt.days  

features = ["Stars", "Forks", "Description Length", "Age"]
target = ["Stars", "Forks"]

X = df[features]
y = df[target]

# Min max scaler kullanarak verileri normalize etme
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Z-score değerlerinin  hesaplanması ve silinmesi
z_scores = np.abs((X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0))
threshold = 3
anomalies = (z_scores > threshold).any(axis=1)
df_no_outliers = df[~anomalies]
X_no_outliers = df_no_outliers[features]
y_no_outliers = df_no_outliers[target]

# Anomali değerler çıkarıldıktan sonra normalize etme
X_no_outliers_scaled = scaler.fit_transform(X_no_outliers)
X_train, X_test, y_train, y_test = train_test_split(X_no_outliers_scaled, y_no_outliers, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Model tanımı
model = tf.keras.Sequential([
    # 1. katman
    tf.keras.layers.Dense(1024, activation='relu', input_dim=X_train.shape[1],
                          kernel_regularizer=regularizers.l2(0.03)),
    tf.keras.layers.Dropout(0.4),

    # 2. katman
    tf.keras.layers.Dense(512, activation='relu',
                          kernel_regularizer=regularizers.l2(0.03)),
    tf.keras.layers.Dropout(0.4),

    # 3. katman
    tf.keras.layers.Dense(256, activation='relu',
                          kernel_regularizer=regularizers.l2(0.03)),
    tf.keras.layers.Dropout(0.4),

    # 4. katman
    tf.keras.layers.Dense(128, activation='relu',
                          kernel_regularizer=regularizers.l2(0.03)),

    # Çıktı katmanı
    tf.keras.layers.Dense(2) #Star ve fork çıktıları
])

# Model dserlemesi
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='mse',  # Ortalama kare hata (MSE)
              metrics=['mae'])  # Ortalama mutlak fark (MAE)

# Erken durdurma ekleme işlemi
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Model eğitimi
history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_val, y_val), 
                    callbacks=[early_stopping], verbose=1)

# Model testi
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {test_loss}, Test MAE: {test_mae}')
predictions = model.predict(X_test) #tahminler

# Bazı tahminleri gösterme işlemi
for i in range(5):
    print(f"Predicted Stars: {predictions[i][0]}, Actual Stars: {y_test.iloc[i]['Stars']}")
    print(f"Predicted Forks: {predictions[i][1]}, Actual Forks: {y_test.iloc[i]['Forks']}")

# performans sonuçları (MSE, MAE, R2)
y_pred_stars = predictions[:, 0]
y_pred_forks = predictions[:, 1]

# MSE
mse_stars = mean_squared_error(y_test['Stars'], y_pred_stars)
mse_forks = mean_squared_error(y_test['Forks'], y_pred_forks)

# MAE
mae_stars = mean_absolute_error(y_test['Stars'], y_pred_stars)
mae_forks = mean_absolute_error(y_test['Forks'], y_pred_forks)

# R2 değeri
r2_stars = r2_score(y_test['Stars'], y_pred_stars)
r2_forks = r2_score(y_test['Forks'], y_pred_forks)

print(f'Mean Squared Error (Stars): {mse_stars}')
print(f'Mean Squared Error (Forks): {mse_forks}')
print(f'Mean Absolute Error (Stars): {mae_stars}')
print(f'Mean Absolute Error (Forks): {mae_forks}')
print(f'R2 Score (Stars): {r2_stars}')
print(f'R2 Score (Forks): {r2_forks}')

# Tüm değerlerin grafikleştirilmesi
history_loss = history.history['loss']
history_val_loss = history.history['val_loss']
history_mae = history.history['mae']
history_val_mae = history.history['val_mae']

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history_loss, label='Training Loss')
plt.plot(history_val_loss, label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_mae, label='Training MAE')
plt.plot(history_val_mae, label='Validation MAE')
plt.title('MAE Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()

# Tahmin edilen ve gerçek starlar
plt.figure(figsize=(10, 6))
plt.scatter(y_test['Stars'], predictions[:, 0], alpha=0.5, color='blue')
plt.plot([min(y_test['Stars']), max(y_test['Stars'])], 
         [min(y_test['Stars']), max(y_test['Stars'])], 'r--', lw=2)
plt.title('Predicted vs Actual Stars')
plt.xlabel('Actual Stars')
plt.ylabel('Predicted Stars')
plt.show()

# Tahmin edilen ve gerçek forklar
plt.figure(figsize=(10, 6))
plt.scatter(y_test['Forks'], predictions[:, 1], alpha=0.5, color='green')
plt.plot([min(y_test['Forks']), max(y_test['Forks'])], 
         [min(y_test['Forks']), max(y_test['Forks'])], 'r--', lw=2)
plt.title('Predicted vs Actual Forks')
plt.xlabel('Actual Forks')
plt.ylabel('Predicted Forks')
plt.show()
