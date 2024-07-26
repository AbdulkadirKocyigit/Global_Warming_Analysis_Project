
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.ion()

## veri seti yükleme
df = pd.read_csv('C:/Users/Kadir Koçyiğit/Desktop/Python-Miiul/GlobalTemperatures.csv')

## ilk 5 satır görüntüleme
print(df.head())

## eksik verileri görüntüleme
print(df.isnull().sum())

## eksik verileri temizleme
df.dropna(inplace=True)

## temizlenmiş verilerin ilk 5 satırını görüntüleme
print(df.head())

## yıl ve ay sütunları oluşturuldu
df['year'] = pd.DatetimeIndex(df['dt']).year
df['month'] = pd.DatetimeIndex(df['dt']).month

## yıllık ortalama sıcaklık trendlerini görselleştirme
annual_avg_temp = df.groupby('year')['LandAverageTemperature'].mean().reset_index()
print(annual_avg_temp.head())

# grafik oluşturma
plt.figure(figsize=(10, 6))
sns.lineplot(x='year', y='LandAverageTemperature', data=annual_avg_temp)
plt.title('Yıllık Ortalama Sıcaklık Değişimi')
plt.xlabel('Yıl')
plt.ylabel('Ortalama Sıcaklık (Celsius)')
plt.show()


## mevsimsel değişimleri inceleme
seasonal_avg_temp = df.groupby('month')['LandAverageTemperature'].mean().reset_index()
print(seasonal_avg_temp)

## grafik oluşturma
plt.figure(figsize=(10, 6))
sns.lineplot(x='month', y='LandAverageTemperature', data=seasonal_avg_temp)
plt.title('Mevsimsel Ortalama Sıcaklık Değişimi')
plt.xlabel('Ay')
plt.ylabel('Ortalama Sıcaklık (Celsius)')
plt.show()


## zaman serisi analizi
from statsmodels.tsa.seasonal import seasonal_decompose

## zaman serisi oluşturma ve decompose etme
ts = df.set_index('dt')['LandAverageTemperature']
result = seasonal_decompose(ts, model='additive', period=365)

## grafik oluşturma
result.plot()
plt.show()

## sıcaklık anomalilerinin hesaplanması ve görselleştirilmesi

# referans dönem ortalamasını hesaplama (örneğin 1951-1990)
reference_period = df[(df['year'] >= 1951) & (df['year'] <= 1990)]
reference_mean = reference_period['LandAverageTemperature'].mean()

## sıcaklık anomalilerini hesaplama
df['TemperatureAnomaly'] = df['LandAverageTemperature'] - reference_mean

## yıllık sıcaklık anomalileri
annual_anomalies = df.groupby('year')['TemperatureAnomaly'].mean().reset_index()

## grafik ve görselleştirme
plt.figure(figsize=(10, 6))
sns.lineplot(x='year', y='TemperatureAnomaly', data=annual_anomalies)
plt.axhline(0, color='r', linestyle='--')
plt.title('Yıllık Sıcaklık Anomalileri')
plt.xlabel('Yıl')
plt.ylabel('Sıcaklık Anomalisi (Celsius)')
plt.show()


## makine öğrenmesi ile tahminleme
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

## veriyi hazırlama - eğitim ve test dosyalarına bölme
X = df[['year']].values
y = df['LandAverageTemperature'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## model eğitimi
model = LinearRegression()
model.fit(X_train, y_train)

## tahminler
predictions = model.predict(X_test)

## sonuçları görselleştirme
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Gerçek Değerler')
plt.plot(X_test, predictions, color='red', label='Tahminler')
plt.title('Sıcaklık Tahminleri')
plt.xlabel('Yıl')
plt.ylabel('Ortalama Sıcaklık (Celsius)')
plt.legend()
plt.show()