# Netflix'e eklenen içerik sayısı tahmini

Hedef, CSV'deki `date_added` üzerinden ay başına katalog eklemesidir. `release_year` orijinal yayın yılıdır ve tarihsel içerik gecikmesinde kullanılır.
Son kayıt **2021-09-25**, veri kesimi **2021-09-25**. Tam ay modelleme penceresi **2016-01–2021-08**. Pencere sonrasındaki kısmi ay (partial month) kayıtları EDA kapsamında görülebilir.

## Doğrulama ve bağımsız test

Tahmin başlangıcı **2020-12**. Bağımsız test **2021-01–2021-08** (8 tam ay). Model seçimi, aşağıdaki eğitim içi katların ortalama RMSE değerine göre yapıldı. Eşitlikte MAE ve model adı kullanıldı.
Bir kat boyunca bütün adaylar aynı geçmişe ve aynı aylara erişir. XGBoost ve Random Forest bir sonraki ayın lag ve rolling özelliklerini kendi tahminleriyle günceller.

| Kat | Eğitim başlangıcı | Son eğitim ayı | Doğrulama başlangıcı | Doğrulama bitişi | Ufuk |
|---|---|---|---|---|---:|
| 1 | 2016-01 | 2018-12 | 2019-01 | 2019-12 | 12 |
| 2 | 2016-01 | 2019-03 | 2019-04 | 2020-03 | 12 |
| 3 | 2016-01 | 2019-06 | 2019-07 | 2020-06 | 12 |
| 4 | 2016-01 | 2019-09 | 2019-10 | 2020-09 | 12 |
| 5 | 2016-01 | 2019-12 | 2020-01 | 2020-12 | 12 |

| Seri | Seçilen model | CV RMSE | En iyi CV referansı | Referans CV RMSE | Test MAE | Test RMSE | Referans test RMSE |
|---|---|---:|---|---:|---:|---:|---:|
| All | RandomForest | 37.67 | SeasonalNaive | 43.74 | 41.01 | 48.20 | 54.83 |
| Movie | XGBoost | 34.00 | SeasonalNaive | 35.73 | 27.74 | 33.37 | 34.20 |
| TV Show | MovingAverage | 10.42 | MovingAverage | 10.42 | 17.17 | 19.46 | 19.46 |

Naive son değeri, SeasonalNaive son 12 ayı ve MovingAverage son eğitim aylarının sabit ortalamasını kullanır.
MASE ölçeği her başlangıçta, o ana kadarki eğitim verisinin 12 aylık mutlak fark ortalamasıdır.

## Tarihsel projeksiyon ve belirsizlik

Tam seriyle yeniden eğitim sonrası **2021-09–2022-08** için 12 aylık projeksiyon üretildi. Bağımsız test 8 adımlık kapsama sahip. Daha uzun adımlarda dayanak eğitim içi backtest'tir. 2026 için tahmin üretilmedi.
Toplam projeksiyon her ay film ve dizi tahminlerinin toplamı. Bu toplamın bağımsız test RMSE'si 50.24. Doğrudan toplam modelinin tahmini ve test skorları ayrı tanı çıktılarında saklanır.

| Seri | Yayımlanan model | Ufuk toplamı |
|---|---|---:|
| All | BottomUp | 2243.89 |
| Movie | XGBoost | 1315.93 |
| TV Show | MovingAverage | 927.96 |

Keşif amaçlı %80 ve %95 bantları eğitim içi doğrulama hatalarının mutlak değerlerine dayanır. Ardışık dörder adım havuzlanır. Toplam için aynı kat ve adımlardaki alt seri hataları toplanır. Katlar örtüşebildiği için örnekler bağımlıdır. Yetersiz gözlemli adımların bantları boş bırakılır. Bağımsız testteki gözlenen kapsama adetleri `evaluation_summary.json` dosyasında yer alır. Bu kısa test, nominal kapsama düzeylerini doğrulamak için yetersizdir.

## Veri kapsamı ve tarihsel desen

Katalog gözlemleri sağlanan CSV ile sınırlıdır. CSV'nin dağıtım lisansı doğrulanmamıştır.
Her dört çeyreğin de gözlendiği **2016–2020** tam yılları için Q1 1669, Q2 1574, Q3 1709, Q4 2202. Bu sayımlar betimleyicidir.
