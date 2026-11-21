# Netflix'e eklenen içerik sayısı tahmini

Auspify Technologies Machine Learning Internship · Task 5 (Advanced)

Görevin amacı, tarihsel katalog verisinden zaman özellikleri çıkararak Netflix’e eklenen içerik sayısını tahmin etmek ve modelleri kronolojik olarak değerlendirmektir.

Bu proje, eldeki Netflix katalog verisinde `date_added` alanına göre **aylık eklenen içerik sayısını** tahmin eder. `release_year` içeriğin orijinal yayın yılıdır ve keşif analizinde eklenme gecikmesini hesaplamak için kullanılır.

## Klasör yapısı

    main.py                          CLI giriş noktası
    configs/config.yaml              Veri, dönem, model ve çıktı ayarları
    src/                             Veri yükleme, EDA, modelleme ve değerlendirme
    tests/                           Veri dosyası gerektirmeyen regresyon testleri
    Netflix_Trend_Forecasting.ipynb   CLI ile aynı akışı kullanan notebook
    outputs/figures/                 Çalıştırmadan üretilen grafikler
    outputs/metrics/                 Toplulaştırılmış metrikler ve projeksiyonlar
    outputs/reports/                 Yönetici raporu
    docs/screenshots/                Gerçek çıktıların rapor ekran görüntüleri
    data/raw/Dataset.csv             Kullanıcının yerel olarak sağladığı dosya

## Kurulum ve çalıştırma

Doğrulama Windows üzerinde Python 3.14.0 ile, sistem paketlerini devralmayan bir sanal ortamda yapıldı. Temiz proje kopyasında PowerShell komutları:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\pyrefly.exe check --python-interpreter-path .\.venv\Scripts\python.exe
.\.venv\Scripts\python.exe main.py
```

Netflix katalog CSV’sini kullanmaya yetkili olduğunuz bir kaynaktan temin edin ve yerel olarak `data/raw/Dataset.csv` konumuna yerleştirin. Gerekli alanlar `type` ve `date_added`. Varsayılan tarih biçimi `M/D/YYYY`, içerik türleri `Movie` ve `TV Show`. Veri dosyasının dağıtım lisansı doğrulanmamıştır. Ham veri bu depoya dahil edilmez. Sentetik testler veri dosyası olmadan çalışır. Gerçek CLI çalıştırması ve notebook yerel CSV gerektirir.

Varsayılan CLI üç seriyi çalıştırır. Tek kırılım için `main.py --type Movie` veya `main.py --type "TV Show"` kullanılabilir. Bu komutlarda ayrı çıktı dizinleri kullanmak için `--config` ile farklı bir YAML dosyası verin. [Notebook](Netflix_Trend_Forecasting.ipynb) proje dizininden Python kernel ile çalıştırılır. Aynı `ForecastingPipeline` üzerinden yalnızca toplulaştırılmış sonuçlar gösterir ve varsayılan olarak geçici bir dizine yazar.

Doğrulama ortamının başlıca sürümleri pandas 3.1.1, NumPy 2.5.5, scikit-learn 1.9.1, XGBoost 3.4.1, statsmodels 0.15.0, SciPy 1.18.1, matplotlib 3.11.2 ve PyYAML 6.0.4. Her iki ağaç modeli YAML'daki tohumları ve tek iş parçacığını kullanır.

## Veri kapsamı

- CSV, 1 Ocak 2008–25 Eylül 2021 arasında 8.790 katalog kaydı içeriyor. 6.126 film ve 2.664 dizi var. Zaman kapsamı, kaynağın gözlemlediği kayıtlarla sınırlı.
- 2008–2014 arasında 84 ayın 27'si kayıt içeriyor. Ocak 2016 seçildi çünkü o aydan Ağustos 2021'e kadar her iki kategoride art arda 68 ay kayıt var. Varsayılan modelleme politikası, kayıtsız ayları belirsiz kapsam olarak değerlendirip hata veriyor. Kapsamı bilinen özel veri kümeleri açıkça `missing_month_policy: zero` seçebilir.
- Ağustos 2021 son tam ay. Eylül ayının 25'ine kadar olan kayıtlar tarihsel EDA'da yer alıyor. Tam aylık eğitim, doğrulama ve test Ağustos 2021'de bitiyor.
- Mevsimsellik karşılaştırması 2016–2020 tam yıllarının eşit gözlem kapsamına dayanıyor. Çeyrek toplamları Q1 **1.669**, Q2 **1.574**, Q3 **1.709**, Q4 **2.202**. Bunlar gözlenen katalog sayımlarıdır. Yıl sonu nedenleri veya gelecekteki Q4 performansı bu toplamlarla belirlenemez.

## Ortak tahmin ve seçim protokolü

`configs/config.yaml` varsayılan aylık takvimi belirler. İlk 60 ay (Ocak 2016–Aralık 2020) eğitim ve model seçimi için ayrılmıştır. Her genişleyen kat, Ocak 2016'dan kendi başlangıç noktasına kadar eğitim görür. Ardından 12 ayı aynı başlangıç noktasından tahmin eder.

| Kat | Son eğitim ayı | Eğitim ayı | Doğrulama |
|---|---|---:|---|
| 1 | Aralık 2018 | 36 | Ocak–Aralık 2019 |
| 2 | Mart 2019 | 39 | Nisan 2019–Mart 2020 |
| 3 | Haziran 2019 | 42 | Temmuz 2019–Haziran 2020 |
| 4 | Eylül 2019 | 45 | Ekim 2019–Eylül 2020 |
| 5 | Aralık 2019 | 48 | Ocak–Aralık 2020 |

SARIMA ve ETS doğal çok adımlı tahminlerini kullanır. XGBoost ve Random Forest, güncel ayın lag ve rolling özelliklerini önceki adımların tahminleriyle oluşturur. Naive son eğitim değerini, SeasonalNaive son 12 eğitim ayını, MovingAverage ise son üç eğitim ayının **sabit** ortalamasını kullanır. Bütün adaylar aynı geçmişten, aynı tarihler için puanlanır.

Kat RMSE'lerinin ortalaması seçim ölçütüdür. Eşitlikte ortalama MAE, ardından model adı kullanılır. Seçimden sonra tüm sabit adaylar Aralık 2020 sonundan Ocak–Ağustos 2021 için puanlanır. Bu sekiz aylık bağımsız test seçimi veya bant kalibrasyonunu etkilemez. Nihai projeksiyon Ağustos 2021 sonuna kadar yeniden eğitilir.

MAE ve RMSE ekleme adedidir. sMAPE'de gerçekleşen ve tahmin birlikte sıfırsa ayın katkısı sıfırdır. MASE'nin ölçeği her başlangıcın kendi eğitim serisindeki 12 aylık mutlak farkların ortalamasıdır. MAPE sıfır gerçekleşenleri hariç tutar. Hesaplanamayan metrikler JSON'da `null` olur. Metrikler tarih indekslerinin bire bir eşleşmesini gerektirir.

## Doğrulanmış sonuçlar

| Seri | Seçilen model | Ortalama CV RMSE | CV'de en iyi basit referans | Referans CV RMSE | Test MAE | Test RMSE | Referans test RMSE |
|---|---|---:|---|---:|---:|---:|---:|
| Toplam, doğrudan | Random Forest | 37,67 | SeasonalNaive | 43,74 | 41,01 | 48,20 | 54,83 |
| Film | XGBoost | 34,00 | SeasonalNaive | 35,73 | 27,74 | 33,37 | 34,20 |
| Dizi | MovingAverage | 10,42 | MovingAverage | 10,42 | 17,17 | 19,46 | 19,46 |
| Toplam, uzlaştırılmış | Film + dizi | — | — | — | 43,99 | 50,24 | — |

Dizide basit referans doğrulama kazananıdır. Filmde üç aylık ortalama bağımsız testte 31,84 RMSE ile seçilen XGBoost'tan düşük hata verdi. Dizide SARIMA bağımsız testte 18,38 RMSE verdi. Model seçimleri doğrulamada donduruldu. Uzlaştırılmış toplam, tutarlı kategori sayıları sağlar ve bu testte doğrudan toplamdan yüksek RMSE verdi. Eski arşivdeki skorlar, test içi gerçek lag'leri ve kısmi Eylül'ü içeren farklı bir değerlendirme protokolünden gelir.

### Yedi adayın RMSE karşılaştırması

CV sütunları beş 12 aylık katın ortalama RMSE’sini, test sütunları sekiz aylık bağımsız testin RMSE’sini gösterir.

| Model | Toplam CV | Film CV | Dizi CV | Toplam test | Film test | Dizi test |
|---|---:|---:|---:|---:|---:|---:|
| SARIMA | 50,60 | 48,13 | 11,28 | 53,91 | 37,45 | 18,38 |
| ETS | 42,40 | 36,44 | 10,89 | 55,49 | 41,69 | 19,70 |
| XGBoost | 39,22 | 34,00 | 11,15 | 53,01 | 33,37 | 21,72 |
| Random Forest | 37,67 | 34,12 | 10,55 | 48,20 | 31,00 | 19,16 |
| Naive | 49,10 | 42,23 | 11,66 | 49,01 | 32,81 | 23,35 |
| SeasonalNaive | 43,74 | 35,73 | 16,39 | 54,83 | 34,20 | 23,80 |
| MovingAverage | 48,11 | 40,74 | 10,42 | 48,80 | 31,84 | 19,46 |

## 12 aylık tarihsel projeksiyon ve belirsizlik

Ağustos 2021 sonunda başlayan projeksiyon **Eylül 2021–Ağustos 2022** arasındadır. Eylül 2021 satırı tam ay projeksiyonudur. Elde bu dönem için tam aylık gerçekleşenler yok. Bu çıktı 2026 için güncel tahmin içermez.

Film **1.315,93**, dizi **927,96** ve uzlaştırılmış toplam **2.243,89** içerik. Yayımlanan toplam her ay iki kategori tahmininin toplamıdır. Doğrudan toplam modelinin **1.837,08** projeksiyonu tanı dosyasında saklanır.

Keşif amaçlı %80 ve %95 bantları eğitim içi mutlak doğrulama hatalarının ampirik yüzdeliklerinden hesaplanır. 1–4, 5–8 ve 9–12 adımları ayrı havuzlarda tutulur. Her havuzda 20 hata bulunur ve örtüşen katlar nedeniyle bu hatalar bağımsız değildir. Toplam bantları, aynı kat ve adımdaki film ve dizi artıklarının toplamından kurulur. Alt sınır sıfıra kırpılır. Kalibrasyon adımı veya yeterli örneği bulunmayan bantlar boş bırakılır. Sekiz aylık testte gözlenen kapsama adetleri sonuç JSON'unda verilir. Bu gözlem sayısı 12 aylık nominal kapsama kalibrasyonunu doğrulamak için yetersizdir.

## Çıktılar ve kontroller

| Çıktı | İçerik |
|---|---|
| [Yönetici raporu](outputs/reports/insights.md) | Kronoloji, seçilen modeller, test ve projeksiyon özeti |
| [Değerlendirme özeti](outputs/metrics/evaluation_summary.json) | Seçim, bağımsız test, uzlaştırma ve gözlenen bant kapsaması |
| `outputs/metrics/backtest_folds_*.csv`, `backtest_mean_*.json` | Model ve kat tarihleri, kat skorları ve ortalamalar |
| `outputs/metrics/backtest_errors_*.csv`, `holdout_predictions_*.csv` | Ay bazında doğrulama hataları ve bağımsız test tahminleri |
| [Uzlaştırılmış projeksiyon](outputs/metrics/future_forecasts.csv) ve [doğrudan toplam](outputs/metrics/future_forecast_direct_all.csv) | Aylık nokta tahminleri ve keşif amaçlı bantlar |
| `outputs/figures/` | Tarihsel veri, doğrulama karşılaştırmaları, test grafikleri ve projeksiyonlar |

Doğrulamada 50 pytest testi geçti. Pyrefly 1.5.0 `default` profili sıfır tip hatası bildirdi. PyYAML ve seaborn için beş üçüncü taraf tip bildirimi uyarısı var. CLI ile temiz notebook kernel çıktıları arasında metrik dosyaları ve rapor eşleşti.

## Gerçek teslim ekran görüntüleri

Aşağıdaki görünümler, gerçek CSV ile yeniden üretilen grafik ve metriklerden tarayıcıda açılan rapor görünümleridir.

| Görünüm | Ekran görüntüsü |
|---|---|
| Tarihsel Netflix katalog trendi | [Görüntüle](docs/screenshots/01-tarihsel-katalog-trendi.png) |
| Model karşılaştırması ve bağımsız test | [Görüntüle](docs/screenshots/02-model-karsilastirmasi.png) |
| 12 aylık tarihsel projeksiyon | [Görüntüle](docs/screenshots/03-on-iki-aylik-tahmin.png) |
