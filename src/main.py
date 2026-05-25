# Kütüphaneleri içe aktar
import os           # Dosya işlemleri için
import re           # Metin işleme (regex) için
import json         # Sonuçları JSON olarak kaydetmek için
import numpy as np  # Sayısal hesaplamalar için

# Makine öğrenmesi
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, classification_report
)

# Grafik çizmek için
import matplotlib
matplotlib.use('Agg')   # Ekrana değil dosyaya kaydetmek için
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


def conll_oku(dosya_yolu):
    """
    CoNLL formatındaki dosyayı okur.
    Çıktı: belge listesi. Her belge = {text, tokens} şeklinde sözlük.
    """
    belgeler = []
    simdiki_tokenler = []
    simdiki_metin = ""

    with open(dosya_yolu, encoding='utf-8') as f:
        for satir in f:
            satir = satir.rstrip('\n')

            # "# text = ..." satırı → metnin kendisi
            if satir.startswith('# text ='):
                simdiki_metin = satir.replace('# text =', '').strip()

            # Boş satır veya yorum satırı → cümle bitti
            elif satir == '' or satir.startswith('#'):
                if simdiki_tokenler:
                    belgeler.append({
                        'metin': simdiki_metin,
                        'tokenler': simdiki_tokenler
                    })
                    simdiki_tokenler = []
                    simdiki_metin = ""

            # Normal satır → bir kelime
            else:
                parcalar = satir.split('\t')
                if len(parcalar) >= 3:
                    simdiki_tokenler.append({
                        'id': int(parcalar[0]),
                        'kelime': parcalar[1],
                        'etiket': parcalar[2].strip()
                    })

    # Dosyanın sonundaki son belgeyi de ekle
    if simdiki_tokenler:
        belgeler.append({
            'metin': simdiki_metin,
            'tokenler': simdiki_tokenler
        })

    return belgeler


# Türkçe zamir listesi
ZAMIRLER = {
    'o', 'ona', 'onu', 'onun', 'onlar', 'onlara', 'onları',
    'ben', 'bana', 'beni', 'biz', 'bize', 'bizi',
    'sen', 'sana', 'seni', 'siz', 'size', 'sizi',
    'kendisi', 'kendisinde', 'kendisine', 'kendisini',
    'ikisi', 'hepsi', 'o', 'şu', 'bu'
}

def ozellik_cikar(tokenler, idx):
    """
    idx numaralı token için özellik sözlüğü döner.
    """
    token = tokenler[idx]
    kelime = token['kelime']
    n = len(tokenler)

    ozellikler = {}

    # ── Kelimenin kendisi hakkında ──
    ozellikler['zamir_mi'] = 1 if kelime.lower() in ZAMIRLER else 0
    ozellikler['ozel_isim_mi'] = 1 if kelime[0].isupper() else 0
    ozellikler['kelime_uzunlugu'] = len(kelime)
    ozellikler['konum'] = idx / max(n - 1, 1)  # 0.0 ile 1.0 arası

    # ── Son ekler (Türkçe morfoloji) ──
    ozellikler['son2'] = hash(kelime[-2:]) % 500 if len(kelime) >= 2 else 0
    ozellikler['son3'] = hash(kelime[-3:]) % 500 if len(kelime) >= 3 else 0

    # Yaygın Türkçe ekler
    ozellikler['nin_eki'] = 1 if kelime.endswith(('nın','nin','nun','nün')) else 0
    ozellikler['da_eki']  = 1 if kelime.endswith(('da','de','ta','te')) else 0
    ozellikler['dan_eki'] = 1 if kelime.endswith(('dan','den','tan','ten')) else 0
    ozellikler['a_eki']   = 1 if kelime.endswith(('a','e')) else 0
    ozellikler['i_eki']   = 1 if kelime.endswith(('ı','i','u','ü')) else 0

    # ── Komşu kelimeler (bağlam penceresi) ──
    for fark in [-2, -1, 1, 2]:
        komsu_idx = idx + fark
        anahtar = f'komsu_{fark}'
        if 0 <= komsu_idx < n:
            komsu = tokenler[komsu_idx]['kelime']
            ozellikler[f'{anahtar}_zamir'] = 1 if komsu.lower() in ZAMIRLER else 0
            ozellikler[f'{anahtar}_ozel']  = 1 if komsu[0].isupper() else 0
            ozellikler[f'{anahtar}_son2']  = hash(komsu[-2:]) % 500 if len(komsu) >= 2 else 0
        else:
            ozellikler[f'{anahtar}_zamir'] = 0
            ozellikler[f'{anahtar}_ozel']  = 0
            ozellikler[f'{anahtar}_son2']  = 0

    return ozellikler


def veri_hazirla(belgeler):
    """
    Belgelerden örnek (X) ve etiket (y) listesi oluşturur.
    """
    ornekler = []
    etiketler = []

    for belge in belgeler:
        tokenler = belge['tokenler']
        for idx, token in enumerate(tokenler):
            # Özellikleri çıkar
            ozellik = ozellik_cikar(tokenler, idx)

            # Etiketi belirle
            ham_etiket = token['etiket']
            if ham_etiket == '_':
                etiket = 'O'  # coreference yok
            else:
                # (1) → COREF_1, (2) → COREF_2, vb.
                eslesme = re.search(r'\((\d+)\)', ham_etiket)
                etiket = f"COREF_{eslesme.group(1)}" if eslesme else 'O'

            ornekler.append(ozellik)
            etiketler.append(etiket)

    return ornekler, etiketler


def sayisallastir(ornekler, anahtar_listesi=None):
    """
    Özellik sözlüklerini numpy dizisine (sayı tablosu) çevirir.
    """
    if anahtar_listesi is None:
        anahtar_listesi = sorted(ornekler[0].keys())

    X = []
    for ornek in ornekler:
        satir = [ornek.get(k, 0) for k in anahtar_listesi]
        X.append(satir)

    return np.array(X, dtype=np.float32), anahtar_listesi


class CoreferenceModeli:
    """
    Coreference resolution için makine öğrenmesi modeli.
    """

    def __init__(self, model_turu='mlp'):
        self.model_turu = model_turu
        self.olcekleyici = StandardScaler()  # Sayıları normalize eder
        self.etiket_encoder = LabelEncoder() # Etiketleri sayıya çevirir
        self.anahtar_listesi = None

        if model_turu == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=42
            )
        else:  # mlp
            self.model = MLPClassifier(
                hidden_layer_sizes=(64, 32),  # 2 gizli katman
                activation='relu',
                max_iter=300,
                random_state=42
            )

    def egit(self, ornekler, etiketler):
        """Modeli eğitir."""
        X, self.anahtar_listesi = sayisallastir(ornekler)
        X = self.olcekleyici.fit_transform(X)    # Normalize et
        y = self.etiket_encoder.fit_transform(etiketler)  # Etiketleri sayıya çevir
        self.model.fit(X, y)
        print(f"Model eğitildi! Sınıf sayısı: {len(self.etiket_encoder.classes_)}")

    def tahmin_et(self, ornekler):
        """Yeni veriler için tahmin yapar."""
        X, _ = sayisallastir(ornekler, self.anahtar_listesi)
        X = self.olcekleyici.transform(X)
        y_pred = self.model.predict(X)
        return self.etiket_encoder.inverse_transform(y_pred)

    def degerlendir(self, ornekler, gercek_etiketler, klasor='results'):
        """Modeli test eder, metrikleri hesaplar, grafik çizer."""
        tahmin_etiketler = self.tahmin_et(ornekler)

        # Metrikleri hesapla
        acc  = accuracy_score(gercek_etiketler, tahmin_etiketler)
        prec = precision_score(gercek_etiketler, tahmin_etiketler,
                               average='weighted', zero_division=0)
        rec  = recall_score(gercek_etiketler, tahmin_etiketler,
                            average='weighted', zero_division=0)
        f1   = f1_score(gercek_etiketler, tahmin_etiketler,
                        average='weighted', zero_division=0)

        # Ekrana yazdır
        print(f"\n{'='*45}")
        print(f"  {self.model_turu.upper()} - SONUÇLAR")
        print(f"{'='*45}")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1-Score  : {f1:.4f}")
        print(f"{'='*45}")
        print(classification_report(gercek_etiketler, tahmin_etiketler, zero_division=0))

        # Grafikleri çiz
        os.makedirs(klasor, exist_ok=True)
        self._karisiklik_matrisi(gercek_etiketler, tahmin_etiketler, klasor)
        self._metrik_grafigi(acc, prec, rec, f1, klasor)

        return {
            'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1': f1,
            'gercek': gercek_etiketler,
            'tahmin': list(tahmin_etiketler)
        }

    def _karisiklik_matrisi(self, gercek, tahmin, klasor):
        siniflar = sorted(set(gercek) | set(tahmin))
        cm = confusion_matrix(gercek, tahmin, labels=siniflar)

        boyut = max(6, len(siniflar))
        plt.figure(figsize=(boyut, boyut - 1))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=siniflar, yticklabels=siniflar)
        plt.title(f'Karışıklık Matrisi - {self.model_turu.upper()}', fontsize=13)
        plt.xlabel('Tahmin Edilen')
        plt.ylabel('Gerçek')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        yol = os.path.join(klasor, f'confusion_matrix_{self.model_turu}.png')
        plt.savefig(yol, dpi=150)
        plt.close()
        print(f"Grafik kaydedildi: {yol}")

    def _metrik_grafigi(self, acc, prec, rec, f1, klasor):
        isimler = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        degerler = [acc, prec, rec, f1]
        renkler = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

        plt.figure(figsize=(8, 5))
        cubuklar = plt.bar(isimler, degerler, color=renkler, edgecolor='white')
        for cubuk, deger in zip(cubuklar, degerler):
            plt.text(cubuk.get_x() + cubuk.get_width()/2,
                     cubuk.get_height() + 0.01,
                     f'{deger:.3f}', ha='center', fontsize=12, fontweight='bold')
        plt.ylim(0, 1.2)
        plt.title(f'Performans Metrikleri - {self.model_turu.upper()}', fontsize=13)
        plt.ylabel('Değer')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        yol = os.path.join(klasor, f'metrikler_{self.model_turu}.png')
        plt.savefig(yol, dpi=150)
        plt.close()
        print(f"Grafik kaydedildi: {yol}")


def model_karsilastir(lr_sonuc, mlp_sonuc, klasor='results'):
    """İki modeli tek grafikte karşılaştırır."""
    metrikler = ['accuracy', 'precision', 'recall', 'f1']
    isimler = ['Accuracy', 'Precision', 'Recall', 'F1']

    x = np.arange(len(metrikler))
    genislik = 0.35

    plt.figure(figsize=(10, 6))
    lr_degerler  = [lr_sonuc[m]  for m in metrikler]
    mlp_degerler = [mlp_sonuc[m] for m in metrikler]

    c1 = plt.bar(x - genislik/2, lr_degerler,  genislik, label='Logistic Regression', color='#4C72B0', alpha=0.85)
    c2 = plt.bar(x + genislik/2, mlp_degerler, genislik, label='MLP',                 color='#DD8452', alpha=0.85)

    for cubuk in list(c1) + list(c2):
        plt.text(cubuk.get_x() + cubuk.get_width()/2,
                 cubuk.get_height() + 0.005,
                 f'{cubuk.get_height():.2f}',
                 ha='center', fontsize=9, fontweight='bold')

    plt.xticks(x, isimler, fontsize=11)
    plt.ylim(0, 1.2)
    plt.ylabel('Değer')
    plt.title('Model Karşılaştırması: Logistic Regression vs MLP', fontsize=13)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    yol = os.path.join(klasor, 'model_karsilastirma.png')
    plt.savefig(yol, dpi=150)
    plt.close()
    print(f"Karşılaştırma grafiği: {yol}")


def conll_yaz(belgeler, tahminler, cikti_yolu):
    """
    Tahminleri CoNLL formatında dosyaya yazar.
    Hoca bunu görmek istiyor!
    """
    idx = 0
    with open(cikti_yolu, 'w', encoding='utf-8') as f:
        for belge in belgeler:
            f.write(f"# text = {belge['metin']}\n")
            f.write("# columns = ID FORM GERCEK_ETIKET TAHMIN\n")
            for token in belge['tokenler']:
                gercek = token['etiket']
                tahmin = tahminler[idx]
                # Tahmini CoNLL formatına geri çevir
                if tahmin == 'O':
                    tahmin_conll = '_'
                elif tahmin.startswith('COREF_'):
                    no = tahmin.replace('COREF_', '')
                    tahmin_conll = f'({no})'
                else:
                    tahmin_conll = '_'
                f.write(f"{token['id']}\t{token['kelime']}\t{gercek}\t{tahmin_conll}\n")
                idx += 1
            f.write("\n")
    print(f"CoNLL çıktısı yazıldı: {cikti_yolu}")

def main():
    print("=" * 50)
    print("  Türkçe Coreference Resolution")
    print("  BTÜ NLP Projesi 2025-2026")
    print("=" * 50)

    KLASOR = 'results'
    os.makedirs(KLASOR, exist_ok=True)

    # 1. Veriyi oku
    print("\n[1] Veri yükleniyor...")
    egitim_belgeleri = conll_oku('/Users/bilgenurcakir/PycharmProjects/confusion_nlp/data/train.conll')
    test_belgeleri = conll_oku('/Users/bilgenurcakir/PycharmProjects/confusion_nlp/data/test.conll')
    print(f"    Eğitim: {len(egitim_belgeleri)} belge")
    print(f"    Test:   {len(test_belgeleri)} belge")

    # 2. Özellik çıkar
    print("\n[2] Özellikler çıkarılıyor...")
    egitim_ornekler, egitim_etiketler = veri_hazirla(egitim_belgeleri)
    test_ornekler,   test_etiketler   = veri_hazirla(test_belgeleri)
    print(f"    Eğitim örneği: {len(egitim_ornekler)}")
    print(f"    Test örneği  : {len(test_ornekler)}")

    # 3. Model 1: Logistic Regression
    print("\n[3] Logistic Regression eğitiliyor...")
    lr_model = CoreferenceModeli(model_turu='logistic')
    lr_model.egit(egitim_ornekler, egitim_etiketler)
    lr_sonuc = lr_model.degerlendir(test_ornekler, test_etiketler, KLASOR)

    # 4. Model 2: MLP
    print("\n[4] MLP eğitiliyor...")
    mlp_model = CoreferenceModeli(model_turu='mlp')
    mlp_model.egit(egitim_ornekler, egitim_etiketler)
    mlp_sonuc = mlp_model.degerlendir(test_ornekler, test_etiketler, KLASOR)

    # 5. Karşılaştırma grafiği
    print("\n[5] Karşılaştırma grafiği çiziliyor...")
    model_karsilastir(lr_sonuc, mlp_sonuc, KLASOR)

    # 6. CoNLL çıktıları yaz
    print("\n[6] CoNLL çıktıları yazılıyor...")
    lr_tahminler  = lr_model.tahmin_et(test_ornekler)
    mlp_tahminler = mlp_model.tahmin_et(test_ornekler)
    conll_yaz(test_belgeleri, list(lr_tahminler),  f'{KLASOR}/tahmin_lr.conll')
    conll_yaz(test_belgeleri, list(mlp_tahminler), f'{KLASOR}/tahmin_mlp.conll')

    # 7. JSON özet kaydet
    ozet = {
        'logistic_regression': {
            'accuracy':  round(lr_sonuc['accuracy'],  4),
            'precision': round(lr_sonuc['precision'], 4),
            'recall':    round(lr_sonuc['recall'],    4),
            'f1':        round(lr_sonuc['f1'],        4),
        },
        'mlp': {
            'accuracy':  round(mlp_sonuc['accuracy'],  4),
            'precision': round(mlp_sonuc['precision'], 4),
            'recall':    round(mlp_sonuc['recall'],    4),
            'f1':        round(mlp_sonuc['f1'],        4),
        }
    }
    with open(f'{KLASOR}/ozet.json', 'w', encoding='utf-8') as f:
        json.dump(ozet, f, indent=2, ensure_ascii=False)
    print(f"\nÖzet JSON kaydedildi: {KLASOR}/ozet.json")

    print("\n" + "=" * 50)
    print("  TAMAMLANDI! results/ klasörüne bak.")
    print("=" * 50)


# Programı çalıştır
if __name__ == '__main__':
    main()
