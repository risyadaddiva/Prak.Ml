Nama  : Risyad Addiva Hadid

NIM   : 1217050125

Kelas : A

NIM Ganjil dengan Algoritma Decision Tree

Tahapan dan Langkah Klasifikasi 
1. Persiapan Data
Impor Dataset: Dataset citrus.csv dimuat menggunakan liblary pandas.

Seleksi Data: Data disaring untuk hanya mengambil dua jenis buah, yaitu orange dan grapefruit.

Pemetaan Label: Nilai kategori pada kolom name diubah menjadi numerik: orange = 0 dan grapefruit = 1.

Penghapusan Nilai Kosong: Baris yang memiliki nilai NaN pada label dihapus untuk menghindari error saat pelatihan.

2. Pemisahan Fitur dan Target
Fitur (X) terdiri dari kolom selain name dan label.

Target (y) adalah kolom label yang berisi 0 atau 1.

3. Split Data
Dataset dibagi menjadi dua bagian: data training (70%) dan data testing (30%) menggunakan train_test_split dari Scikit-learn.

4. Pelatihan Model
Model klasifikasi yang dilatih menggunakan algoritma Decision Tree dengan kriteria entropy.

Model dilatih menggunakan data pelatihan (X_train, y_train).

5. Evaluasi Model
Setelah dilatih model digunakan untuk memprediksi data pengujian 

Laporan Evaluasi: Dihasilkan metrik evaluasi berupa akurasi, presisi, recall, dan F1-score menggunakan classification_report.

Confussion Matriks: Digunakan untuk menampilkan sebaran prediksi benar dan salah dalam bentuk heatmap menggunakan seaborn.
