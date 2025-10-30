
## Ringkasan

Repository ini berisi notebook dan data untuk pengerjaan model properti (data preparation dan model implementation). File utama berada di folder `Notebooks/`, data mentah/terproses di `Data Files/`, metadata model di `Model Files/`, dan keluaran hasil perbandingan di `Results/`.

## Prasyarat

- Python 3.8 atau lebih baru
- pip (atau conda) untuk manajemen paket
- Disarankan membuat virtual environment sebelum menginstal dependensi

## Cara install dependencies

1. Buat virtual environment (opsional tapi direkomendasikan):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependensi dari `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Jika Anda menggunakan conda, alternatif singkat:

```bash
conda create -n uts-env python=3.10
conda activate uts-env
pip install -r requirements.txt
```

## Cara menjalankan kode

Repo ini menggunakan Jupyter Notebooks. Langkah singkat untuk menjalankan:

1. Aktifkan virtual environment (jika dibuat):

```bash
source .venv/bin/activate
```

2. Jalankan Jupyter Notebook / Lab dari root project:

```bash
jupyter notebook
# atau
jupyter lab
```

3. Buka dan jalankan (run) notebook berikut secara berurutan:
- `Notebooks/DataPreparation.ipynb` — pembersihan, praproses, dan pembuatan file train/test yang digunakan.
- `Notebooks/ModelImplementation.ipynb` — pelatihan model, evaluasi, dan menyimpan metadata di `Model Files/`.

Catatan: jika kernel tidak tersedia, jalankan:

```bash
python -m ipykernel install --user --name uts-env --display-name "uts-env"
```

Jika Anda lebih suka mengeksekusi notebook secara batch, Anda dapat menggunakan `nbconvert` atau `papermill` (tambahkan ke `requirements.txt` jika perlu).

## Struktur project

- `Data Files/` — dataset yang digunakan dan dibagi (train/test), ada versi original dan versi ter-scaling.
	- `property_full_dataset.csv`
	- `property_train_original.csv` / `property_train_scaled.csv`
	- `property_test_original.csv` / `property_test_scaled.csv`
- `Notebooks/` — notebook untuk persiapan data dan implementasi model:
	- `DataPreparation.ipynb` — langkah praproses dan pembuatan set data untuk pelatihan/tes.
	- `ModelImplementation.ipynb` — kode pelatihan, evaluasi, dan ekspor metadata model.
- `Model Files/` — metadata atau artefak model:
	- `model_metadata.json` — metadata model yang disimpan oleh notebook.
- `Results/` — hasil evaluasi dan perbandingan model:
	- `model_comparison_metrics.csv`
- `Report.md`, `MODEL_README.md` — dokumentasi tambahan / laporan.

## requirements.txt

File `requirements.txt` berada di root repository. Contoh isi (versi disarankan, dapat disesuaikan):

```
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.2.2
matplotlib==3.7.1
seaborn==0.12.2
jupyter==1.0.0
notebook==6.5.4
ipykernel==6.23.0
scipy==1.10.1
joblib==1.2.0
openpyxl==3.1.1
```

Jika Anda ingin versi yang longgar (tanpa pin), hapus `==<versi>` dari setiap baris.

## Catatan tambahan

- Pastikan file CSV yang diperlukan sudah tersedia di `Data Files/` sebelum menjalankan notebook.
- Jika dataset besar, pertimbangkan menjalankan notebook pada mesin dengan memori lebih besar atau secara bertahap memproses subset data.
- Jika ada masalah dependency (versi konflik), cobalah membuat virtual environment baru dan install hanya paket yang diperlukan.


