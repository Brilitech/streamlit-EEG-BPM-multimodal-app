# 🧠 Focus Detection System

Dashboard prediksi tingkat fokus menggunakan **LSTM Deep Learning** 
berbasis data **EEG + BPM** multimodal.

---

## 📁 Struktur Folder

```
streamlit_app/
├── app.py                  ← Main Streamlit app
├── requirements.txt        ← Dependencies
├── .gitignore
├── README.md
└── models/
    ├── best_model.pth      ← LSTM model weights
    ├── scaler.pkl          ← StandardScaler
    ├── pca.pkl             ← PCA transformer
    ├── label_encoder.pkl   ← Label encoder
    └── data_info.pkl       ← Model metadata
```

---

## 🚀 Cara Deploy ke Streamlit Cloud

### 1. Upload Model Files ke GitHub

Copy file-file berikut dari folder `processed/` ke folder `models/`:

```bash
# Di lokal Windows:
D:\! TESIS\Pytorch\data\processed\best_model.pth    → models/best_model.pth
D:\! TESIS\Pytorch\data\processed\scaler.pkl        → models/scaler.pkl
D:\! TESIS\Pytorch\data\processed\pca.pkl           → models/pca.pkl
D:\! TESIS\Pytorch\data\processed\label_encoder.pkl → models/label_encoder.pkl
D:\! TESIS\Pytorch\data\processed\data_info.pkl     → models/data_info.pkl
```

### 2. Push ke GitHub

```bash
git init
git add .
git commit -m "Initial commit: Focus Detection System"
git remote add origin https://github.com/USERNAME/focus-detection.git
git push -u origin main
```

> ⚠️ Jika `best_model.pth` > 100MB, gunakan **Git LFS**:
> ```bash
> git lfs install
> git lfs track "*.pth"
> git add .gitattributes
> git add models/best_model.pth
> git commit -m "Add large model with LFS"
> ```

### 3. Deploy di Streamlit Cloud

1. Buka [share.streamlit.io](https://share.streamlit.io)
2. Login dengan GitHub
3. Klik **New app**
4. Pilih repository, branch `main`, file `app.py`
5. Klik **Deploy!**

---

## 📊 Format Input Data

### EEG CSV:
```
Timestamp, Low Alpha, High Alpha, Low Beta, High Beta
"2024-12-11,10:09:35.203506", 34, 6, 43, 33
...
```

### BPM CSV:
```
Timestamp, BPM, Avg BPM
2024-12-11 10:09:41, 83.45, 77.75
...
```

---

## 🔧 Cara Run Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📝 Dependencies

| Library | Version | Fungsi |
|---------|---------|--------|
| streamlit | 1.32.0 | Web dashboard |
| torch | 2.2.2 | LSTM inference (CPU) |
| pandas | 2.1.4 | Data processing |
| numpy | 1.26.4 | Array operations |
| scikit-learn | 1.4.0 | Scaler, PCA |
| plotly | 5.18.0 | Interactive charts |

---

## 👨‍🔬 Author

Dibuat untuk keperluan **Tesis** — Focus Detection menggunakan EEG + BPM  
dengan arsitektur **LSTM PyTorch**.
