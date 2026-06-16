# 🎙️ Age Estimation from Speech — A Data Science Approach

> **Course:** Data Science Lab — Politecnico di Torino  
> **Authors:** Elias Noorzad (s314515) · Parastoo Alavi (s340942)

---

## 📌 Project Overview

This project builds a machine learning system to **predict a speaker's age** from acoustic and linguistic features extracted from speech recordings. The task is framed as a **regression problem**, where the target variable is the speaker's age as a continuous numerical value.

The dataset contains **3,624 samples** in total:
- `development.csv` — 2,933 labeled samples used for training and validation
- `evaluation.csv` — 691 unlabeled samples used for final evaluation

---

## 📂 Repository Structure

```
├── Task.ipynb                  # Data preprocessing pipeline
├── Project_-_RF_regressor.ipynb  # Model training, tuning & evaluation
├── development.csv             # Labeled development dataset
├── evaluation.csv              # Unlabeled evaluation dataset
├── Project_s_report.pdf        # Full project report
└── README.md
```

---

## 🧠 Features Used

### Acoustic Features
| Feature | Description |
|---|---|
| `mean_pitch`, `max_pitch`, `min_pitch` | Vocal frequency variations (Hz) |
| `jitter` | Pitch instability |
| `shimmer` | Amplitude instability |
| `energy` | Overall power of the speech signal |
| `zcr_mean` | Zero-crossing rate — tonal changes |
| `spectral_centroid_mean` | Center of mass of the frequency spectrum |
| `hnr` | Harmonic-to-noise ratio — voice quality |
| `silence_duration` | Total silence time (seconds) |

### Linguistic Features
| Feature | Description |
|---|---|
| `num_words` | Number of words spoken |
| `num_characters` | Total character count |
| `num_pauses` | Number of pauses |
| `tempo` | Speaking rate (BPM) |

### Demographic Metadata
| Feature | Description |
|---|---|
| `gender` | Speaker's gender |
| `ethnicity` | Speaker's ethnic background |

---

## ⚙️ Preprocessing Pipeline (`Task.ipynb`)

1. **Feature Selection** — Dropped non-informative columns: `Id`, `path`, `sampling_rate`
2. **Data Cleaning** — Fixed data type issues (`tempo` → float) and corrected a typo in the evaluation set (`"famale"` → `"female"`)
3. **Train/Test Split** — 80% training / 20% test (`random_state=42`)
4. **Z-score Normalization** — Applied `fit_transform` on training set, `transform` only on test/evaluation sets to prevent data leakage
5. **One-Hot Encoding** — Encoded `gender` and `ethnicity`; final feature count expanded to **167 columns**

---

## 🤖 Models & Results (`Project_-_RF_regressor.ipynb`)

Two regression models were trained and compared using **RMSE** as the evaluation metric.

### K-Nearest Neighbors (KNN)
| k | Test RMSE |
|---|---|
| 3 | 11.894 |
| 10 | 10.604 |
| **30** | **10.365** ✅ |
| 40 | 10.458 |

- **Best config:** `k = 30`
- **Evaluation set RMSE:** `10.539`

### Random Forest Regressor
| n_estimators | max_depth | min_sample_split | Test RMSE |
|---|---|---|---|
| 100 | None | 2 | 10.043 |
| 200 | None | 2 | 10.003 |
| **500** | **30** | **3** | **9.936** ✅ |
| 500 | 50 | 3 | 9.939 |

- **Best config:** `n_estimators=500`, `max_depth=30`, `min_sample_split=3`, `random_state=42`
- **Evaluation set RMSE:** `10.215`

### 🏆 Winner: Random Forest
Random Forest outperformed KNN on the evaluation set (**RMSE 10.215 vs 10.539**), showing better generalization and robustness.

---

## 📊 Feature Importance

The top predictors identified by the Random Forest model:

1. `silence_duration` — most important
2. `spectral_centroid_mean`
3. `jitter`
4. `num_pauses`
5. `hnr`

Gender and certain ethnicity categories contributed minimally to predictions.

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/age-estimation-from-speech.git
   cd age-estimation-from-speech
   ```

2. Install dependencies:
   ```bash
   pip install pandas scikit-learn matplotlib
   ```

3. Run the notebooks in order:
   - `Task.ipynb` — preprocessing
   - `Project_-_RF_regressor.ipynb` — model training & evaluation

> ⚠️ Update the file paths inside the notebooks to point to your local copy of `development.csv` and `evaluation.csv`.

---

## 📚 References

1. C. T. Ferrand, "Harmonics-to-noise ratio: An index of vocal aging," *Journal of Voice*, 2002.
2. D. Jiao et al., "Age estimation in foreign-accented speech," *Speech Communication*, 2019.
3. J. E. Huber, "Effects of utterance length and vocal loudness on speech breathing in older adults," *Respiratory Physiology and Neurobiology*, 2008.
4. N. S. Altman, "An introduction to kernel and nearest-neighbor nonparametric regression," *The American Statistician*, 1992.
5. L. Breiman, "Random forests," *Machine Learning*, 2001.
