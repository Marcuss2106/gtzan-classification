# GTZAN Binary Classification with Simple Linear Models

This project explores the behavior of simple linear classifiers — Perceptron, Adaline with Batch Gradient Descent (BGD), and Adaline with Stochastic Gradient Descent (SGD) — on the [GTZAN genre classification dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/).

While the GTZAN dataset is originally for music genres, this study uses it as a controlled environment for audio classification experimentation. The long-term vision is to apply these insights toward dialect-aware Text-to-Speech (TTS) systems, where understanding audio classification dynamics is critical.

---

## Notebook
You can view the Analysis through a Jupyter Notebook here: https://nbviewer.org/github/Marcuss2106/gtzan-classification/blob/master/analysis.ipynb

--- 

## 📌 Project Goals
- Analyze how basic linear models learn from audio features
- Compare learning dynamics: convergence speed, decision boundaries, and classification accuracy
- Build intuition on trade-offs between model simplicity, precision, and training efficiency
- Lay foundational insights for future work in dialect-aware TTS modeling

---

## Setup Instructions (Using pip)
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
jupyter notebook analysis.ipynb
```

---

## 📂 Project Structure

```bash
common-voice-eda/
├── data/
│   └── features_30_sec.csv   # Sourced from Kaggle
├── analysis.ipynb            # Main Jupyter Notebook for Analysis
├── main.py   				  # Reusable script version
├── ml_algorithms.py		  # Module containing classes for ML models
├── plots/                    # Saved charts and figures
└── README.md                 # You're here!
```

---

## 🛠️ Tools Used

- Python 3.13
- pandas
- matplotlib
- numpy
- Jupyter Notebook

---

## Models Implemented
| Model           | Highlights                                                                    |
| --------------- | ----------------------------------------------------------------------------- |
| **Perceptron**  | Slow convergence (\~190 epochs) but perfect boundary when separable.          |
| **Adaline BGD** | Quick, smooth convergence (\~10 epochs); boundary slightly off.               |
| **Adaline SGD** | Fastest convergence (\~2 epochs); minor oscillations; near-accurate boundary. |

---

## Data Source & License
This project includes data from the GTZAN Genre Collection
Data is provided for research and non-commercial use. If used, please cite:
> G. Tzanetakis and P. Cook, "Musical Genre Classification of Audio Signals," IEEE Transactions on Speech and Audio Processing, 2002.
Source: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/