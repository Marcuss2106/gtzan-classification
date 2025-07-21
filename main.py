import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import os

from ml_algorithms import *

# set up figure
fig = plt.figure(figsize=(12, 9))
gs = gridspec.GridSpec(3, 3, figure=fig)
ax_top = fig.add_subplot(gs[0, 1])
ax_top.set_title("Root Mean Squared vs Chroma Short-Time Fourier Transform")
ax_bottom = []
for col in range(3):
    for row in range(1, 3):
        ax = fig.add_subplot(gs[row, col])
        ax_bottom.append(ax)
plt.tight_layout()
plt.subplots_adjust(left=0.05, bottom=0.05, wspace=0.2, hspace=0.5)
ax_bottom[0].set_title("Perceptron")
ax_bottom[2].set_title("Adaline Batch Gradient Descent")
ax_bottom[4].set_title("Adaline Stochastic Gradient Descent")

GTZAN_FILE_PATH = "./data/features_30_sec.csv"

assert os.path.exists(GTZAN_FILE_PATH), f"File not found: {GTZAN_FILE_PATH}"

features = [
    "chroma_stft_mean",
    "rms_mean",
]
targets = [
    "classical",
    "metal",
]


def main():
    print("Hello from gtzan-classification!")
    df = load_data(GTZAN_FILE_PATH)
    df = clean_data(df)
    print(df.head())
    y = select_targets(df)
    X = select_features(df, features)
    # standardize features for Adaline models
    X_std = standardize_features(X)

    # Plot original graph
    ax_top.scatter(X[:100, 0], X[:100, 1], color="r", marker="o", label=targets[0])
    ax_top.scatter(X[100:, 0], X[100:, 1], color="b", marker="o", label=targets[1])
    ax_top.set_xlabel(features[0])
    ax_top.set_ylabel(features[1])
    ax_top.legend(loc="upper left")

    # Train models and plot
    train_perceptron(X, y)
    train_AdalineBGD(X_std, y)
    train_AdalineSGD(X_std, y)
    plt.show()


def standardize_features(X) -> np.ndarray:
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    return X_std


def train_perceptron(X, y):
    ppn = Perceptron(eta=0.001, n_epoch=200)
    ppn.fit(X, y)
    plot_epochs(ppn.errors_, "Number of Updates", ax_bottom[0])
    plot_decision_regions(X, y, ppn, ax_bottom[1], 0.1)


def train_AdalineBGD(X, y):
    ada_bgd = AdalineBGD(eta=0.5, n_epoch=20)
    ada_bgd.fit(X, y)
    plot_epochs(ada_bgd.losses_, "Mean Squared Error", ax_bottom[2])
    plot_decision_regions(X, y, ada_bgd, ax_bottom[3], 1)


def train_AdalineSGD(X, y):
    ada_sgd = AdalineSGD(eta=0.01, n_epoch=10)
    ada_sgd.fit(X, y)
    plot_epochs(ada_sgd.losses_, "Mean Squared Error", ax_bottom[4])
    plot_decision_regions(X, y, ada_sgd, ax_bottom[5], 1)


def select_targets(df: pd.DataFrame) -> np.ndarray:
    y = df["label"].values
    y = np.where(y == targets[0], 0, 1)
    return y


def select_features(df: pd.DataFrame, features: list) -> np.ndarray:
    X = df[features].values
    return X


def plot_epochs(errors, ylabel, ax):
    ax.plot(range(1, len(errors) + 1), errors, marker="o")
    ax.set_xlabel("Epochs")
    ax.set_ylabel(ylabel)


def plot_decision_regions(X, y, classifier, ax, padding, resolution=0.001):
    markers = ("o", "s", "^", "v", "<")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    x2_min, x2_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    )
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    ax.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=f"Class {cl} ({targets[idx]})",
            edgecolor="black",
        )

    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.legend(loc="upper left")


def load_data(file_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    features_to_keep = [
        "filename",
        "chroma_stft_mean",
        "rms_mean",
        "spectral_centroid_mean",
        "spectral_bandwidth_mean",
        "rolloff_mean",
        "zero_crossing_rate_mean",
        "tempo",
        "mfcc1_mean",
        "mfcc2_mean",
        "mfcc3_mean",
        "mfcc4_mean",
        "mfcc5_mean",
        "label",
    ]
    if df.empty:
        print("DataFrame is empty.")
        return df
    df = df[df["label"].isin(["metal", "classical"])]
    df = df[features_to_keep]
    return df.reset_index(drop=True)


if __name__ == "__main__":
    main()
