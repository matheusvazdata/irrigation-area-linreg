from pathlib import Path
import matplotlib.pyplot as plt

def save_scatter(x, y, xlabel, ylabel, title, outpath: Path):
    plt.figure(figsize=(8,6))
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def save_line(y_true, y_pred, title, outpath: Path):
    plt.figure(figsize=(10,6))
    plt.plot(range(len(y_true)), y_true, label="True")
    plt.plot(range(len(y_pred)), y_pred, label="Predicted")
    plt.title(title)
    plt.xlabel("Sample index (test set)")
    plt.ylabel("Irrigated Area per Angle")
    plt.grid(True, alpha=0.3)
    plt.legend()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def save_hist(data, title, xlabel, outpath: Path, bins=30):
    plt.figure(figsize=(8,6))
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()