import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(confmat, class_names, save_path="confusion_matrix.png", title="Matrice de confusion"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(5, 4))
    sns.heatmap(confmat, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, cbar=False,
                annot_kws={"size": 16, "weight": "bold"})
    plt.xlabel("Prédiction", fontsize=14)
    plt.ylabel("Vérité", fontsize=14)
    plt.title(title, fontsize=16, weight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()