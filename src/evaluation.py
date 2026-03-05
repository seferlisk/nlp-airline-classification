import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

class Evaluator:
    """Calculates F1, Precision, Recall and plots Confusion Matrix."""

    @staticmethod
    def plot_loss(train_losses, val_losses):
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.legend();
        plt.title("Loss Curve");
        plt.show()

    @staticmethod
    def report(y_true, y_pred, target_names):
        print(classification_report(y_true, y_pred, target_names=target_names))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names)
        plt.title("Confusion Matrix");
        plt.show()