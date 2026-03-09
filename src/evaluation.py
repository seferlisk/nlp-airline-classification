import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

class Evaluator:
    """Calculates F1, Precision, Recall and plots Confusion Matrix."""

    def __init__(self, label_encoder):
        self.label_encoder = label_encoder

    def plot_loss_curve(self, train_losses, val_losses):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
        plt.plot(val_losses, label='Validation Loss', color='orange', linewidth=2)
        plt.title('Model Loss Curve (Monitoring for Overfitting)')
        plt.xlabel('Epochs')
        plt.ylabel('Cross Entropy Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    def display_metrics(self, y_true, y_pred):
        # Get the names of the classes (Negative, Neutral, Positive)
        class_names = self.label_encoder.classes_

        # 1. Classification Report (Precision, Recall, F1)
        print("\n--- Detailed Classification Report ---")
        print(classification_report(y_true, y_pred, target_names=class_names))

        # 2. Confusion Matrix Heatmap
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('Confusion Matrix: Predicted vs Actual')
        plt.ylabel('Actual Sentiment')
        plt.xlabel('Predicted Sentiment')
        plt.show()