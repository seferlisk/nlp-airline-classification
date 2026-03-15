import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

class Evaluator:
    """Calculates F1, Precision, Recall and plots Confusion Matrix."""

    def __init__(self, label_encoder, output_dir='Outputs'):
        self.label_encoder = label_encoder
        self.root_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
        self.output_dir = os.path.join(self.root_dir, output_dir)

        # Ensure the Outputs folder exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def plot_loss_curve(self, train_losses, val_losses):
        """Generates a line graph comparing Training vs. Validation loss over time."""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
        plt.plot(val_losses, label='Validation Loss', color='orange', linewidth=2)
        plt.title('Model Loss Curve (Monitoring for Overfitting)')
        plt.xlabel('Epochs')
        plt.ylabel('Cross Entropy Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # Save the plot
        save_path = os.path.join(self.output_dir, 'loss_curve.png')
        plt.savefig(save_path)
        print(f"Loss curve saved to: {save_path}")

        plt.show()

    def display_metrics(self, y_true, y_pred):
        """Generates a Classification Report (Precision, Recall, F1).
           Creates a Confusion Matrix Heatmap to show misclassification trends.
           Saves all outputs (images and text) to the Outputs/ directory for reporting."""
        # Get the names of the classes (Negative, Neutral, Positive)
        class_names = self.label_encoder.classes_

        # 1. Classification Report (Precision, Recall, F1)
        print("\n--- Detailed Classification Report ---")
        report = classification_report(y_true, y_pred, target_names=class_names)
        print(report)

        # Save the report to a text file
        with open(os.path.join(self.output_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)

        # 2. Confusion Matrix Heatmap
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('Confusion Matrix: Predicted vs Actual')
        plt.ylabel('Actual Sentiment')
        plt.xlabel('Predicted Sentiment')

        # Save the matrix
        save_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(save_path)
        print(f"Confusion matrix saved to: {save_path}")

        plt.show()

