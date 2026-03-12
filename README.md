# ✈️ US Airline Sentiment Classification (ANN)

This repository contains a deep learning project for the classification of **US Airline tweets** into three sentiment categories: **Negative, Neutral, and Positive**. The project implements a custom NLP pipeline and a Multi-Layer Feed-Forward Neural Network (ANN) using **PyTorch**.

---

## 🚀 Project Overview
The goal is to analyze customer feedback on Twitter to help airlines understand their service quality. Given the inherent "noise" in social media data and the significant class imbalance (high volume of negative tweets), this project focuses on:
* **Robust Preprocessing**
* **Imbalance Management**
* **Evaluation via F1-Score**



---

## 🛠️ Key Features
* **Object-Oriented Design:** The code is structured into modular classes: `TextPreprocessor`, `FeatureExtractor`, `SentimentANN`, `ModelTrainer`, and `Evaluator`.
* **Advanced Preprocessing:** Includes URL/Handle removal, custom stopword filtering (preserving negations like "not"), and WordNet lemmatization.
* **Vectorization:** Implements **TF-IDF** to identify high-signal keywords across 5,000 features.
* **Deep Learning Architecture:**
    * Multi-layer perceptron with **Batch Normalization** for training stability.
    * **Dropout layers** (0.3 and 0.2) to mitigate overfitting.
    * **Early Stopping** based on validation loss monitoring.

---

## 📁 Project Structure
* `Tweets.csv`: The raw dataset containing ~14.6k labeled tweets.
* `Sentiment_Analysis.ipynb`: The main Jupyter Notebook containing the full analysis and training pipeline.
* `best_model_weights.pth`: The saved weights from the best training epoch.
* `Outputs/`: Directory containing generated plots and reports.

---

## 📊 Performance Metrics
Since the dataset is imbalanced, the model is evaluated using:
* **Weighted F1-Score** (Primary Metric)
* **Precision & Recall**
* **Confusion Matrix Heatmap**
* **Loss Curves** (Training vs. Validation)



---

## ⚙️ Requirements
To run this project, you will need:
* `Python 3.x`
* `PyTorch`
* `Scikit-Learn`
* `Pandas` / `Numpy`
* `NLTK`
* `Matplotlib` / `Seaborn`

---

## 🚦 Usage
1.  **Clone the repository.**
2.  Ensure `Tweets.csv` is in the Datasets directory.
3.  Run the `Sentiment_Analysis.ipynb` notebook to execute the pipeline from cleaning to evaluation.