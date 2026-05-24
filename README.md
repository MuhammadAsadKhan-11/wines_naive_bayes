# 🍷 Wine Classification using Naive Bayes

A machine learning project that classifies wine types using two Naive Bayes variants — **GaussianNB** and **MultinomialNB** — with comprehensive exploratory data analysis and model evaluation visualizations.

---

## 📌 Project Overview

This project applies the Naive Bayes algorithm to the classic UCI Wine dataset to classify wines into three categories based on 13 chemical properties. The workflow covers the full ML pipeline: data exploration, preprocessing, model training, evaluation, and visualization.

---

## 📁 Project Structure

```
wines_naive_bayes/
│
├── wine_naive_byes.ipynb              # Main Jupyter Notebook
├── model_pickle                       # Serialized trained models
├── wine_naivebayes_report.pdf         # All visualizations compiled as a report
│
├── Before Prediction (EDA)
│   ├── class_distribution.png         # Class balance bar chart
│   ├── feature_correlation.png        # Heatmap of feature correlations
│   ├── pairplot_of_key_feature.png    # Pairplot of top 4 features
│   ├── feature_distribution_by_class.png  # Boxplots per class
│   └── PCA_2D_projection_of_wine.png  # 2D PCA scatter plot
│
└── After Prediction (Evaluation)
    ├── GaussianNb_confusion_matrix.png
    ├── MultinomialNB_Confustion_Matrix.png
    ├── model_Accuracy_comparision.png
    └── per-class_Accuracy.png
```

---

## 📊 Dataset

- **Source:** `sklearn.datasets.load_wine()`
- **Samples:** 178 wine samples
- **Features:** 13 chemical attributes (alcohol, malic acid, ash, flavanoids, proline, etc.)
- **Classes:** 3 wine cultivars (Class 0, Class 1, Class 2)
- **Train/Test Split:** 70% / 30% (`random_state=42`)

---

## 🧪 Models Used

| Model | Preprocessing | Test Accuracy |
|---|---|---|
| **GaussianNB** | StandardScaler | **100%** |
| **MultinomialNB** | MinMaxScaler | **96.3%** |

Both models were wrapped in `sklearn.pipeline.Pipeline` for clean preprocessing and inference.

---

## 📈 Visualizations

### Before Prediction — Exploratory Data Analysis

| Plot | Purpose |
|---|---|
| Class Distribution | Verify class balance across 3 wine types |
| Feature Correlation Heatmap | Identify multicollinearity (e.g., flavanoids & total_phenols: 0.86) |
| Pairplot (4 key features) | Visual separability between wine classes |
| Boxplots (all 13 features) | Feature spread and outliers per class |
| PCA 2D Projection | Dimensionality reduction to confirm class separability |

### After Prediction — Model Evaluation

| Plot | Purpose |
|---|---|
| Confusion Matrix (GaussianNB) | Perfect diagonal — 0 misclassifications |
| Confusion Matrix (MultinomialNB) | 2 misclassifications (class_0 ↔ class_1) |
| Accuracy Comparison | GaussianNB (1.000) vs MultinomialNB (0.963) |
| Per-Class Accuracy | Class-level breakdown for both models |

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn
- **Environment:** Jupyter Notebook
- **Model Persistence:** pickle

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/wines_naive_bayes.git
   cd wines_naive_bayes
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. **Launch the notebook**
   ```bash
   jupyter notebook wine_naive_byes.ipynb
   ```

4. **Run all cells** — figures will be saved automatically alongside the PDF report.

---

## 💡 Key Findings

- **GaussianNB achieved perfect accuracy (100%)** on the test set, making it the superior choice for this dataset. Its assumption of normally distributed features aligns well with the wine data's chemical properties.
- **MultinomialNB achieved 96.3% accuracy** — a strong result, though it confused 2 samples between Class 0 and Class 1, likely because MinMaxScaling can distort the count-like feature distributions that MultinomialNB expects.
- The **PCA projection** revealed that PC1 alone captures 99.8% of the variance, strongly driven by the `proline` feature.
- **Flavanoids and total_phenols** show the highest correlation (0.86), confirming feature redundancy in the dataset.

---

## 📌 Note on Saving Figures

When saving matplotlib figures, always call `plt.savefig()` **before** `plt.show()`. Calling it after clears the figure buffer and produces a blank image.

```python
# ✅ Correct
plt.tight_layout()
plt.savefig('plot.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 👤 Author

**[Your Name]**
- GitHub: [MuhammadAsadkhan-11](https://github.com/MuhammadAsadkhan-11)
- LinkedIn: [Linkedln](www.linkedin.com/in/muhammad-asad-khan-9a70243a3)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
