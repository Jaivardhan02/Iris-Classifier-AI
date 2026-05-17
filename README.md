# 🔮 Iris Classifier AI (Streamlit Deployment Ready)

Welcome to the **Iris Classifier AI** project! This repository contains a production-ready Machine Learning application built in Python and Streamlit, complete with its training pipeline and raw dataset. It is fully pre-configured and ready to be deployed to **Streamlit Community Cloud**.

---

## 📂 Project Structure

Below is the systematic layout of the repository showing how the dataset, code, models, and diagnostic plots are organized:

```text
IRIS/
├── dataset/
│   └── Iris.csv                  # Raw flower dimensions dataset
├── project/
│   ├── app.py                    # Interactive Streamlit Web UI (Iris Classifier AI)
│   ├── pipeline.py               # ML Engine (EDA, Preprocessing, SVM training, plots)
│   ├── requirements.txt          # App dependencies (pip install list)
│   ├── models/                   # Serialized inference objects (produced by pipeline.py)
│   │   ├── best_model.pkl        # Trained Support Vector Machine model
│   │   ├── label_encoder.pkl     # Encodes target class names
│   │   └── scaler.pkl            # Scales numerical features
│   └── plots/                    # Diagnostics and analysis figures (displayed in app)
│       ├── boxplots.png
│       ├── confusion_matrix.png
│       ├── correlation_matrix.png
│       ├── distributions.png
│       ├── learning_curve.png
│       └── roc_curve.png
└── README.md                     # Project documentation (this file)
```

---

## 💻 Local Setup & Execution

### 1. Install Dependencies
Ensure you have Python 3.8+ installed, then open your terminal and install the requirements:
```bash
cd project
pip install -r requirements.txt
```

### 2. Train the Model (Optional)
To run the Exploratory Data Analysis, generate diagnostic plots, and re-train the SVM model:
```bash
python pipeline.py
```
*Note: This generates or updates the `.pkl` files in the `models/` directory and `.png` plots in the `plots/` directory.*

### 3. Launch the App
To start the Streamlit web server on your local machine:
```bash
streamlit run app.py
```
This opens the dynamic "Iris Classifier AI" dashboard in your default browser at `http://localhost:8501`.

---

## 🚀 Quick Deploy to Streamlit Cloud

Since the pre-trained models and visualization assets are fully version-tracked within the repository, deploying this live to the web takes less than a minute:

1. **Push to GitHub**: Push this entire `IRIS` folder structure to a public repository on your GitHub account.
2. **Log into Streamlit**: Head over to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. **Configure App**:
   - **Repository**: Select your repository (e.g., `Iris-Classifier-AI`).
   - **Branch**: `main`
   - **Main file path**: Type `project/app.py` (since the app sits in the `project/` subfolder).
4. **Deploy**: Click the **Deploy!** button.

---

## 🧠 Machine Learning Engine Stats
- **Core Algorithm**: Support Vector Machine (SVM) with RBF kernel.
- **Accuracy Metric**: 96.67%
- **Overfitting Prevention**: Utilized `GridSearchCV` to optimize the `C` regularization parameter, successfully balancing the bias-variance trade-off on this small dataset without over-memorizing training data.

---
*Developed & Maintained by [Jaivardhan02](https://github.com/Jaivardhan02)*
