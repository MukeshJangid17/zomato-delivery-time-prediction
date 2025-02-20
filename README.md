# Zomato Delivery Time Prediction

## 🚀 Project Overview
This project aims to predict the delivery time for Zomato orders using machine learning. The model considers multiple factors such as traffic conditions, weather, vehicle type, and order details to provide accurate delivery time estimates.

The project follows **MLflow** for experiment tracking and **DVC (Data Version Control)** for managing datasets stored in **Google Drive**. The model is deployed as a **Streamlit web application**, making it easy for users to input order details and get predictions.

---

## 📌 Tech Stack
- **Python** (pandas, scikit-learn, joblib, mlflow, dvc)
- **Machine Learning** (Stacking Regressor, Data Preprocessing)
- **Streamlit** (Web App for Model Deployment)
- **MLflow & DagsHub** (Model Tracking & Registry)
- **DVC & Google Drive** (Dataset Versioning)
- **Pipeline Framework** (Sklearn Pipeline for model deployment)

---

## 🔧 Installation & Setup
Follow these steps to set up the project on your local machine:

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/mukeshjangid7877/zomato-delivery-time-prediction.git
cd zomato-delivery-time-prediction
```

### 2️⃣ Create & Activate a Virtual Environment
```sh
python -m venv venv
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate  # For Windows
```

### 3️⃣ Install Dependencies
```sh
pip install -r requirements-dev.txt
```

### 4️⃣ Configure DVC & Pull Data from Google Drive
```sh
dvc pull
```
*(Ensure you have access to the dataset stored in Google Drive.)*

### 5️⃣ Run the Streamlit App
```sh
streamlit run streamlit_app.py
```
The web app will open in your browser at `http://localhost:8501/`.

---

## 🏆 Key Features & Skills Demonstrated
✅ **Data Engineering**: Feature engineering, preprocessing, and dataset versioning with **DVC**.
✅ **Machine Learning**: Trained a **Stacking Regressor** model with optimized hyperparameters.
✅ **Model Tracking & Deployment**: Used **MLflow & DagsHub** for experiment tracking and model registry.
✅ **Web App Development**: Built an interactive **Streamlit** application for real-time delivery time prediction.
✅ **Reproducibility & Version Control**: Managed dataset and code versions using **DVC & Git**.

---

## 📬 Contact
If you have any questions or would like to discuss my work, feel free to connect with me:
- **GitHub**: [mukeshjangid7877](https://github.com/MukeshJangid17)
- **LinkedIn**: [Your LinkedIn Profile](www.linkedin.com/in/mukesh-jangid-825b7b240)
- **Email**: mukeshjangid727680@gmail.com











_______________________________________________________________________
THE PROJECT STRUCTURE , TO UNDERSTAND THE FIL SYSTEM BETTER 

Zomato-Delivery-Time-Prediction
==============================

this model predicts the delivery time

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
