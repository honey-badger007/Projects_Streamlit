import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


with open("Project_Car_crashes/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🚗 Crash Severity Classification Dashboard")

# File upload
uploaded_file = st.file_uploader("📥 Upload Training Dataset CSV", type="csv")
uploaded_file2 = st.file_uploader("📥 Upload Testing Dataset CSV", type="csv")

if uploaded_file is not None and uploaded_file2 is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    df2 = pd.read_csv(uploaded_file2)
    
    # Preprocessing function
    def preprocess_data(df):
        df = df.copy()
        df.drop(columns=["Distraction Level"], inplace=True, errors='ignore')
        
        # Convert to categorical
        cols = ["Airbag Deployed", "Seatbelt Used", "Weather Conditions", "Road Conditions", 
                "Crash Type", "Vehicle Type", "Brake Condition", "Tire Condition", 
                "Severity", "Traffic Density", "Time of Day"]
        df[cols] = df[cols].astype("category")
        
        return df
    
    df = preprocess_data(df)
    df2 = preprocess_data(df2)
    
    # Prepare features and target
    x = df.drop(columns="Severity")
    y = df["Severity"]
    x1 = df2.drop(columns="Severity")
    y1 = df2["Severity"]
    
    # Ordinal Encoding
    categorical_columns = ["Traffic Density", "Time of Day", "Tire Condition", "Brake Condition", 
                          "Vehicle Type", "Crash Type", "Road Conditions", "Weather Conditions", 
                          "Airbag Deployed", "Seatbelt Used"]
    
    oe = OrdinalEncoder()
    x[categorical_columns] = oe.fit_transform(x[categorical_columns])
    x1[categorical_columns] = oe.transform(x1[categorical_columns])
    
    # Label Encoding
    le = LabelEncoder()
    y = le.fit_transform(y)
    y1 = le.transform(y1)
    
    # Handle class imbalance
    smote = SMOTE(k_neighbors=1, random_state=42)
    x, y = smote.fit_resample(x, y)
    x1, y1 = smote.fit_resample(x1, y1)
    
    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    
    # Store feature names
    feature_names = x_train.columns.tolist()
    
    # PCA (Only for KNN)
    pca = PCA(n_components=10)
    X_train_pca = pca.fit_transform(x_train)
    X_test_pca = pca.transform(x_test)
    X_test_pca1 = pca.transform(x1)
    
    # Model training functions
    def train_knn():
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train_pca, y_train)
        return knn
    
    def train_xgb():
        param_grid = {
            "n_estimators": [100],
            "max_depth": [4],
            "learning_rate": [0.13],
            "subsample": [0.8],
            "colsample_bytree": [0.7],
            "gamma": [0.01],
            "scale_pos_weight": [1],
            "reg_lambda": [0, 1]
        }
        xgb = XGBClassifier(objective="multi:softmax", num_class=len(np.unique(y_train)), 
                           random_state=42, eval_metric="mlogloss")
        grid = GridSearchCV(xgb, param_grid, scoring='accuracy', cv=10, verbose=0, n_jobs=-1)
        grid.fit(x_train, y_train)
        return grid.best_estimator_
    
    def train_rf():
        rf_param_grid = {
            'n_estimators': [200],
            'max_depth': [10],
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'bootstrap': [True]
        }
        rf = RandomForestClassifier(random_state=42)
        grid = GridSearchCV(rf, rf_param_grid, cv=15, scoring='accuracy', n_jobs=-1, verbose=0)
        grid.fit(x_train, y_train)
        return grid.best_estimator_
    
    # Model selection
    model_choice = st.selectbox("🔍 Select a model to evaluate", ["KNN", "XGBoost", "Random Forest"])
    
    try:
        if model_choice == "KNN":
            model = train_knn()
            X_eval = X_test_pca
            X_train_eval = X_train_pca
        elif model_choice == "XGBoost":
            model = train_xgb()
            X_eval = x_test
            X_train_eval = x_train
        else:
            model = train_rf()
            X_eval = x_test
            X_train_eval = x_train
        
        # Evaluation function
        def evaluate_model(model, X_train, y_train, X_test, y_test):
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_acc = accuracy_score(y_train, y_train_pred)
            train_f1 = f1_score(y_train, y_train_pred, average="micro")
            train_cm = confusion_matrix(y_train, y_train_pred)
            
            test_acc = accuracy_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred, average="micro")
            test_cm = confusion_matrix(y_test, y_test_pred)
            
            return (train_acc, train_f1, train_cm), (test_acc, test_f1, test_cm)
        
        # Run evaluation
        (train_acc, train_f1, train_cm), (test_acc, test_f1, test_cm) = evaluate_model(
            model, X_train_eval, y_train, X_eval, y_test
        )
        
        # Display results
        st.subheader(f"📊 {model_choice} - Evaluation Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Accuracy", f"{train_acc:.4f}")
            st.text("Training Classification Report:")
            st.text(classification_report(y_train, model.predict(X_train_eval)))
        
        with col2:
            st.metric("Testing Accuracy", f"{test_acc:.4f}")
            st.text("Testing Classification Report:")
            st.text(classification_report(y_test, model.predict(X_eval)))
        
        # Confusion matrices
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        sns.heatmap(train_cm, annot=True, fmt="d", cmap="Blues", ax=ax[0])
        ax[0].set_title("Training Confusion Matrix")
        sns.heatmap(test_cm, annot=True, fmt="d", cmap="Blues", ax=ax[1])
        ax[1].set_title("Testing Confusion Matrix")
        st.pyplot(fig)
        
        # Test file evaluation
        st.subheader("🔎 Test File Evaluation")
        
        if model_choice == "KNN":
            y_pred_test = model.predict(X_test_pca1)
        else:
            y_pred_test = model.predict(x1)
        
        test_acc = accuracy_score(y1, y_pred_test)
        test_f1 = f1_score(y1, y_pred_test, average="micro")
        test_cm = confusion_matrix(y1, y_pred_test)
        
        st.metric("Test File Accuracy", f"{test_acc:.4f}")
        st.text("Test File Classification Report:")
        st.text(classification_report(y1, y_pred_test))
        
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(test_cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
        ax2.set_title("Test File Confusion Matrix")
        st.pyplot(fig2)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your data and model parameters")
else:
    st.warning("⚠️ Please upload both training and testing datasets.")
