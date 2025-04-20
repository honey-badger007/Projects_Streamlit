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


# Title with custom styling using HTML
st.markdown("<h1 style='color: navy;'>🚗 Crash Severity Classification Dashboard</h1>", unsafe_allow_html=True)

# File upload with icons
uploaded_file = st.file_uploader("📥 Upload Training Dataset CSV", type="csv")
uploaded_file2 = st.file_uploader("📥 Upload Testing Dataset CSV", type="csv")

if uploaded_file is not None and uploaded_file2 is not None:
    df = pd.read_csv(uploaded_file)
    df2 = pd.read_csv(uploaded_file2)

    # Drop unnecessary columns
    df.drop(columns=["Distraction Level"], inplace=True)
    df2.drop(columns=["Distraction Level"], inplace=True)

    # Convert to categorical
    cols = ["Airbag Deployed", "Seatbelt Used", "Weather Conditions", "Road Conditions", 
            "Crash Type", "Vehicle Type", "Brake Condition", "Tire Condition", 
            "Severity", "Traffic Density", "Time of Day"]
    df[cols] = df[cols].astype("category")
    df2[cols] = df2[cols].astype("category")

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
    y = LabelEncoder().fit_transform(y)
    y1 = LabelEncoder().fit_transform(y1)

    # Handle class imbalance
    smote = SMOTE(k_neighbors=1, random_state=42)
    x, y = smote.fit_resample(x, y)
    x1, y1 = smote.fit_resample(x1, y1)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    # PCA (Only for KNN)
    pca = PCA(n_components=10)
    X_train_pca = pca.fit_transform(x_train)
    X_test_pca = pca.transform(x_test)
    X_test_pca1 = pca.transform(x1)

    # Models
    def train_knn():
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train_pca, y_train)  # Use PCA-transformed data
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
        xgb = XGBClassifier(objective="multi:softmax", num_class=len(set(y_train)), 
                           random_state=42, eval_metric="mlogloss")
        grid = GridSearchCV(xgb, param_grid, scoring='accuracy', cv=10, verbose=0, n_jobs=-1)
        grid.fit(x_train, y_train)  # Use original features
        return grid.best_estimator_

    def train_rf():
        rf_param_grid = {
            'n_estimators': [120],
            'max_depth': [8],
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'bootstrap': [True]
        }
        rf = RandomForestClassifier()
        grid = GridSearchCV(rf, rf_param_grid, cv=15, scoring='accuracy', n_jobs=-1, verbose=0)
        grid.fit(x_train, y_train)  # Use original features
        return grid.best_estimator_

    # Model selection with icon
    model_choice = st.selectbox("🔍 Select a model to evaluate", ["KNN", "XGBoost", "Random Forest"])

    if model_choice == "KNN":
        model = train_knn()
        X_eval = X_test_pca  # Use PCA for evaluation
        X_train_eval = X_train_pca  # Use PCA for training evaluation
    elif model_choice == "XGBoost":
        model = train_xgb()
        X_eval = x_test  # Use original features
        X_train_eval = x_train  # Use original features
    else:
        model = train_rf()
        X_eval = x_test  # Use original features
        X_train_eval = x_train  # Use original features

    # Evaluate on both training and testing sets
    def evaluate_model(model, X_train, y_train, X_test, y_test):
        try:
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Training set evaluation
            train_acc = accuracy_score(y_train, y_train_pred)
            train_f1 = f1_score(y_train, y_train_pred, average="micro")
            train_cm = confusion_matrix(y_train, y_train_pred)

            # Testing set evaluation
            test_acc = accuracy_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred, average="micro")
            test_cm = confusion_matrix(y_test, y_test_pred)

            return (train_acc, train_f1, train_cm), (test_acc, test_f1, test_cm)
        except Exception as e:
            st.error(f"Evaluation error: {str(e)}")
            return (0, 0, None), (0, 0, None)

    # Run evaluation
    (train_acc, train_f1, train_cm), (test_acc, test_f1, test_cm) = evaluate_model(
        model, X_train_eval, y_train, X_eval, y_test
    )

    # Display training and testing results
    st.subheader(f"📊 {model_choice} - Evaluation Comparison")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Accuracy", f"{train_acc:.4f}")
        st.text("Training Classification Report:")
        st.text(classification_report(y_train, model.predict(X_train_eval)))
        
    with col2:
        st.metric("Testing Accuracy", f"{test_acc:.4f}")
        st.text("Testing Classification Report:")
        st.text(classification_report(y_test, model.predict(X_eval)))

    # Plot confusion matrices
    if train_cm is not None and test_cm is not None:
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        sns.heatmap(train_cm, annot=True, fmt="d", cmap="YlGnBu", ax=ax[0])
        ax[0].set_xlabel("Predicted")
        ax[0].set_ylabel("True")
        ax[0].set_title(f"{model_choice} - Training Confusion Matrix")

        sns.heatmap(test_cm, annot=True, fmt="d", cmap="YlGnBu", ax=ax[1])
        ax[1].set_xlabel("Predicted")
        ax[1].set_ylabel("True")
        ax[1].set_title(f"{model_choice} - Testing Confusion Matrix")

        st.pyplot(fig)

    # Test File Section with icon
    st.subheader("🔎 Apply Selected Model to Test File")

    try:
        if model_choice == "KNN":
            y_pred_testfile = model.predict(X_test_pca1)  # Use PCA for KNN
        else:
            y_pred_testfile = model.predict(x1)  # Use original features for others

        acc_testfile = accuracy_score(y1, y_pred_testfile)
        f1_testfile = f1_score(y1, y_pred_testfile, average="micro")
        cm_testfile = confusion_matrix(y1, y_pred_testfile)

        st.metric("Test File Accuracy", f"{acc_testfile:.4f}")
        st.text("Test File Classification Report:")
        st.text(classification_report(y1, y_pred_testfile))

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_testfile, annot=True, fmt="d", cmap="YlGnBu", ax=ax2)
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("True")
        ax2.set_title("Test File Confusion Matrix")
        st.pyplot(fig2)

        # Show feature importance for tree-based models
        if model_choice in ["XGBoost", "Random Forest"]:
            st.subheader("📈 Feature Importance")
            if model_choice == "XGBoost":
                importance = model.feature_importances_
                features = x_train.columns
            else:
                importance = model.feature_importances_
                features = x_train.columns
                
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.barplot(x=importance, y=features, ax=ax3, palette="viridis")
            ax3.set_title("Feature Importance Scores")
            st.pyplot(fig3)

    except Exception as e:
        st.error(f"Test file prediction error: {str(e)}")

else:
    # Warning with icon
    st.warning("⚠️ Please upload both training and testing datasets.")
