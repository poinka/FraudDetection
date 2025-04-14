import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.utils.class_weight import compute_class_weight
from catboost import CatBoostClassifier
import json
import os

# --- Configuration & Constants ---
DATA_DIR = 'data'
SUBMISSIONS_DIR = 'submissions'
MODELS_DIR = '.' # Models saved in root in the notebook
GA_SVM_FEATURES_FILE = 'ga_selected_features_for_svm.json'
GA_CATBOOST_FEATURES_FILE = 'ga_selected_features_for_catboost.json'
N_COMPONENTS = 30

# --- Data Loading ---
def load_data():
    """Loads preprocessed validation data and target."""
    try:
        X_val_scaled = pd.read_csv(os.path.join(DATA_DIR, "preprocessed_val.csv"))
        y_val = pd.read_csv(os.path.join(DATA_DIR, "y_val.csv")).values.ravel()
        X_train_scaled = pd.read_csv(os.path.join(DATA_DIR, "preprocessed_train.csv")) # Needed for fitting SVM Full
        y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel() # Needed for fitting SVM Full
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Make sure preprocessed files exist in '{DATA_DIR}'.")
        return None, None, None, None
    return X_train_scaled, y_train, X_val_scaled, y_val

def load_ga_features(filename):
    """Loads GA selected feature indices and names from JSON."""
    try:
        with open(filename, 'r') as f:
            feature_mapping = json.load(f)
        return feature_mapping['indices'], feature_mapping['names']
    except FileNotFoundError:
        print(f"Error: GA features file '{filename}' not found.")
        return None, None
    except KeyError:
        print(f"Error: GA features file '{filename}' is missing 'indices' or 'names' key.")
        return None, None

# --- Model Training/Loading Helpers ---
def get_class_weights(y_train):
    """Computes class weights for imbalanced data."""
    unique_classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=y_train
    )
    return {int(k): float(v) for k, v in zip(unique_classes, class_weights)}

def get_svm_predictions(X_train, y_train, X_val, class_weights_dict, reduction_method='pca', ga_indices=None):
    """Trains/Applies dimensionality reduction and gets SVM predictions."""
    print(f"Generating SVM predictions for: {reduction_method.upper()}")

    if reduction_method == 'pca':
        reducer = PCA(n_components=N_COMPONENTS, random_state=42)
    elif reduction_method == 'ica':
        reducer = FastICA(n_components=N_COMPONENTS, random_state=42)
    elif reduction_method == 'svd':
        reducer = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
    elif reduction_method == 'ga':
        if ga_indices is None:
            print("Error: GA indices required for 'ga' method.")
            return None
        X_train_reduced = X_train.iloc[:, ga_indices]
        X_val_reduced = X_val.iloc[:, ga_indices]
    else:
        print(f"Error: Unknown reduction method '{reduction_method}'")
        return None

    if reduction_method not in ['ga', 'full']:
        X_train_reduced = reducer.fit_transform(X_train)
        X_val_reduced = reducer.transform(X_val)

    # Train SVM
    # Using default parameters as in the notebook baseline, wrapped for probabilities
    base_model = LinearSVC(class_weight=class_weights_dict) # Notebook baseline defaults
    model = CalibratedClassifierCV(base_model, cv=3) # Wrap with CalibratedClassifierCV to get predict_proba
    model.fit(X_train_reduced, y_train)
    y_pred_proba = model.predict_proba(X_val_reduced)[:, 1]
    return y_pred_proba

# --- Plot 1: ROC Curves for Dimensionality Reduction (SVM) ---
def plot_roc_svm_dimensionality_reduction(X_train, y_train, X_val, y_val, ga_svm_indices):
    """Generates Plot 1: ROC curves comparing dimensionality reduction with SVM."""
    print("Generating Plot 1: SVM ROC Curves for Dimensionality Reduction...")
    plt.figure(figsize=(10, 8))

    class_weights_dict = get_class_weights(y_train)

    methods = ['pca', 'ica', 'svd', 'ga', 'full']
    labels = {
        'pca': f'PCA ({N_COMPONENTS} components)',
        'ica': f'ICA ({N_COMPONENTS} components)',
        'svd': f'SVD ({N_COMPONENTS} components)',
        'ga': f'GA-selected ({len(ga_svm_indices)} features)',
    }
    colors = {'pca': 'blue', 'ica': 'green', 'svd': 'red', 'ga': 'purple'}

    all_preds = {}
    for method in methods:
        ga_idx = ga_svm_indices if method == 'ga' else None
        preds = get_svm_predictions(X_train, y_train, X_val, class_weights_dict, reduction_method=method, ga_indices=ga_idx)
        if preds is not None:
            all_preds[method] = preds
        else:
            print(f"Could not generate predictions for {method.upper()}. Skipping.")

    # Plot ROC curves
    for method, y_pred in all_preds.items():
        fpr, tpr, _ = roc_curve(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred)
        plt.plot(fpr, tpr, color=colors[method], lw=2, label=f'{labels[method]} (AUC = {auc:.4f})')

    plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--', label='Chance (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('Validation ROC Curves: SVM with Dimensionality Reduction', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot1_svm_dimensionality_reduction_roc.png")
    print("Plot 1 saved as 'plot1_svm_dimensionality_reduction_roc.png'")
    plt.close()


# --- Plot 2: Bar Chart for Model Comparison ---
def plot_model_comparison_bars():
    """Generates Plot 2: Bar chart comparing Validation and Kaggle AUC scores."""
    print("Generating Plot 2: Model Comparison Bar Chart...")

    # Data extracted from the notebook (Validation AUCs)
    # Note: Kaggle scores for CatBoost+PSO/ACO are placeholders (np.nan) due to the identified bug.
    # Note: Using CatBoost (GA features) as the CatBoost baseline.
    configurations = [
        'SVM (GA features)', 'SVM+PSO', 'SVM+ACO',
        'CatBoost (GA features)', 'CatBoost+PSO', 'CatBoost+ACO',
    ]
    validation_auc = [
        0.828816, # SVM (GA features) - Cell db8d7506
        0.828657, # SVM+PSO - Cell c0457ddb (final eval)
        0.803286, # SVM+ACO - Cell 2b8cd73f (final eval - note: score in markdown is higher, using code output)
        0.925935, # CatBoost (GA features) - Cell 71b02543
        0.787996, # CatBoost+PSO - Cell 31454f41 (final eval)
        0.933243, # CatBoost+ACO - Cell e4e018e9 (final eval)
    ]
    # Kaggle Public Scores from notebook markdown (using np.nan for corrected ones)
    kaggle_auc = [
        0.849871, # SVM (GA features) - Cell 5f1ee265
        0.849987, # SVM+PSO - Cell 60866062
        0.826630, # SVM+ACO - Cell fcf6e3b9
        0.906293, # CatBoost (GA features) - Cell 2c86040e
        0.814647,   # CatBoost+PSO
        0.912401,   # CatBoost+ACO
    ]

    x = np.arange(len(configurations))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width/2, validation_auc, width, label='Validation AUC', color='tab:blue')
    rects2 = ax.bar(x + width/2, kaggle_auc, width, label='Kaggle Public AUC', color='tab:orange')

    ax.set_ylabel('ROC AUC Score', fontsize=12)
    ax.set_title('Model Performance Comparison: Validation vs. Kaggle Public AUC', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configurations, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0.7, 1.0) # Adjust based on score range

    # Add labels to bars
    def autolabel(rects, fmt="{:.4f}"):
        for rect in rects:
            height = rect.get_height()
            if not np.isnan(height): # Only label non-NaN bars
                ax.annotate(fmt.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig("plot2_model_comparison_auc_bars.png")
    print("Plot 2 saved as 'plot2_model_comparison_auc_bars.png'")
    plt.close()

# --- Main Execution ---
if __name__ == "__main__":
    # print("Loading data...")
    # X_train, y_train, X_val, y_val = load_data()

    # if X_val is None or y_val is None or X_train is None or y_train is None:
    #     print("Failed to load data. Exiting.")
    # else:
    #     print("Loading GA features for SVM...")
    #     ga_svm_indices, _ = load_ga_features(GA_SVM_FEATURES_FILE)

    #     if ga_svm_indices:
    #         plot_roc_svm_dimensionality_reduction(X_train, y_train, X_val, y_val, ga_svm_indices)
    #     else:
    #         print("Skipping Plot 1 due to missing GA SVM features.")

        # Plot 2 doesn't require data loading here as scores are hardcoded from notebook
        plot_model_comparison_bars()

        print("\nScript finished.")
