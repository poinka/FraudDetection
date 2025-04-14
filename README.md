# Fraud Detection using Machine Learning and Nature-Inspired Computing

This project explores the effectiveness of different machine learning classifiers (Support Vector Machines - SVM and CatBoost) combined with various dimensionality reduction and feature selection techniques for detecting fraudulent transactions. It leverages the IEEE-CIS Fraud Detection dataset from Kaggle. A key focus is comparing classical dimensionality reduction methods (PCA, ICA, Truncated SVD) against a nature-inspired approach (Genetic Algorithm - GA) for feature selection, and further optimizing classifier hyperparameters using Particle Swarm Optimization (PSO) and Ant Colony Optimization (ACO).

## Table of Contents

1.  [Introduction](#introduction)
2.  [Dataset](#dataset)
3.  [Methodology](#methodology)
    *   [Data Preprocessing](#data-preprocessing)
    *   [Dimensionality Reduction & Feature Selection](#dimensionality-reduction--feature-selection)
    *   [Classification Models](#classification-models)
    *   [Hyperparameter Optimization](#hyperparameter-optimization)
4.  [Implementation Details](#implementation-details)
5.  [Results](#results)
    *   [SVM Performance](#svm-performance)
    *   [CatBoost Performance](#catboost-performance)
    *   [Comparative Analysis](#comparative-analysis)
    *   [Feature Importance](#feature-importance)
6.  [Conclusion](#conclusion)

## Introduction

The goal of this project is to build an effective fraud detection model by comparing the performance of SVM and CatBoost classifiers under different feature engineering scenarios. We investigate:
*   The impact of classical dimensionality reduction techniques (PCA, ICA, SVD).
*   The effectiveness of a Genetic Algorithm (GA) for selecting an optimal feature subset.
*   The ability of Particle Swarm Optimization (PSO) and Ant Colony Optimization (ACO) to fine-tune classifier hyperparameters for improved performance, particularly when combined with GA-selected features.

## Dataset

The project utilizes the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection).

*   **Source:** Kaggle Competition.
*   **Size:** Contains 590,540 transaction records.
*   **Features:** Over 400 features, split into transaction details (`train_transaction.csv`, `test_transaction.csv`) and identity information (`train_identity.csv`, `test_identity.csv`). Features include transaction amount, time, product codes, device types, email domains, etc.
*   **Target Variable:** `isFraud` (binary: 0 for non-fraudulent, 1 for fraudulent).
*   **Challenge:** Highly imbalanced dataset (approx. 3.5% fraudulent transactions) and high dimensionality with many missing values.

## Methodology

### Data Preprocessing

1.  **Merging:** Transaction and identity datasets were merged based on `TransactionID`.
2.  **Missing Value Imputation:**
    *   **Numeric Features:** Imputed with a constant value of 0 (acting as an indicator for missingness where 0 wasn't a natural value).
    *   **Categorical Features:** Imputed with the string 'missing'.
    *   Columns with >90% missing values were initially kept for GA to potentially evaluate.
3.  **Feature Engineering:**
    *   `id_33` (screen resolution) was split into `Screen_Width` and `Screen_Height`.
    *   Numeric columns with very few unique values (<20) were identified and treated as categorical.
4.  **Encoding:**
    *   **Categorical Features:** Weight of Evidence (WOE) encoding was used to handle high cardinality and prepare data for SVM.
5.  **Scaling:**
    *   StandardScaler was applied to all features after encoding and imputation.
6.  **Train/Validation Split:** Data was split into 80% training and 20% validation sets, stratified by the `isFraud` target variable.
7.  **Class Imbalance Handling:** `class_weight='balanced'` was used for both SVM and CatBoost during training to address the severe class imbalance.

### Dimensionality Reduction & Feature Selection

A fixed target of 30 features/dimensions was set for comparison.

1.  **Principal Component Analysis (PCA):** Standard PCA applied to reduce dimensions to 30 components.
2.  **Independent Component Analysis (ICA):** FastICA used to find 30 independent components.
3.  **Truncated Singular Value Decomposition (SVD):** Applied to reduce dimensions to 30 components.
4.  **Genetic Algorithm (GA):**
    *   **Representation:** Each individual in the population represented a subset of 30 feature indices.
    *   **Fitness Function:** Evaluated individuals based on the ROC AUC score achieved by a classifier (LinearSVC or CatBoost) trained on the corresponding feature subset using a train/validation split.
    *   **Operators:** Used standard crossover (union of parent features followed by random sampling to maintain size) and mutation (randomly replacing a feature index).
    *   **Selection:** Truncation selection (top 50% survive).
    *   **Goal:** Evolve a population of feature subsets to find the one maximizing the classifier's ROC AUC. Separate GA runs were performed for SVM and CatBoost.

### Classification Models

1.  **Support Vector Machine (SVM):**
    *   Due to the dataset size, `LinearSVC` was used instead of standard `SVC`.
    *   `CalibratedClassifierCV` was wrapped around `LinearSVC` to enable probability predictions (`predict_proba`) required for ROC AUC calculation.
2.  **CatBoost Classifier:**
    *   A gradient boosting library known for its handling of categorical features and good performance. Used with `Logloss` and evaluated using `AUC`.

### Hyperparameter Optimization

PSO and ACO were applied *after* feature selection with GA to tune the hyperparameters of the respective classifiers (SVM and CatBoost) using only the GA-selected features.

1.  **Particle Swarm Optimization (PSO):**
    *   Used the `pyswarm` library.
    *   Defined bounds for relevant hyperparameters (e.g., `C`, `max_iter` for SVM; `learning_rate`, `depth`, `l2_leaf_reg`, etc., for CatBoost).
    *   The objective function trained the classifier with a given set of parameters and returned the validation ROC AUC score.
    *   PSO iteratively adjusted particle positions (hyperparameter sets) to maximize the objective function.
    *   Handled parameter type conversions (float to int) and dependencies (e.g., `max_leaves` only for `Lossguide` policy in CatBoost).
2.  **Ant Colony Optimization (ACO):**
    *   A custom `ACO_HyperparameterOptimizer` class was implemented.
    *   **Parameter Grid:** Defined discrete values for each hyperparameter to explore.
    *   **Pheromone Trails:** Maintained pheromone levels for each parameter-value pair.
    *   **Solution Construction:** Ants probabilistically selected parameter values based on pheromone levels (`alpha` influence).
    *   **Objective Function:** Similar to PSO, evaluated a constructed hyperparameter set by training the classifier and returning the validation ROC AUC.
    *   **Pheromone Update:** Trails were updated based on evaporation (`rho`) and deposition (`Q` scaled by solution score).

## Implementation Details

*   **Languages:** Python 3
*   **Core Libraries:** pandas, numpy, scikit-learn, CatBoost, category_encoders, matplotlib, seaborn, pyswarm, json.
*   **Environment:** Jupyter Notebook (`main.ipynb`)

## Results

Performance was primarily evaluated using the ROC AUC score on the validation set and Kaggle public/private leaderboards.

### SVM Performance

*(Validation ROC AUC / Kaggle Public Score / Kaggle Private Score)*

*   **PCA + SVM:** 0.8126 / 0.837181 / 0.798548
*   **ICA + SVM:** 0.8126 / 0.837181 / 0.798548
*   **SVD + SVM:** 0.8127 / 0.837300 / 0.798838
*   **GA + SVM:** **0.8288** / **0.849871** / **0.808483**
*   **GA + PSO + SVM:** 0.8287 / 0.849987 / 0.808601
*   **GA + ACO + SVM:** 0.8033 (Validation) / 0.826630 / 0.780363 *(Note: Validation score dropped after final ACO model training)*

**SVM Insights:**
*   GA provided a noticeable improvement over classical dimensionality reduction methods for SVM.
*   PSO tuning on GA features yielded a marginal improvement on Kaggle scores but not validation AUC compared to just GA.
*   ACO tuning significantly degraded SVM performance in the final model evaluation, despite finding a good score during the optimization phase itself, suggesting potential overfitting during the ACO search or instability.

### CatBoost Performance

*(Validation ROC AUC / Kaggle Public Score / Kaggle Private Score)*

*   **PCA + CatBoost:** 0.8802 / 0.363070 / 0.364691 *(Poor Kaggle score suggests issues)*
*   **ICA + CatBoost:** 0.8041 / 0.528929 / 0.540807 *(Poor Kaggle score suggests issues)*
*   **SVD + CatBoost:** 0.7955 / 0.363238 / 0.364369 *(Poor Kaggle score suggests issues)*
*   **GA + CatBoost (Baseline):** **0.9259** / **0.906293** / **0.849037**
*   **GA + PSO + CatBoost:** 0.7880 / 0.906293 / 0.849037 *(Lower validation AUC, same Kaggle score)*
*   **GA + ACO + CatBoost:** **0.9332** / **0.906293** / **0.849037** *(Highest validation AUC, same Kaggle score)*

**CatBoost Insights:**
*   CatBoost significantly outperformed SVM across most configurations, especially when combined with GA.
*   Classical DR methods (PCA, ICA, SVD) performed poorly with CatBoost based on Kaggle scores, indicating potential issues with how these transformations interacted with the CatBoost model, possibly due to bad interpretability caused by transformations or overall complexity of the final model.
*   GA feature selection worked very well with CatBoost, yielding the best baseline performance.
*   ACO tuning on GA features provided the highest validation ROC AUC for CatBoost.
*   PSO tuning on GA features resulted in a lower validation AUC compared to the baseline GA+CatBoost.
*   Interestingly, despite different validation scores, the Kaggle scores for GA+CatBoost, GA+PSO+CatBoost, and GA+ACO+CatBoost were identical, suggesting the leaderboard test set might not have been sensitive to the hyperparameter changes found by PSO/ACO or that the baseline GA model was already near optimal for that specific test set.

### Comparative Analysis

*   **Classifier:** CatBoost generally outperformed LinearSVC (calibrated) on this dataset, especially after feature selection/reduction.
*   **Feature Selection:** GA proved more effective than PCA, ICA, or SVD, particularly for CatBoost, leading to significantly better results. For SVM, GA also showed a slight edge.
*   **Hyperparameter Optimization:**
    *   ACO showed promise, achieving the best validation score for CatBoost, although it degraded SVM performance.
    *   PSO provided marginal or no improvement over the GA baseline for both classifiers in terms of validation AUC, and sometimes resulted in worse performance.
    *   The identical Kaggle scores for the optimized CatBoost models suggest the gains seen in validation might not always translate directly to unseen data or specific leaderboard splits.

### Feature Importance

Analysis of feature importance for the GA-selected features using the optimized CatBoost models (PSO and ACO) revealed the top contributing features. While the exact ranking differed slightly between the PSO and ACO models, common important features included `C1`, `C11`, `C14`, `card2`, `addr1`, `V283`, `V70`, `D1`, `V54`, and `C2`. This indicates that these specific transaction counts, card details, address information, and Vesta-engineered features were highly relevant for fraud detection within the subset identified by GA.

## Conclusion

This project demonstrated that combining CatBoost with a Genetic Algorithm for feature selection yielded the most effective fraud detection model on the IEEE-CIS dataset among the tested configurations. While classical dimensionality reduction methods were less successful, especially with CatBoost, GA effectively identified a high-performing 30-feature subset.

Nature-inspired hyperparameter optimization (PSO and ACO) showed mixed results. ACO successfully tuned CatBoost to achieve the highest validation AUC, while PSO struggled to improve upon the baseline GA models. However, these validation improvements didn't translate to better Kaggle scores in this instance.

The results highlight the potential of GAs for feature selection in high-dimensional, imbalanced datasets and suggest that while PSO/ACO can find good hyperparameter sets, careful validation and consideration of potential overfitting during the search process are crucial. CatBoost proved to be a robust classifier for this task.
