### Suzanne Foley MSc Thesis 2026
### A Novel Calibration-Robustness Metric to Improve Trustworthiness in Predictive Models for Surgical Outcomes 

# This script evaluates a range of probabilistic and ensemble models under data corruption and domain shift scenarios to assess calibration robustness.

### --- 1. Setup and configuration --- ###
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

# Define seeds to re-run analysis multiple times to assess stability
seeds = [42, 43, 44, 45, 46]

# Define size of test dataset
test_size = 0.2

# Define levels of data corruption for corruption experiments
# Gaussian noise levels
noise_levels = [0.05, 0.1, 0.2, 0.3]
# Missingness (MCAR) proportions
missing_levels = [0.05, 0.1, 0.2, 0.3]
# Label inversion proportions
label_inv_levels = [0.05, 0.1, 0.2, 0.3]

# Fractions of training data to include for training size experiment
training_fracs = [0.3, 0.5, 0.7, 1.0]

# All candidate features in the dataset
candidate_features=['age','sex','weight','race','asa', 'emop', 'department', 'antype', 'icd10_pcs']
# Candidate features with those deemed to be biased by hospital protocol (i.e. 'department' and 'icd10_pcs') removed 
candidate_features_no_protocol=['age','sex','weight','race','asa', 'emop', 'antype']
# Define continuous and categorical features used in the dataset
continuous_features = ['age', 'weight']
categorical_features = ['sex', 'race', 'asa', 'emop', 'antype']
# Features to use for missingness experiment
missing_features = ["age", "weight"]


### --- 2. Helper Functions --- ###

# compute Expected Calibration Error metric
def compute_ece(y_true, y_prob, n_bins=10):
    """
    Compute Expected Calibration Error
    ECE divides predicted probabilities into bins and computes for each bin the average predicted probability
    and the true fraction of positive outcomes. It then calculates the weighted average of the absolute
    difference between the confidence and accuracy across all bins.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins = n_bins)
    bin_counts, bin_edges = np.histogram(y_prob, bins=n_bins)
    weights = bin_counts / len(y_prob)
    ece = np.sum(np.abs(prob_true - prob_pred) * weights[:len(prob_true)])
    return ece

# compute evaluation metrics for models
def compute_metrics(y_true, y_prob):
    """
    Compute AUROC, Brier score and ECE.
    """
    auroc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ece = compute_ece(y_true, y_prob)
    return auroc, brier, ece

# apply Gaussian noise corruption
def apply_gaussian_noise(X, features, noise_fraction):
    """
    Apply Gaussian noise to feature in dataset X.
    Gaussian noise sampled from a normal distribution with mean 0 and defined fraction of the
    feature standard deviation.
    """
    X_noisy = X.copy()
    for f in features:
        std = X_noisy[f].std()
        noise = np.random.normal(loc=0, scale=noise_fraction*std, size=X_noisy.shape[0])
        X_noisy[f] += noise

    return X_noisy

def gaussian_wrapper(X, level):
    """
    Apply Gaussian noise at a defined level to the continuous features of a copy of dataset X. 
    """
    return apply_gaussian_noise(X.copy(), continuous_features, level)

def hybrid_gaussian_wrapper(X, level):
    """
    Apply Gaussian noise at a defined level to the continuous features of a copy of dataset X.
    In addition, add the Bayesian logistic regression probabilities as a feature to a new 
    copy of the dataset to be used in hybrid model. 
    """
    X_corr = apply_gaussian_noise(X.copy(), continuous_features, level)
    
    X_corr_hybrid = X_corr.copy()

    # use trained BLR model
    blr = trained_models["Bayesian Logistic Regression"]
    X_corr_hybrid["blr_prob"] = blr.predict_proba(X_corr)[:, 1]
    
    return X_corr_hybrid

# apply missingness corruption
def apply_missingness(X, features, fraction):
    """
    Apply missingness corruption to feature in dataset X.
    Feature records removed at random using MCAR technique.  
    """
    X_missing = X.copy()
    
    n_missing = int(fraction * len(X_missing))
    idx = np.random.choice(X_missing.index, n_missing, replace=False)
    
    for feature in features:
        X_missing.loc[idx, feature] = np.nan
    
    return X_missing

def missing_wrapper(X, level):
    """
    Apply missingness corruption at a defined level to the continuous features of a copy 
    of dataset X. 
    """
    X_corr = apply_missingness(X.copy(), missing_features, level)
    
    # impute for missing values
    X_corr["age"] = X_corr["age"].fillna(impute_values["age"])
    X_corr["weight"] = X_corr["weight"].fillna(impute_values["weight"])
    
    return X_corr

def hybrid_missing_wrapper(X, level):
    """
    Apply missingness corruption at a defined level to the continuous features of a copy of dataset X.
    Impute for missing values using same approach as pre-processing pipeline (i.e. mean imputation).
    In addition, add the Bayesian logistic regression probabilities as a feature to a new 
    copy of the dataset to be used in hybrid model. 
    """
    X_corr = apply_missingness(X.copy(), missing_features, level)

    # impute for missing values
    X_corr["age"] = X_corr["age"].fillna(impute_values["age"])
    X_corr["weight"] = X_corr["weight"].fillna(impute_values["weight"])
    
    X_corr_hybrid = X_corr.copy()
    # use trained model
    blr = trained_models["Bayesian Logistic Regression"]
    X_corr_hybrid["blr_prob"] = blr.predict_proba(X_corr)[:, 1]
    
    return X_corr_hybrid

# apply label inversion corruption
def apply_label_noise(y, fraction):
    """
    Apply label inversion to outcome variable y.
    Outcomes inverted at random.
    """
    y_noisy = y.copy()
    
    n_flip = int(fraction * len(y_noisy))
    idx = np.random.choice(y_noisy.index, n_flip, replace=False)
    
    y_noisy.loc[idx] = 1 - y_noisy.loc[idx]
    
    return y_noisy

# new Calibration Stability Score, including flexibility for compution of CSS for hybrid model
def compute_css(model, X_base, y_test, corruption_func, levels):
    """
    Compute novel Calibration Stability Score (CSS).
    Calculate ECE for clean dataset. Calculate ECE at each corruption level for corrupted dataset.
    Integrate the absolute difference using the trapezoidal rule. 
    """
    
    # clean input (no corruption)
    X_clean = corruption_func(X_base, 0)  
    y_clean = model.predict_proba(X_clean)[:, 1]
    auroc_clean, brier_clean, ece_clean = compute_metrics(y_test, y_clean)

    deltas = []

    # corrupted ECE calculation
    for level in levels:
        X_corr = corruption_func(X_base, level)
        y_corr = model.predict_proba(X_corr)[:, 1]

        auroc_corr, brier_corr, ece_corr = compute_metrics(y_test, y_corr)

        delta = abs(ece_corr - ece_clean)
        deltas.append(delta)

    # apply trapezoidal rule
    css = np.trapezoid(deltas, levels)

    return css, deltas


# need a different CSS function for label noise because need to retrain the model after each corruption
def compute_css_label_noise(model_class, X_train, y_train, X_test, y_test, levels, is_hybrid=False):
    """
    Compute novel Calibration Stability Score (CSS) for label inversion.
    Separate function needed as the outcome variable changes under label inversion and so need
    to retrain the model each time label inversion is done.
    """
    deltas = []
    
    # generic creation of new model object (as models are looped through in the main code block)
    model_clean = model_class()
    
    # clean input (no corruption)
    # add BLR probabilities as dataset feature if model is hybrid model
    if is_hybrid:
        # train BLR first
        blr_local = LogisticRegression(max_iter=2000)
        blr_local.fit(X_train, y_train)
        
        X_train_h = X_train.copy()
        X_test_h = X_test.copy()
        
        X_train_h["blr_prob"] = blr_local.predict_proba(X_train)[:, 1]
        X_test_h["blr_prob"] = blr_local.predict_proba(X_test)[:, 1]
        
        model_clean.fit(X_train_h, y_train)
        y_clean = model_clean.predict_proba(X_test_h)[:, 1]
    else:
        model_clean.fit(X_train, y_train)
        y_clean = model_clean.predict_proba(X_test)[:, 1]
    
    auroc_clean, brier_clean, ece_clean = compute_metrics(y_test, y_clean)
    
    # calculate ECE for corrupted datasets
    for level in levels:
        
        # apply label inversion
        y_train_noisy = apply_label_noise(y_train, level)
        
        # generic creation of new model object (as models are looped through in the main code block)
        model = model_class()
        
        if is_hybrid:
            blr_local = LogisticRegression(max_iter=2000)
            blr_local.fit(X_train, y_train_noisy)
            
            X_train_h = X_train.copy()
            X_test_h = X_test.copy()
            
            X_train_h["blr_prob"] = blr_local.predict_proba(X_train)[:, 1]
            X_test_h["blr_prob"] = blr_local.predict_proba(X_test)[:, 1]
            
            model.fit(X_train_h, y_train_noisy)
            y_pred = model.predict_proba(X_test_h)[:, 1]
        
        else:
            model.fit(X_train, y_train_noisy)
            y_pred = model.predict_proba(X_test)[:, 1]
        
        auroc_corr, brier_corr, ece_corr = compute_metrics(y_test, y_pred)
        
        delta = abs(ece_corr - ece_clean)
        deltas.append(delta)
    
    css = np.trapezoid(deltas, levels)
    
    return css, deltas


def get_domain_shift_subgroups(X_raw, X_encoded, y, domain_shifts):
    """
    Get data subgroups for the domain shifts to be used in the Platt scaling experiment.
    Domain shifts defined in main code block.

    """
    
    subgroups = []
    
    for name, config in domain_shifts.items():
        # continuous feature subgroups based on threshold value, i.e. age and weight
        if config["type"] == "numeric":
            feature = config["feature"]
            threshold = config["threshold"]
            
            # extract subgroups
            mask_high = X_raw[feature] > threshold
            mask_low = X_raw[feature] <= threshold
            
            # add to subgroups list
            subgroups.append((
                f"{name} > {threshold:.1f}",
                X_encoded.loc[mask_high],
                y.loc[mask_high]
            ))
            
            subgroups.append((
                f"{name} ≤ {threshold:.1f}",
                X_encoded.loc[mask_low],
                y.loc[mask_low]
            ))
        
        # binary variable subgroups based on binary value, i.e. emergency operation
        elif config["type"] == "binary":
            feature = config["feature"]
            
            # extract subgroups
            mask_1 = X_raw[feature] == 1
            mask_0 = X_raw[feature] == 0
            
            # add to subgroups list
            subgroups.append((
                "Emergency",
                X_encoded.loc[mask_1],
                y.loc[mask_1]
            ))
            
            subgroups.append((
                "Elective",
                X_encoded.loc[mask_0],
                y.loc[mask_0]
            ))
        
        # ASA physical status variable based on explicitly defined classification value groups
        elif config["type"] == "asa_split":
            feature = config["feature"]

            # extract subgroups
            mask_low = X_raw[feature] <= 2
            mask_high = X_raw[feature] >= 3
            
            # add to subgroups list
            subgroups.append(("ASA ≤ 2", X_encoded.loc[mask_low], y.loc[mask_low]))
            subgroups.append(("ASA ≥ 3", X_encoded.loc[mask_high], y.loc[mask_high]))
    
    return subgroups

def apply_platt_scaling(base_model, X_train, y_train):
    """
    Apply the Platt scaling sigmoid function to the baseline model.
    """
    calibrated_model = CalibratedClassifierCV(
        base_model,
        method="sigmoid", 
        cv=5
    )
    calibrated_model.fit(X_train, y_train)
    return calibrated_model

### --- 3. Dataset Pre-Processing --- ###
# read in PhysioNet INSPRE dataset
data = pd.read_csv("operations.csv")

# Convert outcome columns to binary (if time exists, the event occurred)
data['icu_admitted'] = data['icuin_time'].notna().astype(int)
data['in_hospital_death'] = data['inhosp_death_time'].notna().astype(int)
data['allcause_death'] = data['allcause_death_time'].notna().astype(int)

# Convert emop column to boolean (whether operation was an emergency or not)
data['emop'] = data['emop'].astype(bool)

# Calculate number of outcome events 
total_cases = len(data)
icu_count = data['icu_admitted'].sum()
inhosp_death_count = data['in_hospital_death'].sum()
allcause_death_count = data['allcause_death'].sum()

# Calculate 30-day mortality based on death time stamps
thirty_days_mins = 30*24*60
data['mortality_30d'] = (
    (data['allcause_death_time'].notna()) &
    (data['allcause_death_time'] < thirty_days_mins)
).astype(int)

mortality_30d_count = data['mortality_30d'].sum()

# Calculate 7-day ICU rate based on ICU admission time stamps
seven_days_mins = 7*24*60
data['icu_7d'] = (
    (data['icuin_time'].notna()) &
    (data['icuin_time'] < seven_days_mins)
).astype(int)

icu_7d_count = data['icu_7d'].sum()

# Calculate outcome event rates
icu_rate = icu_count / total_cases
inhosp_death_rate = inhosp_death_count / total_cases
allcause_death_rate = allcause_death_count / total_cases
mortality_30d_rate = mortality_30d_count / total_cases
icu_7d_rate = icu_7d_count / total_cases

# Create summary table of outcome measures, counts and event rate for export to csv
outcome_summary = pd.DataFrame({
    "Outcome": [
        "Total operations",
        "ICU admission",
        "In-hospital death",
        "All-cause death",
        "30-day mortality",
        "7-day ICU admission"
    ],
    "Count": [
        total_cases,
        icu_count,
        inhosp_death_count,
        allcause_death_count,
        mortality_30d_count,
        icu_7d_count
    ],
    "Rate (%)": [
        None,
        icu_rate * 100,
        inhosp_death_rate * 100,
        allcause_death_rate * 100,
        mortality_30d_rate * 100,
        icu_7d_rate * 100
    ]
})

# Round percentages
outcome_summary["Rate (%)"] = outcome_summary["Rate (%)"].round(2)

# Save to csv
outcome_summary.to_csv("outcome_summary.csv", index=False)

# Create a table with the baseline characteristics of the relevant features in the dataset 
# Includes: variable name, category (continuous/categorical), summary statistics and % missingness
summary_rows = []

# Loop through each variable in the dataset
for col in data.columns:
    # Calculate % missing values in that variable
    missing_pct = data[col].isna().mean() * 100
    
    # For continuous variables calculate the mean and standard deviation
    if col in continuous_features:
        mean = data[col].mean()
        std = data[col].std()
        
        summary_rows.append({
            "Variable": col,
            "Category": "",
            "Summary": f"{mean:.2f} ± {std:.2f}",
            "Missing (%)": f"{missing_pct:.1f}"
        })
    
    # For categorical variables calculate the number of values and proportion in each category
    elif col in categorical_features:
        counts = data[col].value_counts(dropna=True)
        proportions = data[col].value_counts(normalize=True, dropna=True) * 100
        
        # Loop through each category in that variable and calculate proportion of values in that category and show missingness % for the full variable
        for i, cat in enumerate(counts.index):
            summary_rows.append({
                "Variable": col if i == 0 else "",
                "Category": cat,
                "Summary": f"{counts[cat]} ({proportions[cat]:.1f}%)",
                "Missing (%)": f"{missing_pct:.1f}" if i == 0 else "" # missingness for full variable
            })

summary_table = pd.DataFrame(summary_rows)

# Save baseline characteristics table to csv
summary_table.to_csv('table_baseline_characteristics.csv')

### --- 4. Setup Results Storage --- ###
# Dictionary to store clean results from all seeds
clean_results_all_seeds = {}

# Dictionaries to store results of data corruption experiments
css_noise_results_all_seeds = {} # Gaussian noise results
css_missing_results_all_seeds = {} # Missingness results
css_label_results_all_seeds = {} # Label results

# Dictionary to store results of domain shift experiment
domain_shift_results_all_seeds = {}

# Dictionary to store results of running analysis for different fractions of training data to assess robustness for smaller data sizes
train_size_all_seeds = {}

# Dictionary to store results of running analysis to assesss how training size impacts CSS
css_train_size_all_seeds = {}

# Dictionaries to store results to assess impact of implementing Platt Scaling for XGBoost and Gaussian Naive Bayes
xgb_platt_results_all_seeds = {}
gnb_platt_results_all_seeds = {}

### --- 5. Main Loop --- ### 
# Loop through each seed
for seed in seeds: 
    print(f"\nRunning seed {seed}")
    np.random.seed(seed)

    # create base dataset for model with the relevant input features and 7-day ICU admissions as outcome variable
    model_data = data[candidate_features_no_protocol + ['icu_7d']].copy()

    # Create input X features and outcome y variable
    X = model_data[candidate_features_no_protocol]
    y = model_data['icu_7d']

    # Save a raw copy before one-hot-encoding of categorical variables
    X_raw = X.copy()

    # Create a stratified training and test split in the dataset
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    # Handle missing continuous values - impute based on mean of the training dataset only to avoid any data leakage
    impute_values = {
        "age": X_train_raw["age"].mean(),
        "weight": X_train_raw["weight"].mean()
    }

    # Apply imputation to both train and test 
    for col in ["age", "weight"]:
        X_train_raw[col] = X_train_raw[col].fillna(impute_values[col])
        X_test_raw[col] = X_test_raw[col].fillna(impute_values[col])
    

    # ASA is the only categorical field with missing values - impute with 'Unknown'
    X_train_raw['asa'] = X_train_raw['asa'].fillna('Unknown') 
    X_test_raw['asa'] = X_test_raw['asa'].fillna('Unknown') 

    # One-hot encode categorical variables as models require numerical input
    X_train = pd.get_dummies(X_train_raw, columns=categorical_features, drop_first=True)
    X_test = pd.get_dummies(X_test_raw, columns=categorical_features, drop_first=True)

    # Align columns to ensure that they match in train and test sets
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    ### Train Models ###
    # Define models for analysis
    # Logistic Regression
    logreg_model = LogisticRegression(max_iter=2000, random_state=seed)

    # Gaussian Naive Bayes
    gnb_model = GaussianNB()

    # Bayesian Logistic Regression
    blr_model = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=2000,
        random_state=seed
    )

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=200, random_state=seed)

    # XGBoost
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=seed
    )

    # Create dictionary with model set to loop through
    models = {
        "Logistic Regression": logreg_model,
        "Random Forest": rf_model,
        "Gaussian Naive Bayes": gnb_model,
        "Bayesian Logistic Regression": blr_model,
        "XGBoost": xgb_model,
    } 

    # Train models on clean dataset
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    # Hybrid model handled separately as needs specific consideration
    # Create hybrid model (XGBoost + BLR)
    # Create copies of train and test datasets
    X_train_hybrid = X_train.copy()
    X_test_hybrid = X_test.copy()

    # Add BLR probability feature
    X_train_hybrid["blr_prob"] = trained_models["Bayesian Logistic Regression"].predict_proba(X_train)[:,1]
    X_test_hybrid["blr_prob"] = trained_models["Bayesian Logistic Regression"].predict_proba(X_test)[:,1]

    # Hybrid model
    xgb_hybrid_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=seed
    )

    xgb_hybrid_model.fit(X_train_hybrid, y_train)

    # Evaluate model performance on clean dataset
    clean_results = {}
    
    print("\nEvaluating performance on clean dataset")

    # Standard models
    for name, model in trained_models.items():
        
        y_prob = model.predict_proba(X_test)[:, 1]
        auroc, brier, ece = compute_metrics(y_test, y_prob)
        
        clean_results[name] = {
            "auroc": auroc,
            "brier": brier,
            "ece": ece,
            "y_prob": y_prob,
            "y_true": y_test
        }

    # Hybrid model (has different input X dataset so handle separately)
    y_prob_hybrid = xgb_hybrid_model.predict_proba(X_test_hybrid)[:, 1]
    auroc_h, brier_h, ece_h = compute_metrics(y_test, y_prob_hybrid)

    clean_results["Hybrid (XGBoost+BLR)"] = {
        "auroc": auroc_h,
        "brier": brier_h,
        "ece": ece_h,
        "y_prob": y_prob_hybrid,
        "y_true": y_test
    }

    # Save clean performance results for this seed 
    clean_results_all_seeds[seed] = clean_results

    ### Apply Gaussian Noise Data Corruption ###
    print("\nEvaluating robustness under Gaussian Noise data corruption")

    css_noise_results = {}

    # Calculate CSS for standard models
    for name, model in trained_models.items():

        # Calculate CSS when Gaussian noise is applied
        css, deltas = compute_css(
            model,
            X_test,
            y_test,
            gaussian_wrapper,
            noise_levels
        )

        css_noise_results[name] = {
            "css": css,
            "deltas": deltas,
            "mean_delta": np.mean(deltas),
            "max_delta": np.max(deltas)
        }

    # Implement Gaussian noise data corruption for hybrid model separately (different input dataset and wrapper)
    css_hybrid_noise, deltas_hybrid_noise = compute_css(
        xgb_hybrid_model,
        X_test,
        y_test,
        hybrid_gaussian_wrapper, 
        noise_levels
    )

    css_noise_results["Hybrid (XGBoost+BLR)"] = {
        "css": css_hybrid_noise,
        "deltas": deltas_hybrid_noise,
        "mean_delta": np.mean(deltas_hybrid_noise),
        "max_delta": np.max(deltas_hybrid_noise)
    }

    css_noise_results_all_seeds[seed] = css_noise_results

    ### Apply Missingness Data Corruption ###
    # Calculate CSS when MCAR (missingness) data corruption is applied
    print("\nEvaluating robustness under MCAR (missingness) data corruption")

    css_missing_results = {}

    # Calculate CSS for standard models
    for name, model in trained_models.items():

        # Calculate CSS when Gaussian noise is applied
        css, deltas = compute_css(
            model,
            X_test,
            y_test,
            missing_wrapper,
            missing_levels
        )

        css_missing_results[name] = {
            "css": css,
            "deltas": deltas,
            "mean_delta": np.mean(deltas),
            "max_delta": np.max(deltas)
        }

    # Implement MCAR data corruption for hybrid model separately (different input dataset and wrapper)
    css_hybrid_missing, deltas_hybrid_missing = compute_css(
        xgb_hybrid_model,
        X_test,
        y_test,
        hybrid_missing_wrapper, 
        missing_levels
    )

    css_missing_results["Hybrid (XGBoost+BLR)"] = {
        "css": css_hybrid_missing,
        "deltas": deltas_hybrid_missing,
        "mean_delta": np.mean(deltas_hybrid_missing),
        "max_delta": np.max(deltas_hybrid_missing)
    }

    css_missing_results_all_seeds[seed] = css_missing_results

    ### Apply Label Inversion Data Corruption ###
    # Calculate CSS when label inversion data corruption is applied
    print("\nEvaluating robustness under label inversion data corruption")

    css_label_results = {}

    # Need to re-train models each time labels are changed
    model_fns = [
        ("Logistic Regression", lambda: LogisticRegression(penalty=None, max_iter=2000, solver='lbfgs')),
        ("Random Forest", lambda: RandomForestClassifier(random_state=seed)),
        ("Gaussian Naive Bayes", lambda: GaussianNB()),
        ("Bayesian Logistic Regression", lambda: LogisticRegression(
            penalty='l2', C=1.0, max_iter=2000, solver='lbfgs', random_state=seed
        )),
        ("XGBoost", lambda: XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=seed
        ))
    ]

    # Calculate CSS for standard models
    for name, model_fn in model_fns:
        
        # Calculate CSS when label inversion corruption is applied
        css, deltas = compute_css_label_noise(
            model_fn,
            X_train,
            y_train,
            X_test,
            y_test,
            label_inv_levels,
            is_hybrid=False
        )
        
        css_label_results[name] = {
            "css": css,
            "deltas": deltas,
            "mean_delta": np.mean(deltas),
            "max_delta": np.max(deltas)
        }

    # Implement label inversion data corruption for hybrid model separately (different input dataset and wrapper)
    css_hybrid_label, deltas_hybrid_label = compute_css_label_noise(
        lambda: XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=seed
        ),
        X_train,
        y_train,
        X_test,
        y_test,
        label_inv_levels,
        is_hybrid=True
    )

    css_label_results["Hybrid (XGBoost+BLR)"] = {
        "css": css_hybrid_label,
        "deltas": deltas_hybrid_label,
        "mean_delta": np.mean(deltas_hybrid_label),
        "max_delta": np.max(deltas_hybrid_label)
    }

    css_label_results_all_seeds[seed] = css_label_results


    ### Apply Domain Shift Analysis ###
    print("\nEvaluating robustness under domain shift analysis")
    all_domain_results = {}

    # Domain shifts to use for domain shift analysis
    domain_shifts = {
        "Age": {
            "type": "numeric",
            "feature": "age",
            "threshold": 65
        },
        "Weight": {
            "type": "numeric",
            "feature": "weight",
            "threshold": X_train_raw["weight"].median()
        },
        "ASA": {
            "type": "asa_split",
            "feature": "asa"
        },
        "Emergency": {
            "type": "binary",
            "feature": "emop"
        }
    }

    # ensure ASA is numeric
    X_train_raw["asa"] = pd.to_numeric(X_train_raw["asa"], errors="coerce")
    X_test_raw["asa"] = pd.to_numeric(X_test_raw["asa"], errors="coerce")

    # For each domain shift
    for shift_name, params in domain_shifts.items():
        print(f"\nImplementing domain shift {shift_name}")

        # Split the dataset based on the specific domain shift
        # If variable is numeric, use the defined threshold   
        if params["type"] == "numeric":
            feature = params["feature"]
            threshold = params["threshold"]
            
            print(f"{feature} threshold: {threshold:.2f}")
            
            # train and test on same subgroup (baseline); train one one subgroup and test on other subgroup (shift test)
            train_mask = X_train_raw[feature] < threshold
            test_mask_A = X_test_raw[feature] < threshold   # baseline test
            test_mask_B = X_test_raw[feature] >= threshold  # shift test
        
        # Split ASA physical status into subgroups
        elif params["type"] == "asa_split":
            feature = params["feature"]
            
            print("ASA split: <=2 vs >=3")
            
            train_mask = X_train_raw[feature] <= 2
            test_mask_A = X_test_raw[feature] <= 2
            test_mask_B = X_test_raw[feature] >= 3
        
        
        elif params["type"] == "binary":
            feature = params["feature"]
            
            print(f"{feature}: False vs True")
            
            train_mask = X_train_raw[feature] == False
            test_mask_A = X_test_raw[feature] == False
            test_mask_B = X_test_raw[feature] == True
        
        # Map back to encoded data
        X_train_A = X_train.loc[X_train_raw[train_mask].index]
        y_train_A = y_train.loc[X_train_A.index]
        
        X_test_A = X_test.loc[X_test_raw[test_mask_A].index]
        y_test_A = y_test.loc[X_test_A.index]
        
        X_test_B = X_test.loc[X_test_raw[test_mask_B].index]
        y_test_B = y_test.loc[X_test_B.index]
        
        # Check for any empty splits and if they exist ignore them
        if len(X_train_A) == 0 or len(X_test_A) == 0 or len(X_test_B) == 0:
            print("Skipping due to empty split")
            continue
        
        
        shift_results = {}
        
        # Train standard models
        for name, model_fn in model_fns:
            
            # Train model on subgroup A
            model = model_fn()
            model.fit(X_train_A, y_train_A)
            
            # Test model on subgroup A (baseline)
            y_prob_A = model.predict_proba(X_test_A)[:, 1]
            _, _, ece_A = compute_metrics(y_test_A, y_prob_A)
            
            # Test model on subgroup B (domain shift test)
            y_prob_B = model.predict_proba(X_test_B)[:, 1]
            _, _, ece_B = compute_metrics(y_test_B, y_prob_B)
            
            # Delta ECE between the two tests
            delta = abs(ece_B - ece_A)
            
            # Store results
            shift_results[name] = {
                "delta_ece": delta,
                "ece_A": ece_A,
                "ece_B": ece_B
            }
    
        # Implement domain shift for hybrid model separately
        
        # train BLR on subgroup A
        blr_local = LogisticRegression(max_iter=2000)
        blr_local.fit(X_train_A, y_train_A)
        
        # Add BLR probability as feature to dataset
        X_train_h = X_train_A.copy()
        X_test_A_h = X_test_A.copy()
        X_test_B_h = X_test_B.copy()
        
        X_train_h["blr_prob"] = blr_local.predict_proba(X_train_A)[:, 1]
        X_test_A_h["blr_prob"] = blr_local.predict_proba(X_test_A)[:, 1]
        X_test_B_h["blr_prob"] = blr_local.predict_proba(X_test_B)[:, 1]
        
        # Train hybrid model
        xgb_hybrid = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=seed
        )
        
        xgb_hybrid.fit(X_train_h, y_train_A)
        
        # Test model on subgroup A (baseline)
        y_prob_A = xgb_hybrid.predict_proba(X_test_A_h)[:, 1]
        _, _, ece_A = compute_metrics(y_test_A, y_prob_A)
        
        # Test model on subgroup B (domain shift test)
        y_prob_B = xgb_hybrid.predict_proba(X_test_B_h)[:, 1]
        _, _, ece_B = compute_metrics(y_test_B, y_prob_B)
        
        # Delta ECE between the two tests
        delta = abs(ece_B - ece_A)
        
        # Store results
        shift_results["Hybrid (XGBoost+BLR)"] = {
            "delta_ece": delta,
            "ece_A": ece_A,
            "ece_B": ece_B
        }

        all_domain_results[shift_name] = shift_results

    # Store results per seed
    domain_shift_results_all_seeds[seed] = all_domain_results

    ### --- Training Size Experiment --- ###
    train_size_results = {}

    # create nested ordering so that they aren't all random subsets of the data but build on each other
    shuffled_idx = X_train.sample(frac=1, random_state=seed).index
    
    for frac in training_fracs:
        
        n = int(frac * len(X_train))
        selected_idx = shuffled_idx[:n]
        
        X_train_sub = X_train.loc[selected_idx]
        y_train_sub = y_train.loc[selected_idx]
        
        model_results = {}
        
        # test training size effect on logistic regression and XGBoost
        selected_model_fns = [
            ("Logistic Regression", lambda: LogisticRegression(
                penalty=None,
                max_iter=2000,
                solver='lbfgs',
                random_state=seed
            )),
            ("XGBoost", lambda: XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='logloss',
                random_state=seed
            ))
        ]
        
        for name, model_fn in selected_model_fns:
            
            model = model_fn()
            model.fit(X_train_sub, y_train_sub)

            # clean performance
            y_prob = model.predict_proba(X_test)[:, 1]
            
            auroc, brier, ece = compute_metrics(y_test, y_prob)

            # Calculate CSS with Gaussian noise applied
            css, deltas = compute_css(
                model,
                X_test,
                y_test,
                gaussian_wrapper,
                noise_levels
            )
            
            model_results[name] = {
                "auroc": auroc,
                "brier": brier,
                "ece": ece,
                "css_noise": css
            }

        train_size_results[frac] = model_results
    
    # Store results for each seed
    train_size_all_seeds[seed] = train_size_results

    ### --- Platt Scaling Experiment --- ###
    print("\n--- Platt Scaling (XGBoost) ---")

    # Create the domain shift subsets
    domain_shift_subsets = get_domain_shift_subgroups(X_test_raw, X_test, y_test, domain_shifts)

    # Train original XGBoost model
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=seed
    )

    xgb_model.fit(X_train, y_train)

    # Calculate clean ECE
    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    _, _, ece_orig = compute_metrics(y_test, y_prob)

    # Apply Platt Scaling to XGBoost model
    xgb_platt = CalibratedClassifierCV(
        XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=seed
        ),
        method="sigmoid",
        cv=5
    )

    xgb_platt.fit(X_train, y_train)

    # Calculate ECE with Platt Scaling
    y_prob_platt = xgb_platt.predict_proba(X_test)[:, 1]
    _, _, ece_platt = compute_metrics(y_test, y_prob_platt)

    # Assess calibration under domain shifts
    ece_shift_orig = []
    ece_shift_platt = []
    shift_names = []

    # For each subgroup by domain shift
    for name, X_shift, y_shift in domain_shift_subsets:
        
        shift_names.append(name)
        
        # Train XGBoost
        y_prob = xgb_model.predict_proba(X_shift)[:, 1]
        _, _, ece = compute_metrics(y_shift, y_prob)
        ece_shift_orig.append(ece)
        
        # Apply Platt Scaling
        y_prob_platt = xgb_platt.predict_proba(X_shift)[:, 1]
        _, _, ece_platt_val = compute_metrics(y_shift, y_prob_platt)
        ece_shift_platt.append(ece_platt_val)

    # Store results
    xgb_platt_results_all_seeds[seed] = {
        "ece_orig": ece_orig,
        "ece_platt": ece_platt,
        "ece_shift_orig": ece_shift_orig,
        "ece_shift_platt": ece_shift_platt,
        "shift_names": shift_names
    }

    # Train original Gaussian Naive Bayes model
    print("\n--- Platt Scaling (GNB) ---")

    gnb_model = GaussianNB()
    gnb_model.fit(X_train, y_train)

    # Calculate clean ECE
    y_prob = gnb_model.predict_proba(X_test)[:, 1]
    _, _, ece_orig = compute_metrics(y_test, y_prob)

    # Apply Platt Scaling
    gnb_platt = CalibratedClassifierCV(
        GaussianNB(),
        method="sigmoid",
        cv=5
    )

    gnb_platt.fit(X_train, y_train)

    # Clean ECE (Platt)
    y_prob_platt = gnb_platt.predict_proba(X_test)[:, 1]
    _, _, ece_platt = compute_metrics(y_test, y_prob_platt)

    # Domain shift evaluation
    ece_shift_orig = []
    ece_shift_platt = []
    shift_names = []

    for name, X_shift, y_shift in domain_shift_subsets:
        
        shift_names.append(name)
        
        # Train original model
        y_prob = gnb_model.predict_proba(X_shift)[:, 1]
        _, _, ece = compute_metrics(y_shift, y_prob)
        ece_shift_orig.append(ece)
        
        # Apply Platt Scaling
        y_prob_platt = gnb_platt.predict_proba(X_shift)[:, 1]
        _, _, ece_platt_val = compute_metrics(y_shift, y_prob_platt)
        ece_shift_platt.append(ece_platt_val)

    # Store results
    gnb_platt_results_all_seeds[seed] = {
        "ece_orig": ece_orig,
        "ece_platt": ece_platt,
        "ece_shift_orig": ece_shift_orig,
        "ece_shift_platt": ece_shift_platt,
        "shift_names": shift_names
    }


### --- 6. Aggregation --- ###
# Aggregate clean performance results for all seeds
summary_rows = []

for model in clean_results_all_seeds[seeds[0]].keys():
    
    aurocs = [clean_results_all_seeds[s][model]["auroc"] for s in seeds]
    briers = [clean_results_all_seeds[s][model]["brier"] for s in seeds]
    eces = [clean_results_all_seeds[s][model]["ece"] for s in seeds]
    
    summary_rows.append({
        "Model": model,
        "AUROC Mean": np.mean(aurocs),
        "AUROC Std": np.std(aurocs),
        "Brier Mean": np.mean(briers),
        "Brier Std": np.std(briers),
        "ECE Mean": np.mean(eces),
        "ECE Std": np.std(eces),
    })

summary_df = pd.DataFrame(summary_rows)

# Format output
summary_df["AUROC Mean"] = summary_df["AUROC Mean"].round(4)
summary_df["AUROC Std"] = summary_df["AUROC Std"].round(4)
summary_df["Brier Mean"] = summary_df["Brier Mean"].round(4)
summary_df["Brier Std"] = summary_df["Brier Std"].round(4)
summary_df["ECE Mean"] = summary_df["ECE Mean"].round(4)
summary_df["ECE Std"] = summary_df["ECE Std"].round(4)

summary_df.to_csv("clean_performance_summary.csv", index=False)

print("\nSaved clean performance summary table to clean_performance_summary.csv")

# Plot clean performance calibration curves
for seed_to_plot in seeds:

    plt.figure(figsize=(8, 6))

    for model in clean_results_all_seeds[seed_to_plot].keys():
        
        y_true = clean_results_all_seeds[seed_to_plot][model]["y_true"]
        y_prob = clean_results_all_seeds[seed_to_plot][model]["y_prob"]
        
        frac_pos, mean_pred = calibration_curve(
            y_true,
            y_prob,
            n_bins=10
        )
        
        plt.plot(mean_pred, frac_pos, marker='o', label=model)
        

    # Perfect calibration line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfect Calibration")

    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Outcome Frequency")
    # plt.title("Clean Performance Reliability Diagram")
    plt.legend(frameon=False)
    plt.tight_layout()

    plt.savefig(f"clean_calibration_single_seed_{seed_to_plot}.pdf", bbox_inches="tight")

# Aggregate Gaussian noise data corruption results for all seeds
summary_rows = []

for model in css_noise_results_all_seeds[seeds[0]].keys():
    
    css_vals = [css_noise_results_all_seeds[s][model]["css"] for s in seeds]
    mean_delta_vals = [css_noise_results_all_seeds[s][model]["mean_delta"] for s in seeds]
    max_delta_vals = [css_noise_results_all_seeds[s][model]["max_delta"] for s in seeds]
    
    summary_rows.append({
        "Model": model,
        "CSS Mean": np.mean(css_vals),
        "CSS Std": np.std(css_vals),
        "Mean Delta_ECE": np.mean(mean_delta_vals),
        "Max Delta_ECE": np.mean(max_delta_vals)
    })

noise_summary_df = pd.DataFrame(summary_rows)

# Format output
noise_summary_df["CSS Mean"] = noise_summary_df["CSS Mean"].round(6)
noise_summary_df["CSS Std"] = noise_summary_df["CSS Std"].round(6)
noise_summary_df["Mean Delta_ECE"] = noise_summary_df["Mean Delta_ECE"].round(6)
noise_summary_df["Max Delta_ECE"] = noise_summary_df["Max Delta_ECE"].round(6)

noise_summary_df.to_csv("gaussian_noise_summary.csv", index=False)
print("\nSaved Gaussian noise corruption summary table to gaussian_noise_summary.csv")

# Plot DeltaECE vs Gaussian noise level
plt.figure(figsize=(8, 6))

handles = []
labels = []

for model in css_noise_results_all_seeds[seeds[0]].keys():
    
    all_curves = []
    
    for seed in seeds:
        deltas = css_noise_results_all_seeds[seed][model]["deltas"]
        all_curves.append(deltas)
    
    all_curves = np.array(all_curves)  # shape: (num_seeds, num_noise_levels)
    
    mean_curve = np.mean(all_curves, axis=0)
    std_curve = np.std(all_curves, axis=0)
    
    # Plot mean curve
    plt.plot(noise_levels, mean_curve, marker='o', label=model)
    
    # Error bars
    err = plt.errorbar(
        noise_levels,
        mean_curve,
        yerr=std_curve,
        marker='o',
        capsize=4,
        label=model
    )

    # Only keep the line (not the error bars)
    handles.append(err.lines[0])
    labels.append(model)

plt.xlabel("Gaussian Noise Level")
plt.ylabel("ΔECE")
# plt.title("Calibration Robustness under Gaussian Noise")
plt.legend(handles, labels)
plt.tight_layout()

plt.savefig("delta_ece_vs_gaussian_noise.pdf", bbox_inches="tight")

# Aggregate missingness data corruption results for all seeds
summary_rows = []

for model in css_missing_results_all_seeds[seeds[0]].keys():
    
    css_vals = [css_missing_results_all_seeds[s][model]["css"] for s in seeds]
    mean_delta_vals = [css_missing_results_all_seeds[s][model]["mean_delta"] for s in seeds]
    max_delta_vals = [css_missing_results_all_seeds[s][model]["max_delta"] for s in seeds]
    
    summary_rows.append({
        "Model": model,
        "CSS Mean": np.mean(css_vals),
        "CSS Std": np.std(css_vals),
        "Mean Delta_ECE": np.mean(mean_delta_vals),
        "Max Delta_ECE": np.mean(max_delta_vals)
    })

missing_summary_df = pd.DataFrame(summary_rows)

# Format output
missing_summary_df["CSS Mean"] = missing_summary_df["CSS Mean"].round(6)
missing_summary_df["CSS Std"] = missing_summary_df["CSS Std"].round(6)
missing_summary_df["Mean Delta_ECE"] = missing_summary_df["Mean Delta_ECE"].round(6)
missing_summary_df["Max Delta_ECE"] = missing_summary_df["Max Delta_ECE"].round(6)

missing_summary_df.to_csv("missingness_summary.csv", index=False)
print("\nSaved missingness corruption summary table to missingness_summary.csv")

# Plot DeltaECE vs Missingness level
plt.figure(figsize=(8, 6))

handles = []
labels = []

for model in css_missing_results_all_seeds[seeds[0]].keys():
    
    all_curves = []
    
    for seed in seeds:
        deltas = css_missing_results_all_seeds[seed][model]["deltas"]
        all_curves.append(deltas)
    
    all_curves = np.array(all_curves)  # shape: (num_seeds, num_noise_levels)
    
    mean_curve = np.mean(all_curves, axis=0)
    std_curve = np.std(all_curves, axis=0)
    
    # Plot mean curve
    plt.plot(missing_levels, mean_curve, marker='o', label=model)
    
    # Error bars
    err = plt.errorbar(
        missing_levels,
        mean_curve,
        yerr=std_curve,
        marker='o',
        capsize=4,
        label=model
    )

    # Only keep the line (not the error bars)
    handles.append(err.lines[0])
    labels.append(model)

plt.xlabel("Missingness Level")
plt.ylabel("ΔECE")
# plt.title("Calibration Robustness under Missingness")
plt.legend(handles, labels)
plt.tight_layout()

plt.savefig("delta_ece_vs_missingness.pdf", bbox_inches="tight")


# Aggregate label inversion data corruption results for all seeds
summary_rows = []

for model in css_label_results_all_seeds[seeds[0]].keys():
    
    css_vals = [css_label_results_all_seeds[s][model]["css"] for s in seeds]
    mean_delta_vals = [css_label_results_all_seeds[s][model]["mean_delta"] for s in seeds]
    max_delta_vals = [css_label_results_all_seeds[s][model]["max_delta"] for s in seeds]
    
    summary_rows.append({
        "Model": model,
        "CSS Mean": np.mean(css_vals),
        "CSS Std": np.std(css_vals),
        "Mean Delta_ECE": np.mean(mean_delta_vals),
        "Max Delta_ECE": np.mean(max_delta_vals)
    })

label_noise_summary_df = pd.DataFrame(summary_rows)

# Format output
label_noise_summary_df["CSS Mean"] = label_noise_summary_df["CSS Mean"].round(6)
label_noise_summary_df["CSS Std"] = label_noise_summary_df["CSS Std"].round(6)
label_noise_summary_df["Mean Delta_ECE"] = label_noise_summary_df["Mean Delta_ECE"].round(6)
label_noise_summary_df["Max Delta_ECE"] = label_noise_summary_df["Max Delta_ECE"].round(6)

label_noise_summary_df.to_csv("label_inversion_summary.csv", index=False)
print("\nSaved label inversion corruption summary table to label_inversion_summary.csv")

# Plot DeltaECE vs Label Inversion level
plt.figure(figsize=(8, 6))

handles = []
labels = []

for model in css_label_results_all_seeds[seeds[0]].keys():
    
    all_curves = []
    
    for seed in seeds:
        deltas = css_label_results_all_seeds[seed][model]["deltas"]
        all_curves.append(deltas)
    
    all_curves = np.array(all_curves)  # shape: (num_seeds, num_noise_levels)
    
    mean_curve = np.mean(all_curves, axis=0)
    std_curve = np.std(all_curves, axis=0)
    
    # Plot mean curve
    plt.plot(label_inv_levels, mean_curve, marker='o', label=model)
    
    # Error bars
    err = plt.errorbar(
        label_inv_levels,
        mean_curve,
        yerr=std_curve,
        marker='o',
        capsize=4,
        label=model
    )

    # Only keep the line (not the error bars)
    handles.append(err.lines[0])
    labels.append(model)

plt.xlabel("Label Inversion Level")
plt.ylabel("ΔECE")
# plt.title("Calibration Robustness under Label Inversion")
plt.legend(handles, labels)
plt.tight_layout()

plt.savefig("delta_ece_vs_label_inversion.pdf", bbox_inches="tight")

# Aggregate domain shift results for all seeds
seeds_list = list(domain_shift_results_all_seeds.keys())

shifts = list(domain_shift_results_all_seeds[seeds_list[0]].keys())
models = list(domain_shift_results_all_seeds[seeds_list[0]][shifts[0]].keys())

summary_rows = []

for model in models:
    for shift in shifts:
        
        deltas = [
            domain_shift_results_all_seeds[s][shift][model]["delta_ece"]
            for s in seeds_list
        ]
        
        summary_rows.append({
            "Model": model,
            "Shift": shift,
            "Delta_ECE Mean": np.mean(deltas),
            "Delta_ECE Std": np.std(deltas)
        })


domain_summary_df = pd.DataFrame(summary_rows)

# Format ouptut
domain_summary_df["Delta_ECE Mean"] = domain_summary_df["Delta_ECE Mean"].round(6)
domain_summary_df["Delta_ECE Std"] = domain_summary_df["Delta_ECE Std"].round(6)


domain_summary_df.to_csv("domain_shift_summary.csv", index=False)
print("\nSaved domain shift summary table to domain_shift_summary.csv")

# Plot DeltaECE vs categorical domain shifts
shifts = domain_summary_df["Shift"].unique()
models = domain_summary_df["Model"].unique()

x = np.arange(len(shifts))
width = 0.8 / len(models)  # space bars nicely

plt.figure(figsize=(8, 6))

for i, model in enumerate(models):
    subset = domain_summary_df[domain_summary_df["Model"] == model]
    
    means = subset["Delta_ECE Mean"].values
    stds = subset["Delta_ECE Std"].values
    
    plt.bar(
        x + i * width,
        means,
        width,
        yerr=stds,
        capsize=3,
        label=model
    )

plt.xticks(x + width * (len(models)-1)/2, shifts, rotation=30)
plt.ylabel("ΔECE")
plt.xlabel("Domain Shift")
# plt.title("Calibration under Domain Shift")
plt.legend()
plt.tight_layout()

plt.savefig("delta_ece_vs_domain_shift.pdf", bbox_inches="tight")


# Aggregate training size experiment results for all seeds
summary_rows = []

seeds_list = list(train_size_all_seeds.keys())
fracs = list(train_size_all_seeds[seeds_list[0]].keys())
models = list(train_size_all_seeds[seeds_list[0]][fracs[0]].keys())

total_sample = len(X_train)

for model in models:
    for frac in fracs:

        # Compute number of sample for this fraction
        n = int(frac * total_sample)
        
        aurocs = [
            train_size_all_seeds[s][frac][model]["auroc"]
            for s in seeds_list
        ]
        
        eces = [
            train_size_all_seeds[s][frac][model]["ece"]
            for s in seeds_list
        ]
        
        css_vals = [
            train_size_all_seeds[s][frac][model]["css_noise"]
            for s in seeds_list
        ]
        
        summary_rows.append({
            "Model": model,
            "Train Fraction": frac,
            "Sample Size": n,
            "AUROC Mean": np.mean(aurocs),
            "AUROC Std": np.std(aurocs),
            "ECE Mean": np.mean(eces),
            "ECE Std": np.std(eces),
            "CSS Mean": np.mean(css_vals),
            "CSS Std": np.std(css_vals),
        })

train_size_summary_df = pd.DataFrame(summary_rows)

for col in ["AUROC Mean", "AUROC Std", "ECE Mean", "ECE Std", "CSS Mean", "CSS Std"]:
    train_size_summary_df[col] = train_size_summary_df[col].round(4)

train_size_summary_df.to_csv("training_size_summary.csv", index=False)
print("\nSaved training size summary table to training_size_summary.csv")

# Plot training size impact on metrics
# AUROC
plt.figure(figsize=(8,6))

for model in train_size_summary_df["Model"].unique():
    subset = train_size_summary_df[train_size_summary_df["Model"] == model]
    subset = subset.sort_values("Train Fraction")
    
    plt.errorbar(
        subset["Train Fraction"],
        subset["AUROC Mean"],
        yerr=subset["AUROC Std"],
        marker='o',
        label=model
    )

plt.xlabel("Train Fraction")
plt.ylabel("AUROC")
# plt.title("AUROC vs Training Size")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("train_size_auroc.pdf")

# ECE
plt.figure(figsize=(8,6))

for model in train_size_summary_df["Model"].unique():
    subset = train_size_summary_df[train_size_summary_df["Model"] == model]
    subset = subset.sort_values("Train Fraction")
    
    plt.errorbar(
        subset["Train Fraction"],
        subset["ECE Mean"],
        yerr=subset["ECE Std"],
        marker='o',
        label=model
    )

plt.xlabel("Train Fraction")
plt.ylabel("ECE")
# plt.title("ECE vs Training Size")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("train_size_ece.pdf")

# CSS
plt.figure(figsize=(8,6))

for model in train_size_summary_df["Model"].unique():
    subset = train_size_summary_df[train_size_summary_df["Model"] == model]
    subset = subset.sort_values("Train Fraction")
    
    plt.errorbar(
        subset["Train Fraction"],
        subset["CSS Mean"],
        yerr=subset["CSS Std"],
        marker='o',
        label=model
    )

plt.xlabel("Train Fraction")
plt.ylabel("CSS")
# plt.title("CSS vs Training Size")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("train_size_css.pdf")

# Aggregate Platt Scaling experiment results for all seeds
# Aggregated clean table
summary_rows = []

for model_name, results_dict in [
    ("XGBoost", xgb_platt_results_all_seeds),
    ("GaussianNB", gnb_platt_results_all_seeds)
]:
    
    seeds_list = list(results_dict.keys())
    
    ece_orig = [results_dict[s]["ece_orig"] for s in seeds_list]
    ece_platt = [results_dict[s]["ece_platt"] for s in seeds_list]
    
    summary_rows.append({
        "Model": model_name,
        "Original ECE Mean": np.mean(ece_orig),
        "Original ECE Std": np.std(ece_orig),
        "Platt ECE Mean": np.mean(ece_platt),
        "Platt ECE Std": np.std(ece_platt)
    })

platt_clean_df = pd.DataFrame(summary_rows)

platt_clean_df.to_csv("platt_clean_summary.csv", index=False)
print("\nSaved clean Platt Scaling summary table to platt_clean_summary.csv")

# Plot clean Platt Scaling results
models = platt_clean_df["Model"]
x = np.arange(len(models))
width = 0.35

orig_mean = platt_clean_df["Original ECE Mean"]
orig_std = platt_clean_df["Original ECE Std"]

platt_mean = platt_clean_df["Platt ECE Mean"]
platt_std = platt_clean_df["Platt ECE Std"]

fig, ax = plt.subplots(figsize=(8, 6))

ax.bar(x - width/2, orig_mean, width, yerr=orig_std, capsize=3, label="Original")
ax.bar(x + width/2, platt_mean, width, yerr=platt_std, capsize=3, label="Platt")

ax.set_xticks(x)
ax.set_xticklabels(models)

ax.set_ylabel("ECE")
# ax.set_title("Effect of Platt Scaling on Calibration (Clean Data)")
ax.legend(frameon=False)

plt.tight_layout()
plt.savefig("platt_clean.pdf", bbox_inches="tight")

# Aggregated domain shift table
summary_rows = []

# Use shift names from first seed (same ordering assumed)
shift_names = xgb_platt_results_all_seeds[list(xgb_platt_results_all_seeds.keys())[0]]["shift_names"]

for model_name, results_dict in [
    ("XGBoost", xgb_platt_results_all_seeds),
    ("GaussianNB", gnb_platt_results_all_seeds)
]:
    
    seeds_list = list(results_dict.keys())
    
    for i, shift in enumerate(shift_names):
        
        ece_orig = [
            results_dict[s]["ece_shift_orig"][i]
            for s in seeds_list
        ]
        
        ece_platt = [
            results_dict[s]["ece_shift_platt"][i]
            for s in seeds_list
        ]
        
        summary_rows.append({
            "Model": model_name,
            "Shift": shift,
            "Original ECE Mean": np.mean(ece_orig),
            "Original ECE Std": np.std(ece_orig),
            "Platt ECE Mean": np.mean(ece_platt),
            "Platt ECE Std": np.std(ece_platt)
        })

platt_shift_df = pd.DataFrame(summary_rows)

platt_shift_df.to_csv("platt_domain_shift_summary.csv", index=False)
print("\nSaved domain shift Platt Scaling summary table to platt_domain shift_summary.csv")
 
#  Plot XGBoost Platt Scaling
fig, ax = plt.subplots(figsize=(8, 6))

subset = platt_shift_df[platt_shift_df["Model"] == "XGBoost"]

shifts = subset["Shift"].unique()
x = np.arange(len(shifts))
width = 0.35

orig_mean = subset["Original ECE Mean"].values
orig_std = subset["Original ECE Std"].values

platt_mean = subset["Platt ECE Mean"].values
platt_std = subset["Platt ECE Std"].values

ax.bar(x - width/2, orig_mean, width, yerr=orig_std, capsize=3, label="Original")
ax.bar(x + width/2, platt_mean, width, yerr=platt_std, capsize=3, label="Platt")

ax.set_xticks(x)
ax.set_xticklabels(shifts, rotation=30, ha="right")

ax.set_ylabel("ECE")
# ax.set_title("XGBoost: Platt Scaling under Domain Shift")
ax.legend(frameon=False)

plt.tight_layout()
plt.savefig("platt_xgb_shift.pdf", bbox_inches="tight")

# Plot Gaussian NB Platt Scaling
fig, ax = plt.subplots(figsize=(8, 6))

subset = platt_shift_df[platt_shift_df["Model"] == "GaussianNB"]

shifts = subset["Shift"].unique()
x = np.arange(len(shifts))
width = 0.35

orig_mean = subset["Original ECE Mean"].values
orig_std = subset["Original ECE Std"].values

platt_mean = subset["Platt ECE Mean"].values
platt_std = subset["Platt ECE Std"].values

ax.bar(x - width/2, orig_mean, width, yerr=orig_std, capsize=3, label="Original")
ax.bar(x + width/2, platt_mean, width, yerr=platt_std, capsize=3, label="Platt")

ax.set_xticks(x)
ax.set_xticklabels(shifts, rotation=30, ha="right")

ax.set_ylabel("ECE")
# ax.set_title("Gaussian Naive Bayes: Platt Scaling under Domain Shift")
ax.legend(frameon=False)

plt.tight_layout()
plt.savefig("platt_gnb_shift.pdf", bbox_inches="tight")