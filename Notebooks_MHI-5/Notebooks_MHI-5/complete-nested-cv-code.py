import numpy as np
import optuna
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score, accuracy_score
from collections import defaultdict

# Define all models and their hyperparameter grids (simpler grids for GridSearchCV phase)
models = {
    'Logistic Regression': (LogisticRegression(max_iter=1000), {
        'model__C': [0.0001, 0.001, 0.01, 1, 10, 100]
    }),
    'SVM': (SVC(), {
        'model__C': [0.1, 1, 10, 100],
        'model__kernel': ['linear', 'rbf'],
        'model__gamma': ['scale', 'auto'],
        'model__class_weight': [None, 'balanced']
    }),
    'Random Forest': (RandomForestClassifier(), {
        'model__n_estimators': [100],
        'model__max_features': ['auto'],
        'model__max_depth': [None],
        'model__min_samples_split': [2],
        'model__min_samples_leaf': [1]
    }),
    'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {
        'model__n_estimators': [100],
        'model__learning_rate': [0.1],
        'model__max_depth': [6],
        'model__subsample': [0.8],
        'model__colsample_bytree': [0.8]
    })
}

# Define balancing techniques
balancing_techniques = {
    'None': None,
    'SMOTE-Tomek': SMOTETomek(),
    'Random Undersampling': RandomUnderSampler()
}

# Function for Optuna optimization for finer hyperparameter tuning
def optuna_objective(trial, model_name, X, y, balancing_method, cv):
    if model_name == 'Random Forest':
        params = {
            'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 150, 200, 250, 300]),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
            'max_depth': trial.suggest_categorical('max_depth', [None, 5, 10, 15, 20, 25, 30]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced', 'balanced_subsample'])
        }
        base_model = RandomForestClassifier(**params)
    
    elif model_name == 'XGBoost':
        params = {
            'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 200, 300, 400, 500]),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10)
        }
        base_model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
    
    # Create a pipeline with the balancer and model
    if balancing_method is None:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', base_model)
        ])
    else:
        pipeline = ImbPipeline([
            ('balancer', balancing_method),
            ('scaler', StandardScaler()),
            ('model', base_model)
        ])
    
    # Cross-validate
    scores = []
    kf = KFold(n_splits=cv)
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)
        scores.append(f1_score(y_val, preds))
    
    return np.mean(scores)

def run_nested_cv(X, y):
    # Outer loop: 5-Fold Nested Cross Validation
    outer_kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = defaultdict(list)
    
    for outer_fold, (train_idx, test_idx) in enumerate(outer_kf.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Inner loop: 3-Fold Cross Validation for model selection & hyperparameter tuning
        inner_kf = KFold(n_splits=3, shuffle=True, random_state=42)
        
        print(f"\n========== Outer Fold {outer_fold} ==========")
        print("\nPHASE 1: Finding best classifier and balancing technique combination with GridSearchCV")
        
        # Dictionary to store the best model for each classifier-balancing combination
        best_models_for_fold = {}
        grid_search_results = {}
        
        # PHASE 1: Test all combinations of classifiers and balancing techniques with GridSearchCV
        for bal_name, bal_method in balancing_techniques.items():
            for model_name, (model, param_grid) in models.items():
                # Create pipeline with balancing technique
                pipeline = ImbPipeline([
                    ('balancer', bal_method),
                    ('scaler', StandardScaler()),
                    ('model', model)
                ])
                
                # Perform GridSearchCV
                grid_search = GridSearchCV(
                    pipeline, 
                    param_grid, 
                    cv=inner_kf, 
                    scoring='f1', 
                    n_jobs=-1
                )
                
                # Train and find best hyperparameters
                grid_search.fit(X_train, y_train)
                
                # Get the key for this classifier-balancing combination
                model_key = f"{model_name} ({bal_name})"
                
                # Store the best model for this combination
                best_models_for_fold[model_key] = grid_search.best_estimator_
                grid_search_results[model_key] = grid_search.best_score_
                
                # Evaluate on the test set
                y_pred = grid_search.predict(X_test)
                f1 = f1_score(y_test, y_pred)
                results[f"GridSearch-{model_key}"].append(f1)
                
                # Print result for this fold and model
                print(f"  {model_key}: F1 = {f1:.4f} (Inner CV best: {grid_search.best_score_:.4f})")
        
        # Find the best classifier and balancing technique combination
        best_combo = max(grid_search_results.items(), key=lambda x: x[1])
        best_combo_name = best_combo[0]
        best_combo_score = best_combo[1]
        
        print(f"\nBest combination from GridSearchCV: {best_combo_name} with inner CV F1 = {best_combo_score:.4f}")
        
        # Extract classifier name and balancing technique from the best combination
        best_classifier = best_combo_name.split(' (')[0]
        best_balancer_name = best_combo_name.split('(')[1].replace(')', '')
        best_balancer = balancing_techniques[best_balancer_name]
        
        # PHASE 2: If the best classifier is RF or XGBoost, use Optuna for finer hyperparameter tuning
        if best_classifier in ['Random Forest', 'XGBoost']:
            print(f"\nPHASE 2: Using Optuna for finer hyperparameter tuning of {best_classifier} with {best_balancer_name}")
            
            # Create and optimize the Optuna study
            study = optuna.create_study(direction='maximize')
            study.optimize(
                lambda trial: optuna_objective(
                    trial, 
                    best_classifier, 
                    X_train, 
                    y_train, 
                    best_balancer, 
                    cv=3
                ), 
                n_trials=30
            )
            
            # Get best parameters
            best_params = study.best_params
            print(f"  Best parameters found by Optuna: {best_params}")
            
            # Create model with best parameters
            if best_classifier == 'Random Forest':
                best_model = RandomForestClassifier(**best_params)
            else:  # XGBoost
                best_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
            
            # Create pipeline with best balancing technique and optimized model
            if best_balancer is None:
                final_pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', best_model)
                ])
            else:
                final_pipeline = ImbPipeline([
                    ('balancer', best_balancer),
                    ('scaler', StandardScaler()),
                    ('model', best_model)
                ])
            
            # Fit the optimized model
            final_pipeline.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = final_pipeline.predict(X_test)
            optuna_f1 = f1_score(y_test, y_pred)
            results[f"Optuna-{best_classifier} ({best_balancer_name})"].append(optuna_f1)
            
            print(f"  Optuna-optimized {best_classifier} with {best_balancer_name}: F1 = {optuna_f1:.4f}")
        else:
            print(f"\nPHASE 2: Skipping Optuna since best classifier is {best_classifier}, not RF or XGBoost")
    
    # Final Summary
    print("\n========== Final Results ==========")
    print("\nMean F1 Scores Across All Outer Folds:")
    
    # Group results by GridSearch and Optuna
    grid_results = {k: v for k, v in results.items() if k.startswith('GridSearch')}
    optuna_results = {k: v for k, v in results.items() if k.startswith('Optuna')}
    
    print("\nGridSearchCV Results:")
    for model_name, scores in grid_results.items():
        if len(scores) > 0:  # Only print if we have results
            mean_f1 = np.mean(scores)
            std_f1 = np.std(scores)
            model_name_clean = model_name.replace('GridSearch-', '')
            print(f"  {model_name_clean}: {mean_f1:.4f} ± {std_f1:.4f}")
    
    if optuna_results:
        print("\nOptuna Results:")
        for model_name, scores in optuna_results.items():
            if len(scores) > 0:  # Only print if we have results
                mean_f1 = np.mean(scores)
                std_f1 = np.std(scores)
                model_name_clean = model_name.replace('Optuna-', '')
                print(f"  {model_name_clean}: {mean_f1:.4f} ± {std_f1:.4f}")
    
    # Find the best overall model across all methods
    all_results = {}
    for model_type, results_dict in [("GridSearch", grid_results), ("Optuna", optuna_results)]:
        for model_name, scores in results_dict.items():
            if len(scores) > 0:
                all_results[model_name] = np.mean(scores)
    
    if all_results:
        best_model_name = max(all_results.items(), key=lambda x: x[1])[0]
        print(f"\nBest overall model: {best_model_name} with mean F1 = {all_results[best_model_name]:.4f}")
    
    return results

# To use this function:
# results = run_nested_cv(X, y)
