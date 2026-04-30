import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from src.utils import load_config, get_logger, save_object
from src.data_preprocessing import get_preprocessor, split_data

logger = get_logger('train_pipeline')

def train_model():
    config = load_config()
    
    logger.info("Loading data...")
    df = pd.read_csv(config['data']['raw_data_path'])
    
    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(
        df, 
        test_size=config['model']['test_size'], 
        random_state=config['model']['random_state']
    )
    
    logger.info("Building and fitting preprocessor...")
    preprocessor = get_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save the preprocessor
    save_object(preprocessor, config['model']['preprocessor_path'])
    logger.info(f"Preprocessor saved to {config['model']['preprocessor_path']}")
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=config['model']['random_state']),
        'Decision Tree': DecisionTreeClassifier(random_state=config['model']['random_state']),
        'Random Forest': RandomForestClassifier(random_state=config['model']['random_state'])
    }
    
    best_model = None
    best_f1 = 0
    best_model_name = ""
    
    logger.info("Evaluating baseline models...")
    for name, model in models.items():
        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_test_processed)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        logger.info(f"[{name}] Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
        
    logger.info("Tuning Random Forest...")
    rf_params = config['hyperparameters']['random_forest']
    
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=config['model']['random_state']),
        param_grid=rf_params,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    
    grid_search.fit(X_train_processed, y_train)
    
    best_rf = grid_search.best_estimator_
    logger.info(f"Best Random Forest Params: {grid_search.best_params_}")
    
    y_pred_rf = best_rf.predict(X_test_processed)
    best_f1 = f1_score(y_test, y_pred_rf)
    
    logger.info(f"Tuned Random Forest F1-score: {best_f1:.4f}")
    
    # Save best model
    save_object(best_rf, config['model']['model_path'])
    logger.info(f"Best model saved to {config['model']['model_path']}")

if __name__ == '__main__':
    train_model()
