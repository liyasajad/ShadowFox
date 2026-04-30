import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from src.utils import load_config, get_logger, load_object
from src.data_preprocessing import split_data

logger = get_logger('evaluate_pipeline')

def evaluate_model():
    config = load_config()
    
    logger.info("Loading test data...")
    df = pd.read_csv(config['data']['raw_data_path'])
    _, X_test, _, y_test = split_data(
        df, 
        test_size=config['model']['test_size'], 
        random_state=config['model']['random_state']
    )
    
    logger.info("Loading preprocessor and model...")
    preprocessor = load_object(config['model']['preprocessor_path'])
    model = load_object(config['model']['model_path'])
    
    X_test_processed = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_processed)
    y_prob = model.predict_proba(X_test_processed)[:, 1]
    
    logger.info("--- Evaluation Metrics ---")
    logger.info("\n" + classification_report(y_test, y_pred))
    
    auc = roc_auc_score(y_test, y_prob)
    logger.info(f"ROC-AUC Score: {auc:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")
    
    # Feature Importances
    if hasattr(model, 'feature_importances_'):
        logger.info("Extracting feature importances...")
        # Get feature names from preprocessor
        col_trans = preprocessor.named_steps['preprocessor']
        num_cols = col_trans.transformers_[0][2]
        cat_encoder = col_trans.transformers_[1][1].named_steps['onehot']
        cat_cols = cat_encoder.get_feature_names_out(col_trans.transformers_[1][2])
        
        all_features = list(num_cols) + list(cat_cols)
        importances = model.feature_importances_
        
        feature_importance_df = pd.DataFrame({
            'Feature': all_features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        logger.info("\nTop 10 Important Features:\n" + feature_importance_df.head(10).to_string(index=False))

if __name__ == '__main__':
    evaluate_model()
