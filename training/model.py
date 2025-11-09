import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb 

def create_nn_model(input_dim, units_1, units_2, learning_rate):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(units_1, activation='relu', name='hidden_layer_1'),
        Dropout(0.2),
        Dense(units_2, activation='relu', name='hidden_layer_2'),
        Dropout(0.2),
        Dense(1, activation='sigmoid', name='output_layer') # Binary classification
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
    return model


def create_rf_model(n_estimators=100, max_depth=None, random_state=42):
    """Kreira model Random Forest Klasifikatora."""
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight='balanced' 
    )
    return rf_model


def create_xgb_model(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
    """Kreira model XGBoost Klasifikatora."""
    xgb_model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=random_state,
        base_score=0.5 
    )
    return xgb_model