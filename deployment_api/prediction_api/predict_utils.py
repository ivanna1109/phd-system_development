import os
import pickle
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def predict_model(model, input_df):
    pred = model.predict(input_df)
    
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)[0]
        if probs.ndim == 0:  
            probs = np.array([1 - probs, probs])
    else:
        if isinstance(pred, (float, int, np.floating, np.integer)):
            probs = np.array([1 - pred, pred])
        elif isinstance(pred, np.ndarray) and pred.ndim == 1:
            probs = np.array([1 - pred[0], pred[0]])
        else:
            probs = pred 
    
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    
    return pred_idx, confidence, probs


def predict_keras_model(model, input_array):
    p = model.predict(input_array)[0][0] 
    pred_idx = 1 if p >= 0.5 else 0
    confidence = max(p, 1-p)
    probs = np.array([1-p, p])
    return pred_idx, confidence, probs
