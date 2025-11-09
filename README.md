# PhD Studies - Subject: System Development
## Topic - Integration of MLOps Techniques in Biomedicine by Creating a Regulatory Compliant AI Lifecycle

## Overview

This project implements a comprehensive MLOps framework to establish a robust and regulatory compliant AI lifecycle for biomedicine. Here is utilized MLflow for full pipeline management, 
including experiment tracking and model versioning, focusing on the deployment of a Random Forest, XGBoost and Deep Neural Network model. A key part is the integration of LIME (Local Interpretable Model-agnostic Explanations) 
directly into the production API, ensuring that every prediction is accompanied by a clinically relevant feature-based justification. This architecture prioritizes transparency and auditability, confirming that model decisions 
are based on expected clinical markers (validated via SHAP analysis).

### Repository Contents
  * **`data_processing/`**
    * **`data/`**
        * Contains csv file of original dataset
    * **`prepared_data/`**
        * Contains csv files related to originally splited dataset into three sets: train, validation and test
    * **`plots/`**
        * images of generated plots in EDA process
    * analyze_data.ipynb - Contains EDA process
    * preprocessing.py - Contains data preprocessing
  * **`training/`**
    * **`model.py`** - Contains definition of used models
    * **`model_train.py`** - Contains training process, loging everything into MLFlow, including XAI metrics
    * **`xai_metrics.py`** - Contains definition of LIME and SHAP and its calculating
    * **`plots/`** - plots related to XAI results
  * **`mlruns/`** - Contains info about MLFlow logged models and MLFlow runs
  * **`deployment_api/`** - Contains Docker container setup
     *  **`prediction_api/`** - Files defining Django application
     *  **`ml_loader.py`** - Contains process of retrieving production models from MLFlow
      
