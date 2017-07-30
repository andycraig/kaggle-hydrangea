##################
# Create folds.
python create_folds.py dummy.yaml

# Generate features.
python preprocess_nn.py dummy.yaml
python preprocess_gbt.py dummy.yaml
# SVM uses GBT features.

# Find hyperparameters.
# Only for SVM for now.
python fit_hyperparams.py dummy.yaml 2 # SVM

# Fit models.
python main.py dummy.yaml 0 # NN
python main.py dummy.yaml 1 # xgboost
python main.py dummy.yaml 2 # SVM

# Combine models.
python stack.py config.yaml
