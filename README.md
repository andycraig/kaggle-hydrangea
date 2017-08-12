xgboost code based on:

https://www.kaggle.com/the1owl/fractals-of-nature-blend-0-90050

CNN code based on code by Finlay Liu, from:

https://www.kaggle.com/finlay/naive-bagging-cnn-pb0-985?scriptVersionId=1187890

```
# Create folds.
python create_folds.py config.yaml

# Generate features.
python preprocess_nn.py config.yaml
python preprocess_gbt.py config.yaml
# SVM uses GBT features.

# Find hyperparameters.
python fit_hyperparams.py config.yaml 2 # SVM

# Fit models.
python train_loop.py config.yaml 0 # NN
python train_loop.py config.yaml 1 # GBT
python train_loop.py config.yaml 2 # SVM
python train_loop.py config.yaml 3 # NN (extra layers)
python train_loop.py config.yaml 4 # GBT (log loss)

# Combine models.
python stack.py config.yaml
```
