##################
# Testing
python create_folds.py dummy.yaml
python preprocess_nn.py dummy.yaml
python preprocess_gbt.py dummy.yaml
# SVM uses GBT features.

python main.py dummy.yaml 0 # NN
python main.py dummy.yaml 1 # xgboost
python main.py dummy.yaml 2 # SVM

python stack.py config.yaml


##################
# Training
./create_folds.py config.yaml
./preprocess_nn.py config.yaml
./preprocess_gbt.py config.yaml

./main_nn.py config.yaml
./main_gbt.py config.yaml
./train_svn.py config.yaml

./stack.py config.yaml
