##################
# Testing
python create_folds.py dummy.yaml
python preprocess_nn.py dummy.yaml
python preprocess_gbt.py dummy.yaml

python main_nn.py dummy.yaml
python main_gbt.py dummy.yaml
python train_svn.py dummy.yaml

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