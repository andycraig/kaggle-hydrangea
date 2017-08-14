# Kaggle invasive plant classification competition

This is my solution to the invasive plant classification challenge on Kaggle, which ended mid August 2017.

Techniques used:
- Convolutional deep neural networks
- Gradient boosting trees
- Stacking
- Bagging
- Cross-validation

Tools used:
- Keras (on top of TensorFlow)
- XGBoost
- Scikit-learn
- Python

# To run

To run the whole process (assuming files have been placed as specified in config.yaml):

```
# Create folds.
python create_folds.py config.yaml

# Generate features.
python preprocess_nn.py config.yaml
python preprocess_gbt.py config.yaml

# Find hyperparameters.
python fit_hyperparams.py config.yaml 1 # GBT

# Fit models.
python train_loop.py config.yaml 0 # NN
python train_loop.py config.yaml 1 # GBT
python train_loop.py config.yaml 3 # NN (extra layers)
python train_loop.py config.yaml 4 # GBT (log loss)

# Combine models.
python stack.py config.yaml
```

# Classifier overview

My ensemble consisted of four base models:
- A convolutional neural networks (CNN), based on code from Finlay Liu (
https://www.kaggle.com/finlay/naive-bagging-cnn-pb0-985?scriptVersionId=1187890
);
- Another CNN, with an extra layer;
- A gradient boosting trees (GBT) (trained using XGBoost) with AUC score, based on code from ( https://www.kaggle.com/the1owl/fractals-of-nature-blend-0-90050
);
- Another GBT, with log likelihood loss.

From each of these I created a bagging ensemble, and then I stacked those together using regularised logistic regression.

Hyperparameters for the GBTs were chosen using cross-validation.

I originally included a support vector machine (SVM), but it performed so badly that I dropped it.

# Discussion

I was fairly sure this would be a competition dominated by deep neural networks, as these have dominated recent image-related competitions.  I did not expect a  high position in this competition, as the computers I had access to lacked the GPU RAM to run code made publicly available by Finlay Liu (
https://www.kaggle.com/finlay/naive-bagging-cnn-pb0-985?scriptVersionId=1187890
), which reported achieved 98.5% accuracy on the public test set. Rather, I saw this competition as a good chance to try linking a variety of machine learning tools together using the Python library Scikit-learn, which I had not used before, and in this sense I felt it was a great success.

There were several more approaches I would have liked to have tried in working on this data set:
- Optimising hyperparameters using Gaussian processes, rather than cross-validation;
- Trying different stacking functions;
- Optimising the hyperparameters of the stacking function;
- Trying to see what performance I could draw out of a SVM;
- Trying more modifications of the DNNs. The largest DNN I was able to run seemed to perform worse than a smaller one, which I would have liked to have looked into further.
