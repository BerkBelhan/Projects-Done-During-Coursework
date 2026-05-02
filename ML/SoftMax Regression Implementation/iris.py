"""
I have Implemented this code during CMPE442 Machine Learning course. 
The code implements a Softmax Regression model with polynomial features and regularization on the Iris dataset. 
It includes cross-validation to select the best polynomial degree and compares different regularization techniques (Ridge, Lasso, Elastic Net) with various learning rates. 
The final model is evaluated on the test set using accuracy, precision, recall, and F1 score metrics.

"""
import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
torch.manual_seed(42)
np.random.seed(42)
# Load the iris dataset
data = load_iris()
X = data.data
y = data.target

#First split into train and temp (70% train, 30% temp), then split temp into val and test (15% each)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
#Second split of temp into val and test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

def normalize(train, val, test):
    mean = train.mean(0)
    std = train.std(0) + 1e-8
    return (train - mean)/std, (val - mean)/std, (test - mean)/std

X_train, X_val, X_test = normalize(X_train, X_val, X_test)

#POLYNOMIAL FEATURES
def polynomial_features(X, degree):
    n_samples, n_features = X.shape
    features = [X]
    #if linear we just return the original features
    #if degree is 2 
    if degree >= 2:
        for i in range(n_features):
            features.append((X[:, i:i+1] ** 2))#we take each individual feature, raise it to the power of 2 and append it to the list of features
        for i, j in itertools.combinations(range(n_features), 2):#we take all combinations of 2 features
            features.append((X[:, i:i+1] * X[:, j:j+1]))#multiply them together and append them to the list of features
    #same stuff applied below for degree 3 but we also have to consider the combinations of 3 features(its the last loop)
    if degree >= 3:
        for i in range(n_features):
            features.append((X[:, i:i+1] ** 3))
        for i, j in itertools.combinations(range(n_features), 2):
            features.append((X[:, i:i+1]**2 * X[:, j:j+1]))
        for i, j, k in itertools.combinations(range(n_features), 3):
            features.append((X[:, i:i+1] * X[:, j:j+1] * X[:, k:k+1]))
    return torch.cat(features, dim=1)

#MODEL
class SoftmaxRegression:
    def __init__(self, input_dim, num_classes):
        self.W = torch.randn(input_dim, num_classes, requires_grad=True)#initialize the weights randomly with requires_grad=True so that we can compute gradients during backpropagation
        self.b = torch.zeros(num_classes, requires_grad=True)#initialize the bias to zeros with requires_grad=True for the same reason as above

    def forward(self, X):
        logits = X @ self.W + self.b#compute the logits by multiplying the input features with the weights and adding the bias
        return torch.softmax(logits, dim=1)#apply the softmax function to the logits to get the predicted probabilities for each class
    
#loss tells the model how to improve
def crossEntropy(y_pred, y_true):
    return -torch.mean(torch.log(y_pred[range(len(y_true)), y_true] + 1e-9))

#regularization
def regularization(model, reg_type, lambda_= 0.01, alpha = 0.5):
    if reg_type == 'ridge':
        return lambda_ * torch.sum(model.W ** 2)
    elif reg_type == 'lasso':
        return lambda_ * torch.sum(torch.abs(model.W))
    elif reg_type == 'elasticnet':
        return lambda_ * (alpha * torch.sum(torch.abs(model.W)) +
                           (1 - alpha) * torch.sum(model.W ** 2))
    return 0



#Training
def train(model, X_train, y_train, X_val, y_val, X_test, y_test, lr, reg_type, epochs=50):
    train_losses = []
    val_losses = []
    test_losses = []

    for epoch in range(epochs):
        y_pred = model.forward(X_train)
        #calculate the cross entropy (how wrong the predictions are)
        loss = crossEntropy(y_pred, y_train) + regularization(model, reg_type)

        loss.backward()#pytorch calculates the gradient of the loss with respect to the model parameters (weights and bias)
        with torch.no_grad():
            model.W -= lr * model.W.grad
            model.b -= lr * model.b.grad
            model.W.grad.zero_()#we have to reset at the end of every epoch 
            model.b.grad.zero_()
        with torch.no_grad():
            val_loss = crossEntropy(model.forward(X_val), y_val)
            test_loss = crossEntropy(model.forward(X_test), y_test)
        
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())  
        test_losses.append(test_loss.item())

    return train_losses, val_losses, test_losses

#cross validation
def cross_validation(X, y, degree):
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    losses = []

    for train_index, val_index in kf.split(X):
        X_tr, X_vl = X[train_index], X[val_index]
        y_tr, y_vl = y[train_index], y[val_index]

        X_tr = polynomial_features(X_tr, degree)
        X_vl = polynomial_features(X_vl, degree)

        model = SoftmaxRegression(X_tr.shape[1], 3)

        for _ in range(50):
            y_pred = model.forward(X_tr)
            loss = crossEntropy(y_pred, y_tr)
            loss.backward()

            with torch.no_grad():
                model.W -= 0.01 * model.W.grad
                model.b -= 0.01 * model.b.grad
            model.W.grad.zero_()
            model.b.grad.zero_()
        val_loss = crossEntropy(model.forward(X_vl), y_vl)
        losses.append(val_loss.item())
    return np.mean(losses)

#selecting the best model
degrees = [1, 2, 3]
cv_results = {}

for d in degrees:
    cv_results[d] = cross_validation(X_train, y_train, d)

best_degree = min(cv_results, key=cv_results.get)
print(f"Best degree: {best_degree} with CV loss: {cv_results[best_degree]:.4f}")

X_train = polynomial_features(X_train, best_degree)
X_val = polynomial_features(X_val, best_degree) 
X_test = polynomial_features(X_test, best_degree)

#train 9 models
learning_rates = [1.5e-5, 1.5e-3, 0.15]
reg_types = ['ridge', 'lasso', 'elasticnet']
results = {}

for lr in learning_rates:
    for reg in reg_types:
        key = f"{reg}_lr{lr}"
        print("Training:", key)

        model = SoftmaxRegression(X_train.shape[1], 3)

        train_l, val_l, test_l = train(
            model, X_train, y_train, X_val,
            y_val, X_test, y_test, lr, reg)

        results[key] = {
            "train": train_l,
            "val": val_l,
            "test": test_l,
            "model": model
        } 

def plot_losses(results, loss_type):
    plt.figure()
    for key in results:
        plt.plot(results[key][loss_type], label=key)
    plt.xlabel("Epoch")
    plt.ylabel(f"{loss_type.capitalize()} Loss")
    plt.legend()
    plt.title(f"{loss_type.capitalize()} Loss vs Epoch")
    plt.show()
plot_losses(results, "train")
plot_losses(results, "val")
plot_losses(results, "test")

#find the best model based on the lowest validation loss
best_model_key = min(results, key=lambda k: min(results[k]['val']))
best_model = results[best_model_key]['model']
print("Best model:", best_model_key)

#evaluate the best model on the test set
with torch.no_grad():
    X_pred = best_model.forward(X_test)
    y_pred_classes = torch.argmax(X_pred, dim=1).numpy()

y_true = y_test.numpy()

acc = accuracy_score(y_true, y_pred_classes)
prec = precision_score(y_true, y_pred_classes, average='macro')
rec = recall_score(y_true, y_pred_classes, average='macro')
f1 = f1_score(y_true, y_pred_classes, average='macro')

print("\n Final Metrics (Test Set):")
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)
