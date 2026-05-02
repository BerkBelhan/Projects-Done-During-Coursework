"""
I have Implemented this code during CMPE442 Machine Learning course. 
The code compares the performance of a Decision Tree Classifier and two Multi-Layer Perceptrons (MLPs) with ReLU and Tanh activation functions on the MNIST dataset.
The MNIST dataset is loaded and preprocessed, including normalization and stratified splitting into training, validation, and test sets.
The Decision Tree is trained with and without pre-pruning (max depth of 5), and the MLPs are trained for 20 epochs using the Adam optimizer and Cross-Entropy Loss.
After training, the models are evaluated on the test set using classification reports that include precision, recall, and F1 scores. Finally, the training and validation loss curves for the MLPs are plotted for visual comparison.

"""
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Seed 
SEED = 43906121950 % 200 
np.random.seed(SEED)
torch.manual_seed(SEED)

# Load MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist['data'], mnist.target.astype(int)

#Normalize
X = X / 255.0

#Select 10,000 samples (stratified)
X_subset, _, y_subset, _ = train_test_split(
    X, y, 
    train_size=10000, 
    stratify=y, 
    random_state=SEED
    )
#Split daata 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X_subset, y_subset, 
    test_size=0.3, 
    stratify=y_subset, 
    random_state=SEED
    )
#Split temp 15% val, 15 test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, 
    test_size=0.5, 
    stratify=y_temp, 
    random_state=SEED
    )
print("Dataset shapes:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

#DECISION TREE CLASSIFIER
print("\nDecision Tree")

#pre-puned tree, I chose to go with 5 depth hyperparameter.
dt_pruned = DecisionTreeClassifier(max_depth=5, random_state=SEED)
dt_pruned.fit(X_train, y_train)
#full tree
dt_full = DecisionTreeClassifier(random_state=SEED)
dt_full.fit(X_train, y_train)

train_acc_pruned = accuracy_score(y_train, dt_pruned.predict(X_train))
val_acc_pruned = accuracy_score(y_val, dt_pruned.predict(X_val))

train_acc_full = accuracy_score(y_train, dt_full.predict(X_train))
val_acc_full = accuracy_score(y_val, dt_full.predict(X_val))

print("\nPre-Puned Tree:")
print("Train Accuracy:", train_acc_pruned)  
print("Validation Accuracy:", val_acc_pruned)

print("\nFull Tree:")
print("Train Accuracy:", train_acc_full)
print("Validation Accuracy:", val_acc_full)

#Pytorch SETUP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)

X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

#MODELS
model_relu = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
).to(device)

model_tanh = nn.Sequential(
    nn.Linear(784, 128),
    nn.Tanh(),
    nn.Linear(128, 10)
).to(device)

#Training function 
def train_model(model, name, X_train, y_train, X_val, y_val, epochs=20):
    print(f"\nTraining {name}...")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        #Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())

        print(f"{name} Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
    return train_losses, val_losses

#TRAIN MLPs
loss_relu, val_relu = train_model(
    model_relu, "ReLU", 
    X_train_t, y_train_t, 
    X_val_t, y_val_t
)
loss_tanh, val_tanh = train_model(
    model_tanh, "Tanh",
    X_train_t, y_train_t,
    X_val_t, y_val_t
)

#Evaluation
def evaluate(model, name, X_test, y_test):
    print(f"\n{name} Test Results:")

    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1).cpu().numpy()
        print(classification_report(y_test, preds))
evaluate(model_relu, "MLP (Relu)", X_test_t, y_test)
evaluate(model_tanh, "MLP (Tanh)", X_test_t, y_test)

#Decision Tree test evaluation
print("\nDecision Tree (Pruned) Test Results:")
print(classification_report(y_test, dt_pruned.predict(X_test)))

print("\nDecision Tree (Full) Test Results:")
print(classification_report(y_test, dt_full.predict(X_test)))

#Plot loss curves
plt.figure()
plt.plot(loss_relu, label='Train ReLU')
plt.plot(val_relu, label='Validation ReLU')
plt.plot(loss_tanh, label='Train Tanh')
plt.plot(val_tanh, label='Validation Tanh')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('MLP Loss Curves')
plt.legend()
plt.show()
