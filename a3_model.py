import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
# Whatever other imports you need
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
# You can implement classes and helper functions here too.

le = LabelEncoder()

class PerceptronModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, nonlinearity=None):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, output_size)
        if nonlinearity == "relu":
            self.nonlinearity = nn.ReLU()
        elif nonlinearity == "sigmoid":
            self.nonlinearity = nn.Sigmoid()
        else:
            self.nonlinearity = nn.Identity()
        self.logsoftmax = nn.LogSoftmax(dim=1)


    def forward(self, data):
        after_input_layer = self.input_layer(data)
        after_hidden_layer = self.nonlinearity(after_input_layer)
        output = self.hidden_layer(after_hidden_layer)
        output = self.logsoftmax(output)
        
        return output
    
def train(tx, ty, epochs, batch_size):
    train_dataset = TensorDataset(train_vectors_tensored, train_labels_tensored)
    train_dataset = [sample for sample in train_dataset if sample[1].numel() > 0]
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    m = model

    optimizer = optim.Adam(m.parameters(), lr=0.1)
    loss_function = nn.NLLLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            model_input = batch[0]
            ground_truth = batch[1]
            output = model(model_input)
            loss = loss_function(output, ground_truth.long())
            total_loss += loss.item()
            print(f'epoch {epoch},', f'batch {i}:', round(total_loss / (i + 1), 4), end='\r')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return m

def print_confusion_matrix(model, test_vectors, test_labels):
    model.eval()
    predictions = model(test_vectors).argmax(dim=1)
    cm = confusion_matrix(test_labels, predictions) 
    print(f"Confusion matrix:\n{cm}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    parser.add_argument("--epochs", dest="epochs", type=int, default="4", help="The number of epochs used to train the model.")
    parser.add_argument("--linearity", dest="nonlinearity", type=str, default=None, help="The name of the non linear activation function.")
    # Add options here for part 3 -- hidden layer and nonlinearity,
    # and any other options you may think you want/need.  Document
    # everything.
    
    args = parser.parse_args()
    print("Reading {}...".format(args.featurefile))
    myfile = args.featurefile
    file_path = os.path.abspath(myfile)
    data = pd.read_csv(file_path)
    # implement everything you need here
    ## extracting the values needed for the upcoming functions and the training/evaluating of the model
    vectors = data.iloc[:, 2:]
    labels = data['author']
    training_set = data["set"] == "training set"
    testing_set = data["set"] == "testing set"
    vectors_train = vectors[training_set]
    vectors_test = vectors[testing_set]
    labels_train = labels[training_set]
    labels_test = labels[testing_set]
    le = LabelEncoder()
    labels_train_encoded = le.fit_transform(labels_train)
    labels_test_encoded = le.transform(labels_test)
    ## getting the input and output sizes to put in my model layers
    number_of_features = vectors.shape[1]
    number_of_unique_authors = len(labels.unique())
    ## I turn my extracted values into tensors so as to use in my PyTorch model after coverting them into numpy arrays
    train_vectors_tensored = torch.Tensor(vectors_train.to_numpy())
    train_labels_tensored = torch.LongTensor(np.array(labels_train_encoded))
    test_vectors_tensored = torch.Tensor(vectors_test.to_numpy())
    test_labels_tensored = torch.LongTensor(np.array(labels_test_encoded))
    ## establishing my model
    model = PerceptronModel(input_size=number_of_features, output_size=number_of_unique_authors, hidden_size=50, nonlinearity=args.nonlinearity)
    ## training my model on the data I processed and tensored
    trained_model = train(train_vectors_tensored, train_labels_tensored, args.epochs, batch_size=10)
    ## evaluating and printing my confusion matrix
    print_confusion_matrix(trained_model, test_vectors_tensored, test_labels_tensored)