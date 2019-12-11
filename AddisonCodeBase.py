import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from sklearn.metrics import confusion_matrix

def read_data(fname):
    '''Takes a filename, and gives back x and y of the dataset'''
    f = open(fname, 'r')
    f.readline()
    
    x = []
    y = []
    
    for line in f:
        line = line.replace("\n", "").split(",")
        label = int(line[0])
        image = line[1:]
        y.append(label)
        # take flattend input, and reshape to 28x28 image
        x.append(np.array(image, dtype=np.float).reshape(28,28))
    
    
    x = np.array(x)
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
    # Normalize the pixels to be between 0 and 1.
    x /= 255.
    # Swap axes for pytorch model
    x = np.swapaxes(x, 1, 3)
    
    return x, np.array(y)


def get_mapping_dictionary(labels):
    '''Returns the dictionary mapping labels to letters'''
    mappings = {}
    for x in range(max(labels) + 2):
        mappings[x] = chr(ord('A') + int(x))
    
    
    return mappings



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from sklearn.metrics import confusion_matrix


class Net(nn.Module):
    
    def __init__(self, N, n_input):
        
        super(Net, self).__init__()
        
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(2304, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, N)
        )
        
    def forward(self, x):
            
        x = self.conv_layer(x)
            
        x = x.view(x.size(0), -1) # Flatten output of conv layer
            
        x = self.fc_layer(x)
            
        return x #logits
        
    

class SignLanguageClassifier(object):
    
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = torch.from_numpy(x_train)
        self.x_test = torch.from_numpy(x_test)
        self.y_train = torch.from_numpy(y_train)
        self.y_test = torch.from_numpy(y_test)
        self.N = 25
        self.n_input = x_train.shape[1]
        self.model = self.get_model()
        self.training_accs = None
        self.test_accs = None
        self.loss = None
    
    def get_model(self):
        return Net(self.N, self.n_input) # Return a new instance of Net(N, n_input)
    
    def train(self):

        # Pass x_train, y_train, x_test, y_test tensors into a data loader object
        training_data = TensorDataset(self.x_train, self.y_train)
        test_data = TensorDataset(self.x_test, self.y_test)
        
        # Batch size
        batch_size = 256
        
        train_loader = torch.utils.data.DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True)
        
        batch_size = 256
        
        
        test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
        
        # Loss function
        criterion = torch.nn.CrossEntropyLoss()
        
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        epochs = 15
        
        training_accs = []
        test_accs = []
        loss_tracker = []
        
        for epoch in range(epochs):
            self.model.train()
            for d,t in train_loader:
                
                # Zero out the optimizer
                optimizer.zero_grad()
                
                # Make prediction based on the model
                outputs = self.model(d.float())
                
                
                # Compute the loss
                loss = criterion(outputs, t)
                
                
                # Compute the derivative with respect to params
                loss.backward()
                
                # Update the parameters
                optimizer.step()
            
            # Track the loss at this epoch.
            loss_tracker.append(loss.item())
            
            # Get the training set accuracy
            total = 0
            correct = 0
            
            for d,t in train_loader:
                outputs = self.model(d.float())
                
                _, predicted = torch.max(outputs.data, 1)
                
                total += t.size(0)
                correct += (predicted == t).sum()
                
            training_accs.append(100.*correct / total)
            
            # Get test set accuracy
            total = 0
            correct = 0
            
            for d,t in test_loader:
                outputs = self.model(d.float())
                
                _, predicted = torch.max(outputs.data, 1)
                
                total += t.size(0)
                correct += (predicted == t).sum()
            
            test_accs.append(100.*correct / total)
            
            
            print("Epoch: %d, Training Accuracy: %0.3f, Test Accuracy: %0.3f" %(epoch+1, training_accs[-1], test_accs[-1]))
        
        self.model.eval()
        
        self.training_accs = training_accs
        self.test_accs = test_accs
        self.loss = loss_tracker
    
    def confusion_M(self):
        y_pred_test = self.model(self.x_test.float()).detach().numpy()
        y_pred_test = np.argmax(y_pred_test, axis=1) # Get classes from logits.
        
        return confusion_matrix(y_pred_test, self.y_test.detach().numpy())

    def train_test_chart(self):
        # Show the training/test accuracy per epoch
        x = np.linspace(1, 15, 15)
        plt.title("Training vs Test Set Accuracy")
        plt.plot(x, self.training_accs, 'r-', label='training')
        plt.plot(x, self.test_accs, 'g-', label='test')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend(loc='lower right')
        plt.show()

    def loss_epoch_chart(self):
        # Show the loss per epoch
        x = np.linspace(1, 15, 15)
        loss = self.loss
        plt.title("Loss vs Epoch")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.plot(x, loss, 'r-')
        plt.show()

    def show_confusion_matrix(self):
        #Show a graphical representation of the confusion matrix
        confusion_M = self.confusion_M()

        plt.title("Confusion Matrix")
        axis = np.linspace(0, 25, 26)
        plt.xlabel("Class")
        plt.ylabel("Class")
        plt.xticks(axis)
        plt.yticks(axis)
        for i in range(len(confusion_M)):
            for j in range(len(confusion_M[i])):
            
            
                if(confusion_M[i][j] != 0):
                    if(i >= 9 and j >= 9):
                        plt.plot(i+1,j+1, 'r.')
                        plt.text(i+1, j+1, confusion_M[i][j], horizontalalignment='right', verticalalignment='top')
                    elif(i >= 9):
                        plt.plot(i+1,j, 'r.')
                        plt.text(i+1, j, confusion_M[i][j], horizontalalignment='right', verticalalignment='top')
                    elif(j >= 9):
                        plt.plot(i,j+1, 'r.')
                        plt.text(i, j+1, confusion_M[i][j], horizontalalignment='right', verticalalignment='top')
                    else:
                        plt.plot(i,j, 'r.')
                        plt.text(i, j, confusion_M[i][j], horizontalalignment='right', verticalalignment='top')
                    
                
        plt.grid(b=False, linestyle='-', color='lightgray')
        plt.show()

    def get_precision_recall(self):
        #Get the precision and recall chart
        mappings = get_mapping_dictionary(self.y_train.detach().numpy())
        confusion_M = self.confusion_M()
        # Precision and Recall
        precision = np.diag(confusion_M)/confusion_M.sum(axis=1)
        recall = np.diag(confusion_M)/confusion_M.sum(axis=0)
        print("Precision" + '\t' + "Recall" + '\t\t' + 'Letter')
        for x in range(len(precision)):
            
            print("%0.2f" % (precision[x]*100), end='\t\t')
            print("%0.2f" % (recall[x]*100), end='\t\t')
            if(x >= 9):
                print(mappings[x+1], end='\n')
            else:
                print(mappings[x], end='\n')
        
