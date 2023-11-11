'''
*****************************************************************************************
*
*        		===============================================
*           		GeoGuide(GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 1A of GeoGuide(GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:			[ gg_3120 ]
# Author List:		[ Jasraj Chouhan ]
# Filename:			task_1a.py
# Functions:	    [`ideantify_features_and_targets`, `load_as_tensors`,
# 					 `model_loss_function`, `model_optimizer`, `model_number_of_epochs`, `training_function`,
# 					 `validation_functions` ]

####################### IMPORT MODULES #######################
import pandas
import torch
import numpy
###################### Additional Imports ####################
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader



'''s
You can import any additional modules that you require from
torch, matplotlib or sklearn.
You are NOT allowed to import any other libraries. It will
cause errors while running the executable
'''
##############################################################

################# ADD UTILITY FUNCTIONS HERE #################

##############################################################


def data_preprocessing(task_1a_dataframe):
    # Create a copy of the original DataFrame to avoid modifying the original data
    # global encoded_dataframe
    encoded_dataframe = task_1a_dataframe.copy()
    # Normalize your input data if it's not already normalized

    # Initialize a LabelEncoder for encoding textual features
    label_encoder = LabelEncoder()

    # Encode the "Education" column
    encoded_dataframe['Education'] = label_encoder.fit_transform(encoded_dataframe['Education'])

    encoded_dataframe['JoiningYear'] = label_encoder.fit_transform(encoded_dataframe['JoiningYear'])
    encoded_dataframe['PaymentTier'] = label_encoder.fit_transform(encoded_dataframe['PaymentTier'])
    # Encode the "City" column
    encoded_dataframe['City'] = label_encoder.fit_transform(encoded_dataframe['City'])
    encoded_dataframe['Age'] = label_encoder.fit_transform(encoded_dataframe['Age'])
    # Encode the "Gender" column
    encoded_dataframe['Gender'] = label_encoder.fit_transform(encoded_dataframe['Gender'])

    # Encode the "EverBenched" column if it contains textual values
    if encoded_dataframe['EverBenched'].dtype == 'object':
        encoded_dataframe['EverBenched'] = label_encoder.fit_transform(encoded_dataframe['EverBenched'])


    return encoded_dataframe

# Example call:
# encoded_dataframe = data_preprocessing(task_1a_dataframe)


def identify_features_and_targets(encoded_dataframe):
    '''
    Purpose:
    ---
    The purpose of this function is to define the features and
    the required target labels. The function returns a python list
    in which the first item is the selected features and the second 
    item is the target label

    Input Arguments:
    ---
    `encoded_dataframe` : [ Dataframe ]
                    Pandas dataframe that has all the features mapped to 
                    numbers starting from zero

    Returns:
    ---
    `features_and_targets` : [ list ]
                        Python list in which the first item is the 
                        selected features and the second item is the target label

    Example call:
    ---
    features_and_targets = identify_features_and_targets(encoded_dataframe)
    '''

    #################	ADD YOUR CODE HERE	##################

    # drop the target (leaveOrNot)
    features = encoded_dataframe.drop(columns=['LeaveOrNot'])

    # store in target = 'LeaveOrNot'
    target = encoded_dataframe['LeaveOrNot']

    # Create a list containing the features and target label
    features_and_targets = [features, target]

    ##########################################################

    return features_and_targets

# Example call:
# features_and_targets = identify_features_and_targets(encoded_dataframe)

def load_as_tensors(features_and_targets):
    '''
    Purpose:
    ---
    This function aims at loading your data (both training and validation)
    as PyTorch tensors. Here you will have to split the dataset for training 
    and validation and then load them as tensors. 
    Training of the model requires iterating over the training tensors. 
    Hence the training tensors need to be converted to an iterable dataset
    object.
    
    Input Arguments:
    ---
    `features_and_targets` : [ list ]
                            Python list in which the first item is the 
                            selected features, and the second item is the target label
    
    Returns:
    ---
    `tensors_and_iterable_training_data` : [ list ]
                                            Items:
                                            [0]: X_train_tensor: Training features loaded into PyTorch array
                                            [1]: X_test_tensor: Feature tensors in validation data
                                            [2]: y_train_tensor: Training labels as PyTorch tensor
                                            [3]: y_test_tensor: Target labels as tensor in the validation data
                                            [4]: Iterable dataset object and iterating over it in 
                                                 batches, which are then fed into the model for processing

    Example call:
    ---
    tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
    '''

    #################	ADD YOUR CODE HERE	##################

    # Extract features and target from the input list
    features, target = features_and_targets

    # Split the data into training and validation sets (adjust the ratio as needed)
    train_ratio = 0.8  # 80% for training, 20% for validation
    num_samples = len(features)
    num_train_samples = int(train_ratio * num_samples)

    X_train = features[:num_train_samples]
    y_train = target[:num_train_samples]

    X_val = features[num_train_samples:]
    y_val = target[num_train_samples:]

    # Convert the training and validation data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)  # Reshape target to [batch_size, 1]
    X_val_tensor = torch.FloatTensor(X_val.values)
    y_val_tensor = torch.FloatTensor(y_val.values).view(-1, 1)  # Reshape target to [batch_size, 1]

    # Create a TensorDataset for training data
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    # Create DataLoader for training data to make it iterable in batches
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    tensors_and_iterable_training_data = [X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor, train_loader]

    return tensors_and_iterable_training_data

# Example call:
# tensors_and_iterable_training_data = load_as_tensors(features_and_targets)

class Salary_Predictor(nn.Module):
    # input_dim = 4634 
    
    def __init__(self, input_dim):
        super(Salary_Predictor, self).__init__()

        self.fc1 = nn.Linear(len(encoded_dataframe.columns) -1, 512)
        self.relu1 = nn.ReLU()               # Activation function (ReLU)
        self.fc2 = nn.Linear(512, 256)        # Hidden layer
        self.relu2 = nn.ReLU()               # Activation function (ReLU)
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()               # Activation function (ReLU)
        self.fc4 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(64, 32)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(32, 1)         # Output layer
        
        # self.fc1 = nn.Linear(len(encoded_dataframe.columns) -1, 30)
        # self.relu1 = nn.ReLU()               # Activation function (ReLU)
        # self.fc2 = nn.Linear(30, 30)
        # self.relu2 = nn.ReLU()               # Activation function (ReLU)
        # self.fc3 = nn.Linear(30, 1)
        # # self.relu3 = nn.ReLU()               # Activation function (ReLU)
        # # self.fc4 = nn.Linear(32, 1)
       
        
      
    def forward(self, x):
        # Define the forward pass
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        x = self.relu5(x)
        x = self.fc6(x)
        
        return torch.sigmoid(x)  # Apply sigmoid activation for binary classification

# Example usage:
# input_dim should match the number of input features
# model = Salary_Predictor(input_dim)
# predicted_output = model(input_data)



def model_loss_function():
    '''
    Purpose:
    ---
    To define the loss function for the model. The loss function measures 
    how well the predictions of a model match the actual target values 
    in training data.
    
    Input Arguments:
    ---
    None

    Returns:
    ---
    `loss_function`: This is a pre-defined loss function in PyTorch
                    suitable for binary classification tasks

    Example call:
    ---
    loss_function = model_loss_function()
    '''

    #################	ADD YOUR CODE HERE	##################

    # Define the loss function for binary classification (cross-entropy loss)
    loss_function = nn.BCELoss()  # BCELoss stands for Binary Cross-Entropy Loss
    # loss_function = nn.MSELoss()
    ##########################################################

    return loss_function

# Example call:
# loss_function = model_loss_function()



def model_optimizer(model):
    '''
    Purpose:
    ---
    To define the optimizer for the model. The optimizer is responsible 
    for updating the parameters (weights and biases) in a way that 
    minimizes the loss function.
    
    Input Arguments:
    ---
    `model`: An object of the 'Salary_Predictor' class

    Returns:
    ---
    `optimizer`: Pre-defined optimizer from PyTorch (e.g., SGD or Adam)

    Example call:
    ---
    optimizer = model_optimizer(model)
    '''

    #################	ADD YOUR CODE HERE	##################

    # Define the optimizer
    learning_rate = 0.001  # You can adjust the learning rate as needed
    optimizer = optim.Adamax(model.parameters(), lr=learning_rate)

    ##########################################################

    return optimizer

# Example call:
# optimizer = model_optimizer(model)

def model_number_of_epochs():
    '''
    Purpose:
    ---
    To define the number of epochs for training the model

    Input Arguments:
    ---
    None

    Returns:
    ---
    `number_of_epochs`: [integer value]

    Example call:
    ---
    number_of_epochs = model_number_of_epochs()
    '''

    #################	ADD YOUR CODE HERE	##################

    # Define the number of epochs
    number_of_epochs = 20  # we can adjust this value based on your experimentation

    ##########################################################

    return number_of_epochs

# Example call:
# number_of_epochs = model_number_of_epochs()


def training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer):
    '''
    Purpose:
    ---
    All the required parameters for training are passed to this function.

    Input Arguments:
    ---
    1. `model`: An object of the 'Salary_Predictor' class
    2. `number_of_epochs`: For training the model
    3. `tensors_and_iterable_training_data`: list containing training and validation data tensors 
                                             and an iterable dataset object of training tensors
    4. `loss_function`: Loss function defined for the model
    5. `optimizer`: Optimizer defined for the model

    Returns:
    ---
    trained_model: The model after training

    Example call:
    ---
    trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer)
    '''

    #################	ADD YOUR CODE HERE	##################

    # Extract training data and DataLoader
    X_train_tensor, _, y_train_tensor, _, train_loader = tensors_and_iterable_training_data

    # Training loop
    for epoch in range(number_of_epochs):
        model.train()  # Set the model to training mode

        for inputs, targets in train_loader:
            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            outputs = model(inputs)
            loss = loss_function(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Print training loss for each epoch (optional)
        print(f"Epoch [{epoch + 1}/{number_of_epochs}] - Loss: {loss.item()}")

    return model

# Example call:
# trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer)

def validation_function(trained_model, tensors_and_iterable_training_data):
    '''
    Purpose:
    ---
    This function will utilize the trained model to make predictions on the
    validation dataset. This will enable us to understand the accuracy of
    the model.

    Input Arguments:
    ---
    1. `trained_model`: The model returned from the training function
    2. `tensors_and_iterable_training_data`: List containing training and validation data tensors 
                                             and an iterable dataset object of training tensors

    Returns:
    ---
    model_accuracy: Accuracy on the validation dataset

    Example call:
    ---
    model_accuracy = validation_function(trained_model, tensors_and_iterable_training_data)

    '''   
    #################    ADD YOUR CODE HERE    ##################

    # Extract validation data and DataLoader
    _, X_val_tensor, _, y_val_tensor, _ = tensors_and_iterable_training_data

    # Set the model to evaluation mode
    trained_model.eval()

    # Disable gradient computation during validation
    with torch.no_grad():
        # Forward pass on the validation data
        outputs = trained_model(X_val_tensor)
        
        # Convert predicted probabilities to binary predictions (0 or 1)
        predicted_labels = (outputs >= 0.5).float()
        
        # Calculate accuracy
        correct_predictions = (predicted_labels == y_val_tensor).sum().item()
        total_samples = y_val_tensor.size(0)
        model_accuracy = correct_predictions / total_samples

    ##########################################################

    return model_accuracy

# Example call:
# model_accuracy = validation_function(trained_model, tensors_and_iterable_training_data)

'''
	Purpose:
	---
	The following is the main function combining all the functions
	mentioned above. Go through this function to understand the flow
	of the script

'''


if __name__ == "__main__":

	# reading the provided dataset csv file using pandas library and 
	# converting it to a pandas Dataframe
	task_1a_dataframe = pandas.read_csv('task_1a_dataset.csv')

	# data preprocessing and obtaining encoded data
	encoded_dataframe = data_preprocessing(task_1a_dataframe)

	# selecting required features and targets
	features_and_targets = identify_features_and_targets(encoded_dataframe)

	# obtaining training and validation data tensors and the iterable
	# training data object
	tensors_and_iterable_training_data = load_as_tensors(features_and_targets)


	model = Salary_Predictor(9)


	# obtaining loss function, optimizer and the number of training epochs
	loss_function = model_loss_function()
	optimizer = model_optimizer(model)
	number_of_epochs = model_number_of_epochs()

	# training the model
	trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data,loss_function, optimizer)

	# validating and obtaining accuracy
	model_accuracy = validation_function(trained_model,tensors_and_iterable_training_data)
	print(f"Accuracy on the test set = {model_accuracy}")

	X_train_tensor = tensors_and_iterable_training_data[0]
	x = X_train_tensor[0]
	jitted_model = torch.jit.save(torch.jit.trace(model, (x)), "task_1a_trained_model.pth")
