

In this lab, we will perform one **regression** and one **multi-class classification** tasks. We will consider the Boston house Price Dataset for the regression problem, and for multi-class classification, we will consider Fashion-MNIST Dataset. For Boston house Price dataset details visit - https://scikit-learn.org/stable/datasets/toy_dataset.html. For Fashion-MNIST dataset you may get it using keras (see documentation) Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist

---



Section 1:  **(Total points = 50)**

---

Q1. Develop a Multi-Layer Perceptron (MLP) Neural Network to predict the house prices (Dataset loading function and accessing data matrix and target values have been provided in the code section) *Use of built-in library functions for the specific implementation as asked in the questions are not allowed in this section*

1. Perform an exploratory analysis on the given dataset. Print the avg, max & min values of each column. Also, show the coorelation of each column with the target values in the dataset using multi-plots. **(3 point)**
2. Perform data pre-processing operations like standardization and splitting (80:20) of data.  **(2 points)**
3. Implement the three layer fully connected MLP feedforward model with only one hidden layer having 15 hidden units + bias. For this problem make your decision on the number of output units. Also, your implementation should facilitate a choice between Sigmoid & Tanh actionations at each layer of MLP for the user.  **( 15 points)**
4. Implement Back-propagation algorithm to train the parameters of the MLP created in the previous section. The Backpropagation should support gradient flow for both Sigmoid and Tanh activation functions. **(15 points)**
5. Train your model using the Mean Sqaured Errors. Mention your choices of the hyperparameters for training. Perform traning with batch gradient descent and stochastic gradeint descent. Plot the graph of traning error versus Epochs for both the training methods. Report the final accuracy you achieved on the Test Data using both the traning methods. **(2+3 = 5)**
6. Using the best traning method from the above traning, train your MLP with different learning rates given as [ 0.5, 0.1, 0.01, 0.001, 0.0001].  **(5 points)**
7. Plot the training error versus epochs for each learning rate in a single line graph. Also, plot accuracy versus lerning rate. Comment on your observations. **(3+2 = 5 points)**


#Declaration Block
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np

housing = load_boston() # Loading the housing data
#housing

data = pd.DataFrame(housing["data"]) # Input data 
data.columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
data

housing["feature_names"] # List of features

target = pd.DataFrame(housing["target"]) # Ground truth house prices for each row of data
target

"""# Part 1: Exploratory data analysis on Boston Housing Price Dataset"""

# Exploratory data analysis on Boston Housing Price Dataset
# Summary information
data.info()

# Missing Values
data.isnull().sum()

# Basic Descriptive Statistics Value and the avg, max & min values of each column
data.describe()

# Histogram Distribution
import matplotlib.pylab as plt
for name in data.columns:
    plt.title(name)
    plt.hist(data[name], bins=50)
    plt.show()

"""From the above figure, we can infer that the data of “CHAS” and “RAD” are NOT continuous values. Generally, such data that is not continuous is called a categorical variable. For the above reasons, let’s check the categorical variables individually."""

data['RAD'].value_counts()

# It is important to check the data visually.
data['CHAS'].value_counts().plot.bar(title="CHAS")

data['RAD'].value_counts().plot.bar(title="RAD")

# Correlation of Variables
data.corr()

import seaborn as sns

data['MEDV'] = housing.target
correlation_matrix = data.corr().round(2)
# annot = True to print the values inside the square
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(data=correlation_matrix, annot=True,ax=ax)

"""Correlation of each column with target column."""

sns.pairplot(data=data,y_vars=['MEDV'],x_vars=list(housing["feature_names"]),height=2)

"""# Part 2: Perform data pre-processing operations like standardization and splitting (80:20) of data."""

# Standardize and Splitting Dataset
def train_test_split(df1):
    shuffle_df1 = df1.sample(frac=1)
    train_size = int(0.8 * len(df1))
    df_train = shuffle_df1[:train_size]
    df_test = shuffle_df1[train_size:]
    return df_train,df_test
def dataset_split(df):
    X = df.drop([df.columns[-1]], axis = 1)
    y = df[df.columns[-1]]
    return X, y

# Standardize
for column in data:
    data[column] = (data[column] - data[column].mean()) / data[column].std()
data['Prices']=target
data_train,data_test=train_test_split(data)
X_train,y_train=dataset_split(data_train)
X_test,y_test=dataset_split(data_test)
X_test = X_test.drop(columns=['MEDV'])
X_train = X_train.drop(columns=['MEDV'])

X_train=X_train.to_numpy()
X_test = X_test.to_numpy()

y_train=y_train.to_numpy()
y_test=y_test.to_numpy()

y_test = (y_test-np.mean(y_test,axis=0))/np.std(y_test,axis=0)
y_train = (y_train-np.mean(y_train,axis=0))/np.std(y_train,axis=0)

y_train = y_train.reshape(1,y_train.shape[0])
y_test=y_test.reshape(1,y_test.shape[0])

print(type(X_train), X_train.shape)
print(type(y_train), y_train.shape)
print(type(X_test), X_test.shape)
print(type(y_test), y_test.shape)

"""# Part 3:Implement the three layer fully connected MLP feedforward model with only one hidden layer having 15 hidden units + bias. 
  
# Part 4: Implement Back-propagation algorithm to train the parameters of the MLP created in the previous section.


Both part in one section
"""

# We will perform customized forward and backward propagation




# Define Abstract class
# class Layer():


# Define Fully connected Layer
# class FCLayer():

  # Forward Propagation

  # Backward Propagation



# Activation Layer (ReLU) 
# class ActivationLayer():
    # Forward Propagation

    # Backward Propagation

# Define activation function and its derivative
# Logistic


# Hyperbolic tangent


# Rectified linear
class MLP :
  def create(self,input_size,output_size,hidden_dims,output_type,seed=None,activation='sigmoid'):
    self.layer_dims=[input_size]+hidden_dims+[output_size]
    self.W = {}
    self.b = {}
    self.activation = activation
    self.output_type = output_type
    self.train_cost=[]
    self.L = len(self.layer_dims)-1

    if seed == None:
      np.random.seed(seed)

    for i in range(self.L):
      self.W[i+1] = np.random.randn(self.layer_dims[i+1],self.layer_dims[i])
      self.b[i+1] = np.zeros((self.layer_dims[i+1],1))
    for i in range(self.L):
      # self.W[i+1] = self.W[i+1]*np.sqrt(2/(self.layer_dims[i]))
      self.W[i+1] = self.W[i+1]*np.sqrt(1/(self.layer_dims[i]))

  def sigmoid(self,X):
    return 1/(1+np.exp(-X))

  def sigmoid_grad(self,X):
    return self.sigmoid(X)*(1-self.sigmoid(X))

  def tanh(self,X):
    return np.tanh(X)

  def tanh_grad(self,X):
    return 1-((self.tanh(X))**2)

  def forward_propagation(self,X,dropout=False):
    self.Z = {}
    self.A = {}

    self.A[0] = X

    for i in range (len(self.layer_dims)-2):
      self.Z[i+1] = np.matmul(self.W[i+1],self.A[i])+self.b[i+1]
      _ = "self.A[i+1] = self."+self.activation+"(self.Z[i+1])"
      exec(_)

    self.Z[self.L] = np.matmul(self.W[self.L],self.A[len(self.layer_dims)-2])+self.b[self.L]
    self.A[self.L] = self.Z[self.L] 
    return self.A


# Define Loss function (Use mean square error)
  def compute_cost(self,Y_pred,Y_true):
    cost =(1/(2*Y_true.shape[1]))*np.sum((Y_pred-Y_true)**2)
    return cost
  
  
  # Implement Backpropagation
  def backward_propagation(self,Y):
    self.dZ = {}
    self.dA = {}
    self.dW = {}
    self.db = {}

    self.dZ[self.L] = self.A[self.L]-Y

    for i in range(self.L,0,-1):

      self.dW[i] = (1/self.dZ[i].shape[1])*np.matmul(self.dZ[i],self.A[i-1].T)

      self.dW[i] += 0.1*self.W[i]
        
      self.db[i] = (1/self.dZ[i].shape[1])*np.sum(self.dZ[i],axis=1,keepdims=True)
      _ = "self.dZ[i-1] = np.matmul(self.W[i].T,self.dZ[i])*self."+self.activation+"_grad(self.A[i-1])"
      exec(_) 
      
    return (self.dW,self.db)


# Training Network
  def train(self,X_train,Y_train,X_val,Y_val,optimizer='gd',
            keep_probs=[],mini_batch_size=32,epochs=100,learning_rate=0.01,
            print_loss_freq=100,plot_loss=True):
    

    if keep_probs != []:
      self.keep_probs = keep_probs
    else:
      self.keep_probs = [1]*(len(self.layer_dims)-2)

    self.print_loss_freq = print_loss_freq  

    self.Mw = {}
    self.Mb = {}
    self.Vw = {}
    self.Vb = {}

    for i in range(self.L):
      self.Mw[i+1] = np.zeros(shape=self.W[i+1].shape)
      self.Mb[i+1] = np.zeros(shape=self.b[i+1].shape)
      self.Vw[i+1] = np.zeros(shape=self.W[i+1].shape)
      self.Vb[i+1] = np.zeros(shape=self.b[i+1].shape)

    # train_cost = []
    val_cost = []
    train_acc = []
    val_acc = []
    m = X_train.shape[1]

    drop = False
    # if(self.regularizer == 'dropout'):
    #   drop = True

    t = 1
    
    for e in range(epochs):

      mask = np.random.permutation(m)

      X_train = X_train[:,mask]
      Y_train = Y_train[:,mask]

      if optimizer == 'gd':

        for i in range(0,m,mini_batch_size):

          _ = self.forward_propagation(X_train[:,i:(i+mini_batch_size)],drop)
          _ = self.backward_propagation(Y_train[:,i:(i+mini_batch_size)])
          
          for i in range(self.L):
            self.W[i+1] -= learning_rate*self.dW[i+1]
            self.b[i+1] -= learning_rate*self.db[i+1]
      elif optimizer =='sgd':
        p= learning_rate / (1 + epochs * 1)
        for i in range(X_train.shape[1]):
          indexes = np.random.randint(0, m, mini_batch_size) # random sample
          Xs = np.take(X_train.T, indexes)
          ys = np.take(Y_train, indexes)
          _ = self.forward_propagation(X_train[:,i:(i+mini_batch_size)],drop)
          _ = self.backward_propagation(Y_train[:,i:(i+mini_batch_size)])
          
          for i in range(self.L):
            if learning_rate < 1*e^-7:
              p=learning_rate
            self.W[i+1] -= p*self.dW[i+1]
            self.b[i+1] -= p*self.db[i+1]

      Y_pred_train = self.forward_propagation(X_train)[self.L]
      Y_pred_val = self.forward_propagation(X_val)[self.L]
                                   
      self.train_cost.append(self.compute_cost(Y_pred_train,Y_train))
      val_cost.append(self.compute_cost(Y_pred_val,Y_val))

      if (e+1)%self.print_loss_freq==0:
        print("After "+str(e+1)+" epochs :    Training Loss = "+str(self.train_cost[e]) + 
                "    Validation Loss = "+str(val_cost[e])+'\n')  

    if plot_loss == True:

      r = list(range(1,epochs+1))
      plt.plot(r,self.train_cost,'r',label="Training Error")
      # plt.plot(r,val_cost,'b',label="Validation error")
      plt.xlabel('Epochs')
      if self.output_type == 'regression':
        plt.ylabel('Error')
      plt.legend()
      plt.show()

      print("\nTraining Error : ",self.train_cost[-1])
      # print("\nValidation Error : ",val_cost[-1]) 

    return (self.train_cost,val_cost,train_acc,val_acc)   

  def predict(self,X):
    A = self.forward_propagation(X)

    if self.output_type == 'regression':
      return A[self.L]

"""# Part 5: Train your model using the Mean Sqaured Errors. Mention your choices of the hyperparameters for training.

#Using Batch Gradient descent:
Hyperparameter: learning rate is 0.01 and activation function is tanh and 100 epoch
"""

# Using Batch Gradient descent
model = MLP()

model.create(13,1,[15,],output_type='regression',activation='tanh')

costs = model.train(X_train.T,y_train,X_test.T,y_test,optimizer='gd',
                    mini_batch_size=X_train.shape[0],epochs=100,print_loss_freq=100,
                    learning_rate=0.01)

Y_p = model.predict(X_test.T)
print("Test error value: ",model.compute_cost(Y_p,y_test.reshape(1,102)))

p=((y_test-np.mean(y_test))**2)
q=((y_test-Y_p)**2)
print("Model Accuracy using gradient: ",1-model.compute_cost(Y_p,y_test.reshape(1,102)))

"""#Using stochastic Gradient descent:
Hyperparameter: learning rate is 0.01 and activation function is tanh and iteration=100
"""

# Using stochastic gradeint descent
model = MLP()

model.create(13,1,[15,],output_type='regression',activation='tanh')

costs = model.train(X_train.T,y_train,X_test.T,y_test,optimizer='sgd',
                    mini_batch_size=1,epochs=100,print_loss_freq=10,
                    learning_rate=0.01)

# Test the Model
Y_p = model.predict(X_test.T)
print("Test model error: ",model.compute_cost(Y_p,y_test.reshape(1,102)))

p=((y_test-np.mean(y_test))**2)
q=((y_test-Y_p)**2)
# print(np.sum(q))
# print(np.sum(p))
# print(np.sum(q)/np.sum(p))
print("Model Accuracy using stochastic gradient descent: ",1-model.compute_cost(Y_p,y_test.reshape(1,102)))

"""#Part 6: Using the best traning method from the above traning, train your MLP with different learning rates given as [ 0.5, 0.1, 0.01, 0.001, 0.0001]

Hyperparameter of best model used in this experiment is: Activation function Tanh, epoch is 100 and optimizer is Stochastic gradient descent.
"""

# Using the best model for learning rate 0.0001
model = MLP()

model.create(13,1,[15,],output_type='regression',activation='tanh')

costs = model.train(X_train.T,y_train,X_test.T,y_test,optimizer='sgd',
                    mini_batch_size=1,epochs=100,print_loss_freq=10,
                    learning_rate=0.0001,plot_loss=False)
st1=model.train_cost
Y_p = model.predict(X_test.T)
pt1=1-model.compute_cost(Y_p,y_test.reshape(1,102))

# Using the best model for learning rate 0.001
model = MLP()

model.create(13,1,[15,],output_type='regression',activation='tanh')

costs = model.train(X_train.T,y_train,X_test.T,y_test,optimizer='sgd',
                    mini_batch_size=1,epochs=100,print_loss_freq=10,
                    learning_rate=0.001,plot_loss=False)
st2=model.train_cost
Y_p = model.predict(X_test.T)
pt2=1-model.compute_cost(Y_p,y_test.reshape(1,102))

# Using the best model for learning rate 0.01
model = MLP()

model.create(13,1,[15,],output_type='regression',activation='tanh')

costs = model.train(X_train.T,y_train,X_test.T,y_test,optimizer='sgd',
                    mini_batch_size=1,epochs=100,print_loss_freq=10,
                    learning_rate=0.01,plot_loss=False)
st3=model.train_cost
Y_p = model.predict(X_test.T)
pt3=1-model.compute_cost(Y_p,y_test.reshape(1,102))

# Using the best model for learning rate 0.1
model = MLP()

model.create(13,1,[15,],output_type='regression',activation='tanh')

costs = model.train(X_train.T,y_train,X_test.T,y_test,optimizer='sgd',
                    mini_batch_size=4,epochs=100,print_loss_freq=10,
                    learning_rate=0.1,plot_loss=False)
st4=model.train_cost
Y_p = model.predict(X_test.T)
pt4=1-model.compute_cost(Y_p,y_test.reshape(1,102))

# Using the best model for learning rate 0.5
model = MLP()

model.create(13,1,[15,],output_type='regression',activation='tanh')

costs = model.train(X_train.T,y_train,X_test.T,y_test,optimizer='sgd',
                    mini_batch_size=4,epochs=100,print_loss_freq=10,
                    learning_rate=0.5,plot_loss=False)
st5=model.train_cost
Y_p = model.predict(X_test.T)
pt5=1-model.compute_cost(Y_p,y_test.reshape(1,102))

"""# Part 7: Plot the training error versus epochs for each learning rate in a single line graph. Also, plot accuracy versus lerning rate."""

# Plots
# Plot for training error against each learning rate
r =list(range(1,101))
plt.plot(st1,label="lr=0.0001")
plt.plot(st2,label="lr=0.001")
plt.plot(st3,label="lr=0.01")
plt.plot(st4,label="lr=0.1")
plt.plot(st5,label="lr=0.5")
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend(loc='upper right')
plt.title("Training Error vs Epoch")
# plt.legend(['train','test'], loc='upper left')
plt.show()

#print(pt1,pt2,pt3,pt4,pt5)

# Plot for training accuracy against each learning rate
r =list(range(1,6))
q=[pt1,pt2,pt3,pt4,pt5]
plt.plot(r,q)
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.xticks([r + 0.25 for r in range(1,len(q)+1)],['0.0001','0.001','0.01','0.1','0.5'])
# plt.legend(loc='upper right')
# plt.legend(['train','test'], loc='upper left')
plt.title("Accuracy vs Learning Rate")
plt.show()

# Challenges and Observations

"""#Part 7: Observation
1. From above experiment on different learning rate we can observe that high learning rate for a regression type of problem solving model can cause unexpected behaviour for learning cure of a neural network model. 

2.  We know that the learning rate controls how quickly the model is adapted to the problem. Smaller learning rates require more training epochs given the smaller changes made to the weights each update, whereas larger learning rates result in rapid changes and require fewer training epochs.

3. The plots show oscillations in behavior for the too-large learning rate of 0.5 and the inability of the model to learn anything with the too-small learning rates of 0.0001. 

4. On several experiment it has been found that learning rate that is too large can cause the model to converge too quickly to a suboptimal solution, whereas a learning rate that is too small can cause the process to get stuck. 

5. The challenge of training deep learning neural networks involves carefully selecting the learning rate. It may be the most important hyperparameter for the model. 

6.  Above experiment gives us a nice intution about how a learning rate for a model should be chosen. As we can see for our model learning rate of value 0.01 semms to be well tuned for all of the experiment. Along with SGD optimizer and Tanh activation function gives us a preety good result for this problem.

---



Section 2:  **(Total points = 50)**

---
Q2. In this question, we will learn to perform multi-class classification on Fashion-MNIST using a convolutional neural network. 


1. Explore the Dataset. Display one randomly selected image from each fashion class. **(5 points)**

2. Plot the distribution of number of images in each fashion class. Is the distribution uniform? Comment. **(5 points)**

3. Implememnt the 2D convolution function using a kernel size of 3x3. Use the [sobel kernel](https://en.wikipedia.org/wiki/Sobel_operator) and display the feature map for one example from each fashion class. For using sobel kernel, you need to calculate two convolutions, one for x-direction (x) and one for y-direction (y), the feature map then can be calculated as $F_m = \sqrt(x^2 + y^2)$**(10 points)**

4. Using the Keras library, implement a CNN model for classification. Use the following network architecture:  **(5 points)**
*  Input layer
*  Con2D with 32 3x3 kernals and ReLU activation
*  Max Pooling layer with pool-size 2x2
*  Con2D with 64 3x3 kernals and ReLU ReLU activation
*  Max Pooling layer with size 2x2
*  Dropout
*  Fully Connected Layer with softmax activation

3. Implement a custom cross-entropy loss (error function) for the multi-class classification. Use it for traning the model. **(5 points)**

4. Compile and train your model with four different optimizers viz. SGD
RMSprop, Adam, Adagrad. Plot the training loss for all four optimizers. Comment on your observations. **(10 points)**

5. Choose different hyperparameters for Conv Layers, change number of Conv layer and drop-out rate and train your model. Plot training and test accuracies and losses wrt epochs for different hyper-parameters. Do you find any improvement in classification performance. Report your analysis. **(10 points)**
"""

# import Fashion MNIST dataset
import keras
from keras.datasets import fashion_mnist
import numpy as np

fashion_data = fashion_mnist.load_data() #load dataset
fashion_data

"""# Part 1: Explore the Dataset. Display one randomly selected image from each fashion class."""

#Dataset Exploration
import matplotlib.pylab as plt
(x_train, y_train), (x_test, y_test)=fashion_data
x_train = x_train / 255.0
x_test = x_test / 255.0
print("Train dataset shape: ",x_train.shape)
print("Test dataset shape: ",x_test.shape)
print("Number of Classes: ",len(np.unique(y_train)))
fig, axes = plt.subplots(2, 5, figsize=(3*5,3*2))
for i in range(len(np.unique(y_train))):
  indx=np.random.choice(np.where(y_train==i)[0])
  ax = axes[i//5, i%5]
  ax.imshow(np.squeeze(x_train[indx])) #,cmap=plt.get_cmap('gray'))
  ax.set_title('Label: {}'.format(y_train[indx]))
plt.tight_layout()
plt.show()

"""# Part 2: Plot the distribution of number of images in each fashion class."""

# Plot the distribution of number of images in each fashion class
labels_map = {0 : "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
          5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}

def get_classes_distribution(data):
    # Get the count for each label
    unique, counts = np.unique(data, return_counts=True)
    label_counts=dict(zip(unique, counts))
    #label_counts = data.value_counts()
    print(label_counts)
    # Get total number of samples
    total_samples = len(data)


    # Count the number of items in each class
    for i in range(len(label_counts)):
        label = labels_map[i]
        count = label_counts[i]
        #count=np.where(y_train==i)[0]
        percent = (count / total_samples) * 100
        print("{:<20s}:   {} or {}%".format(label, count, percent))

print("Class distribution of train data: ")
get_classes_distribution(y_train)
print("Class distribution of test data: ")
get_classes_distribution(y_test)

#print(y_test.label)
import pandas as pd
import seaborn as sns
def plot_label_per_class(data):
    f, ax = plt.subplots(1,1, figsize=(12,4))
    g = sns.countplot(data.label, order = [x for x in range(10)])
    g.set_title("Number of labels for each class")
    i=0
    for p, label in zip(g.patches, data["label"].value_counts().index):
        g.annotate(labels_map[i], (p.get_x(), p.get_height()+0.1))
        i+=1
    plt.show()  
df = pd.DataFrame(y_test,columns =["label"])
print("Class distribution plot of test data: ")
plot_label_per_class(df)
df = pd.DataFrame(y_train,columns =["label"])   
print("Class distribution plot of train data: ")
plot_label_per_class(df)

"""# Part 2: Observation 
1. From above distribution plot and also the data extrated from the given dataset we can see that all data points of test dataset and train dataset is equally distributed among all the classes. 
2. For train dataset all classes have 6000 data points each and in test data all classes has 1000 data points each.

So, using this experiment we can ensure that dataset is uniformly distributed among each class of this dataset.

# Part 3:
# Implememnt the 2D convolution function using a kernel size of 3x3. Use the sobel kernel and display the feature map for one example from each fashion class.
"""

def convolution(image, kernel, average=False):
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
    output = np.zeros(image.shape)
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image
    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]

    return output

# Sobel Kernel
def sobel_kernel(image, filter):
    new_image_x = convolution(image, filter)
    new_image_y = convolution(image, np.flip(filter.T, axis=0))
    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    return gradient_magnitude

# Display the feature map for one example from each fashion class
filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
fig, axes = plt.subplots(2, 5, figsize=(3*5,3*2))
for i in range(len(np.unique(y_train))):
  indx=np.random.choice(np.where(y_train==i)[0])
  ax = axes[i//5, i%5]
  a=sobel_kernel(x_train[indx], filter)
  ax.imshow(np.squeeze(a),cmap=plt.cm.binary) #,cmap=plt.get_cmap('gray'))
  ax.set_title('Label: {}'.format(labels_map[y_train[indx]]))
plt.tight_layout()
plt.show()

"""# Part 4: Using the Keras library, implement a CNN model for classification."""

# Model Structure 
model = keras.models.Sequential()
model.add(
        keras.layers.Conv2D(
        filters=32, # How many filters we will learn 
        kernel_size=(3, 3), # Size of feature map that will slide over image
        #strides=(1, 1), # How the feature map "steps" across the image
        #padding='valid', # We are not using padding
        activation='relu', # Rectified Linear Unit Activation Function
        input_shape=(28, 28, 1) # The expected input shape for this layer
    )
) 
model.add(
    keras.layers.MaxPooling2D(
        pool_size=(2, 2), # Size feature will be mapped to
        #strides=(2, 2) # How the pool "steps" across the feature
    )
)
model.add(
        keras.layers.Conv2D(
        filters=64, # How many filters we will learn 
        kernel_size=(3, 3), # Size of feature map that will slide over image
        #strides=(1, 1), # How the feature map "steps" across the image
        #padding='valid', # We are not using padding
        activation='relu', # Rectified Linear Unit Activation Function
        #input_shape=(28, 28, 1) # The expected input shape for this layer
    )
)
model.add(
    keras.layers.MaxPooling2D(
        pool_size=(2, 2), # Size feature will be mapped to
        #strides=(2, 2) # How the pool "steps" across the feature
    )
)
model.add(
    keras.layers.Dropout(
        rate=0.25 # Randomly disable 25% of neurons
    )
)
model.add(
    keras.layers.Flatten()
)

# A dense (interconnected) layer is added for mapping the derived features 
# to the required class.
# model.add(
#     tf.keras.layers.Dense(
#         units=128, # Output shape
#         activation='relu' # Rectified Linear Unit Activation Function
#     )
# )
model.add(
    keras.layers.Dense(
        units=10, # Output shape
        activation='softmax' # Softmax Activation Function
    )
)

"""# Part 5: Implement a custom cross-entropy loss (error function) for the multi-class classification"""

# Implement a custom cross-entropy loss (error function) for the multi-class classification
import tensorflow as tf
def cross_entropy(y,y_pre):
  # print("y true shape",y.shape, y)
  # print("y pred shape",y_pre.shape, y_pre)
  #y=tf.cast(y, tf.float32)
  # print("y true shape",y)
  loss=-tf.math.reduce_sum(y*tf.math.log(y_pre))
  # print("y pred shape",y_pre.shape, y_pre)
  return loss/float(y_pre.shape[1])

"""# Part 6: Compile and train your model with four different optimizers viz. SGD RMSprop, Adam, Adagrad. Plot the training loss for all four optimizers"""

# Training of the model 
# With adagrad optimizer
model.compile(
    loss=cross_entropy, # loss function
    optimizer='adagrad', # optimizer function
    metrics=['accuracy'] # reporting metric
)
print(model.summary())

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
#print(x_train.shape)
# Train the CNN on the training data
history = model.fit(
    
      # Training data : features (images) and classes.
      x_train, y_train,
                    
      # number of samples to work through before updating the 
      # internal model parameters via back propagation.
      batch_size=256, 

      # An epoch is an iteration over the entire training data.
      epochs=10,

      # The model will set apart his fraction of the training 
      # data, will not train on it, and will evaluate the loss
      # and any model metrics on this data at the end of 
      # each epoch. 
      #validation_split=0.2, 
      validation_data=(x_test, y_test))

## Plots
for_adagrad=history.history['loss']
print(y_train.shape)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

# with SGD optimizer
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
model.compile(
    loss=cross_entropy, # loss function
    optimizer='sgd', # optimizer function
    metrics=['accuracy'] # reporting metric
)
#print(model.summary())
history = model.fit(x_train, y_train,batch_size=256, epochs=10, validation_data=(x_test, y_test))
for_sgd=history.history['loss']
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

# RmsProp
model.compile(
    loss=cross_entropy, # loss function
    optimizer='rmsprop', # optimizer function
    metrics=['accuracy'] # reporting metric
)
# print(model.summary())
print(model.summary())
history = model.fit(x_train, y_train,batch_size=256, epochs=10,validation_data=(x_test, y_test))
for_rmsprop=history.history['loss']
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# With Adam optimizer
model.compile(
    loss=cross_entropy, # loss function
    optimizer='adam', # optimizer function
    metrics=['accuracy'] # reporting metric
)
# print(model.summary())
print(model.summary())
history = model.fit(x_train, y_train,batch_size=256, epochs=10,validation_data=(x_test, y_test))
for_adam=history.history['loss']
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Plot of training loss for different optimizer
plt.plot(for_adagrad)
plt.plot(for_adam)
plt.plot(for_rmsprop)
plt.plot(for_sgd)
plt.title('model loss for different optimizer')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Adagrad', 'Adam','RmsProp','SGD'], loc='upper right')
plt.show()

"""# Part 6: Observation
1.  From above error graph plot and also above individual error graph plot we observe that Adam optimizer has produced some of the best result using this dataset. 

2.  Although this optimizer is computationally costly but this method is too fast and converges rapidly. It also rectifies vanishing learning rate, high variance quickly. 

3.  We have also noticed that Adagrad has worst result compared to the all other optimizer as the learning rate is always decreasing results in slow training. 

4.  For SGD we can say that it has high variance in model parameters and may shoot even after achieving global minima.If we want to get the same convergence as gradient descent we need to slowly reduce the value of learning rate.

5.  Although RmsProp also produces some good result, but in some senarios it may fall behind Adam as Adam optimizer combines the heuristics of both Momentum and RMSProp.

Adam is the best optimizers if one wants to train the neural network in less time and more efficiently.

# Part 7: Choose different hyperparameters for Conv Layers, change number of Conv layer and drop-out rate and train your model.

Hyperparameter: 3 convolutional layer size 32, 64, 128 and kernal size 3X3 and activation function as relu, 2 maxpooling layer with size (2,2), 2 dropout layer with dropout rate 0.5, one flatten layer, one dense layer with unit=128, and one output layer with softmax activation function. we have compile this code with adam optimizer and custom cross entropy function and fit the model with batch size 512 and 30 epochs.
"""

conv1 = keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1) )
conv2 = keras.layers.Conv2D(64, (3,3), activation='relu')
conv3 = keras.layers.Conv2D(128, (3,3), activation='relu')
max_pool_1 = keras.layers.MaxPooling2D((2,2))
max_pool_2 = keras.layers.MaxPooling2D((2,2))
max_pool_3 = keras.layers.MaxPooling2D((2,2))
flat_layer = keras.layers.Flatten()
fc = keras.layers.Dense(128, activation='relu')
output = keras.layers.Dense(10, 'softmax')
drop_1 = keras.layers.Dropout(0.5)
drop_2 = keras.layers.Dropout(0.5)
drop_3 = keras.layers.Dropout(0.5)
#model = keras.models.Sequential()

new_model = keras.models.Sequential()

new_model.add(conv1)
new_model.add(conv2)
new_model.add(max_pool_2)
new_model.add(drop_2)
new_model.add(conv3)
new_model.add(max_pool_3)
new_model.add(drop_3)
new_model.add(flat_layer)
new_model.add(fc)
new_model.add(output)
new_model.summary()

new_model.compile(optimizer='adam',
              loss=cross_entropy,
              metrics=['accuracy'])
history=new_model.fit(x_train, y_train, epochs=30, batch_size=512, shuffle=True, validation_split=0.2)

test_loss, test_accuracy = new_model.evaluate(x_test, y_test)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(['train', 'test'], loc='upper left')
plt.show()



# Challenges and observations

"""#Part 7: Challenges
1.  If we want to increase the number of Conv2D layer and maxplooing we can but it  will make the evaluation model more complex. 
2.  Eventually it creates limitation such that it will be somewhat more time taking as the model complexity goes on. 
3.  And sometimes using of 5 convolutional layers or more cannot create any type of improvement if you run the model less number of epochs. 

So we have to balance the model complexity as well as number of epochs to get the best performance out of it.

# Part 7: Observation and Analysis


1.   From above experiment we can say that we have achieved some amount of performance improment compared to what the default model was given in the question.

2.   We have already mentioned that Adam optimizer can give better result so we have chosen that for this experiment. Also as mentioned in the question we have tried to change the hyperparameters for this experiment. Such that, we have used 3 Convolutional layer with different sizes of 32, 64 and 128 with relu activation. Along with 2 maxpooling layer of size 2X2 and 2 dropout layer of value 0.5. And we have also used two dense l flatten layer and one dense layer with relu activation function and size 128. At last we have used one output layer with 10 units and softmax activation function.

3.   We have noticed that after running for 30 epochs it produced some significant improvement in classification using this model. 

4.   The addition of one Conv2D layer and maxpolling layer although makes the model complex but eventually it generating good output for our experiment. 

5.   Also we have to run for more about 100 epochs to get effective accuracy using these models. 

So we can say that  even with these model configuration we can see more improvement if we train it for more number of epochs.
"""