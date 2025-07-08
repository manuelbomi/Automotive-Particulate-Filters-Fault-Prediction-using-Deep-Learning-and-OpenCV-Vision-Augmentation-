# Automotive Diesel Particulate (Cordierite Filter based Catalytic Converter) Filters Images based Fault Identification using Deep Learning and OpenCV Vision Filters
---
In this project, deep learning is used to learn the features of faults on cordierite filters images. Manufacturing defects or faults from returned filters are divided into categories such as 'Axial Crack', 'Skin Crack', 'Melting Defect', 'Inlet','Outlet', 'Radial Crack'. Images corresponding to these faults are converted into numpy arrays, and the arrays are used used to train deep learning networks. The image categories are converted into deep learning training columns using one-hot encoding process as shown below:

---

![Image](https://github.com/user-attachments/assets/82e2e0f0-999c-4069-99f5-1d8a1ab203d6)
---

All images are converted to same size. The images are then divided into X-Train and X_test sets in ratio 0.9:0.1

---

![Image](https://github.com/user-attachments/assets/ff96579a-58d6-437f-aa1d-e4f64d78a490)
---

The images 'ID' (images ID such as 'img1', 'img2' etc) and 'Defects' (defect description columns such as 'Axial Crack') are used as target (Y) columns. They are divdied into Y-train and Y_test in ratio 09:0.1

---
![Image](https://github.com/user-attachments/assets/bda5b48b-7d35-4e82-8216-79070d132f0d)
---
## Model Architecture
---

 The output layer will have 6 neurons equal to the number of identifying metadata hot columns. Use sigmoid as activation. 
 Model can be optimized later by modifying the hypaparameters, e.g. number of hidden layers; etc.

```ruby


#### Initialize model constructor, sequential is a linear stack of neural network layers
model = Sequential() 
#### Add an input layer (optimization can be done here: input shape, etc can be modified). Input layer contains 16 hidden unit in this instance. Each image is 400x400 pixels in RGB (3 channels),
model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(400,400,3))) 
#### Covolutional and maxpooling will allow us to efficiently train on image data
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.25))
#### Add further layers. We can try softmax activation later to see if model performs better
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu')) 
#### Max pooling is a sample-based discretization process. Pooling is required to down sample the detection of features in feature maps. The objective is to down-sample an input representation (image, hidden-layer output matrix, etc.), reducing its dimensionality and allowing for assumptions to be made about features contained in the sub-regions binned
#### Reduce nos of parameters With max pooling,slide a 2x2 filter across the previous layer and take the max of the 4 values in the 2x2 filter
model.add(MaxPooling2D(pool_size=(2, 2))) 
#### Add dropout to prevent overfitting
model.add(Dropout(0.25)) 
#### RELU (Rectified linear activation unit activation function overcomes the vanishing gradient problem. It is the default activation when developing multilayer Perceptron and convolutional neural networks). Sigmoid (sigmoid()) and hyperbolic tangent (tahn()) activation functions cannot be used in networks with many layers due to the vanishing gradient problem
#### Try ELU or SELU (scaled exponential linear unit) activation function later?
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu")) 
#### Try average pooling or global max pooling later to reduce computation time? 
#### (GlobalMaxPool2D(); AveragePooling2D())
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.25)) 
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.25)) 
model.add(Flatten()) 
#### Add hidden layers
model.add(Dense(128, activation='relu')) 
model.add(Dropout(0.5)) 
#### Add hidden layers 
model.add(Dense(64, activation='relu')) 
model.add(Dropout(0.5)) 
#### model output layer, correspond to the nos of the hot encoding training column in our image .csv file. Modify this if you need to increase or decrease the nos of training columns
model.add(Dense(6, activation='sigmoid'))

```

---
## Benefits of Using RELU (Rectified Linear Unit) in the Model
---
f(x) = max(0, x)
ReLU (Rectified Linear Unit) is an activation function defined as f(x) = max(0, x). This means that for positive inputs, the gradient is always 1, and for negative inputs, the gradient is 0. 


#### Advantages of Using RELU
--
ReLU avoids the vanishing gradient problem: ReLU activation functions help mitigate the vanishing gradient problem by avoiding saturation for positive inputs, unlike sigmoid and tanh functions. ReLU's derivative is 1 for positive values, allowing gradients to flow more easily during backpropagation, which can help prevent the gradients from becoming extremely small in deep networks. 

#### Vanishing Gradient Problem
-
In deep neural networks, during backpropagation, gradients are multiplied across multiple layers. If the activation functions in these layers have gradients that are very small (especially close to zero), the gradients can become vanishingly small as they propagate backward, hindering learning in the early layers. 

#### How ReLU Helps with the Vanishing Gradient Problem
-
The constant gradient of 1 for positive inputs prevents the gradients from shrinking exponentially as they pass through ReLU layers. This allows the network to learn more effectively, especially in deep architectures. 

#### Benefits of Leaky ReLU and Parametric ReLU
-
A potential issue with ReLU is the "dying ReLU" problem. If a neuron consistently receives negative input, it will always output 0, and its gradient will be 0, effectively becoming inactive. This can be mitigated by using techniques like Leaky ReLU or Parametric ReLU, which introduce a small slope for negative inputs, ensuring that the neuron can still be activated and contribute to learning.

## Benefits of Using Sigmoid and Softmax for some layers
---
#### Sigmoid Function
Sigmoid is used for binary classification, while softmax is used for multi-class classification. Softmax can be seen as a generalization of sigmoid for multiple classes.

In image classification, the choice between using sigmoid or softmax activation functions in different layers, particularly the output layer, depends on the nature of the classification problem. 

#### Sigmoid for Hidden Layers and Multi-Label Classification:
Introducing Non-linearity: Sigmoid functions, like tanh, are often used in hidden layers because of their non-linear nature. This non-linearity is crucial for neural networks to approximate a wide range of functions and learn complex patterns in data, such as those present in images.

#### Multi-Label Classification: 
When an image can belong to multiple categories simultaneously (e.g., a photo containing both a cat and a dog), sigmoid activation is suitable for the output layer. Each sigmoid unit in the output layer predicts the probability of an image belonging to a specific class independently, without being constrained by other classes. For example, in chest X-ray image analysis, where an image might show signs of various diseases, using sigmoid for each disease allows independent assessment of each condition's probability. 

#### Softmax for Output Layers in Multi-Class Classification:
Mutually Exclusive Classes: In a typical image classification scenario, where an image belongs to only one category (e.g., classifying a handwritten digit as either a 7 or an 8, but not both), softmax is the preferred activation function for the output layer.
Probability Distribution: Softmax converts a vector of raw scores (logits) into a probability distribution over the classes. This means the outputs of the softmax function are non-negative and sum to 1. An increased probability for one class necessitates a decrease in the probability of other classes. For instance, in classifying handwritten digits, if the model assigns a high probability to the image being an "8," the probabilities for other digits (0-7 and 9) will be correspondingly lower. 

#### In summary:
#### Sigmoid: 
Good for introducing non-linearity in hidden layers and for multi-label classification where outputs are not mutually exclusive.
#### Softmax: 
Ideal for the output layer in multi-class classification where outputs are mutually exclusive and a probability distribution across classes is desired. 
Note: For binary classification (two mutually exclusive classes), a single sigmoid unit can be used in the output layer to estimate the probability of one class, with the probability of the other class being 1 minus that value. Softmax with two output units in this scenario is mathematically equivalen


## Compile, Fit and Train the Model
---
![Image](https://github.com/user-attachments/assets/6bd6adbd-fce8-4d9e-b883-e2c3998aaffb)

## Make Predictions with Unseen Images
---

![Image](https://github.com/user-attachments/assets/b3326d97-527e-4d11-9779-88037eaa5993)

