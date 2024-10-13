[Back to Index](./index.md)  [Previous.Page](./Page-5_Cost-Functions-Gradient-Descent.md)  [Next.Page](./Page-1_Single_AND_MultiLayers_Perceptrons.md)

### Backpropagation

**Backpropagation** (short for "backward propagation of errors") is an algorithm used in training artificial neural networks, particularly for multilayer networks. It allows the network to adjust its weights and biases in order to minimize the error (or cost) between the predicted output and the actual output. Backpropagation uses **gradient descent** to compute the gradient of the cost function with respect to each weight in the network.

### Overview of Backpropagation

1. **Forward Pass**:
   - The input data is fed through the network, layer by layer, to compute the output (predictions).
   - This involves calculating the weighted sum of inputs at each neuron and applying an activation function.
   
2. **Calculate the Error**:
   - Once the predicted output is obtained, the error (difference between predicted and actual values) is computed using a **cost function** (e.g., Mean Squared Error for regression or Cross-Entropy for classification).
   
3. **Backward Pass (Backpropagation)**:
   - The error is propagated backward through the network to calculate how much each weight contributed to the error.
   - Partial derivatives (gradients) of the cost function with respect to each weight are calculated using the **chain rule** of calculus.
   - These gradients are used to update the weights of the network to reduce the error.

### Steps in Backpropagation

#### 1. **Forward Propagation**:
In a neural network with multiple layers, the input passes through the layers (each having weights and biases) to produce the final output.

- For each layer:
  - Compute the weighted sum of the inputs:
    \[
    z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}
    \]
    where:
    - \( W^{(l)} \) is the weight matrix of layer \( l \),
    - \( a^{(l-1)} \) is the activation of the previous layer,
    - \( b^{(l)} \) is the bias vector of layer \( l \),
    - \( z^{(l)} \) is the weighted sum before applying the activation function.
  
  - Apply the **activation function** (e.g., sigmoid, ReLU):
    \[
    a^{(l)} = \text{activation}(z^{(l)})
    \]
    where \( a^{(l)} \) is the output of the layer after applying the activation function.

#### 2. **Compute the Error**:
At the final layer, compute the cost (loss) using a cost function. For instance, if the network is solving a classification problem, you might use the cross-entropy loss:
\[
J(W, b) = - \frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]
\]
where:
- \( y^{(i)} \) is the true label,
- \( \hat{y}^{(i)} \) is the predicted output from the network,
- \( m \) is the number of training examples.

#### 3. **Backpropagation**:
Using the chain rule, backpropagation computes the gradient of the cost function with respect to each weight and bias in the network. These gradients tell us how to change each weight and bias to reduce the cost.

- **At the output layer**:
  - Compute the error at the output:
    \[
    \delta^{(L)} = a^{(L)} - y
    \]
    where \( \delta^{(L)} \) is the error in the output layer, \( a^{(L)} \) is the predicted output, and \( y \) is the true output.
  
  - Update the weights and biases:
    \[
    \frac{\partial J}{\partial W^{(L)}} = \delta^{(L)} \cdot (a^{(L-1)})^T
    \]
    \[
    \frac{\partial J}{\partial b^{(L)}} = \delta^{(L)}
    \]

- **For hidden layers**:
  - Compute the error at the previous layer using the error from the next layer:
    \[
    \delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \cdot f'(z^{(l)})
    \]
    where \( f'(z^{(l)}) \) is the derivative of the activation function at layer \( l \), and \( \delta^{(l)} \) is the error at layer \( l \).

  - Update the weights and biases for each layer:
    \[
    \frac{\partial J}{\partial W^{(l)}} = \delta^{(l)} \cdot (a^{(l-1)})^T
    \]
    \[
    \frac{\partial J}{\partial b^{(l)}} = \delta^{(l)}
    \]

#### 4. **Update Weights and Biases**:
After calculating the gradients, the weights and biases are updated using gradient descent:
\[
W := W - \eta \frac{\partial J}{\partial W}
\]
\[
b := b - \eta \frac{\partial J}{\partial b}
\]
where \( \eta \) is the **learning rate** that controls how large the updates are.

---

### Example of Backpropagation in a Neural Network

Consider a simple neural network with:
- **Input layer** (2 neurons),
- **Hidden layer** (2 neurons, with sigmoid activation),
- **Output layer** (1 neuron, with sigmoid activation).

#### 1. **Forward Propagation**:
Let’s say we have the following:
- Input: \( \mathbf{x} = [x_1, x_2] \)
- Weights and biases for the hidden layer: \( W_1 \) (2x2 matrix), \( b_1 \) (2x1 vector)
- Weights and biases for the output layer: \( W_2 \) (1x2 matrix), \( b_2 \) (scalar)

For the hidden layer:
\[
z_1 = W_1 \mathbf{x} + b_1
\]
Apply the sigmoid activation function:
\[
a_1 = \frac{1}{1 + e^{-z_1}}
\]

For the output layer:
\[
z_2 = W_2 a_1 + b_2
\]
Apply the sigmoid activation function again:
\[
\hat{y} = \frac{1}{1 + e^{-z_2}}
\]

#### 2. **Compute the Error**:
The error (using mean squared error as an example):
\[
J(W, b) = \frac{1}{2} (\hat{y} - y)^2
\]
where \( y \) is the true label.

#### 3. **Backpropagation**:
- Compute the error at the output layer:
  \[
  \delta_2 = (\hat{y} - y) \cdot \hat{y}(1 - \hat{y})
  \]

- Compute the error at the hidden layer:
  \[
  \delta_1 = (W_2)^T \delta_2 \cdot a_1(1 - a_1)
  \]

- Compute the gradients for the weights and biases:
  \[
  \frac{\partial J}{\partial W_2} = \delta_2 \cdot a_1^T
  \]
  \[
  \frac{\partial J}{\partial b_2} = \delta_2
  \]
  \[
  \frac{\partial J}{\partial W_1} = \delta_1 \cdot \mathbf{x}^T
  \]
  \[
  \frac{\partial J}{\partial b_1} = \delta_1
  \]

#### 4. **Update Weights**:
Using gradient descent, update the weights and biases for both layers.

---

### Key Concepts of Backpropagation

1. **Chain Rule**: The backpropagation algorithm relies heavily on the chain rule of calculus to compute the gradients of the cost function with respect to the weights in each layer.
   
2. **Local Gradients**: Each layer’s gradients are computed based on the local error (how much that layer’s output contributed to the overall error).

3. **Efficiency**: Backpropagation is computationally efficient because it reuses computations of gradients during the backward pass, making it suitable for deep networks.

---

### Example in Real-World Applications

1. **Image Recognition**: Neural networks with backpropagation are used for tasks like classifying images into categories (e.g., recognizing handwritten digits in the MNIST dataset).
   
2. **Natural Language Processing (NLP)**: Neural networks with backpropagation are widely used in text classification tasks like sentiment analysis, machine translation, etc.

3. **Speech Recognition**: Backpropagation is also used in deep neural networks for recognizing speech patterns and converting spoken language into text.

---

### Summary
Backpropagation is the key algorithm for training deep learning models. It allows neural networks to "learn" by minimizing the error between predicted and actual outputs. By adjusting the weights using gradient descent, the network gradually improves its predictions.

[Back to Index](./index.md)  [Previous.Page](./Page-5_Cost-Functions-Gradient-Descent.md)  [Next.Page](./Page-1_Single_AND_MultiLayers_Perceptrons.md)
