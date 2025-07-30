# 🧠 Backpropagation from Scratch

This folder contains my Week 4 project from my Deep Learning Journey — where I implemented **Backpropagation** from scratch using pure Python and NumPy.

---

## 📘 What You'll Find Here

### ✅ `Backprop_from_Scratch_Regression.ipynb`
- Implements backpropagation for a **regression task**
- Uses **Mean Squared Error (MSE)** as the loss function
- Step-by-step forward and backward pass
- Final weight updates using gradient descent

### ✅ `Backprop_from_Scratch_Classification.ipynb`
- Implements backpropagation for a **binary classification task**
- Uses **Binary Cross-Entropy Loss**
- Includes sigmoid activation 

---

## 📚 What I Learned

- How backpropagation works under the hood
- How gradients flow through layers using the **chain rule**
- How to compute **partial derivatives** for weights and biases
- Importance of memoization for efficient backward passes
- Logic behind the update rule:
  ```python
  weight -= learning_rate * gradient
  
## 🚀 Why This Matters
Most deep learning libraries (like TensorFlow or PyTorch) hide these internals.
Building it myself helped me truly understand:

How learning happens in neural networks

What gradients really mean

Why we use optimizers

### ✍️ Blog Post
👉 I also wrote a detailed blog post about Backpropagation on Medium:  
[📖 Read my blog on Backpropagation]([https://medium.com/@divyanshu1331/week-4-backpropagation-from-scratch-how-neural-networks-learn-979e9673d180](https://medium.com/@divyanshu1331/week-4-backpropagation-from-scratch-how-neural-networks-learn-979e9673d180))

