# PyTorch Projects 🚀  

This repository contains hands-on deep learning projects implemented in **PyTorch**, covering Artificial Neural Networks (ANNs), Convolutional Neural Networks (CNNs), and hyperparameter optimization with **Optuna**.  

## 📂 Project Structure  

PyTorch-Projects/

│── 01-ANN_using_Pytorch.ipynb

│── 02-Hyperparameter_tunning_of_ANN.ipynb

│── 03-CNN_on_FashionMNIST_using_pytorch.ipynb

---

## 🔹 01. ANN using PyTorch  
In this project, an **Artificial Neural Network (ANN)** is trained on the **Adult Income dataset**.  
- Built a **custom Dataset class** and applied a **transformation function** inside `__getitem__()`.  
- Showed how **Dataset, DataLoader, and transformation pipeline** work together.  
- **Final Test Accuracy:** **80.94%**  

📘 [View Notebook](01-ANN_using_Pytorch.ipynb)  

---

## 🔹 02. Hyperparameter Tuning of ANN with Optuna  
This project focuses on **improving model performance** using **Optuna** for hyperparameter tuning.  

**Initial Model Performance:**  
- Training Accuracy: **80.61%**  
- Testing Accuracy: **80.95%**  

**After Optuna Optimization (best params → `num_hidden_layer=3`, `neuron_per_layer=38`):**  
- Training Accuracy: **82.22%**  
- Testing Accuracy: **82.13%**  

✅ Clear improvement shows the effectiveness of **hyperparameter optimization** with Optuna.  

📘 [View Notebook](02-Hyperparameter_tunning_of_ANN.ipynb)  

---

## 🔹 03. CNN on Fashion-MNIST using PyTorch  
In this project, a **Convolutional Neural Network (CNN)** is trained on the **Fashion-MNIST dataset**.  
- Used **torchvision’s built-in dataset** and applied **image transformations** (normalization).  
- Demonstrated **Dataset and DataLoader pipeline** for image data.  
- Trained the model on **GPU** for faster computation.  
- **Final Test Accuracy:** **91.17%**  

📘 [View Notebook](03-CNN_on_FashionMNIST_using_pytorch.ipynb)  

---

## ✨ Key Learnings Across Projects  
- Building **custom Datasets** and applying **transformations** inside `__getitem__()`.  
- Leveraging **DataLoader** for batching, shuffling, and efficient training.  
- Applying **data transformations** for encoding & scaling.  
- Using **GPU acceleration** for deep learning tasks.  
- Applying **Optuna** for **hyperparameter optimization** to improve ANN performance.  

---

## ✍️ Blogs on PyTorch  
Alongside these projects, I’ve also written detailed blogs on PyTorch fundamentals on **Medium**.  
They’re a great way to dive deeper into the concepts before exploring the code in this repo:  

- 📖 [Week 13: PyTorch Essentials — Tensors, Autograd, and Perceptrons](https://medium.com/@divyanshu1331/week-13-pytorch-essentials-tensors-autograd-and-perceptrons-df14e7d415e7)  
- 📖 [Week 14: Dataset, DataLoader, and Final Steps in PyTorch](https://medium.com/@divyanshu1331/week-14-dataset-dataloader-and-final-steps-in-pytorch-dd94bf4479d5)  

👉 Check them out if you’d like a **step-by-step explanation of the concepts** that power these projects. More blogs are on the way — next up, I’ll be covering **Vision Transformers (ViTs)** 🚀  

---

