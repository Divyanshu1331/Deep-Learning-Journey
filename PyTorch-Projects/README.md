# PyTorch Projects 🚀  

This repository contains hands-on deep learning projects implemented in **PyTorch**, covering Artificial Neural Networks (ANNs), Convolutional Neural Networks (CNNs), and hyperparameter optimization with **Optuna**.  

### 📂 Project Structure  

PyTorch-Projects

│── 01-ANN_using_Pytorch.ipynb

│── 02-Hyperparameter_tunning_of_ANN.ipynb

│── 03-CNN_on_FashionMNIST_using_pytorch.ipynb

---

---

# 💼 Adult Income Classification using Artificial Neural Network (ANN)

## 🧠 Overview
This project predicts whether an individual's **income exceeds $50K per year** based on demographic and employment-related attributes from the **UCI Adult Income dataset**.  
An **Artificial Neural Network (ANN)** is implemented to perform binary classification, demonstrating the application of deep learning in structured (tabular) data.

📘 [View Notebook](01-ANN_using_Pytorch.ipynb) 
---

## 🧩 Project Workflow

### 🧮 1. Data Loading
Loaded the **Adult Income dataset** containing features such as age, education, occupation, workclass, marital status, and hours worked per week, along with the target variable — *income category (>50K or ≤50K)*.

---

### 🧹 2. Data Preprocessing
Performed essential data cleaning and preparation steps:
- Handled missing or ambiguous entries (e.g., '?')
- Encoded **categorical variables** using one-hot encoding
- Normalized numerical features to bring them to a similar scale
- Split the dataset into **training (80%)** and **testing (20%)** subsets for model evaluation

---

### ⚙️ 3. Feature Scaling
Applied **StandardScaler** to standardize all numerical columns.  
This helps stabilize gradient descent during ANN training and improves convergence speed.

---

### 🧠 4. ANN Model Building
Constructed a feed-forward **Artificial Neural Network** using the following structure:
- **Input Layer:** Accepts all encoded and scaled features  
- **Hidden Layers:** Two dense layers with ReLU activation to capture nonlinear relationships  
- **Output Layer:** Single neuron with sigmoid activation for binary classification  

The model was compiled with:
- **Optimizer:** Adam  
- **Loss Function:** Binary Crossentropy  
- **Metrics:** Accuracy  

---

### 🏋️‍♂️ 5. Model Training
Trained the ANN on the training dataset for multiple epochs while monitoring validation performance to prevent overfitting.  
Used **batch-based learning** and **early stopping** to enhance efficiency and model generalization.

---

### 📊 6. Model Evaluation
Evaluated the model using key metrics:
- **Accuracy Score**
- **Confusion Matrix**
- **Precision, Recall, and F1-score**

Compared predictions on test data with actual labels to assess model reliability in predicting income category.

---

### 📈 7. Results Summary

| Metric | Training Accuracy | Testing Accuracy | Remark |
|--------|--------------------|------------------|---------|
| ANN Model | 86.45% | **85.27%** | Performs well with moderate complexity and balanced accuracy |

---

### 🏁 8. Conclusion
The **Artificial Neural Network** achieved a strong accuracy in predicting high-income individuals based on demographic and professional features.  
This project highlights the practical use of deep learning for **socioeconomic data analysis and income classification**.

---

📘 **Note:**  
This README summarizes three Jupyter Notebooks included in this directory:
1. Data Preprocessing & Encoding  
2. ANN Model Construction & Training  
3. Model Evaluation & Performance Comparison


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

