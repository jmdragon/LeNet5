# LeNet-5

We implemented both:

1. **LeNet1** — the *original* LeNet-5 architecture from  
   *Gradient-Based Learning Applied to Document Recognition (LeCun et al., 1998)*  
   using scaled tanh activations, average subsampling, and RBF-style output units.

2. **LeNet2** — a *modified* and more robust CNN using  
   ReLU, max-pooling, cross-entropy loss, and geometric data augmentation  
   to handle **unseen / transformed MNIST** digits.

Both models are trained, evaluated, and saved for grading and testing.

---

## Group Members
- **Joshua Mondragon** 
- **Yohanan Pinto** 

---

## Project Structure

HW4/
│
├── data/
│ ├── train/ # 10,000 MNIST training PNGs (28×28 → padded to 32×32)
│ ├── test/ # 10,000 MNIST test PNGs
│ ├── train_label.txt # One label per line (0–9)
│ ├── test_label.txt
│ └── digits_updated/ # DIGIT dataset used to build RBF prototypes
│
├── models/
│ ├── LeNet1.pth # Trained original LeNet-5
│ └── LeNet2.pth # Trained modified LeNet-5
│
├── results/
│ ├── lenet1_errors.npz # Train/test error curves for LeNet1
│ ├── lenet2_errors.npz # Train/test error curves for LeNet2
│ ├── lenet1_confusion_matrix.txt
│ └── most_confusing_lenet1/ # Worst misclassified examples for LeNet1
│
├── src/
│ ├── data.py
│ ├── mnist.py
│ ├── prototypes_from_digit.py
│ ├── lenet1.py
│ ├── lenet2.py
│ ├── train_lenet1.py
│ ├── train_lenet2.py
│ ├── eval_lenet1.py
│ ├── test1.py
│ └── test2.py
│
└── README.md


---

## Model Summaries

### **LeNet1 (Original Architecture)**
Implements the exact LeNet-5 structure:
- Scaled tanh activations  
- Average subsampling  
- C1 → S2 → C3 → S4 → C5 → F6  
- **RBF output layer**:  
  \( y_k = \| f_6 - \mu_k \|^2 \)  
- Trained with **MAP loss** (Eq. 9 in the paper)  
- Batch size = **1**, learning rate ≈ **0.001**

**Prototype Generation:**  
RBF prototypes (10 × 84) are generated from the **DIGIT** dataset by resizing digits to **7×12**, averaging many samples, and binarizing to ±1.

### **LeNet2 (Modified Architecture)**
A more modernized and robust variant:
- ReLU activations  
- Max pooling instead of average subsampling  
- Standard fully connected classifier  
- Cross-entropy loss  
- **RandomAffine** data augmentation:
  - Rotation ±30°  
  - Translation ±10%  
  - Scaling 0.8–1.2  
- Batch size = 64, SGD + Momentum

This model is more robust to unseen geometric distortions, as required by Problem 2.

---

## How to Run 

### **1. Train LeNet1 (original architecture)**  
```bash
python src/train_lenet1.py


### **2. 	Test LeNet1 (for grading)**
python src/test1.py

### **3.    Evaluate LeNet1 (confusion matrix + worst errors)**
python src/eval_lenet1.py

### **4.    Train LeNet2 (modified architecture)**
python src/train_lenet2.py

### **5.    Test LeNet2 (for grading)**
python src/test2.py

### **REMEMBER TO INSTALL**
pip install torch torchvision numpy pillow matplotlib
pip install huggingface_hub  # required for data.py (MNIST parquet)
