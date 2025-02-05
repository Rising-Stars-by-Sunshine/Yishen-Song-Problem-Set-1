# **Income Prediction with Fairness-Aware Machine Learning**

## **Overview**
This project focuses on income prediction using fairness-aware machine learning models. It integrates structured census data, external economic indicators, and NLP-based bias detection to improve accuracy and fairness in AI-driven decision-making.

## **System Configuration and Setup**
The system can be set up and executed in both **local environments** and **cloud-based platforms** such as Google Colab or AWS. Below are the detailed setup instructions for both environments.

---
## **1. Local Environment Setup**

### **1.1 System Requirements**
Ensure your system meets the following requirements:
- **Operating System:** Windows 10/11, macOS, or Linux
- **Python Version:** 3.8 or higher
- **RAM:** Minimum 8GB (Recommended: 16GB for large datasets)
- **Storage:** At least 5GB free space for dataset and model files

### **1.2 Installation Steps**
#### **Step 1: Clone the Repository**
```bash
git clone https://github.com/your-repo/income-prediction.git
cd income-prediction
```

#### **Step 2: Set Up a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
```

#### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

#### **Step 4: Download the Dataset**
Ensure the **Adult Income Dataset** is available in the `data/` directory. If not, download it manually:
```bash
mkdir data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data -P data/
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test -P data/
```

#### **Step 5: Run Preprocessing and Training**
```bash
python preprocess.py  # Cleans and preprocesses the dataset
python train.py  # Trains fairness-aware machine learning models
```

#### **Step 6: Evaluate Model Performance**
```bash
python evaluate.py  # Generates performance and fairness reports
```

---
## **2. Cloud Environment Setup**

### **2.1 Using Google Colab**
Google Colab provides a free cloud-based Jupyter Notebook environment with GPU support.

#### **Step 1: Open the Colab Notebook**
- Open [Google Colab](https://colab.research.google.com/)
- Upload the project notebook (`income_prediction.ipynb`)

#### **Step 2: Install Dependencies in Colab**
```python
!pip install -r requirements.txt
```

#### **Step 3: Download the Dataset in Colab**
```python
!mkdir data
!wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data -P data/
!wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test -P data/
```

#### **Step 4: Run Preprocessing and Model Training in Colab**
```python
!python preprocess.py
!python train.py
!python evaluate.py
```

---
### **2.2 Using AWS EC2 (Ubuntu 20.04)**

#### **Step 1: Launch an AWS EC2 Instance**
- **Instance Type:** `t3.medium` (for CPU) or `g4dn.xlarge` (for GPU)
- **AMI:** Ubuntu 20.04 LTS
- **Storage:** 10GB EBS volume

#### **Step 2: Connect to the Instance**
```bash
ssh -i your-key.pem ubuntu@your-ec2-instance-ip
```

#### **Step 3: Install System Dependencies**
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv -y
```

#### **Step 4: Clone the Repository and Set Up Virtual Environment**
```bash
git clone https://github.com/your-repo/income-prediction.git
cd income-prediction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### **Step 5: Run Model Training on AWS**
```bash
python preprocess.py
python train.py
python evaluate.py
```

---
## **3. Additional Notes**
### **3.1 Key Dependencies**
The project uses the following key Python libraries:
- **Scikit-learn** (Machine learning models)
- **Pandas & NumPy** (Data processing)
- **Matplotlib & Seaborn** (Visualization)
- **Fairlearn** (Bias mitigation techniques)
- **Hugging Face Transformers** (NLP-based bias detection)

### **3.2 Troubleshooting**
- If **pip install** fails, ensure you have the latest **pip version**:
```bash
pip install --upgrade pip
```
- If the dataset is missing, manually download it using the links in **Step 4**.
- If you encounter GPU-related issues on AWS, ensure **CUDA drivers** are installed:
```bash
!nvidia-smi
```

---
## **4. Contributors**
- **Your Name** (Lead Researcher)
- **Advisor / Supervisor** (Institution Name)
- **Contributors** (Team Members)



---
## **5. License**
This project is licensed under the **MIT License**. See `LICENSE.md` for details.
