# 🌷 Flower Classification with Deep Learning

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Accuracy](https://img.shields.io/badge/Test_Accuracy-53.93%25-success)

> **Automated flower species classification using transfer learning and deep convolutional neural networks**

A production-ready deep learning system that classifies 102 flower species with 53.93% accuracy using transfer learning with MobileNetV2. Built as a portfolio project demonstrating end-to-end ML engineering skills.

![Sample Predictions](docs/images/sample_predictions.png)

---

## 📊 Project Overview

### **Business Context**
Developed for a hypothetical start-up building automated plant identification systems for:
- 🌱 Smart gardening applications
- 🛒 E-commerce flower cataloging
- 📱 Mobile plant identification apps
- 🔬 Botanical research automation

### **The Challenge**
- **102 flower species** to classify
- **Limited training data**: Only 10 images per class (~1,020 total)
- **Class imbalance**: 40-258 images per category
- **Visual complexity**: Similar-looking flowers, varying lighting, backgrounds, angles

### **The Solution**
Transfer learning with MobileNetV2 pre-trained on ImageNet, achieving **53.93% test accuracy** on unseen data - **55x better than random guessing** with minimal training data.

---

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 53.93% |
| **Validation Accuracy** | 55.49% |
| **Precision (macro avg)** | 54.74% |
| **Recall (macro avg)** | 54.53% |
| **F1-Score (macro avg)** | 50.32% |
| **Model Size** | 11.20 MB |
| **Training Time** | ~2 minutes (5 epochs) |

### **Model Performance**
![Training Curves](docs/images/training_curves.png)

**Best Performing Classes:**
- Classes 48, 63: **100% accuracy** (perfect classification!)
- Classes 28, 62, 57: **96-98% accuracy**

**Most Challenging Classes:**
- Classes 90, 54, 41: **0% accuracy** (likely due to visual similarity or data imbalance)

![Confusion Matrix](docs/images/confusion_matrix.png)

---

## 🛠️ Technical Skills Demonstrated

### **Deep Learning & ML**
- ✅ Convolutional Neural Networks (CNNs)
- ✅ Transfer Learning with MobileNetV2
- ✅ Model architecture design & optimization
- ✅ Hyperparameter tuning
- ✅ Overfitting detection & mitigation
- ✅ Gradient descent optimization (Adam)
- ✅ Multi-class classification (102 classes)

### **Data Engineering**
- ✅ TensorFlow Datasets (TFDS) integration
- ✅ Image preprocessing pipelines
- ✅ Data augmentation (rotation, flipping, brightness)
- ✅ Batch processing & prefetching
- ✅ Train/validation/test splitting

### **Model Evaluation**
- ✅ Confusion matrix analysis
- ✅ Precision, recall, F1-score calculation
- ✅ Per-class performance analysis
- ✅ Generalization gap assessment
- ✅ Cross-dataset validation

### **Development Tools & Practices**
- ✅ Python 3.11 with modern package management (uv)
- ✅ Jupyter notebooks for experimentation
- ✅ Git version control
- ✅ Professional documentation
- ✅ Reproducible research practices

### **Libraries & Frameworks**
```python
TensorFlow 2.20          # Deep learning framework
Keras                    # High-level neural network API
TensorFlow Datasets      # Dataset loading
NumPy                    # Numerical computing
Pandas                   # Data manipulation
Matplotlib & Seaborn     # Visualization
scikit-learn             # Metrics & evaluation
```

---

## 📁 Project Structure
```
flower-classification/
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA & dataset analysis
│   ├── 02_data_preprocessing.ipynb     # Preprocessing & augmentation
│   └── 03_model_building.ipynb         # Model training & evaluation
├── models/
│   ├── flower_classifier_mobilenetv2.keras  # Best model (11 MB) ⭐
│   └── baseline_cnn.keras                   # Baseline experiment (128 MB)
├── docs/
│   └── images/                         # Visualizations for README
├── app/                                # Streamlit deployment (coming soon)
├── pyproject.toml                      # Project dependencies (uv)
├── uv.lock                             # Locked dependencies
└── README.md                           # This file
```

---

## 🚀 Methodology

### **3. Model Development** (`03_model_building.ipynb`)

#### **Experiment 1: Baseline CNN (From Scratch)** ❌
- Simple CNN with 3 convolutional blocks
- 11M trainable parameters
- **Result: 0.69% validation accuracy** (random guessing)
- **Conclusion:** Insufficient for 102 classes with limited data
- **Model saved:** `models/baseline_cnn.keras` (128 MB)

#### **Experiment 2: Transfer Learning (Final Model)** ✅
```python
Architecture:
- Base: MobileNetV2 (pre-trained on ImageNet, 14M images)
- Frozen base model: 2,257,984 parameters
- Custom classifier:
  - GlobalAveragePooling2D
  - Dense(128, relu)
  - Dropout(0.5)
  - Dense(102, softmax)
- Trainable parameters: 177,126
```

**Training Configuration:**
- Optimizer: Adam (learning_rate=0.001)
- Loss: Sparse Categorical Crossentropy
- Batch size: 32
- Epochs: 5
- **Result: 55.49% validation accuracy** ✅
- **Model saved:** `models/flower_classifier_mobilenetv2.keras` (11 MB) ⭐

#### **Experiment 3: Fine-Tuning Attempt** ⚠️
- **Strategy:** Unfroze top 54 layers of MobileNetV2
- **Learning rate:** 0.0001 (reduced)
- **Additional epochs:** 10
- **Result: 45.10% validation accuracy** (degraded from 55%)
- **Diagnosis:** Overfitting - model memorized training data
- **Root cause:** Too many parameters (~600K) for limited training data (1,020 images)
- **Decision:** Reverted to Experiment 2 model (55%) ✅
- **Key learning:** Fine-tuning requires careful layer selection with small datasets
- **Model NOT saved** (inferior performance)

#### **Model Selection & Justification**
```
┌─────────────────────────────────────────────────────────────┐
│                    Model Comparison                          │
├──────────────────┬──────────┬──────────┬──────────┬─────────┤
│ Model            │ Val Acc  │ Test Acc │ Size     │ Status  │
├──────────────────┼──────────┼──────────┼──────────┼─────────┤
│ Baseline CNN     │  0.69%   │    -     │  128 MB  │ Failed  │
│ Transfer Learning│ 55.49%   │ 53.93%   │   11 MB  │ SELECTED│
│ Fine-tuned       │ 45.10%   │    -     │   11 MB  │ Rejected│
└──────────────────┴──────────┴──────────┴──────────┴─────────┘
```

**Final Decision:** Deploy Transfer Learning model (55% validation, 54% test accuracy)
- Best performance with strong generalization (1.56% gap)
- Lightweight and deployment-ready (11 MB)
- Demonstrates transfer learning effectiveness (80x improvement over baseline)

### **4. Evaluation**
- Test set accuracy: **53.93%**
- Generalization gap: Only 1.56% (excellent!)
- Precision/Recall: Balanced performance (~54% each)
- Confusion matrix: Identified problematic classes

---

## 💻 Setup & Installation

### **Prerequisites**
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (modern package manager)

### **Installation**
```bash
# Clone repository
git clone https://github.com/Akakinad/flower-classification.git
cd flower-classification

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync

# Run Jupyter notebooks
jupyter notebook
```

### **Dataset**
The Oxford Flowers 102 dataset is automatically downloaded via TensorFlow Datasets on first run (~330 MB).

---

## 🔮 Future Enhancements

- [ ] **Streamlit web app** for real-time predictions
- [ ] **Model optimization** with EfficientNet or Vision Transformers
- [ ] **Grad-CAM visualizations** to explain predictions
- [ ] **Mobile deployment** with TensorFlow Lite
- [ ] **API development** with FastAPI
- [ ] **Docker containerization**
- [ ] **Class balancing** techniques for underrepresented species

---

## 📚 Key Learnings

1. **Transfer learning is essential** with limited data - achieved 80x improvement over baseline
2. **Fine-tuning can backfire with small datasets** - attempted optimization degraded performance from 55% → 45% due to overfitting; knowing when NOT to tune is as important as knowing how to tune
3. **Data augmentation helps** but can't replace having more training examples
4. **Model size matters** - MobileNetV2 (11 MB) is deployment-ready, unlike larger architectures
5. **Class imbalance impacts performance** - some species need more training data

---

## 📊 Dataset

**Oxford Flowers 102**
- **Source:** [Visual Geometry Group, University of Oxford](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- **Classes:** 102 flower categories
- **Images:** 8,189 total (1,020 train / 1,020 val / 6,149 test)
- **Challenge:** Significant scale, pose, and lighting variations

**Citation:**
```
Nilsback, M-E. and Zisserman, A.
Automated Flower Classification over a Large Number of Classes
Proceedings of the Indian Conference on Computer Vision, Graphics and Image Processing (2008)
```

---

## 👨‍💻 Author

**Akakinad**
- Machine Learning Engineer
- [GitHub](https://github.com/Akakinad)

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- TensorFlow & Keras teams for excellent deep learning frameworks
- Oxford Visual Geometry Group for the Flowers 102 dataset

---

**⭐ If you found this project helpful, please consider starring the repository!**