# Hybrid Water Quality Assessment Using K-Means, Random Forest, SVM, and k-NN: A Case Study of the Mahakali River, Nepal

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Status](https://img.shields.io/badge/Status-Completed-green.svg)]()

## 📋 Overview

This repository contains the complete implementation of a hybrid machine learning approach for water quality assessment of the Mahakali River in Nepal. The project combines unsupervised learning (K-means clustering), Water Quality Index (WQI) labeling, and supervised classification algorithms to achieve **99.31% accuracy** in water quality classification.

## 🎯 Key Achievements

- **99.31% Accuracy** using k-Nearest Neighbors (k-NN) classifier
- **720 samples** analyzed across 3 districts over 3 years
- **10 physicochemical parameters** monitored from 15 stations
- **70% cost reduction** potential with 3-parameter monitoring
- Identified spatial and temporal water quality patterns

## 🔬 Methodology

### 1. Data Preparation
- KNN imputation for missing values (12.5%)
- Z-score normalization
- PCA dimensionality reduction (92.75% variance retained)
- SMOTE for class balancing

### 2. Unsupervised Learning
- K-means clustering (k=2) to identify water quality groups
- Cluster 0: Poor quality (38% samples, WQI=62.90)
- Cluster 1: Good quality (62% samples, WQI=36.54)

### 3. Supervised Classification
- **k-NN (k=7)**: 99.31% accuracy ⭐
- **Random Forest**: 97.92% accuracy
- **SVM (RBF kernel)**: 97.22% accuracy

## 📁 Repository Structure

```
├── Assignment Question/       # Project requirements
├── Assignment Report/         # Complete thesis report
├── Code Screenshots/          # Code implementation screenshots
├── Original Dataset/          # Raw water quality data
├── Proposal/                  # Project proposal
├── machine_learning_assignment.ipynb  # Main Jupyter notebook
├── machine_learning_assignment.py     # Python script version
├── cleaned_water_quality_mahakali_river.xlsx  # Preprocessed data
├── data_with_clusters.xlsx    # K-means clustering results
├── data_with_labels.xlsx      # WQI-labeled dataset
├── *.pkl                      # Trained models (k-NN, RF, SVM, PCA, Scaler)
└── *.png                      # Visualization outputs
```

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pandas
numpy
scikit-learn
matplotlib
seaborn
openpyxl
```

### Installation
```bash
# Clone the repository
git clone https://github.com/tek-raj-bhatt-250069/STW7072CEM-Machine-Learning.git

# Navigate to directory
cd STW7072CEM-Machine-Learning

# Install dependencies
pip install -r requirements.txt
```

### Usage
```bash
# Run Jupyter notebook
jupyter notebook machine_learning_assignment.ipynb

# Or run Python script
python machine_learning_assignment.py
```

## 📊 Key Results

| Model | Accuracy | Precision | F1-Score | AUC-ROC |
|-------|----------|-----------|----------|---------|
| k-NN  | 99.31%   | 99.30%    | 99.31%   | 0.9931  |
| Random Forest | 97.92% | 97.93% | 97.92% | 0.9792 |
| SVM   | 97.22%   | 97.26%    | 97.22%   | 0.9722  |

### Most Important Features
1. **Ammonia** (32.18%) - Primary pollution indicator
2. **Iron** (20.23%) - Natural and industrial sources
3. **Chloride** (13.22%) - Mineral dissolution

## 🌍 Impact

- Enables cost-effective water quality monitoring
- Identifies high-risk districts: Baitadi (43% poor), Dadeldhura (43% poor)
- Reveals monsoon impact: 60% quality degradation (June-September)
- Ready for deployment to protect 500,000+ residents

## 👨‍🎓 Author

**Tek Raj Bhatt**  
Student ID: 250069 | CUID: 16544288  
MSc. Data Science and Computational Intelligence  
Softwarica College / Coventry University  
📧 250069@softwarica.edu.np

## 📄 License

This project is part of academic coursework at Softwarica College/Coventry University.

## 🙏 Acknowledgments

- Supervisor and faculty at Softwarica College
- Nepal Department of Water and Sewerage Management
- All contributors to the water quality monitoring program

---

⭐ **If you find this project useful, please consider giving it a star!**
