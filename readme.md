## CoSemiGNN
> Official PyTorch implementation of the paper:
> **"CoSemiGNN: Blockchain Fraud Detection with Dynamic Graph Neural Networks Based on Co-association of
Semi-Supervised"**

---

### 📌 Table of Contents

- [CoSemiGNN](#cosemignn)
  - [📌 Table of Contents](#-table-of-contents)
  - [🧩 Requirements](#-requirements)
  - [📁 Dataset](#-dataset)
- [Run data\_clear.py to clean up the missing values.You can then inspect the cleaned data.](#run-data_clearpy-to-clean-up-the-missing-valuesyou-can-then-inspect-the-cleaned-data)
    - [Train](#train)
    - [Inference](#inference)
  - [📊 Results](#-results)
- [](#)
  - [📄 License](#-license)
  - [🙏 Acknowledgments](#-acknowledgments)

---


### 🧩 Requirements

* Python == 3.8
* PyTorch: 1.10.0+cu113
* GPU: NVIDIA GeForce RTX 3060 GPU
* CUDA: 11.3
* Operating system: windows

You can install dependencies using:

```bash
pip install -r requirements.txt
```

---

### 📁 Dataset

To use the Elliptic and Elliptic++ datasets:

* Elliptic: available upon request at https://www.kaggle.com/datasets/ellipticco/elliptic-data-set/data
* Elliptic++: available at the GitHub repository https://github.com/git-disl/EllipticPlusPlus

Please follow their respective licenses and data usage agreements. Place the downloaded files under the data/ directory.

Run data_clear.py to clean up the missing values.You can then inspect the cleaned data.
---

#### Train

```bash
 python CoSemiGNN_main.py --tr_feature "custom_data/features.csv" --tr_edge "custom_data/edges.csv"
```

#### Inference

```bash
python CoSemiGNN_main.py --tr_feature "new_data/features.csv" --tr_edge "new_data/edges.csv" --load_pretrain
```

---
### 📊 Results
![bar.png](image%2Fbar.png)
![line.png](image%2Fline.png)
---


---

### 📄 License

Please see the [LICENSE](LICENSE) file for details.

---

### 🙏 Acknowledgments

This codebase is built upon and inspired by several great projects. We extend our sincere thanks to the contributors of:
PyTorch Geometric: For providing a powerful and flexible framework for graph neural networks.
PyTorch Geometric Temporal: For their excellent work on temporal graph networks, which provided valuable inspiration.
EvolveGCN: For their seminal work on dynamic graph convolution.
Elliptic & Elliptic++ Dataset: For providing the dataset used for our experiments.

---

