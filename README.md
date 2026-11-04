# 2025 Machine Learning Exercises

Independent coursework notebooks and scripts covering core machine learning methods.

## Exercise Index

| File | Exercise |
|---|---|
| `2-way and 3-way splitting, k folds evaluation.py` | Compare train/test splitting strategies and k-fold evaluation |
| `BDT.ipynb` | Bayesian decision classification using email data |
| `K-Means_Clustering.ipynb` | K-means clustering |
| `MNIST_dataset.ipynb` | PyTorch classification on selected MNIST digits |
| `linear_regression.ipynb` | Linear regression |
| `regularization.ipynb` | Regularization methods |

Each file is a standalone exercise; there is no shared application or single end-to-end pipeline.

## Tech Stack

Python, Jupyter Notebook, NumPy, pandas, scikit-learn, Matplotlib, Seaborn, PyTorch, and torchvision.

## Run an Exercise

Install Jupyter and the libraries used by the notebooks:

```bash
python -m pip install jupyter numpy pandas scikit-learn matplotlib seaborn torch torchvision
jupyter notebook
```

Open the chosen notebook and run its cells in order. The split-evaluation exercise expects `dummy_binary_classification.csv`, and `BDT.ipynb` reads `emails.csv`; neither dataset is included. MNIST and Iris examples load data through their libraries.

## Author

**Gireesh Sadanand Patgar** · [GitHub](https://github.com/gireeshpatgar17)
