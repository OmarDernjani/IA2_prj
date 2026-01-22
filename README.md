# IA2 Project - Machine learning applied to MathE Dataset


## Project Description
This project, developed for the Artificial Intelligence 2 (IA2) course, focuses on Knowledge Tracing using the MathE Dataset. The goal is to train a Machine Learning model to predict the correctness (True/False) of a student's next answer. Please consult the LaTeX documentation for full project details.

### Key Features
* Modular training pipeline using `scikit-learn` and `pytorch-lightning`.
* Data analysis using `Pandas`, `Numpy`, `Matplotlib` and `Seaborn`.
* Experiment tracking (Loss, Accuracy) via logging.
* Data preprocessing and augmentation pipeline.
* Reproducible environment configuration.
* Full set of feature engineering.

---

##  Project Structure

```text
IA2_prj/
├── data/
│   ├──processed/           #ignored by gitignore
│   └──raw/                 #ignored by gitignore
├── logs/                   #hyperparams and logging once started training
├── results/
│   ├──eda_res/
│   │     ├── figure1.jpg
│   │     .
│   │     .
│   │     └── figure10.jpg
│   └──model_res/
│        ├── all_results.csv
│        ├── baseline_results.csv
│        ├── feature_importance.csv
│        └── tuned_results.csv
├──src/
│   ├── data_preprocessing.py
│   ├── generate_plot.py
│   ├── helper_fun.py
│   ├── mlp.py
│   └── models.py
├── .python-version
├── model_best.pt
├── requirements.txt
├── report.pdf
└── README.md 
```

##  Getting Started

### Prerequisites
Ensure you have **Python 3.8+** installed. It is highly recommended to run this project inside a virtual environment to isolate dependencies.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd IA2_prj
    ```

2.  **Create and Activate a Virtual Environment:**

    * **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

    * **macOS / Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install Dependencies:**
    Once the environment is active, install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

**1. Data Preparation**
Ensure your raw data is located in `data/raw/` or run the preprocessing script to generate the necessary files:
```bash
python src/data_preprocessing.py
```

### Documentation

For a detailed explanation of the theoretical background, model architecture, and experimental results, please refer to the Project Report located in the root directory.
