# IA2 Project - Machine learning applied to MathE Dataset


## Project Description
This project, developed for the Artificial Intelligence 2 (IA2) course, focuses on **Question Difficulty Classification** using the MathE Dataset. The goal is to train a Machine Learning model to predict the difficulty level (Basic/Advanced) of mathematical questions based on their characteristics and historical student performance. Please consult the LaTeX documentation for full project details.

### Key Features
* Modular training pipeline using `scikit-learn` and `pytorch-lightning`.
* Question-level feature engineering based on performance statistics, topic analysis, and cross-country variance.
* Data analysis using `Pandas`, `Numpy`, `Matplotlib` and `Seaborn`.
* Experiment tracking (Loss, Accuracy) via logging.
* Data preprocessing and aggregation pipeline.
* Reproducible environment configuration.

---

##  Project Structure

```text
IA2_prj/
├── data/
│   ├──processed/           
│   └──raw/                
├── logs/                  
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
