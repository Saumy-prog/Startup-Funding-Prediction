# Startup Funding Prediction and Investment Analytics

## 📌 Project Overview

Startup Funding Prediction and Investment Analytics is a Python-based data science project that analyzes startup funding data to identify investment trends and discover patterns in the startup ecosystem.

The project combines data preprocessing, exploratory data analysis, statistical analysis, machine learning, and clustering techniques to understand startup funding and investment behavior.

It aims to transform raw funding data into meaningful insights that can help explore investment patterns and understand the factors associated with startup funding.

## 🎯 Project Objectives

- Analyze startup funding data to understand investment trends.
- Clean and preprocess raw data for analysis.
- Perform Exploratory Data Analysis (EDA) to discover useful patterns.
- Apply regression algorithms to study relationships between variables.
- Use classification algorithms for predictive analysis.
- Apply clustering techniques to identify groups with similar characteristics.
- Perform statistical analysis to understand relationships within the dataset.
- Generate meaningful insights from startup funding information.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| SciPy | Scientific and statistical computations |
| Statsmodels | Statistical modeling and regression analysis |
| Scikit-learn | Machine learning and clustering |

## 📊 Project Features

### 1. Data Loading and Inspection
- Loads startup funding data from a CSV file.
- Displays dataset dimensions and column information.
- Examines data types, missing values, and basic statistics.

### 2. Data Cleaning and Preprocessing
- Handles missing values.
- Processes data to make it suitable for analysis.
- Prepares relevant features for machine learning models.

### 3. Exploratory Data Analysis (EDA)
- Examines the distribution of funding-related variables.
- Explores patterns and relationships in the dataset.
- Uses graphs and visualizations to understand the data.

### 4. Regression Analysis
The project includes the following regression techniques:

- Simple Linear Regression
- Multiple Linear Regression
- Polynomial Regression
- Ordinary Least Squares (OLS) Regression

These techniques help explore relationships between numerical variables and evaluate how well a model explains the observed data.

### 5. Classification Models
The project explores multiple classification algorithms:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Decision Tree
- Support Vector Machine (SVM)

These models can be used to study classification problems based on the available dataset and target variable.

### 6. Clustering Analysis
The project uses clustering techniques to explore similarities among data points.

**K-Means Clustering**
- Groups similar observations into clusters.

**Hierarchical Clustering**
- Single linkage
- Complete linkage
- Average linkage
- Centroid linkage

### 7. Association Rule Analysis
- Explores associations between categorical patterns in the data.
- Uses a manually implemented approach to identify potential relationships without requiring the `mlxtend` library.

## 📂 Project Structure

```text
Startup-Funding-Prediction/
│
├── pred.py
├── india_startup_funding_sample.csv
├── README.md
├── .gitignore
└── results/
```

**File descriptions:**

- `pred.py` — Main Python program containing the analysis and machine learning implementation.
- `india_startup_funding_sample.csv` — Dataset used for the project, if included.
- `README.md` — Project documentation.
- `.gitignore` — Specifies files and folders Git should ignore.
- `results/` — Folder for generated outputs, if the program saves them.

*Note: The folder structure may vary depending on how the project is organized.*

## 💻 Installation and Setup

### Prerequisites

Make sure the following are installed:

- Python 3.11 or a compatible version
- pip
- A Python editor such as IDLE, VS Code, or PyCharm

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Startup-Funding-Prediction.git
```

Replace `YOUR-USERNAME` with your GitHub username.

### Step 2: Navigate to the Project Directory

```bash
cd Startup-Funding-Prediction
```

### Step 3: Install Required Libraries

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy
```

If you use the Windows Python launcher, you can also run:

```bash
py -m pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy
```

### Step 4: Configure the Dataset

Place the CSV dataset in the project directory.

If necessary, configure the dataset path in `pred.py`:

```python
from pathlib import Path

DATA_FILE = Path(__file__).parent / "india_startup_funding_sample.csv"
```

This relative path allows the program to locate the dataset in the same folder as the Python script.

If the dataset is not included in the repository, obtain it separately and place it in the project directory.

### Step 5: Run the Project

```bash
python pred.py
```

Alternatively, open `pred.py` in Python IDLE and select **Run → Run Module**.

## 📈 Expected Outcomes

After running the project, users can explore:

- Summary statistics and information about the dataset.
- Data quality and preprocessing results.
- Visualizations showing patterns in startup funding data.
- Regression model results and evaluation metrics, where implemented.
- Classification model performance, where implemented.
- Clusters that reveal similarities between observations.
- Potential associations between selected data features.

Actual outputs depend on the dataset, selected variables, and model configurations.

## 🔍 Applications

This project demonstrates data science techniques that can be applied to:

- Startup ecosystem analysis.
- Investment trend exploration.
- Funding data analysis.
- Statistical modeling.
- Machine learning experimentation.
- Pattern discovery in structured financial datasets.

The results should be treated as analytical findings rather than guaranteed predictions of future startup funding.

## 🚀 Future Improvements

Potential improvements include:

- Developing an interactive dashboard using Streamlit or Power BI.
- Comparing models using suitable evaluation metrics.
- Improving feature engineering and data preprocessing.
- Adding hyperparameter tuning and cross-validation.
- Building an interactive interface for exploring startup funding patterns.
- Integrating updated datasets to analyze changing investment trends.

## 🎓 Learning Outcomes

Through this project, I aim to strengthen my understanding of:

- Data cleaning and preprocessing with Pandas.
- Exploratory data analysis and visualization.
- Statistical analysis and regression modeling.
- Machine learning classification techniques.
- Clustering and pattern discovery.
- Python-based implementation of data science workflows.

## 👨‍💻 Author

**Saumy Dixit**  
B.Tech — Computer Science and Engineering (Data Science)

## 📄 License

This project is intended for educational and learning purposes. Add an appropriate open-source license if you decide to distribute the code under specific reuse terms.# Startup-Funding-Prediction
Startup funding analysis, investment trends, and prediction using Python and machine learning
