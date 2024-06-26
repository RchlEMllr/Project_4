# Crime in Denver
![Denver crimes 2019-2023 color-coded by district](https://raw.githubusercontent.com/RchlEMllr/Project_4/branchel/Resources/2019_2023_crimes_map.jpg)


In this project, our object was to classify instances of murder occurrences using various features such as district ID, precinct ID, neighborhood ID, and victim count. This code utilizes machine learning techniques, including data preprocessing, resampling, and model training, to build a predictive model. The performance of the model is evaluated using classification metrics, including Matthews Correlation Coefficient (MCC) and Cohen's Kappa.

The dataset used was found on Kaggle, and covers Denver crime reports from 2019 through October 2023, as pulled from Denver's open data catalogue. It includes information on what crimes were reported, the date reported, dates the crimes took place, and where the crimes were reported (including latitude and longitude, specific address, neighborhood, precinct, and district), as well as count of victims affected. In order to run a predictive model, we also created a binary column which included whether the crime was a murder or not. 

## Requirements

The following libraries are required to run the code:

- pandas
- scikit-learn
- imbalanced-learn

You can install the necessary libraries using pip:

```bash
pip install pandas scikit-learn imbalanced-learn

```

## Dataset
The dataset should be a pandas DataFrame named murder_target with the following columns:

- district_id: Categorical feature representing the district.
- precinct_id: Numerical feature representing the precinct.
- neighborhood_id: Categorical feature representing the neighborhood.
- victim_count: Numerical feature representing the count of victims.
- murders: Target variable indicating the occurrence of murder (0 or 1).

#Code Explanation
#Data Preparation

- 1.) Selecting Features and Target:
  
```python
X = murder_target[['district_id', 'precinct_id', 'neighborhood_id', 'victim_count']].copy()
y = murder_target['murders']

```
```Handling Missing Values:
X = X.fillna(-1)
```
```One-Hot Encoding Categorical Features:
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
```

# Running the Code
To run the code, ensure you have the murder_target DataFrame loaded in your environment and execute the script. The script will preprocess the data, apply resampling, train a Random Forest classifier with hyperparameter tuning, and evaluate the model using classification metrics.

## Resources

- [Denver Crime Dataset](https://www.kaggle.com/datasets/paultimothymooney/denver-crime-data)
- [SMOTE](https://towardsdatascience.com/imbalanced-classification-in-python-smote-tomek-links-method-6e48dfe69bbc)
- [Linear Relationships (MLR)](https://www.investopedia.com/terms/m/mlr.asp#:~:text=Linear%20regression%20can%20only%20be,extends%20to%20several%20explanatory%20variables)
  - Module_21_02Tuesday_BackToTheMoon
  - Module_21_02Tuesday_AutoOptimization
  - Module_21_02Tuesday_TuneUp
- Hot Encoding
  - Module_21_02Tuesday_GettingReal
- [Matthews Correlation Coefficient](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html)
- Cohen's Kappa
  - [Stack Overflow](https://stackoverflow.com/questions/68898292/is-that-cohen-kappa-score-correct)
- [Data Lab - Linear Regression Tutorial](https://datatab.net/tutorial/linear-regression)

