---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.11.0
  nbformat: 4
  nbformat_minor: 4
  toc-autonumbering: true
  vscode:
    interpreter:
      hash: 5238573367df39f7286bb46f9ff5f08f63a01a80960060ce41e3c79b190280fa
---

::: {.cell .markdown}
## **Fraud Detection in Python**

**Course Description**

A typical organization loses an estimated 5% of its yearly revenue to
fraud. In this course, learn to fight fraud by using data. Apply
supervised learning algorithms to detect fraudulent behavior based upon
past fraud, and use unsupervised learning methods to discover new types
of fraud activities.

Fraudulent transactions are rare compared to the norm. As such, learn to
properly classify imbalanced datasets.

The course provides technical and theoretical insights and demonstrates
how to implement fraud detection models. Finally, get tips and advice
from real-life experience to help prevent common mistakes in fraud
analytics.
:::

::: {.cell .markdown}
**Imports**
:::

::: {.cell .code execution_count="1"}
``` python
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
```
:::

::: {.cell .markdown}
Explain every import by putting a comment in front
:::

::: {.cell .code execution_count="2"}
``` python
import pandas as pd ### importing pandas foe dealing with dataframe
import matplotlib.pyplot as plt ### importing matplotlib.pyplot for graphs
from matplotlib.patches import Rectangle ## The Rectangle class is one such class that can be used to create a rectangle patch.
import numpy as np ### importing numpy 
from pprint import pprint as pp # The pprint module in Python provides a simple way to pretty-print data structures
import csv ## importing csv 
from pathlib import Path ### importing path for path object
import seaborn as sns ## importing seaborn 
from itertools import product ### import product for cartesian product
import string ### import string 

import nltk ## The nltk (Natural Language Toolkit)
from nltk.corpus import stopwords ### importing stopwords
from nltk.stem.wordnet import WordNetLemmatizer ### importing WordNetLemmatizer reduce words to base


### imblearn is a Python library that provides tools for handling imbalanced datasets in machine learning.
## The library includes a number of techniques for oversampling, undersampling, and combining different sampling methods.

from imblearn.over_sampling import SMOTE ### oversampling methods
from imblearn.over_sampling import BorderlineSMOTE ### oversampling methods
from imblearn.pipeline import Pipeline  ### The Pipeline class in imblearn allows you to chain together different sampling and machine learning steps into a single object

### importing sklearn and algorithms and metrics
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import homogeneity_score, silhouette_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans, DBSCAN

### gensim is a Python library for topic modeling, document indexing, and similarity retrieval with large corpora.
#  The library provides tools for building and training models on textual data.
import gensim
from gensim import corpora
```
:::

::: {.cell .markdown}
**Pandas Configuration Options**
:::

::: {.cell .code execution_count="3"}
``` python
pd.set_option('display.max_columns', 700) ### This line of code sets the maximum number of columns that can be displayed when using the pd.DataFrame object in pandas to 700
pd.set_option('display.max_rows', 400) ### This line of code sets the maximum number of rows that can be displayed when using the pd.DataFrame object in pandas to 400
pd.set_option('display.min_rows', 10)### This line of code sets the minimum number of rows that can be displayed when using the pd.DataFrame object in pandas to 10.
pd.set_option('display.expand_frame_repr', True)### is a command in the pandas library for Python that sets an option for how data frames are displayed in the console.
```
:::

::: {.cell .markdown}
**Data Files Location**

-   Most data files for the exercises can be found on the [course
    site](https://www.datacamp.com/courses/fraud-detection-in-python)
    -   [Chapter
        1](https://assets.datacamp.com/production/repositories/2162/datasets/cc3a36b722c0806e4a7df2634e345975a0724958/chapter_1.zip)
    -   [Chapter
        2](https://assets.datacamp.com/production/repositories/2162/datasets/4fb6199be9b89626dcd6b36c235cbf60cf4c1631/chapter_2.zip)
    -   [Chapter
        3](https://assets.datacamp.com/production/repositories/2162/datasets/08cfcd4158b3a758e72e9bd077a9e44fec9f773b/chapter_3.zip)
    -   [Chapter
        4](https://assets.datacamp.com/production/repositories/2162/datasets/94f2356652dc9ea8f0654b5e9c29645115b6e77f/chapter_4.zip)
:::

::: {.cell .markdown}
**Data File Objects**
:::

::: {.cell .code execution_count="72"}
``` python
## This line of code creates a Path object that represents the path to a directory called fraud_detection inside a directory called data in the current working directory (cwd).
data = Path.cwd() / 'data' / 'fraud_detection' 

ch1 = data / 'chapitre_1'
cc1_file = ch1 / 'creditcard_sampledata.csv'
cc3_file = ch1 / 'creditcard_sampledata_3.csv'

ch2 = data / 'chapitre_2'
cc2_file = ch2 / 'creditcard_sampledata_2.csv'

ch3 = data / 'chapitre_3'
banksim_file = ch3 / 'banksim.csv'
banksim_adj_file = ch3 / 'banksim_adj.csv'
db_full_file = ch3 / 'db_full.pickle'
labels_file = ch3 / 'labels.pickle'
labels_full_file = ch3 / 'labels_full.pickle'
x_scaled_file = ch3 / 'x_scaled.pickle'
x_scaled_full_file = ch3 / 'x_scaled_full.pickle'

ch4 = data / 'chapitre_4'
enron_emails_clean_file = ch4 / 'enron_emails_clean.csv'
cleantext_file = ch4 / 'cleantext.pickle'
corpus_file = ch4 / 'corpus.pickle'
dict_file = ch4 / 'dict.pickle'
ldamodel_file = ch4 / 'ldamodel.pickle'
```
:::

::: {.cell .markdown}
# Introduction and preparing your data

Learn about the typical challenges associated with fraud detection.
Learn how to resample data in a smart way, and tackle problems with
imbalanced data.
:::

::: {.cell .markdown}
## Introduction to fraud detection

-   Types:
    -   Insurance
    -   Credit card
    -   Identity theft
    -   Money laundering
    -   Tax evasion
    -   Healthcare
    -   Product warranty
-   e-commerce businesses must continuously assess the legitimacy of
    client transactions
-   Detecting fraud is challenging:
    -   Uncommon; \< 0.01% of transactions
    -   Attempts are made to conceal fraud
    -   Behavior evolves
    -   Fraudulent activities perpetrated by networks - organized crime
-   Fraud detection requires training an algorithm to identify concealed
    observations from any normal observations
-   Fraud analytics teams:
    -   Often use rules based systems, based on manually set thresholds
        and experience
    -   Check the news
    -   Receive external lists of fraudulent accounts and names
        -   suspicious names or track an external hit list from police
            to reference check against the client base
    -   Sometimes use machine learning algorithms to detect fraud or
        suspicious behavior
        -   Existing sources can be used as inputs into the ML model
        -   Verify the veracity of rules based labels
:::

::: {.cell .markdown}
### Checking the fraud to non-fraud ratio

In this chapter, you will work on `creditcard_sampledata.csv`, a dataset
containing credit card transactions data. Fraud occurrences are
fortunately an **extreme minority** in these transactions.

However, Machine Learning algorithms usually work best when the
different classes contained in the dataset are more or less equally
present. If there are few cases of fraud, then there\'s little data to
learn how to identify them. This is known as **class imbalance**, and
it\'s one of the main challenges of fraud detection.

Let\'s explore this dataset, and observe this class imbalance problem.

**Instructions**

-   `import pandas as pd`, read the credit card data in and assign it to
    `df`. This has been done for you.
-   Use `.info()` to print information about `df`.
-   Use `.value_counts()` to get the count of fraudulent and
    non-fraudulent transactions in the `'Class'` column. Assign the
    result to `occ`.
-   Get the ratio of fraudulent transactions over the total number of
    transactions in the dataset.
:::

::: {.cell .code execution_count="5"}
``` python
#df = pd.read_csv(cc3_file)
df = pd.read_csv("chapitre_1/creditcard_sampledata_3.csv")
```
:::

::: {.cell .markdown}
#### Explore the features available in your dataframe
:::

::: {.cell .code execution_count="6"}
``` python
df.info()
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5050 entries, 0 to 5049
    Data columns (total 31 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   Unnamed: 0  5050 non-null   int64  
     1   V1          5050 non-null   float64
     2   V2          5050 non-null   float64
     3   V3          5050 non-null   float64
     4   V4          5050 non-null   float64
     5   V5          5050 non-null   float64
     6   V6          5050 non-null   float64
     7   V7          5050 non-null   float64
     8   V8          5050 non-null   float64
     9   V9          5050 non-null   float64
     10  V10         5050 non-null   float64
     11  V11         5050 non-null   float64
     12  V12         5050 non-null   float64
     13  V13         5050 non-null   float64
     14  V14         5050 non-null   float64
     15  V15         5050 non-null   float64
     16  V16         5050 non-null   float64
     17  V17         5050 non-null   float64
     18  V18         5050 non-null   float64
     19  V19         5050 non-null   float64
     20  V20         5050 non-null   float64
     21  V21         5050 non-null   float64
     22  V22         5050 non-null   float64
     23  V23         5050 non-null   float64
     24  V24         5050 non-null   float64
     25  V25         5050 non-null   float64
     26  V26         5050 non-null   float64
     27  V27         5050 non-null   float64
     28  V28         5050 non-null   float64
     29  Amount      5050 non-null   float64
     30  Class       5050 non-null   int64  
    dtypes: float64(29), int64(2)
    memory usage: 1.2 MB
:::
:::

::: {.cell .code execution_count="7"}
``` python
df.head()
```

::: {.output .execute_result execution_count="7"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>V11</th>
      <th>V12</th>
      <th>V13</th>
      <th>V14</th>
      <th>V15</th>
      <th>V16</th>
      <th>V17</th>
      <th>V18</th>
      <th>V19</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>258647</td>
      <td>1.725265</td>
      <td>-1.337256</td>
      <td>-1.012687</td>
      <td>-0.361656</td>
      <td>-1.431611</td>
      <td>-1.098681</td>
      <td>-0.842274</td>
      <td>-0.026594</td>
      <td>-0.032409</td>
      <td>0.215113</td>
      <td>1.618952</td>
      <td>-0.654046</td>
      <td>-1.442665</td>
      <td>-1.546538</td>
      <td>-0.230008</td>
      <td>1.785539</td>
      <td>1.419793</td>
      <td>0.071666</td>
      <td>0.233031</td>
      <td>0.275911</td>
      <td>0.414524</td>
      <td>0.793434</td>
      <td>0.028887</td>
      <td>0.419421</td>
      <td>-0.367529</td>
      <td>-0.155634</td>
      <td>-0.015768</td>
      <td>0.010790</td>
      <td>189.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>69263</td>
      <td>0.683254</td>
      <td>-1.681875</td>
      <td>0.533349</td>
      <td>-0.326064</td>
      <td>-1.455603</td>
      <td>0.101832</td>
      <td>-0.520590</td>
      <td>0.114036</td>
      <td>-0.601760</td>
      <td>0.444011</td>
      <td>1.521570</td>
      <td>0.499202</td>
      <td>-0.127849</td>
      <td>-0.237253</td>
      <td>-0.752351</td>
      <td>0.667190</td>
      <td>0.724785</td>
      <td>-1.736615</td>
      <td>0.702088</td>
      <td>0.638186</td>
      <td>0.116898</td>
      <td>-0.304605</td>
      <td>-0.125547</td>
      <td>0.244848</td>
      <td>0.069163</td>
      <td>-0.460712</td>
      <td>-0.017068</td>
      <td>0.063542</td>
      <td>315.17</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>96552</td>
      <td>1.067973</td>
      <td>-0.656667</td>
      <td>1.029738</td>
      <td>0.253899</td>
      <td>-1.172715</td>
      <td>0.073232</td>
      <td>-0.745771</td>
      <td>0.249803</td>
      <td>1.383057</td>
      <td>-0.483771</td>
      <td>-0.782780</td>
      <td>0.005242</td>
      <td>-1.273288</td>
      <td>-0.269260</td>
      <td>0.091287</td>
      <td>-0.347973</td>
      <td>0.495328</td>
      <td>-0.925949</td>
      <td>0.099138</td>
      <td>-0.083859</td>
      <td>-0.189315</td>
      <td>-0.426743</td>
      <td>0.079539</td>
      <td>0.129692</td>
      <td>0.002778</td>
      <td>0.970498</td>
      <td>-0.035056</td>
      <td>0.017313</td>
      <td>59.98</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>281898</td>
      <td>0.119513</td>
      <td>0.729275</td>
      <td>-1.678879</td>
      <td>-1.551408</td>
      <td>3.128914</td>
      <td>3.210632</td>
      <td>0.356276</td>
      <td>0.920374</td>
      <td>-0.160589</td>
      <td>-0.801748</td>
      <td>0.137341</td>
      <td>-0.156740</td>
      <td>-0.429388</td>
      <td>-0.752392</td>
      <td>0.155272</td>
      <td>0.215068</td>
      <td>0.352222</td>
      <td>-0.376168</td>
      <td>-0.398920</td>
      <td>0.043715</td>
      <td>-0.335825</td>
      <td>-0.906171</td>
      <td>0.108350</td>
      <td>0.593062</td>
      <td>-0.424303</td>
      <td>0.164201</td>
      <td>0.245881</td>
      <td>0.071029</td>
      <td>0.89</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>86917</td>
      <td>1.271253</td>
      <td>0.275694</td>
      <td>0.159568</td>
      <td>1.003096</td>
      <td>-0.128535</td>
      <td>-0.608730</td>
      <td>0.088777</td>
      <td>-0.145336</td>
      <td>0.156047</td>
      <td>0.022707</td>
      <td>-0.963306</td>
      <td>-0.228074</td>
      <td>-0.324933</td>
      <td>0.390609</td>
      <td>1.065923</td>
      <td>0.285930</td>
      <td>-0.627072</td>
      <td>0.170175</td>
      <td>-0.215912</td>
      <td>-0.147394</td>
      <td>0.031958</td>
      <td>0.123503</td>
      <td>-0.174528</td>
      <td>-0.147535</td>
      <td>0.735909</td>
      <td>-0.262270</td>
      <td>0.015577</td>
      <td>0.015955</td>
      <td>6.53</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .code execution_count="8"}
``` python
# Count the occurrences of fraud and no fraud and print them
occ = df['Class'].value_counts()
occ
```

::: {.output .execute_result execution_count="8"}
    0    5000
    1      50
    Name: Class, dtype: int64
:::
:::

::: {.cell .code execution_count="9"}
``` python
# Print the ratio of fraud cases
print(round(50/5050,2))
```

::: {.output .stream .stdout}
    0.01
:::
:::

::: {.cell .markdown}
**The ratio of fraudulent transactions is very low. This is a case of
class imbalance problem, and you\'re going to learn how to deal with
this in the next exercises.**
:::

::: {.cell .markdown}
### Data visualization

From the previous exercise we know that the ratio of fraud to non-fraud
observations is very low. You can do something about that, for example
by **re-sampling** our data, which is explained in the next video.

In this exercise, you\'ll look at the data and **visualize the fraud to
non-fraud ratio**. It is always a good starting point in your fraud
analysis, to look at your data first, before you make any changes to it.

Moreover, when talking to your colleagues, a picture often makes it very
clear that we\'re dealing with heavily imbalanced data. Let\'s create a
plot to visualize the ratio fraud to non-fraud data points on the
dataset `df`.

The function `prep_data()` is already loaded in your workspace, as well
as `matplotlib.pyplot as plt`.

**Instructions**

-   Define the `plot_data(X, y)` function, that will nicely plot the
    given feature set `X` with labels `y` in a scatter plot. This has
    been done for you.
-   Use the function `prep_data()` on your dataset `df` to create
    feature set `X` and labels `y`.
-   Run the function `plot_data()` on your newly obtained `X` and `y` to
    visualize your results.
:::

::: {.cell .markdown}
#### def prep_data
:::

::: {.cell .code execution_count="10"}
``` python
def prep_data(df):
    """
    Convert the DataFrame into two variable
    X: data columns (V1 - V28)
    y: lable column
    """
    X = df.iloc[:, 2:30].values
    y = df.Class.values
    return X, y
```
:::

::: {.cell .markdown}
#### def plot_data
:::

::: {.cell .code execution_count="11"}
``` python
# Define a function to create a scatter plot of our data and labels
def plot_data(X: np.ndarray, y: np.ndarray):
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c='r')
    plt.legend()
    return plt.show()
```
:::

::: {.cell .code execution_count="12"}
``` python
# Create X and y from the prep_data function 
X, y = prep_data(df)
```
:::

::: {.cell .code execution_count="13"}
``` python
# Plot our data by running our plot data function on X and y
plot_data(X,y)
```

::: {.output .display_data}
![](vertopal_5c32587947084112be3b965be5a32bb5/3985a383bc8a6e3d705d2f2c9f546e0824fbc215.png)
:::
:::

::: {.cell .markdown}
**By visualizing the data, you can immediately see how our fraud cases
are scattered over our data, and how few cases we have. A picture often
makes the imbalance problem clear. In the next exercises we\'ll visually
explore how to improve our fraud to non-fraud balance.**
:::

::: {.cell .markdown}
#### Reproduced using the DataFrame
:::

::: {.cell .code execution_count="14"}
``` python
plt.scatter(df.V2[df.Class == 0], df.V3[df.Class == 0], label="Class #0", alpha=0.5, linewidth=0.15)
plt.scatter(df.V2[df.Class == 1], df.V3[df.Class == 1], label="Class #1", alpha=0.5, linewidth=0.15, c='r')
plt.legend()
plt.show()
```

::: {.output .display_data}
![](vertopal_5c32587947084112be3b965be5a32bb5/3985a383bc8a6e3d705d2f2c9f546e0824fbc215.png)
:::
:::

::: {.cell .markdown}
## Increase successful detections with data resampling

-   resampling can help model performance in cases of imbalanced data
    sets
:::

::: {.cell .markdown}
#### Undersampling

-   ![undersampling](vertopal_5c32587947084112be3b965be5a32bb5/b87ed4f49b10d069879e6e4cddd6cb792098a281.JPG)
-   Undersampling the majority class (non-fraud cases)
    -   Straightforward method to adjust imbalanced data
    -   Take random draws from the non-fraud observations, to match the
        occurences of fraud observations (as shown in the picture)
:::

::: {.cell .markdown}
#### Oversampling

-   ![oversampling](vertopal_5c32587947084112be3b965be5a32bb5/52d6f0f4dd7767c16e847c269d3b66ebcb6b6fdd.JPG)
-   Oversampling the minority class (fraud cases)
    -   Take random draws from the fraud cases and copy those
        observations to increase the amount of fraud samples
-   Both methods lead to having a balance between fraud and non-fraud
    cases
-   Drawbacks
    -   with random undersampling, a lot of information is thrown away
    -   with oversampling, the model will be trained on a lot of
        duplicates
:::

::: {.cell .markdown}
#### Implement resampling methods using Python imblean module

-   compatible with scikit-learn

``` python
from imblearn.over_sampling import RandomOverSampler

method = RandomOverSampler()
X_resampled, y_resampled =  method.fit_sample(X, y)

compare_plots(X_resampled, y_resampled, X, y)
```

![oversampling
plot](vertopal_5c32587947084112be3b965be5a32bb5/b8403c4129910fe26c43f377621ec4e078e80e56.JPG)

-   The darker blue points reflect there are more identical data
:::

::: {.cell .markdown}
#### SMOTE

-   ![smote](vertopal_5c32587947084112be3b965be5a32bb5/2a4f4a63224bd39e9ba64f7e98f9a46a245dbb97.JPG)
-   Synthetic minority Oversampling Technique (SMOTE)
    -   [Resampling strategies for Imbalanced Data
        Sets](https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets)
    -   Another way of adjusting the imbalance by oversampling minority
        observations
    -   SMOTE uses characteristics of nearest neighbors of fraud cases
        to create new synthetic fraud cases
        -   avoids duplicating observations
:::

::: {.cell .markdown}
#### Determining the best resampling method is situational

-   Random Undersampling (RUS):
    -   If there is a lot of data and many minority cases, then
        undersampling may be computationally more convenient
        -   In most cases, throwing away data is not desirable
-   Random Oversampling (ROS):
    -   Straightforward
    -   Training the model on many duplicates
-   SMOTE:
    -   more sophisticated
    -   realistic data set
    -   training on synthetic data
    -   only works well if the minority case features are similar
        -   **if fraud is spread through the data and not distinct,
            using nearest neighbors to create more fraud cases,
            introduces noise into the data, as the nearest neighbors
            might not be fraud cases**
:::

::: {.cell .markdown}
#### When to use resmapling methods

-   Use resampling methods on the training set, not on the test set
-   The goal is to produce a better model by providing balanced data
    -   The goal is not to predict the synthetic samples
-   Test data should be free of duplicates and synthetic data
-   Only test the model on real data
    -   First, spit the data into train and test sets

``` python
# Define resampling method and split into train and test
method = SMOTE(kind='borderline1')
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

# Apply resampling to the training data only
X_resampled, y_resampled = method.fit_sample(X_train, y_train)

# Continue fitting the model and obtain predictions
model = LogisticRegression()
model.fit(X_resampled, y_resampled)

# Get model performance metrics
predicted = model.predict(X_test)
print(classification_report(y_test, predicted))
```
:::

::: {.cell .markdown}
### Resampling methods for imbalanced data

Which of these methods takes a random subsample of your majority class
to account for class \"imbalancedness\"?

**Possible Answers**

-   ~~Random Over Sampling (ROS)~~
-   **Random Under Sampling (RUS)**
-   ~~Synthetic Minority Over-sampling Technique (SMOTE)~~
-   ~~None of the above~~

**By using ROS and SMOTE you add more examples to the minority class.
RUS adjusts the balance of your data by reducing the majority class.**
:::

::: {.cell .markdown}
### Applying Synthetic Minority Oversampling Technique (SMOTE)

In this exercise, you\'re going to re-balance our data using the
**Synthetic Minority Over-sampling Technique** (SMOTE). Unlike ROS,
SMOTE does not create exact copies of observations, but **creates new,
synthetic, samples** that are quite similar to the existing observations
in the minority class. SMOTE is therefore slightly more sophisticated
than just copying observations, so let\'s apply SMOTE to our credit card
data. The dataset `df` is available and the packages you need for SMOTE
are imported. In the following exercise, you\'ll visualize the result
and compare it to the original data, such that you can see the effect of
applying SMOTE very clearly.

**Instructions**

-   Use the `prep_data` function on `df` to create features `X` and
    labels `y`.
-   Define the resampling method as SMOTE of the regular kind, under the
    variable `method`.
-   Use `.fit_sample()` on the original `X` and `y` to obtain newly
    resampled data.
-   Plot the resampled data using the `plot_data()` function.
:::

::: {.cell .code execution_count="15"}
``` python
# Run the prep_data function
X,y=prep_data(df)
```
:::

::: {.cell .code execution_count="16"}
``` python
print(f'X shape: {X.shape}\ny shape: {y.shape}')
```

::: {.output .stream .stdout}
    X shape: (5050, 28)
    y shape: (5050,)
:::
:::

::: {.cell .code execution_count="17"}
``` python
# Define the resampling method
method = SMOTE()
```
:::

::: {.cell .code execution_count="18"}
``` python
# Create the resampled feature set
X_resampled, y_resampled = method.fit_resample(X, y)
```
:::

::: {.cell .code execution_count="19"}
``` python
# Plot the resampled data
plot_data(X_resampled,y_resampled)
```

::: {.output .display_data}
![](vertopal_5c32587947084112be3b965be5a32bb5/28140138f501d0831843babd1f93d6b5226825e9.png)
:::
:::

::: {.cell .markdown}
**The minority class is now much more prominently visible in our data.
To see the results of SMOTE even better, we\'ll compare it to the
original data in the next exercise.**
:::

::: {.cell .markdown}
### Compare SMOTE to original data

In the last exercise, you saw that using SMOTE suddenly gives us more
observations of the minority class. Let\'s compare those results to our
original data, to get a good feeling for what has actually happened.
Let\'s have a look at the value counts again of our old and new data,
and let\'s plot the two scatter plots of the data side by side. You\'ll
use the function compare_plot() for that that, which takes the following
arguments: `X`, `y`, `X_resampled`, `y_resampled`, `method=''`. The
function plots your original data in a scatter plot, along with the
resampled side by side.

**Instructions**

-   Print the value counts of our original labels, `y`. Be mindful that
    `y` is currently a Numpy array, so in order to use value counts,
    we\'ll assign `y` back as a pandas Series object.
-   Repeat the step and print the value counts on `y_resampled`. This
    shows you how the balance between the two classes has changed with
    SMOTE.
-   Use the `compare_plot()` function called on our original data as
    well our resampled data to see the scatterplots side by side.
:::

::: {.cell .code execution_count="20"}
``` python
pd.value_counts(pd.Series(y))
```

::: {.output .execute_result execution_count="20"}
    0    5000
    1      50
    dtype: int64
:::
:::

::: {.cell .code execution_count="21"}
``` python
pd.value_counts(pd.Series(y_resampled))
```

::: {.output .execute_result execution_count="21"}
    0    5000
    1    5000
    dtype: int64
:::
:::

::: {.cell .markdown}
#### def compare_plot
:::

::: {.cell .code execution_count="22"}
``` python
def compare_plot(X: np.ndarray, y: np.ndarray, X_resampled: np.ndarray, y_resampled: np.ndarray, method: str):
    plt.subplot(1, 2, 1)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c='r')
    plt.title('Original Set')
    plt.subplot(1, 2, 2)
    plt.scatter(X_resampled[y_resampled == 0, 0], X_resampled[y_resampled == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15)
    plt.scatter(X_resampled[y_resampled == 1, 0], X_resampled[y_resampled == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c='r')
    plt.title(method)
    plt.legend()
    plt.show()
```
:::

::: {.cell .code execution_count="23"}
``` python
compare_plot(X, y, X_resampled, y_resampled, method='SMOTE')
```

::: {.output .display_data}
![](vertopal_5c32587947084112be3b965be5a32bb5/b2270400532aae31c1abe6ae54ff9f4025faf7e8.png)
:::
:::

::: {.cell .markdown}
**It should by now be clear that SMOTE has balanced our data completely,
and that the minority class is now equal in size to the majority class.
Visualizing the data shows the effect on the data very clearly. The next
exercise will demonstrate multiple ways to implement SMOTE and that each
method will have a slightly different effect.**
:::

::: {.cell .markdown}
## Fraud detection algorithms in action
:::

::: {.cell .markdown}
#### Rules Based Systems

-   ![rules
    based](vertopal_5c32587947084112be3b965be5a32bb5/674ae55734918e8089a247cd7d1dd1451abe2fad.JPG)
-   Might block transactions from risky zip codes
-   Block transactions from cards used too frequently (e.g. last 30
    minutes)
-   Can catch fraud, but also generates false alarms (false positive)
-   Limitations:
    -   Fixed threshold per rule and it\'s difficult to determine the
        threshold; they don\'t adapt over time
    -   Limited to yes / no outcomes, whereas ML yields a probability
        -   probability allows for fine-tuning the outcomes (i.e. rate
            of occurences of false positives and false negatives)
    -   Fails to capture interaction between features
        -   Ex. Size of the transaction only matters in combination to
            the frequency
:::

::: {.cell .markdown}
#### ML Based Systems

-   Adapt to the data, thus can change over time
-   Uses all the data combined, rather than a threshold per feature
-   Produces a probability, rather than a binary score
-   Typically have better performance and can be combined with rules
:::

::: {.cell .code execution_count="24"}
``` python
# Step 1: split the features and labels into train and test data with test_size=0.2
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
```
:::

::: {.cell .code execution_count="25"}
``` python
# Step 2: Define which model to use
model = LinearRegression()
```
:::

::: {.cell .code execution_count="26"}
``` python
# Step 3: Fit the model to the training data
model.fit(X_train,y_train)
```

::: {.output .execute_result execution_count="26"}
```{=html}
<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>
```
:::
:::

::: {.cell .code execution_count="27"}
``` python
# Step 4: Obtain model predictions from the test data
y_predicted = model.predict(X_test)
```
:::

::: {.cell .code execution_count="28"}
``` python
# Step 5: Compare y_test to predictions and obtain performance metrics (r^2 score)
r2_score(y_test,y_predicted)
```

::: {.output .execute_result execution_count="28"}
    0.6217658594718924
:::
:::

::: {.cell .markdown}
### Exploring the traditional method of fraud detection

In this exercise you\'re going to try finding fraud cases in our credit
card dataset the *\"old way\"*. First you\'ll define threshold values
using common statistics, to split fraud and non-fraud. Then, use those
thresholds on your features to detect fraud. This is common practice
within fraud analytics teams.

Statistical thresholds are often determined by looking at the **mean**
values of observations. Let\'s start this exercise by checking whether
feature **means differ between fraud and non-fraud cases**. Then,
you\'ll use that information to create common sense thresholds. Finally,
you\'ll check how well this performs in fraud detection.

`pandas` has already been imported as `pd`.

**Instructions**

-   Use `groupby()` to group `df` on `Class` and obtain the mean of the
    features.
-   Create the condition `V1` smaller than -3, and `V3` smaller than -5
    as a condition to flag fraud cases.
-   As a measure of performance, use the `crosstab` function from
    `pandas` to compare our flagged fraud cases to actual fraud cases.
:::

::: {.cell .code execution_count="29"}
``` python
df.drop(['Unnamed: 0'], axis=1, inplace=True)
```
:::

::: {.cell .code execution_count="30"}
``` python
df.groupby('Class').mean()
```

::: {.output .execute_result execution_count="30"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>V11</th>
      <th>V12</th>
      <th>V13</th>
      <th>V14</th>
      <th>V15</th>
      <th>V16</th>
      <th>V17</th>
      <th>V18</th>
      <th>V19</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
    </tr>
    <tr>
      <th>Class</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.035030</td>
      <td>0.011553</td>
      <td>0.037444</td>
      <td>-0.045760</td>
      <td>-0.013825</td>
      <td>-0.030885</td>
      <td>0.014315</td>
      <td>-0.022432</td>
      <td>-0.002227</td>
      <td>0.001667</td>
      <td>-0.004511</td>
      <td>0.017434</td>
      <td>0.004204</td>
      <td>0.006542</td>
      <td>-0.026640</td>
      <td>0.001190</td>
      <td>0.004481</td>
      <td>-0.010892</td>
      <td>-0.016554</td>
      <td>-0.002896</td>
      <td>-0.010583</td>
      <td>-0.010206</td>
      <td>-0.003305</td>
      <td>-0.000918</td>
      <td>-0.002613</td>
      <td>-0.004651</td>
      <td>-0.009584</td>
      <td>0.002414</td>
      <td>85.843714</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-4.985211</td>
      <td>3.321539</td>
      <td>-7.293909</td>
      <td>4.827952</td>
      <td>-3.326587</td>
      <td>-1.591882</td>
      <td>-5.776541</td>
      <td>1.395058</td>
      <td>-2.537728</td>
      <td>-5.917934</td>
      <td>4.020563</td>
      <td>-7.032865</td>
      <td>-0.104179</td>
      <td>-7.100399</td>
      <td>-0.120265</td>
      <td>-4.658854</td>
      <td>-7.589219</td>
      <td>-2.650436</td>
      <td>0.894255</td>
      <td>0.194580</td>
      <td>0.703182</td>
      <td>0.069065</td>
      <td>-0.088374</td>
      <td>-0.029425</td>
      <td>-0.073336</td>
      <td>-0.023377</td>
      <td>0.380072</td>
      <td>0.009304</td>
      <td>113.469000</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .code execution_count="31"}
``` python
df['flag_as_fraud'] = np.where(np.logical_and(df.V1 < -3, df.V3 < -5), 1, 0)
```
:::

::: {.cell .code execution_count="32"}
``` python
pd.crosstab(df.Class, df.flag_as_fraud, rownames=['Actual Fraud'], colnames=['Flagged Fraud'])
```

::: {.output .execute_result execution_count="32"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Flagged Fraud</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>Actual Fraud</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4984</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>28</td>
      <td>22</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown}
**With this rule, 22 out of 50 fraud cases are detected, 28 are not
detected, and 16 false positives are identified.**
:::

::: {.cell .markdown}
### Using ML classification to catch fraud

In this exercise you\'ll see what happens when you use a simple machine
learning model on our credit card data instead.

Do you think you can beat those results? Remember, you\'ve predicted *22
out of 50* fraud cases, and had *16 false positives*.

So with that in mind, let\'s implement a **Logistic Regression** model.
If you have taken the class on supervised learning in Python, you should
be familiar with this model. If not, you might want to refresh that at
this point. But don\'t worry, you\'ll be guided through the structure of
the machine learning model.

The `X` and `y` variables are available in your workspace.

**Instructions**

-   Split `X` and `y` into training and test data, keeping 30% of the
    data for testing.
-   Fit your model to your training data.
-   Obtain the model predicted labels by running `model.predict` on
    `X_test`.
-   Obtain a classification comparing `y_test` with `predicted`, and use
    the given confusion matrix to check your results.
:::

::: {.cell .code execution_count="33"}
``` python
# Create the training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
```
:::

::: {.cell .code execution_count="34"}
``` python
# Fit a logistic regression model to our data
model=LogisticRegression()
model.fit(X_train,y_train)
```

::: {.output .execute_result execution_count="34"}
```{=html}
<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div></div></div>
```
:::
:::

::: {.cell .code execution_count="35"}
``` python
# Obtain model predictions
predicted = model.predict(X_test)
```
:::

::: {.cell .code execution_count="36"}
``` python
# Print the classifcation report and confusion matrix
print('Classification report:\n', classification_report(y_test, predicted))
####
print('Confusion matrix:\n', confusion_matrix(y_test, predicted))
```

::: {.output .stream .stdout}
    Classification report:
                   precision    recall  f1-score   support

               0       1.00      1.00      1.00      1505
               1       0.89      0.80      0.84        10

        accuracy                           1.00      1515
       macro avg       0.94      0.90      0.92      1515
    weighted avg       1.00      1.00      1.00      1515

    Confusion matrix:
     [[1504    1]
     [   2    8]]
:::
:::

::: {.cell .markdown}
**Do you think these results are better than the rules based model? We
are getting far fewer false positives, so that\'s an improvement. Also,
we\'re catching a higher percentage of fraud cases, so that is also
better than before. Do you understand why we have fewer observations to
look at in the confusion matrix? Remember we are using only our test
data to calculate the model results on. We\'re comparing the crosstab on
the full dataset from the last exercise, with a confusion matrix of only
30% of the total dataset, so that\'s where that difference comes from.
In the next chapter, we\'ll dive deeper into understanding these model
performance metrics. Let\'s now explore whether we can improve the
prediction results even further with resampling methods.**
:::

::: {.cell .markdown}
### Logistic regression with SMOTE

In this exercise, you\'re going to take the Logistic Regression model
from the previous exercise, and combine that with a **SMOTE resampling
method**. We\'ll show you how to do that efficiently by using a pipeline
that combines the resampling method with the model in one go. First, you
need to define the pipeline that you\'re going to use.

**Instructions**

-   Import the `Pipeline` module from `imblearn`, this has been done for
    you.
-   Then define what you want to put into the pipeline, assign the
    `SMOTE` method with `borderline2` to `resampling`, and assign
    `LogisticRegression()` to the `model`.
-   Combine two steps in the `Pipeline()` function. You need to state
    you want to combine `resampling` with the `model` in the respective
    place in the argument. I show you how to do this.
:::

::: {.cell .code execution_count="37"}
``` python
# Define which resampling method and which ML model to use in the pipeline
# resampling = SMOTE(kind='borderline2')  # has been changed to BorderlineSMOTE
resampling = BorderlineSMOTE()
model = LogisticRegression(solver='liblinear')
```
:::

::: {.cell .code execution_count="38"}
``` python
pipeline = Pipeline([('SMOTE', resampling), ('Logistic Regression', model)])
```
:::

::: {.cell .markdown}
### Pipelining

Now that you have our pipeline defined, aka **combining a logistic
regression with a SMOTE method**, let\'s run it on the data. You can
treat the pipeline as if it were a **single machine learning model**.
Our data X and y are already defined, and the pipeline is defined in the
previous exercise. Are you curious to find out what the model results
are? Let\'s give it a try!

**Instructions**

-   Split the data \'X\'and \'y\' into the training and test set. Set
    aside 30% of the data for a test set, and set the `random_state` to
    zero.
-   Fit your pipeline onto your training data and obtain the predictions
    by running the `pipeline.predict()` function on our `X_test`
    dataset.
:::

::: {.cell .code execution_count="39"}
``` python
# Split your data X and y, into a training and a test set and fit the pipeline onto the training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
```
:::

::: {.cell .code execution_count="40"}
``` python
pipeline.fit(X_train, y_train) 
predicted = pipeline.predict(X_test)
```
:::

::: {.cell .code execution_count="41"}
``` python
# Obtain the results from the classification report and confusion matrix 
print('Classifcation report:\n', classification_report(y_test, predicted))
####
print('Confusion matrix:\n', confusion_matrix(y_test, predicted))
```

::: {.output .stream .stdout}
    Classifcation report:
                   precision    recall  f1-score   support

               0       1.00      1.00      1.00      1505
               1       0.62      1.00      0.77        10

        accuracy                           1.00      1515
       macro avg       0.81      1.00      0.88      1515
    weighted avg       1.00      1.00      1.00      1515

    Confusion matrix:
     [[1499    6]
     [   0   10]]
:::
:::

::: {.cell .markdown}
**As you can see, the SMOTE slightly improves our results. We now manage
to find all cases of fraud, but we have a slightly higher number of
false positives, albeit only 7 cases. Remember, resampling doesn\'t
necessarily lead to better results. When the fraud cases are very spread
and scattered over the data, using SMOTE can introduce a bit of bias.
Nearest neighbors aren\'t necessarily also fraud cases, so the synthetic
samples might \'confuse\' the model slightly. In the next chapters,
we\'ll learn how to also adjust our machine learning models to better
detect the minority fraud cases.**
:::

::: {.cell .markdown}
# Fraud detection using labeled data

Learn how to flag fraudulent transactions with supervised learning. Use
classifiers, adjust and compare them to find the most efficient fraud
detection model.
:::

::: {.cell .markdown}
## Review classification methods

-   Classification:
    -   The problem of identifying to which class a new observation
        belongs, on the basis of a training set of data containing
        observations whose class is known
    -   Goal: use known fraud cases to train a model to recognize new
        cases
    -   Classes are sometimes called targets, labels or categories
    -   Spam detection in email service providers can be identified as a
        classification problem
        -   Binary classification since there are only 2 classes, spam
            and not spam
    -   Fraud detection is also a binary classification prpoblem
    -   Patient diagnosis
    -   Classification problems normall have categorical output like
        yes/no, 1/0 or True/False
    -   Variable to predict: $$y\in0,1$$
        -   0: negative calss (\'majority\' normal cases)
        -   1: positive class (\'minority\' fraud cases)
:::

::: {.cell .markdown}
#### Logistic Regression

-   Logistic Regression is one of the most used ML algorithms in binary
    classification
-   ![logistic
    regression](vertopal_5c32587947084112be3b965be5a32bb5/93676fb7d7973e3f5dc0e76112dd9e23d4de03bf.JPG)
-   Can be adjusted reasonably well to work on imbalanced data\...useful
    for fraud detection
:::

::: {.cell .markdown}
#### Neural Network

-   ![neural
    network](vertopal_5c32587947084112be3b965be5a32bb5/c79198d87d63e24a599b8b03afc0dfc3bb44de3c.JPG)
-   Can be used as classifiers for fraud detection
-   Capable of fitting highly non-linear models to the data
-   More complex to implement than other classifiers - not demonstrated
    here
:::

::: {.cell .markdown}
#### Decision Trees

-   ![decision
    tree](vertopal_5c32587947084112be3b965be5a32bb5/243b99d13a1556093976fd66bf1cb72fb21f33ee.JPG)
-   Commonly used for fraud detection
-   Transparent results, easily interpreted by analysts
-   Decision trees are prone to overfit the data
:::

::: {.cell .markdown}
#### Random Forests

-   ![random
    forest](vertopal_5c32587947084112be3b965be5a32bb5/6cfabb2c81a46fef409bb2e085c0354858302e8f.JPG)
-   **Random Forests are a more robust option than a single decision
    tree**
    -   Construct a multitude of decision trees when training the model
        and outputting the class that is the mode or mean predicted
        class of the individual trees
    -   A random forest consists of a collection of trees on a random
        subset of features
    -   Final predictions are the combined results of those trees
    -   Random forests can handle complex data and are not prone to
        overfit
    -   They are interpretable by looking at feature importance, and can
        be adjusted to work well on highly imbalanced data
    -   Their drawback is they\'re computationally complex
    -   Very popular for fraud detection
    -   A Random Forest model will be optimized in the exercises

**Implementation:**

``` python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
predicted = model.predict(X_test)
print(f'Accuracy Score:\n{accuracy_score(y_test, predicted)}')
```
:::

::: {.cell .markdown}
### Natural hit rate

In this exercise, you\'ll again use credit card transaction data. The
features and labels are similar to the data in the previous chapter, and
the **data is heavily imbalanced**. We\'ve given you features `X` and
labels `y` to work with already, which are both numpy arrays.

First you need to explore how prevalent fraud is in the dataset, to
understand what the **\"natural accuracy\"** is, if we were to predict
everything as non-fraud. It\'s is important to understand which level of
\"accuracy\" you need to \"beat\" in order to get a **better prediction
than by doing nothing**. In the following exercises, you\'ll create our
first random forest classifier for fraud detection. That will serve as
the **\"baseline\"** model that you\'re going to try to improve in the
upcoming exercises.

**Instructions**

-   Count the total number of observations by taking the length of your
    labels `y`.
-   Count the non-fraud cases in our data by using list comprehension on
    `y`; remember `y` is a NumPy array so `.value_counts()` cannot be
    used in this case.
-   Calculate the natural accuracy by dividing the non-fraud cases over
    the total observations.
-   Print the percentage.
:::

::: {.cell .code execution_count="42"}
``` python
df2 = pd.read_csv("chapitre_2/creditcard_sampledata_2.csv")
df2.head()
```

::: {.output .execute_result execution_count="42"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>V11</th>
      <th>V12</th>
      <th>V13</th>
      <th>V14</th>
      <th>V15</th>
      <th>V16</th>
      <th>V17</th>
      <th>V18</th>
      <th>V19</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>221547</td>
      <td>-1.191668</td>
      <td>0.428409</td>
      <td>1.640028</td>
      <td>-1.848859</td>
      <td>-0.870903</td>
      <td>-0.204849</td>
      <td>-0.385675</td>
      <td>0.352793</td>
      <td>-1.098301</td>
      <td>-0.334597</td>
      <td>-0.679089</td>
      <td>-0.039671</td>
      <td>1.372661</td>
      <td>-0.732001</td>
      <td>-0.344528</td>
      <td>1.024751</td>
      <td>0.380209</td>
      <td>-1.087349</td>
      <td>0.364507</td>
      <td>0.051924</td>
      <td>0.507173</td>
      <td>1.292565</td>
      <td>-0.467752</td>
      <td>1.244887</td>
      <td>0.697707</td>
      <td>0.059375</td>
      <td>-0.319964</td>
      <td>-0.017444</td>
      <td>27.44</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>184524</td>
      <td>1.966614</td>
      <td>-0.450087</td>
      <td>-1.228586</td>
      <td>0.142873</td>
      <td>-0.150627</td>
      <td>-0.543590</td>
      <td>-0.076217</td>
      <td>-0.108390</td>
      <td>0.973310</td>
      <td>-0.029903</td>
      <td>0.279973</td>
      <td>0.885685</td>
      <td>-0.583912</td>
      <td>0.322019</td>
      <td>-1.065335</td>
      <td>-0.340285</td>
      <td>-0.385399</td>
      <td>0.216554</td>
      <td>0.675646</td>
      <td>-0.190851</td>
      <td>0.124055</td>
      <td>0.564916</td>
      <td>-0.039331</td>
      <td>-0.283904</td>
      <td>0.186400</td>
      <td>0.192932</td>
      <td>-0.039155</td>
      <td>-0.071314</td>
      <td>35.95</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>91201</td>
      <td>1.528452</td>
      <td>-1.296191</td>
      <td>-0.890677</td>
      <td>-2.504028</td>
      <td>0.803202</td>
      <td>3.350793</td>
      <td>-1.633016</td>
      <td>0.815350</td>
      <td>-1.884692</td>
      <td>1.465259</td>
      <td>-0.188235</td>
      <td>-0.976779</td>
      <td>0.560550</td>
      <td>-0.250847</td>
      <td>0.936115</td>
      <td>0.136409</td>
      <td>-0.078251</td>
      <td>0.355086</td>
      <td>0.127756</td>
      <td>-0.163982</td>
      <td>-0.412088</td>
      <td>-1.017485</td>
      <td>0.129566</td>
      <td>0.948048</td>
      <td>0.287826</td>
      <td>-0.396592</td>
      <td>0.042997</td>
      <td>0.025853</td>
      <td>28.40</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26115</td>
      <td>-0.774614</td>
      <td>1.100916</td>
      <td>0.679080</td>
      <td>1.034016</td>
      <td>0.168633</td>
      <td>0.874582</td>
      <td>0.209454</td>
      <td>0.770550</td>
      <td>-0.558106</td>
      <td>-0.165442</td>
      <td>0.017562</td>
      <td>0.285377</td>
      <td>-0.818739</td>
      <td>0.637991</td>
      <td>-0.370124</td>
      <td>-0.605148</td>
      <td>0.275686</td>
      <td>0.246362</td>
      <td>1.331927</td>
      <td>0.080978</td>
      <td>0.011158</td>
      <td>0.146017</td>
      <td>-0.130401</td>
      <td>-0.848815</td>
      <td>0.005698</td>
      <td>-0.183295</td>
      <td>0.282940</td>
      <td>0.123856</td>
      <td>43.20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>201292</td>
      <td>-1.075860</td>
      <td>1.361160</td>
      <td>1.496972</td>
      <td>2.242604</td>
      <td>1.314751</td>
      <td>0.272787</td>
      <td>1.005246</td>
      <td>0.132932</td>
      <td>-1.558317</td>
      <td>0.484216</td>
      <td>-1.967998</td>
      <td>-1.818338</td>
      <td>-2.036184</td>
      <td>0.346962</td>
      <td>-1.161316</td>
      <td>1.017093</td>
      <td>-0.926787</td>
      <td>0.183965</td>
      <td>-2.102868</td>
      <td>-0.354008</td>
      <td>0.254485</td>
      <td>0.530692</td>
      <td>-0.651119</td>
      <td>0.626389</td>
      <td>1.040212</td>
      <td>0.249501</td>
      <td>-0.146745</td>
      <td>0.029714</td>
      <td>10.59</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .code execution_count="43"}
``` python
X, y = prep_data(df2)
print(f'X shape: {X.shape}\ny shape: {y.shape}')
```

::: {.output .stream .stdout}
    X shape: (7300, 28)
    y shape: (7300,)
:::
:::

::: {.cell .code execution_count="44"}
``` python
X[0, :]
```

::: {.output .execute_result execution_count="44"}
    array([ 4.28408570e-01,  1.64002800e+00, -1.84885886e+00, -8.70902974e-01,
           -2.04848888e-01, -3.85675453e-01,  3.52792552e-01, -1.09830131e+00,
           -3.34596757e-01, -6.79088729e-01, -3.96709268e-02,  1.37266082e+00,
           -7.32000706e-01, -3.44528134e-01,  1.02475103e+00,  3.80208554e-01,
           -1.08734881e+00,  3.64507163e-01,  5.19236276e-02,  5.07173439e-01,
            1.29256539e+00, -4.67752261e-01,  1.24488683e+00,  6.97706854e-01,
            5.93750372e-02, -3.19964326e-01, -1.74444289e-02,  2.74400000e+01])
:::
:::

::: {.cell .code execution_count="45"}
``` python
df2.Class.value_counts()
```

::: {.output .execute_result execution_count="45"}
    0    7000
    1     300
    Name: Class, dtype: int64
:::
:::

::: {.cell .code execution_count="46"}
``` python
# Count the total number of observations from the length of y
total_obs = len(y)
total_obs
```

::: {.output .execute_result execution_count="46"}
    7300
:::
:::

::: {.cell .code execution_count="47"}
``` python
# Count the total number of non-fraudulent observations 
non_fraud = [i for i in y if i == 0]
count_non_fraud = non_fraud.count(0)
count_non_fraud
```

::: {.output .execute_result execution_count="47"}
    7000
:::
:::

::: {.cell .code execution_count="48"}
``` python
percentage = count_non_fraud/total_obs * 100
print(f'{percentage:0.2f}%')
```

::: {.output .stream .stdout}
    95.89%
:::
:::

::: {.cell .markdown}
**This tells us that by doing nothing, we would be correct in 95.9% of
the cases. So now you understand, that if we get an accuracy of less
than this number, our model does not actually add any value in
predicting how many cases are correct. Let\'s see how a random forest
does in predicting fraud in our data.**
:::

::: {.cell .markdown}
### Random Forest Classifier - part 1

Let\'s now create a first **random forest classifier** for fraud
detection. Hopefully you can do better than the baseline accuracy
you\'ve just calculated, which was roughly **96%**. This model will
serve as the **\"baseline\" model** that you\'re going to try to improve
in the upcoming exercises. Let\'s start first with **splitting the data
into a test and training set**, and **defining the Random Forest
model**. The data available are features `X` and labels `y`.

**Instructions**

-   Import the random forest classifier from `sklearn`.
-   Split your features `X` and labels `y` into a training and test set.
    Set aside a test set of 30%.
-   Assign the random forest classifier to `model` and keep
    `random_state` at 5. We need to set a random state here in order to
    be able to compare results across different models.
:::

::: {.cell .markdown}
#### X_train, X_test, y_train, y_test
:::

::: {.cell .code execution_count="49"}
``` python
# Split your data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
```
:::

::: {.cell .code execution_count="50"}
``` python
# Define the model as the random forest
model = RandomForestClassifier(random_state=5, n_estimators=20)
```
:::

::: {.cell .markdown}
### Random Forest Classifier - part 2

Let\'s see how our Random Forest model performs **without doing anything
special to it**. The `model` from the previous exercise is available,
and you\'ve already split your data in
`X_train, y_train, X_test, y_test`.

**Instructions 1/3**

-   Fit the earlier defined `model` to our training data and obtain
    predictions by getting the model predictions on `X_test`.
:::

::: {.cell .code execution_count="51"}
``` python
# Fit the model to our training set
model.fit(X_train, y_train)
```

::: {.output .execute_result execution_count="51"}
```{=html}
<style>#sk-container-id-3 {color: black;background-color: white;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier(n_estimators=20, random_state=5)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" checked><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(n_estimators=20, random_state=5)</pre></div></div></div></div></div>
```
:::
:::

::: {.cell .code execution_count="52"}
``` python
# Obtain predictions from the test data 
predicted = model.predict(X_test)
```
:::

::: {.cell .markdown}
**Instructions 2/3**

-   Obtain and print the accuracy score by comparing the actual labels
    `y_test` with our predicted labels `predicted`.
:::

::: {.cell .code execution_count="53"}
``` python
print(f'Accuracy Score:\n{accuracy_score(y_test, predicted):0.3f}')
```

::: {.output .stream .stdout}
    Accuracy Score:
    0.991
:::
:::

::: {.cell .markdown}
**Instructions 3/3**

What is a benefit of using Random Forests versus Decision Trees?

**Possible Answers**

-   ~~Random Forests always have a higher accuracy than Decision
    Trees.~~
-   **Random Forests do not tend to overfit, whereas Decision Trees
    do.**
-   ~~Random Forests are computationally more efficient than Decision
    Trees.~~
-   ~~You can obtain \"feature importance\" from Random Forest, which
    makes it more transparent.~~

**Random Forest prevents overfitting most of the time, by creating
random subsets of the features and building smaller trees using these
subsets. Afterwards, it combines the subtrees of subsamples of features,
so it does not tend to overfit to your entire feature set the way
\"deep\" Decisions Trees do.**
:::

::: {.cell .markdown}
## Perfomance evaluation

-   Performance metrics for fraud detection models
-   There are other performace metrics that are more informative and
    reliable than accuracy
:::

::: {.cell .markdown}
#### Accuracy

![accuracy](vertopal_5c32587947084112be3b965be5a32bb5/991244d9ada62e3a39eec517e935324ca7e924f5.JPG)

-   Accuracy isn\'t a reliable performance metric when working with
    highly imbalanced data (such as fraud detection)
-   By doing nothing, aka predicting everything is the majority class
    (right image), a higher accuracy is obtained than by trying to build
    a predictive model (left image)
:::

::: {.cell .markdown}
#### Confusion Matrix

![advanced confusion
matrix](vertopal_5c32587947084112be3b965be5a32bb5/4f46225ef031fd2585ad07cd547a7205a0074090.JPG)

-   False Positives (FP) / False Negatives (FN) \* Cases of fraud not
    caught by the model \* Cases of \'false alarm\' \* a credit card
    company might want to catch as much fraud as possible and reduce
    false negatives, as fraudulent transactions can be incredibly costly
    \* a false alarm just means a transaction is blocked \* an insurance
    company can\'t handle many false alarms, as it means getting a team
    of investigators involved for each positive prediction

-   True Positives / True Negatives are the cases predicted correctly
    (e.g. fraud / non-fraud)
:::

::: {.cell .markdown}
#### Precision Recall

-   Credit card company wants to optimize for recall
-   Insurance company wants to optimize for precision
-   Precision:
    -   $$Precision=\frac{\#\space True\space Positives}{\#\space True\space Positives+\#\space False\space Positives}$$
    -   Fraction of actual fraud cases out of all predicted fraud cases
        -   true positives relative to the sum of true positives and
            false positives
-   Recall:
    -   $$Recall=\frac{\#\space True\space Positives}{\#\space True\space Positives+\#\space False\space Negatives}$$
    -   Fraction of predicted fraud cases out of all actual fraud cases
        -   true positives relative to the sum of true positives and
            false negative
-   Precision and recall are typically inversely related
    -   As precision increases, recall falls and vice-versa
    -   ![precision recall inverse
        relation](vertopal_5c32587947084112be3b965be5a32bb5/93a29d04ea2890e8a41f16e37d136514297e0390.JPG)
:::

::: {.cell .markdown}
#### F-Score

-   Weighs both precision and recall into on measure

\\begin{align} F-measure =
\\frac{2\\times{Precision}\\times{Recall}}{Precision\\times{Recall}} \\
\\ = \\frac{2\\times{TP}}{2\\times{TP}+FP+FN} \\end{align}

-   is a performance metric that takes into account a balance between
    Precision and Recall
:::

::: {.cell .markdown}
#### Obtaining performance metrics from sklean

``` python
# import the methods
from sklearn.metrics import precision_recall_curve, average_precision_score

# Calculate average precision and the PR curve
average_precision = average_precision_score(y_test, predicted)

# Obtain precision and recall
precision, recall = precision_recall_curve(y_test, predicted)
```
:::

::: {.cell .markdown}
#### Receiver Operating Characteristic (ROC) curve to compare algorithms

-   Created by plotting the true positive rate against the false
    positive rate at various threshold settings
-   ![roc
    curve](vertopal_5c32587947084112be3b965be5a32bb5/15dc954ab1ab4b7b699482d3b5beb761c0719c39.JPG)
-   Useful for comparing performance of different algorithms

``` python
# Obtain model probabilities
probs = model.predict_proba(X_test)

# Print ROC_AUC score using probabilities
print(metrics.roc_auc_score(y_test, probs[:, 1]))
```
:::

::: {.cell .markdown}
#### Confusion matrix and classification report

``` python
from sklearn.metrics import classification_report, confusion_matrix

# Obtain predictions
predicted = model.predict(X_test)

# Print classification report using predictions
print(classification_report(y_test, predicted))

# Print confusion matrix using predictions
print(confusion_matrix(y_test, predicted))
```
:::

::: {.cell .markdown}
### Performance metrics for the RF model

In the previous exercises you obtained an accuracy score for your random
forest model. This time, we know **accuracy can be misleading** in the
case of fraud detection. With highly imbalanced fraud data, the AUROC
curve is a more reliable performance metric, used to compare different
classifiers. Moreover, the *classification report* tells you about the
precision and recall of your model, whilst the *confusion matrix*
actually shows how many fraud cases you can predict correctly. So let\'s
get these performance metrics.

You\'ll continue working on the same random forest model from the
previous exercise. Your model, defined as
`model = RandomForestClassifier(random_state=5)` has been fitted to your
training data already, and `X_train, y_train, X_test, y_test` are
available.

**Instructions**

-   Import the classification report, confusion matrix and ROC score
    from `sklearn.metrics`.
-   Get the binary predictions from your trained random forest `model`.
-   Get the predicted probabilities by running the `predict_proba()`
    function.
-   Obtain classification report and confusion matrix by comparing
    `y_test` with `predicted`.
:::

::: {.cell .code execution_count="54"}
``` python
# Obtain the predictions from our random forest model 
predicted = model.predict(X_test)
```
:::

::: {.cell .code execution_count="55"}
``` python
# Predict probabilities
probs = model.predict_proba(X_test)
```
:::

::: {.cell .code execution_count="56"}
``` python
probs
```

::: {.output .execute_result execution_count="56"}
    array([[1., 0.],
           [1., 0.],
           [1., 0.],
           ...,
           [1., 0.],
           [1., 0.],
           [1., 0.]])
:::
:::

::: {.cell .code execution_count="57"}
``` python
# Print the ROC curve, classification report and confusion matrix
print('ROC Score:')
print(roc_auc_score(y_test, probs[:,1]))
print('\nClassification Report:')
print(classification_report(y_test, predicted))
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, predicted))
```

::: {.output .stream .stdout}
    ROC Score:
    0.9419896444670147

    Classification Report:
                  precision    recall  f1-score   support

               0       0.99      1.00      1.00      2099
               1       0.97      0.80      0.88        91

        accuracy                           0.99      2190
       macro avg       0.98      0.90      0.94      2190
    weighted avg       0.99      0.99      0.99      2190


    Confusion Matrix:
    [[2097    2]
     [  18   73]]
:::
:::

::: {.cell .markdown}
**You have now obtained more meaningful performance metrics that tell us
how well the model performs, given the highly imbalanced data that
you\'re working with. The model predicts 76 cases of fraud, out of which
73 are actual fraud. You have only 3 false positives. This is really
good, and as a result you have a very high precision score. You do
however, miss 18 cases of actual fraud. Recall is therefore not as good
as precision.**
:::

::: {.cell .markdown}
### Plotting the Precision vs. Recall Curve {#plotting-the-precision-vs-recall-curve}

You can also plot a **Precision-Recall curve**, to investigate the
trade-off between the two in your model. In this curve **Precision and
Recall are inversely related**; as Precision increases, Recall falls and
vice-versa. A balance between these two needs to be achieved in your
model, otherwise you might end up with many false positives, or not
enough actual fraud cases caught. To achieve this and to compare
performance, the precision-recall curves come in handy.

Your Random Forest Classifier is available as `model`, and the
predictions as `predicted`. You can simply obtain the average precision
score and the PR curve from the sklearn package. The function
`plot_pr_curve()` plots the results for you. Let\'s give it a try.

**Instructions 1/3**

-   Calculate the average precision by running the function on the
    actual labels `y_test` and your predicted labels `predicted`.
:::

::: {.cell .code execution_count="58"}
``` python
# Calculate average precision and the PR curve
average_precision = average_precision_score(y_test, predicted)
average_precision
```

::: {.output .execute_result execution_count="58"}
    0.7890250388880526
:::
:::

::: {.cell .markdown}
**Instructions 2/3**

-   Run the `precision_recall_curve()` function on the same arguments
    `y_test` and `predicted` and plot the curve (this last thing has
    been done for you).
:::

::: {.cell .code execution_count="59"}
``` python
# Obtain precision and recall 
precision, recall, _ = precision_recall_curve(y_test, predicted)
print(f'Precision: {precision}\nRecall: {recall}')
```

::: {.output .stream .stdout}
    Precision: [0.04155251 0.97333333 1.        ]
    Recall: [1.        0.8021978 0.       ]
:::
:::

::: {.cell .markdown}
#### def plot_pr_curve
:::

::: {.cell .code execution_count="60"}
``` python
def plot_pr_curve(recall, precision, average_precision):
    """
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    """
    from inspect import signature
    plt.figure()
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})

    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title(f'2-class Precision-Recall curve: AP={average_precision:0.2f}')
    return plt.show()
```
:::

::: {.cell .code execution_count="61"}
``` python
# Plot the recall precision tradeoff
plot_pr_curve(recall, precision, average_precision)
```

::: {.output .display_data}
![](vertopal_5c32587947084112be3b965be5a32bb5/f2f410df9fd58e5a760abf6160a51f7e2d4aa19b.png)
:::
:::

::: {.cell .markdown}
**Instructions 3/3**

What\'s the benefit of the performance metric ROC curve (AUROC) versus
Precision and Recall?

**Possible Answers**

-   **The AUROC answers the question: \"How well can this classifier be
    expected to perform in general, at a variety of different baseline
    probabilities?\" but precision and recall don\'t.**
-   ~~The AUROC answers the question: \"How meaningful is a positive
    result from my classifier given the baseline probabilities of my
    problem?\" but precision and recall don\'t.~~
-   ~~Precision and Recall are not informative when the data is
    imbalanced.~~
-   ~~The AUROC curve allows you to visualize classifier performance and
    with Precision and Recall you cannot.~~

**The ROC curve plots the true positives vs. false positives , for a
classifier, as its discrimination threshold is varied. Since, a random
method describes a horizontal curve through the unit interval, it has an
AUC of 0.5. Minimally, classifiers should perform better than this, and
the extent to which they score higher than one another (meaning the area
under the ROC curve is larger), they have better expected performance.**
:::

::: {.cell .markdown}
## Adjusting the algorithm weights

-   Adjust model parameter to optimize for fraud detection.
-   When training a model, try different options and settings to get the
    best recall-precision trade-off
-   sklearn has two simple options to tweak the model for heavily
    imbalanced data
    -   `class_weight`:
        -   `balanced` mode:
            `model = RandomForestClassifier(class_weight='balanced')`
            -   uses the values of y to automatically adjust weights
                inversely proportional to class frequencies in the the
                input data
            -   this option is available for other classifiers
                -   `model = LogisticRegression(class_weight='balanced')`
                -   `model = SVC(kernel='linear', class_weight='balanced', probability=True)`
        -   `balanced_subsample` mode:
            `model = RandomForestClassifier(class_weight='balanced_subsample')`
            -   is the same as the `balanced` option, except weights are
                calculated again at each iteration of growing a tree in
                a the random forest
            -   this option is only applicable for the Random Forest
                model
        -   manual input
            -   adjust weights to any ratio, not just value counts
                relative to sample
            -   `class_weight={0:1,1:4}`
            -   this is a good option to slightly upsample the minority
                class
:::

::: {.cell .markdown}
#### Hyperparameter tuning

-   Random Forest takes many other options to optimize the model

``` python
model = RandomForestClassifier(n_estimators=10, 
                               criterion=’gini’, 
                               max_depth=None, 
                               min_samples_split=2, 
                               min_samples_leaf=1, 
                               max_features=’auto’, 
                               n_jobs=-1, class_weight=None)
```

-   the shape and size of the trees in a random forest are adjusted with
    **leaf size** and **tree depth**
-   `n_estimators`: one of the most important setting is the number of
    trees in the forest
-   `max_features`: the number of features considered for splitting at
    each leaf node
-   `criterion`: change the way the data is split at each node (default
    is `gini` coefficient)
:::

::: {.cell .markdown}
#### GridSearchCV for hyperparameter tuning

-   [sklearn.model_selection.GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
-   `from sklearn.model_selection import GridSearchCV`
-   \`GridSearchCV evaluates all combinations of parameters defined in
    the parameter grid
-   Random Forest Parameter Grid:

``` python
# Create the parameter grid 
param_grid = {'max_depth': [80, 90, 100, 110],
              'max_features': [2, 3],
              'min_samples_leaf': [3, 4, 5],
              'min_samples_split': [8, 10, 12],
              'n_estimators': [100, 200, 300, 1000]}

# Define which model to use
model = RandomForestRegressor()

# Instantiate the grid search model
grid_search_model = GridSearchCV(estimator = model, 
                                 param_grid = param_grid, 
                                 cv = 5,
                                 n_jobs = -1, 
                                 scoring='f1')
```

-   define the ML model to be used
-   put the model into `GridSearchCV`
-   pass in `param_grid`
-   frequency of cross-validation
-   define a scoring metric to evaluate the models
    -   the default option is accuracy which isn\'t optimal for fraud
        detection
    -   use `precision`, `recall` or `f1`

``` python
# Fit the grid search to the data
grid_search_model.fit(X_train, y_train)

# Get the optimal parameters 
grid_search_model.best_params_

{'bootstrap': True,
 'max_depth': 80,
 'max_features': 3,
 'min_samples_leaf': 5,
 'min_samples_split': 12,
 'n_estimators': 100}
```

-   once `GridSearchCV` and `model` are fit to the data, obtain the
    parameters belonging to the optimal model by using the
    `best_params_` attribute
-   `GridSearchCV` is computationally heavy
    -   Can require many hours, depending on the amount of data and
        number of parameters in the grid
    -   ****Save the Results****

``` python
# Get the best_estimator results
grid_search.best_estimator_
grid_search.best_score_
```

-   `best_score_`: mean cross-validated score of the `best_estimator_`,
    which depends on the `scoring` option
:::

::: {.cell .markdown}
### Model adjustments

A simple way to adjust the random forest model to deal with highly
imbalanced fraud data, is to use the **`class_weights` option** when
defining the `sklearn` model. However, as you will see, it is a bit of a
blunt force mechanism and might not work for your very special case.

In this exercise you\'ll explore the `weight = "balanced_subsample"`
mode the Random Forest model from the earlier exercise. You already have
split your data in a training and test set, i.e `X_train`, `X_test`,
`y_train`, `y_test` are available. The metrics function have already
been imported.

**Instructions**

-   Set the `class_weight` argument of your classifier to
    `balanced_subsample`.
-   Fit your model to your training set.
-   Obtain predictions and probabilities from X_test.
-   Obtain the `roc_auc_score`, the classification report and confusion
    matrix.
:::

::: {.cell .code execution_count="62"}
``` python
# Define the model with balanced subsample
model = RandomForestClassifier(class_weight='balanced_subsample', random_state=5, n_estimators=100)

# Fit your training model to your training set
model.fit(X_train, y_train)

# Obtain the predicted values and probabilities from the model 
predicted = model.predict(X_test)
probs = model.predict_proba(X_test)

# Print the ROC curve, classification report and confusion matrix
print('ROC Score:')
print(roc_auc_score(y_test, probs[:,1]))
print('\nClassification Report:')
print(classification_report(y_test, predicted))
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, predicted))
```

::: {.output .stream .stdout}
    ROC Score:
    0.9750299724096771

    Classification Report:
                  precision    recall  f1-score   support

               0       0.99      1.00      1.00      2099
               1       0.99      0.80      0.88        91

        accuracy                           0.99      2190
       macro avg       0.99      0.90      0.94      2190
    weighted avg       0.99      0.99      0.99      2190


    Confusion Matrix:
    [[2098    1]
     [  18   73]]
:::
:::

::: {.cell .markdown}
**You can see that the model results don\'t improve drastically. We now
have 3 less false positives, but now 19 in stead of 18 false negatives,
i.e. cases of fraud we are not catching. If we mostly care about
catching fraud, and not so much about the false positives, this does
actually not improve our model at all, albeit a simple option to try. In
the next exercises you\'ll see how to more smartly tweak your model to
focus on reducing false negatives and catch more fraud.**
:::

::: {.cell .markdown}
### Adjusting RF for fraud detection

In this exercise you\'re going to dive into the options for the random
forest classifier, as we\'ll **assign weights** and **tweak the shape**
of the decision trees in the forest. You\'ll **define weights
manually**, to be able to off-set that imbalance slightly. In our case
we have 300 fraud to 7000 non-fraud cases, so by setting the weight
ratio to 1:12, we get to a 1/3 fraud to 2/3 non-fraud ratio, which is
good enough for training the model on.

The data in this exercise has already been split into training and test
set, so you just need to focus on defining your model. You can then use
the function `get_model_results()` as a short cut. This function fits
the model to your training data, predicts and obtains performance
metrics similar to the steps you did in the previous exercises.

**Instructions**

-   Change the `weight` option to set the ratio to 1 to 12 for the
    non-fraud and fraud cases, and set the split criterion to
    \'entropy\'.
-   Set the maximum depth to 10.
-   Set the minimal samples in leaf nodes to 10.
-   Set the number of trees to use in the model to 20.
:::

::: {.cell .markdown}
#### def get_model_results
:::

::: {.cell .code execution_count="63"}
``` python
def get_model_results(X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray, model):
    """
    model: sklearn model (e.g. RandomForestClassifier)
    """
    # Fit your training model to your training set
    model.fit(X_train, y_train)

    # Obtain the predicted values and probabilities from the model 
    predicted = model.predict(X_test)
    
    try:
        probs = model.predict_proba(X_test)
        print('ROC Score:')
        print(roc_auc_score(y_test, probs[:,1]))
    except AttributeError:
        pass

    # Print the ROC curve, classification report and confusion matrix
    print('\nClassification Report:')
    print(classification_report(y_test, predicted))
    print('\nConfusion Matrix:')
    print(confusion_matrix(y_test, predicted))
```
:::

::: {.cell .code execution_count="64"}
``` python
# Change the model options
model = RandomForestClassifier(bootstrap=True,
                               class_weight={0:1, 1:12},
                               criterion='entropy',
                               # Change depth of model
                               max_depth=10,
                               # Change the number of samples in leaf nodes
                               min_samples_leaf=10, 
                               # Change the number of trees to use
                               n_estimators=20,
                               n_jobs=-1,
                               random_state=5)

# Run the function get_model_results
get_model_results(X_train, y_train, X_test, y_test, model)
```

::: {.output .stream .stdout}
    ROC Score:
    0.9609651901219315

    Classification Report:
                  precision    recall  f1-score   support

               0       0.99      1.00      1.00      2099
               1       0.97      0.85      0.91        91

        accuracy                           0.99      2190
       macro avg       0.98      0.92      0.95      2190
    weighted avg       0.99      0.99      0.99      2190


    Confusion Matrix:
    [[2097    2]
     [  14   77]]
:::
:::

::: {.cell .markdown}
**By smartly defining more options in the model, you can obtain better
predictions. You have effectively reduced the number of false negatives,
i.e. you are catching more cases of fraud, whilst keeping the number of
false positives low. In this exercise you\'ve manually changed the
options of the model. There is a smarter way of doing it, by using
`GridSearchCV`, which you\'ll see in the next exercise!**
:::

::: {.cell .markdown}
### Parameter optimization with GridSearchCV

In this exercise you\'re going to **tweak our model in a less \"random\"
way**, but use `GridSearchCV` to do the work for you.

With `GridSearchCV` you can define **which performance metric to score**
the options on. Since for fraud detection we are mostly interested in
catching as many fraud cases as possible, you can optimize your model
settings to get the best possible Recall score. If you also cared about
reducing the number of false positives, you could optimize on F1-score,
this gives you that nice Precision-Recall trade-off.

`GridSearchCV` has already been imported from `sklearn.model_selection`,
so let\'s give it a try!

**Instructions**

-   Define in the parameter grid that you want to try 1 and 30 trees,
    and that you want to try the `gini` and `entropy` split criterion.
-   Define the model to be simple `RandomForestClassifier`, you want to
    keep the random_state at 5 to be able to compare models.
-   Set the `scoring` option such that it optimizes for recall.
-   Fit the model to the training data `X_train` and `y_train` and
    obtain the best parameters for the model.
:::

::: {.cell .code execution_count="65"}
``` python
# Define the parameter sets to test
param_grid = {'n_estimators': [1, 30],
              'max_features': ['auto', 'log2'], 
              'max_depth': [4, 8, 10, 12],
              'criterion': ['gini', 'entropy']}

# Define the model to use
model = RandomForestClassifier(random_state=5)

# Combine the parameter sets with the defined model
CV_model = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='recall', n_jobs=-1)

# Fit the model to our training data and obtain best parameters
CV_model.fit(X_train,y_train)
CV_model.best_params_
```

::: {.output .execute_result execution_count="65"}
    {'criterion': 'gini',
     'max_depth': 8,
     'max_features': 'log2',
     'n_estimators': 30}
:::
:::

::: {.cell .markdown}
### Model results with GridSearchCV

You discovered that the **best parameters for your model** are that the
split criterion should be set to `'gini'`, the number of estimators
(trees) should be 30, the maximum depth of the model should be 8 and the
maximum features should be set to `"log2"`.

Let\'s give this a try and see how well our model performs. You can use
the `get_model_results()` function again to save time.

**Instructions**

-   Input the optimal settings into the model definition.
-   Fit the model, obtain predictions and get the performance parameters
    with `get_model_results()`.
:::

::: {.cell .code execution_count="66"}
``` python
# Input the optimal parameters in the model
model = RandomForestClassifier(class_weight={0:1,1:12},
                               criterion='gini',
                               max_depth=8,
                               max_features='log2', 
                               min_samples_leaf=10,
                               n_estimators=30,
                               n_jobs=-1,
                               random_state=5)

# Get results from your model
get_model_results(X_train, y_train, X_test, y_test, model)
```

::: {.output .stream .stdout}
    ROC Score:
    0.9749697658225529

    Classification Report:
                  precision    recall  f1-score   support

               0       0.99      1.00      1.00      2099
               1       0.95      0.84      0.89        91

        accuracy                           0.99      2190
       macro avg       0.97      0.92      0.94      2190
    weighted avg       0.99      0.99      0.99      2190


    Confusion Matrix:
    [[2095    4]
     [  15   76]]
:::
:::

::: {.cell .markdown}
**The model has been improved even further. The number of false
positives has now been slightly reduced even further, which means we are
catching more cases of fraud. However, you see that the number of false
positives actually went up. That is that Precision-Recall trade-off in
action. To decide which final model is best, you need to take into
account how bad it is not to catch fraudsters, versus how many false
positives the fraud analytics team can deal with. Ultimately, this final
decision should be made by you and the fraud team together.**
:::

::: {.cell .markdown}
## Ensemble methods

![ensemble](vertopal_5c32587947084112be3b965be5a32bb5/0c6b2afda7e93205608e961c2e6809f726cd68b8.JPG)

-   Ensemble methods are techniques that create multiple machine
    learning models and then combine them to produce a final result
-   Usually produce more accurate predictions than a single model
-   The goal of an ML problem is to find a single model that will best
    predict our wanted outcome
    -   Use ensemble methods rather than making one model and hoping
        it\'s best, most accurate predictor
-   Ensemble methods take a myriad of models into account and average
    them to produce one final model
    -   Ensures the predictions are robust
    -   Less likely to be the result of overfitting
    -   Can improve prediction performance
        -   Especially by combining models with different recall and
            precision scores
    -   Are a winning formula at Kaggle competitions
-   The Random Forest classifier is an ensemble of Decision Trees
    -   **Bootstrap Aggregation** or **Bagging Ensemble** method
    -   In a Random Forest, models are trained on random subsamples of
        data and the results are aggregated by taking the average
        prediction of all the trees
:::

::: {.cell .markdown}
#### Stacking Ensemble Methods

![stacking
ensemble](vertopal_5c32587947084112be3b965be5a32bb5/c06117521782fcba31a724a0b773fb48f21261a6.JPG)

-   Multiple models are combined via a \"voting\" rule on the model
    outcome
-   The base level models are each trained based on the complete
    training set
    -   Unlike the Bagging method, models are not trained on a subsample
        of the data
-   Algorithms of different types can be combined
:::

::: {.cell .markdown}
#### Voting Classifier

-   available in sklearn
    -   easy way of implementing an ensemble model

``` python
from sklearn.ensemble import VotingClassifier

# Define Models
clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()

# Combine models into ensemble
ensemble_model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

# Fit and predict as with other models
ensemble_model.fit(X_train, y_train)
ensemble_model.predict(X_test)
```

-   the `voting='hard'` option uses the predicted class labels and takes
    the majority vote
-   the `voting='soft'` option takes the average probability by
    combining the predicted probabilities of the individual models
-   Weights can be assigned to the `VotingClassifer` with
    `weights=[2,1,1]`
    -   Useful when one model significantly outperforms the others
:::

::: {.cell .markdown}
#### Reliable Labels

-   In real life it\'s unlikely the data will have truly unbiased,
    reliable labels for the model
-   In credit card fraud you often will have reliable labels, in which
    case, use the methods learned so far
-   Most cases you\'ll need to rely on unsupervised learning techniques
    to detect fraud
:::

::: {.cell .markdown}
### Logistic Regression {#logistic-regression}

In this last lesson you\'ll **combine three algorithms** into one model
with the **VotingClassifier**. This allows us to benefit from the
different aspects from all models, and hopefully improve overall
performance and detect more fraud. The first model, the Logistic
Regression, has a slightly higher recall score than our optimal Random
Forest model, but gives a lot more false positives. You\'ll also add a
Decision Tree with balanced weights to it. The data is already split
into a training and test set, i.e. `X_train`, `y_train`, `X_test`,
`y_test` are available.

In order to understand how the Voting Classifier can potentially improve
your original model, you should check the standalone results of the
Logistic Regression model first.

**Instructions**

-   Define a LogisticRegression model with class weights that are 1:15
    for the fraud cases.
-   Fit the model to the training set, and obtain the model predictions.
-   Print the classification report and confusion matrix.
:::

::: {.cell .code execution_count="67"}
``` python
# Define the Logistic Regression model with weights
model = LogisticRegression(class_weight={0:1, 1:15}, random_state=5, solver='liblinear')

# Get the model results
get_model_results(X_train, y_train, X_test, y_test, model)
```

::: {.output .stream .stdout}
    ROC Score:
    0.9722054981702433

    Classification Report:
                  precision    recall  f1-score   support

               0       0.99      0.98      0.99      2099
               1       0.63      0.88      0.73        91

        accuracy                           0.97      2190
       macro avg       0.81      0.93      0.86      2190
    weighted avg       0.98      0.97      0.98      2190


    Confusion Matrix:
    [[2052   47]
     [  11   80]]
:::
:::

::: {.cell .markdown}
**As you can see the Logistic Regression has quite different performance
from the Random Forest. More false positives, but also a better Recall.
It will therefore will a useful addition to the Random Forest in an
ensemble model.**
:::

::: {.cell .markdown}
### Voting Classifier {#voting-classifier}

Let\'s now **combine three machine learning models into one**, to
improve our Random Forest fraud detection model from before. You\'ll
combine our usual Random Forest model, with the Logistic Regression from
the previous exercise, with a simple Decision Tree. You can use the
short cut `get_model_results()` to see the immediate result of the
ensemble model.

**Instructions**

-   Import the Voting Classifier package.
-   Define the three models; use the Logistic Regression from before,
    the Random Forest from previous exercises and a Decision tree with
    balanced class weights.
-   Define the ensemble model by inputting the three classifiers with
    their respective labels.
:::

::: {.cell .code execution_count="68"}
``` python
# Define the three classifiers to use in the ensemble
clf1 = LogisticRegression(class_weight={0:1, 1:15},
                          random_state=5,
                          solver='liblinear')

clf2 = RandomForestClassifier(class_weight={0:1, 1:12}, 
                              criterion='gini', 
                              max_depth=8, 
                              max_features='log2',
                              min_samples_leaf=10, 
                              n_estimators=30, 
                              n_jobs=-1,
                              random_state=5)

clf3 = DecisionTreeClassifier(random_state=5,
                              class_weight="balanced")

# Combine the classifiers in the ensemble model
ensemble_model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('dt', clf3)], voting='hard')

# Get the results 
get_model_results(X_train, y_train, X_test, y_test, ensemble_model)
```

::: {.output .stream .stdout}

    Classification Report:
                  precision    recall  f1-score   support

               0       0.99      1.00      0.99      2099
               1       0.90      0.86      0.88        91

        accuracy                           0.99      2190
       macro avg       0.95      0.93      0.94      2190
    weighted avg       0.99      0.99      0.99      2190


    Confusion Matrix:
    [[2090    9]
     [  13   78]]
:::
:::

::: {.cell .markdown}
**By combining the classifiers, you can take the best of multiple
models. You\'ve increased the cases of fraud you are catching from 76 to
78, and you only have 5 extra false positives in return. If you do care
about catching as many fraud cases as you can, whilst keeping the false
positives low, this is a pretty good trade-off. The Logistic Regression
as a standalone was quite bad in terms of false positives, and the
Random Forest was worse in terms of false negatives. By combining these
together you indeed managed to improve performance.**
:::

::: {.cell .markdown}
### Adjusting weights within the Voting Classifier

You\'ve just seen that the Voting Classifier allows you to improve your
fraud detection performance, by combining good aspects from multiple
models. Now let\'s try to **adjust the weights** we give to these
models. By increasing or decreasing weights you can play with **how much
emphasis you give to a particular model** relative to the rest. This
comes in handy when a certain model has overall better performance than
the rest, but you still want to combine aspects of the others to further
improve your results.

For this exercise the data is already split into a training and test
set, and `clf1`, `clf2` and `clf3` are available and defined as before,
i.e. they are the Logistic Regression, the Random Forest model and the
Decision Tree respectively.

**Instructions**

-   Define an ensemble method where you over weigh the second classifier
    (`clf2`) with 4 to 1 to the rest of the classifiers.
-   Fit the model to the training and test set, and obtain the
    predictions `predicted` from the ensemble model.
-   Print the performance metrics, this is ready for you to run.
:::

::: {.cell .code execution_count="70"}
``` python
# Define the ensemble model
ensemble_model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft', weights=[1, 4, 1], flatten_transform=True)

# Get results 
get_model_results(X_train, y_train, X_test, y_test, ensemble_model)
```

::: {.output .stream .stdout}
    ROC Score:
    0.9739226947421326

    Classification Report:
                  precision    recall  f1-score   support

               0       0.99      1.00      1.00      2099
               1       0.94      0.85      0.89        91

        accuracy                           0.99      2190
       macro avg       0.97      0.92      0.94      2190
    weighted avg       0.99      0.99      0.99      2190


    Confusion Matrix:
    [[2094    5]
     [  14   77]]
:::
:::

::: {.cell .markdown}
**The weight option allows you to play with the individual models to get
the best final mix for your fraud detection model. Now that you have
finalized fraud detection with supervised learning, let\'s have a look
at how fraud detetion can be done when you don\'t have any labels to
train on.**
:::

::: {.cell .markdown}
# Fraud detection using unlabeled data

Use unsupervised learning techniques to detect fraud. Segment customers,
use K-means clustering and other clustering algorithms to find
suspicious occurrences in your data.
:::

::: {.cell .markdown}
## Normal versus abnormal behavior

-   Explore fraud detection without reliable data labels
-   Unsupervised learning to detect suspicious behavior
-   Abnormal behavior isn\'t necessarily fraudulent
-   Challenging because it\'s difficult to validate
:::

::: {.cell .markdown}
#### What\'s normal behavior?

-   thoroughly describe the data:
    -   plot histograms
    -   check for outliers
    -   investigate correlations
-   Are there any known historic cases of fraud? What typifies those
    cases?
-   Investigate whether the data is homogeneous, or whether different
    types of clients display different behavior
-   Check patterns within subgroups of data: is your data homogeneous?
-   Verify data points are the same type:
    -   individuals
    -   groups
    -   companies
    -   governmental organizations
-   Do the data points differ on:
    -   spending patterns
    -   age
    -   location
    -   frequency
-   For credit card fraud, location can be an indication of fraud
-   This goes for e-commerce sites
    -   where\'s the IP address located and where is the product ordered
        to ship?
-   Create a separate model for each segment
-   How to aggregate the many model results back into one final list
:::

::: {.cell .markdown}
### Exploring the data

In the next exercises, you will be looking at bank **payment transaction
data**. The financial transactions are categorized by type of expense,
as well as the amount spent. Moreover, you have some client
characteristics available such as age group and gender. Some of the
transactions are labeled as fraud; you\'ll treat these labels as given
and will use those to validate the results.

When using unsupervised learning techniques for fraud detection, you
want to **distinguish normal from abnormal** (thus potentially
fraudulent) behavior. As a fraud analyst to understand what is
\"normal\", you need to have a good understanding of the data and its
characteristics. Let\'s explore the data in this first exercise.

**Instructions 1/3**

-   Obtain the shape of the dataframe `df` to inspect the size of our
    data and display the first rows to see which features are available.
:::

::: {.cell .code execution_count="74"}
``` python
banksim_df = pd.read_csv("chapitre_3/banksim.csv")
banksim_df.drop(['Unnamed: 0'], axis=1, inplace=True)
banksim_adj_df = pd.read_csv("chapitre_3/banksim_adj.csv")
banksim_adj_df.drop(['Unnamed: 0'], axis=1, inplace=True)
```
:::

::: {.cell .code execution_count="75"}
``` python
banksim_df.shape
```

::: {.output .execute_result execution_count="75"}
    (7200, 5)
:::
:::

::: {.cell .code execution_count="76"}
``` python
banksim_df.head()
```

::: {.output .execute_result execution_count="76"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>gender</th>
      <th>category</th>
      <th>amount</th>
      <th>fraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>F</td>
      <td>es_transportation</td>
      <td>49.71</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>F</td>
      <td>es_health</td>
      <td>39.29</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>F</td>
      <td>es_transportation</td>
      <td>18.76</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>M</td>
      <td>es_transportation</td>
      <td>13.95</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>M</td>
      <td>es_transportation</td>
      <td>49.87</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .code execution_count="77"}
``` python
banksim_adj_df.shape
```

::: {.output .execute_result execution_count="77"}
    (7189, 18)
:::
:::

::: {.cell .code execution_count="78"}
``` python
banksim_adj_df.head()
```

::: {.output .execute_result execution_count="78"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>amount</th>
      <th>fraud</th>
      <th>M</th>
      <th>es_barsandrestaurants</th>
      <th>es_contents</th>
      <th>es_fashion</th>
      <th>es_food</th>
      <th>es_health</th>
      <th>es_home</th>
      <th>es_hotelservices</th>
      <th>es_hyper</th>
      <th>es_leisure</th>
      <th>es_otherservices</th>
      <th>es_sportsandtoys</th>
      <th>es_tech</th>
      <th>es_transportation</th>
      <th>es_travel</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>49.71</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>39.29</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>18.76</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>13.95</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>49.87</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown}
**Instructions 2/3**

-   Group the data by transaction category and take the mean of the
    data.
:::

::: {.cell .code execution_count="79"}
``` python
banksim_df.groupby(['category']).mean()
```

::: {.output .execute_result execution_count="79"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>amount</th>
      <th>fraud</th>
    </tr>
    <tr>
      <th>category</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>es_barsandrestaurants</th>
      <td>43.841793</td>
      <td>0.022472</td>
    </tr>
    <tr>
      <th>es_contents</th>
      <td>55.170000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>es_fashion</th>
      <td>59.780769</td>
      <td>0.020619</td>
    </tr>
    <tr>
      <th>es_food</th>
      <td>35.216050</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>es_health</th>
      <td>126.604704</td>
      <td>0.242798</td>
    </tr>
    <tr>
      <th>es_home</th>
      <td>120.688317</td>
      <td>0.208333</td>
    </tr>
    <tr>
      <th>es_hotelservices</th>
      <td>172.756245</td>
      <td>0.548387</td>
    </tr>
    <tr>
      <th>es_hyper</th>
      <td>46.788180</td>
      <td>0.125000</td>
    </tr>
    <tr>
      <th>es_leisure</th>
      <td>229.757600</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>es_otherservices</th>
      <td>149.648960</td>
      <td>0.600000</td>
    </tr>
    <tr>
      <th>es_sportsandtoys</th>
      <td>157.251737</td>
      <td>0.657895</td>
    </tr>
    <tr>
      <th>es_tech</th>
      <td>132.852862</td>
      <td>0.179487</td>
    </tr>
    <tr>
      <th>es_transportation</th>
      <td>27.422014</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>es_travel</th>
      <td>231.818656</td>
      <td>0.944444</td>
    </tr>
    <tr>
      <th>es_wellnessandbeauty</th>
      <td>66.167078</td>
      <td>0.060606</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown}
**Instructions 3/3**

Based on these results, can you already say something about fraud in our
data?

**Possible Answers**

-   ~~No, I don\'t have enough information.~~
-   **Yes, the majority of fraud is observed in travel, leisure and
    sports related transactions.**
:::

::: {.cell .markdown}
### Customer segmentation

In this exercise you\'re going to check whether there are any **obvious
patterns** for the clients in this data, thus whether you need to
segment your data into groups, or whether the data is rather homogenous.

You unfortunately don\'t have a lot client information available; you
can\'t for example distinguish between the wealth levels of different
clients. However, there is data on **age ** available, so let\'s see
whether there is any significant difference between behavior of age
groups.

**Instructions 1/3**

-   Group the dataframe `df` by the category `age` and get the means for
    each age group.
:::

::: {.cell .code execution_count="80"}
``` python
banksim_df.groupby(['age']).mean()
```

::: {.output .execute_result execution_count="80"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>amount</th>
      <th>fraud</th>
    </tr>
    <tr>
      <th>age</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>49.468935</td>
      <td>0.050000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>35.622829</td>
      <td>0.026648</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37.228665</td>
      <td>0.028718</td>
    </tr>
    <tr>
      <th>3</th>
      <td>37.279338</td>
      <td>0.023283</td>
    </tr>
    <tr>
      <th>4</th>
      <td>36.197985</td>
      <td>0.035966</td>
    </tr>
    <tr>
      <th>5</th>
      <td>37.547521</td>
      <td>0.023990</td>
    </tr>
    <tr>
      <th>6</th>
      <td>36.700852</td>
      <td>0.022293</td>
    </tr>
    <tr>
      <th>U</th>
      <td>39.117000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown}
**Instructions 2/3**

-   Count the values of each age group.
:::

::: {.cell .code execution_count="81"}
``` python
banksim_df.age.value_counts()
```

::: {.output .execute_result execution_count="81"}
    2    2333
    3    1718
    4    1279
    5     792
    1     713
    6     314
    0      40
    U      11
    Name: age, dtype: int64
:::
:::

::: {.cell .markdown}
**Instructions 3/3**

Based on the results you see, does it make sense to divide your data
into age segments before running a fraud detection algorithm?

**Possible Answers**

-   **No, the age groups who are the largest are relatively similar.**
-   ~~Yes, the age group \"0\" is very different and I would split that
    one out.~~

**The average amount spent as well as fraud occurrence is rather similar
across groups. Age group \'0\' stands out but since there are only 40
cases, it does not make sense to split these out in a separate group and
run a separate model on them.**
:::

::: {.cell .markdown}
### Using statistics to define normal behavior

In the previous exercises we saw that fraud is **more prevalent in
certain transaction categories**, but that there is no obvious way to
segment our data into for example age groups. This time, let\'s
investigate the **average amounts spent** in normal transactions versus
fraud transactions. This gives you an idea of how fraudulent
transactions **differ structurally** from normal transactions.

**Instructions**

-   Create two new dataframes from fraud and non-fraud observations.
    Locate the data in `df` with `.loc` and assign the condition \"where
    fraud is 1\" and \"where fraud is 0\" for creation of the new
    dataframes.
-   Plot the `amount` column of the newly created dataframes in the
    histogram plot functions and assign the labels `fraud` and
    `nonfraud` respectively to the plots.
:::

::: {.cell .code execution_count="82"}
``` python
# Create two dataframes with fraud and non-fraud data 
df_fraud = banksim_df.loc[banksim_df["fraud"]==1]
df_non_fraud = banksim_df.loc[banksim_df["fraud"]==0]
```
:::

::: {.cell .code execution_count="88"}
``` python
# Plot histograms of the amounts in fraud and non-fraud data 
plt.hist(df_fraud.amount,bins=20, alpha=0.5, label='fraud')
plt.hist(df_non_fraud.amount,bins=20, alpha=0.5, label='nonfraud')
plt.xlabel('amount')
plt.legend()
plt.show()
```

::: {.output .display_data}
![](vertopal_5c32587947084112be3b965be5a32bb5/bd386384513b14aaf25ee09dab51507ab590f4b2.png)
:::
:::

::: {.cell .markdown}
**As the number fraud observations is much smaller, it is difficult to
see the full distribution. Nonetheless, you can see that the fraudulent
transactions tend to be on the larger side relative to normal
observations. This is good news, as it helps us later in detecting fraud
from non-fraud. In the next chapter you\'re going to implement a
clustering model to distinguish between normal and abnormal
transactions, when the fraud labels are no longer available.**
:::

::: {.cell .markdown}
## Clustering methods to detect fraud
:::

::: {.cell .markdown}
#### K-means clustering

![k-means](vertopal_5c32587947084112be3b965be5a32bb5/19411efbf98061bcba85b24c1535b1adf4e98d58.JPG)

-   The objective of any clustering model is to detect patterns in the
    data
-   More specifically, to group the data into distinct clusters made of
    data points that are very similar to each other, but distinct from
    the points in the other clusters.
-   **The objective of k-means is to minimize the sum of all distances
    between the data samples and their associated cluster centroids**
    -   The score is the inverse of that minimization, so the score
        should be close to 0.
-   **Using the distance to cluster centroids**
    -   Training samples are shown as dots and cluster centroids are
        shown as crosses
    -   Attempt to cluster the data in image A
        -   Start by putting in an initial guess for two cluster
            centroids, as in B
        -   Predefine the number of clusters at the start
        -   Then calculate the distances of each sample in the data to
            the closest centroid
        -   Figure C shows the data split into the two clusters
        -   Based on the initial clusters, the location of the centroids
            can be redefined (fig D) to minimize the sum of all
            distances in the two clusters.
        -   Repeat the step of reassigning points that are nearest to
            the centroid (fig E) until it converges to the point where
            no sample gets reassigned to another cluster (fig F)
        -   ![clustering](vertopal_5c32587947084112be3b965be5a32bb5/5e1770b47b794c32d89bc086efdd8af3aec87cac.JPG)
:::

::: {.cell .markdown}
#### K-means clustering in Python

-   It\'s of utmost importance to scale the data before doing K-means
    clustering, or any algorithm that uses distances
-   Without scaling, features on a larger scale will weight more heavily
    in the algorithm. All features should weigh equally at the initial
    stage
-   fix `random_state` so models can be compared

``` python
# Import the packages
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Transform and scale your data
X = np.array(df).astype(np.float)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Define the k-means model and fit to the data
kmeans = KMeans(n_clusters=6, random_state=42).fit(X_scaled)
```
:::

::: {.cell .markdown}
#### The right amount of clusters

-   The drawback of K-means clustering is the need to assign the number
    of clusters beforehand
-   There are multiple ways to check what the right number of clusters
    should be
    -   Silhouette method
    -   Elbow curve
-   By running a k-means model on clusters varying from 1 to 10 and
    generate an **elbow curve** by saving the scores for each model
    under \"score\".
-   Plot the scores against the number of clusters

``` python
clust = range(1, 10) 
kmeans = [KMeans(n_clusters=i) for i in clust]

score = [kmeans[i].fit(X_scaled).score(X_scaled) for i in range(len(kmeans))]

plt.plot(clust,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()
```

![elbow
curve](vertopal_5c32587947084112be3b965be5a32bb5/cba386f40e87e83d3d7de5ca963febf9f1e039ee.JPG)

-   The slight elbow at 3 means that 3 clusters could be optimal, but
    it\'s not very pronounced
:::

::: {.cell .markdown}
### Scaling the data

For ML algorithms using distance based metrics, it is **crucial to
always scale your data**, as features using different scales will
distort your results. K-means uses the Euclidean distance to assess
distance to cluster centroids, therefore you first need to scale your
data before continuing to implement the algorithm. Let\'s do that first.

Available is the dataframe `df` from the previous exercise, with some
minor data preparation done so it is ready for you to use with
`sklearn`. The fraud labels are separately stored under labels, you can
use those to check the results later.

**Instructions**

-   Import the `MinMaxScaler`.
-   Transform your dataframe `df` into a numpy array `X` by taking only
    the values of `df` and make sure you have all `float` values.
-   Apply the defined scaler onto `X` to obtain scaled values of
    `X_scaled` to force all your features to a 0-1 scale.
:::

::: {.cell .code execution_count="89"}
``` python
labels = banksim_adj_df.fraud
```
:::

::: {.cell .code execution_count="90"}
``` python
cols = ['age', 'amount', 'M', 'es_barsandrestaurants', 'es_contents',
        'es_fashion', 'es_food', 'es_health', 'es_home', 'es_hotelservices',
        'es_hyper', 'es_leisure', 'es_otherservices', 'es_sportsandtoys',
        'es_tech', 'es_transportation', 'es_travel']
```
:::

::: {.cell .code execution_count="91"}
``` python
# Take the float values of df for X
X = banksim_adj_df[cols].values.astype(np.float)
```
:::

::: {.cell .code execution_count="92"}
``` python
X.shape
```

::: {.output .execute_result execution_count="92"}
    (7189, 17)
:::
:::

::: {.cell .code execution_count="93"}
``` python
# Define the scaler and apply to the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```
:::

::: {.cell .markdown}
### K-mean clustering

A very commonly used clustering algorithm is **K-means clustering**. For
fraud detection, K-means clustering is straightforward to implement and
relatively powerful in predicting suspicious cases. It is a good
algorithm to start with when working on fraud detection problems.
However, fraud data is oftentimes very large, especially when you are
working with transaction data. **MiniBatch K-means** is an **efficient
way** to implement K-means on a large dataset, which you will use in
this exercise.

The scaled data from the previous exercise, `X_scaled` is available.
Let\'s give it a try.

**Instructions**

-   Import `MiniBatchKMeans` from `sklearn`.
-   Initialize the minibatch kmeans model with 8 clusters.
-   Fit the model to your scaled data.
:::

::: {.cell .code execution_count="94"}
``` python
# Define the model 
kmeans = MiniBatchKMeans(n_clusters=8)

# Fit the model to the scaled data
kmeans.fit(X_scaled)
```

::: {.output .execute_result execution_count="94"}
```{=html}
<style>#sk-container-id-4 {color: black;background-color: white;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>MiniBatchKMeans()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" checked><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">MiniBatchKMeans</label><div class="sk-toggleable__content"><pre>MiniBatchKMeans()</pre></div></div></div></div></div>
```
:::
:::

::: {.cell .markdown}
**You have now fitted your MiniBatch K-means model to the data. In the
upcoming exercises you\'re going to explore whether this model is any
good at flagging fraud. But before doing that, you still need to figure
our what the right number of clusters to use is. Let\'s do that in the
next exercise.**
:::

::: {.cell .markdown}
### Elbow method

In the previous exercise you\'ve implemented MiniBatch K-means with 8
clusters, without actually checking what the right amount of clusters
should be. For our first fraud detection approach, it is important to
**get the number of clusters right**, especially when you want to use
the outliers of those clusters as fraud predictions. To decide which
amount of clusters you\'re going to use, let\'s apply the **Elbow
method** and see what the optimal number of clusters should be based on
this method.

`X_scaled` is again available for you to use and `MiniBatchKMeans` has
been imported from `sklearn`.

**Instructions**

-   Define the range to be between 1 and 10 clusters.
-   Run MiniBatch K-means on all the clusters in the range using list
    comprehension.
-   Fit each model on the scaled data and obtain the scores from the
    scaled data.
-   Plot the cluster numbers and their respective scores.
:::

::: {.cell .code execution_count="95"}
``` python
# Define the range of clusters to try
clustno = range(1, 10)

# Run MiniBatch Kmeans over the number of clusters
kmeans = [MiniBatchKMeans(n_clusters=i) for i in clustno]

# Obtain the score for each model
score = [kmeans[i].fit(X_scaled).score(X_scaled) for i in range(len(kmeans))]
```
:::

::: {.cell .code execution_count="96"}
``` python
# Plot the models and their respective score 
plt.plot(clustno, score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()
```

::: {.output .display_data}
![](vertopal_5c32587947084112be3b965be5a32bb5/4e18e97183954f55871a23a57b588f66fdf98d34.png)
:::
:::

::: {.cell .markdown}
**Now you can see that the optimal number of clusters should probably be
at around 3 clusters, as that is where the elbow is in the curve. We\'ll
use this in the next exercise as our baseline model, and see how well
this does in detecting fraud**
:::

::: {.cell .markdown}
## Assigning fraud vs. non-fraud {#assigning-fraud-vs-non-fraud}

-   ![clusters](vertopal_5c32587947084112be3b965be5a32bb5/41f6c6f4cc31abb6b784577065017cc2e321ee54.JPG)
-   Take the outliers of each cluster, and flag those as fraud.
-   ![clusters](vertopal_5c32587947084112be3b965be5a32bb5/f2af5d6614628ca2fa166318d0b392f8b6c5be6f.JPG)

1.  Collect and store the cluster centroids in memory
    -   Starting point to decide what\'s normal and not
2.  Calculate the distance of each point in the dataset, to their own
    cluster centroid

-   ![clusters](vertopal_5c32587947084112be3b965be5a32bb5/891c59209d6b3b25c5dc3c153a6f79f6106e8040.JPG)
    -   Euclidean distance is depicted by the circles in this case
    -   Define a cut-off point for the distances to define what\'s an
        outlier
        -   Done based on the distributions of the distances collected
        -   i.e. everything with a distance larger than the top 95th
            percentile, should be considered an outlier
        -   the tail of the distribution of distances
        -   anything outside the yellow circles is an outlier
        -   ![clusters](vertopal_5c32587947084112be3b965be5a32bb5/d098bb27054f7ed9da17e41dda49b5fc219c036f.JPG)
        -   these are definitely outliers and can be described as
            abnormal or suspicious
            -   doesn\'t necessarily mean they are fraudulent
:::

::: {.cell .markdown}
#### Flagging Fraud Based on Distance to Centroid

``` python
# Run the kmeans model on scaled data
kmeans = KMeans(n_clusters=6, random_state=42,n_jobs=-1).fit(X_scaled)

# Get the cluster number for each datapoint
X_clusters = kmeans.predict(X_scaled)

# Save the cluster centroids
X_clusters_centers = kmeans.cluster_centers_

# Calculate the distance to the cluster centroid for each point
dist = [np.linalg.norm(x-y) for x,y in zip(X_scaled, X_clusters_centers[X_clusters])]

# Create predictions based on distance
km_y_pred = np.array(dist)
km_y_pred[dist>=np.percentile(dist, 93)] = 1
km_y_pred[dist<np.percentile(dist, 93)] = 0
```

-   `np.linalg.norm`: returns the vector norm, the vector of distance
    for each datapoint to their assigned cluster
-   use the percentiles of the distances to determine which samples are
    outliers
:::

::: {.cell .markdown}
:::

::: {.cell .markdown}
#### Validating the Model Results

-   without fraud labels, the usual performance metrics can\'t be run
    -   check with the fraud analyst
    -   investigate and describe cases that are flagged in more detail
        -   is it fraudulent or just a rare case of legit data
        -   avoid rare, legit cases by deleting certain features or
            removing the cases from the data
    -   if there are past cases of fraud, see if the model can predict
        them using historic data
:::

::: {.cell .markdown}
### Detecting outliers

In the next exercises, you\'re going to use the K-means algorithm to
predict fraud, and compare those predictions to the actual labels that
are saved, to sense check our results.

The fraudulent transactions are typically flagged as the observations
that are furthest aways from the cluster centroid. You\'ll learn how to
do this and how to determine the cut-off in this exercise. In the next
one, you\'ll check the results.

Available are the scaled observations X_scaled, as well as the labels
stored under the variable y.

**Instructions**

-   Split the scaled data and labels y into a train and test set.
-   Define the MiniBatch K-means model with 3 clusters, and fit to the
    training data.
-   Get the cluster predictions from your test data and obtain the
    cluster centroids.
-   Define the boundary between fraud and non fraud to be at 95% of
    distance distribution and higher.
:::

::: {.cell .code execution_count="98"}
``` python
# Split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, labels, test_size=0.3, random_state=0)

# Define K-means model 
kmeans = MiniBatchKMeans(n_clusters=3, random_state=42).fit(X_train)
# Obtain predictions and calculate distance from cluster centroid
X_test_clusters = kmeans.predict(X_test)
X_test_clusters_centers = kmeans.cluster_centers_
X_test_clusters_centers
```

::: {.output .execute_result execution_count="98"}
    array([[5.01768173e-01, 1.14860872e-01, 1.00000000e+00, 9.16830386e-04,
            1.30975769e-04, 7.85854617e-04, 1.17878193e-03, 9.16830386e-04,
            2.61951539e-04, 0.00000000e+00, 2.61951539e-04, 0.00000000e+00,
            0.00000000e+00, 6.54878847e-04, 2.61951539e-04, 9.93058284e-01,
            2.61951539e-04],
           [4.82413828e-01, 3.56201957e-01, 3.95236660e-01, 7.17515828e-02,
            1.65812481e-02, 7.35604462e-02, 2.41784745e-01, 1.90533615e-01,
            2.35152246e-02, 2.68314742e-02, 6.57220380e-02, 8.74283992e-03,
            4.82363582e-03, 6.90382876e-02, 3.25595418e-02, 0.00000000e+00,
            1.98974977e-02],
           [4.93947663e-01, 1.10954524e-01, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00,
            0.00000000e+00]])
:::
:::

::: {.cell .code execution_count="99"}
``` python
# Split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, labels, test_size=0.3, random_state=0)

# Define K-means model 
kmeans = MiniBatchKMeans(n_clusters=3, random_state=42).fit(X_train)

# Obtain predictions and calculate distance from cluster centroid
X_test_clusters = kmeans.predict(X_test)
X_test_clusters_centers = kmeans.cluster_centers_
dist = [np.linalg.norm(x-y) for x, y in zip(X_test, X_test_clusters_centers[X_test_clusters])]

# Create fraud predictions based on outliers on clusters 
km_y_pred = np.array(dist)
km_y_pred[dist >= np.percentile(dist, 95)] = 1
km_y_pred[dist < np.percentile(dist, 95)] = 0
```
:::

::: {.cell .markdown}
### Checking model results

In the previous exercise you\'ve flagged all observations to be fraud,
if they are in the top 5th percentile in distance from the cluster
centroid. I.e. these are the very outliers of the three clusters. For
this exercise you have the scaled data and labels already split into
training and test set, so y_test is available. The predictions from the
previous exercise, km_y\_pred, are also available. Let\'s create some
performance metrics and see how well you did.

**Instructions 1/3**

-   Obtain the area under the ROC curve from your test labels and
    predicted labels.
:::

::: {.cell .code execution_count="100"}
``` python
def plot_confusion_matrix(cm, classes=['Not Fraud', 'Fraud'],
                          normalize=False,
                          title='Fraud Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    From:
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-
        examples-model-selection-plot-confusion-matrix-py
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
```
:::

::: {.cell .code execution_count="101"}
``` python
# Obtain the ROC score
roc_auc_score(y_test, km_y_pred)
```

::: {.output .execute_result execution_count="101"}
    0.8197704982668266
:::
:::

::: {.cell .markdown}
**Instructions 2/3**

-   Obtain the confusion matrix from the test labels and predicted
    labels and plot the results.
:::

::: {.cell .code execution_count="102"}
``` python
# Create a confusion matrix
km_cm = confusion_matrix(y_test, km_y_pred)

# Plot the confusion matrix in a figure to visualize results 
plot_confusion_matrix(km_cm)
```

::: {.output .stream .stdout}
    Confusion matrix, without normalization
:::

::: {.output .display_data}
![](vertopal_5c32587947084112be3b965be5a32bb5/94185a30e35e197c3db28e5927b4d87a759a2939.png)
:::
:::

::: {.cell .markdown}
**Instructions 3/3**

If you were to decrease the percentile used as a cutoff point in the
previous exercise to 93% instead of 95%, what would that do to your
prediction results?

**Possible Answers**

-   **The number of fraud cases caught increases, but false positives
    also increase.**
-   ~~The number of fraud cases caught decreases, and false positives
    decrease.~~
-   ~~The number of fraud cases caught increases, but false positives
    would decrease.~~
-   ~~Nothing would happen to the amount of fraud cases caught.~~
:::

::: {.cell .markdown}
## Alternate clustering methods for fraud detection

-   In addition to K-means, there are many different clustering methods,
    which can be used for fraud detection
-   ![clustering
    methods](vertopal_5c32587947084112be3b965be5a32bb5/ea8a252754ac63fe51adf13274a460a21477c6e0.JPG)
-   K-means works well when the data is clustered in normal, round
    shapes
-   There are methods to flag fraud other the cluster outliers
-   ![clustering
    outlier](vertopal_5c32587947084112be3b965be5a32bb5/88558d5ceefe8e0fb1ce5d9f45a77827b40c6661.JPG)
    -   Small clusters can be an indication of fraud
    -   This approach can be used when fraudulent behavior has
        commonalities, which cause clustering
    -   The fraudulent data would cluster in tiny groups, rather than be
        the outliers of larger clusters
-   ![typical
    data](vertopal_5c32587947084112be3b965be5a32bb5/0d8af5432451ea1b685273cf0a5834c7fd46a70e.JPG)
    -   In this case there are 3 obvious clusters
    -   The smallest dots are outliers and outside of what can be
        described as normal behavior
    -   There are also small to medium clusters closely connected to the
        red cluster
    -   Visualizing the data with something like
        [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis)
        can be quite helpful
:::

::: {.cell .markdown}
#### DBSCAN: Density-Based Spatial Clustering of Applications with Noise

-   [DBscan](https://en.wikipedia.org/wiki/DBSCAN)
-   DBSCAN vs. K-means
    -   The number of clusters does not need to be predefined
        -   The algorithm finds core samples of high density and expands
            clusters from them
        -   Works well on data containing clusters of similar density
    -   This type of algorithm can be used to identify fraud as very
        small clusters
    -   Maximum allowed distance between points in a cluster must be
        assigned
    -   Minimal number of data points in clusters must be assigned
    -   Better performance on weirdly shaped data
    -   Computationally heavier then MiniBatch K-means
:::

::: {.cell .markdown}
#### Implementation of DBSCAN

``` python
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.5, min_samples=10, n_jobs=-1).fit(X_scaled)

# Get the cluster labels (aka numbers)
pred_labels = db.labels_

# Count the total number of clusters
n_clusters_ = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)

# Print model results
print(f'Estimated number of clusters: {n_clusters_}')
>>> Estimated number of clusters: 31
    
# Print model results
print(f'Silhouette Coefficient: {metrics.silhouette_score(X_scaled, pred_labels):0.3f}')
>>> Silhouette Coefficient: 0.359
    
# Get sample counts in each cluster 
counts = np.bincount(pred_labels[pred_labels>=0])
print(counts)
>>> [ 763, 496, 840, 355 1086, 676, 63, 306, 560, 134, 28, 18, 262, 128,
     332, 22, 22, 13, 31, 38, 36, 28, 14, 12, 30, 10, 11, 10, 21, 10, 5]
```

-   start by defining the epsilon `eps`
    -   Distance between data points allowed from which the cluster
        expands
-   define minimum samples in the clusters
-   conventional DBSCAN can\'t produce the optimal value of epsilon, so
    it requires sophisticated DBSCAN modifications to automatically
    determine the optimal epsilon value
-   Fit DBSCAN to **scaled data**
-   Use `labels_` method to get the assigned cluster label for each data
    point
-   The cluster count can also be determine by counting the unique
    cluster labels from the cluster `label_` predictions
-   Can have performance metrics such as **average silhouette score**
-   The size of each cluster can be calculated with `np.bincount`
    -   counts the number of occurrences of non-negative values in a
        `numpy` array
-   sort `counts` and decide how many of the smaller clusters to flag as
    fraud
    -   selecting the clusters to flag, is a trial-and-error step and
        depends on the number of cases the fraud team can manage
:::

::: {.cell .markdown}
### DB scan

In this exercise you\'re going to explore using a **density based
clustering** method (DBSCAN) to detect fraud. The advantage of DBSCAN is
that you **do not need to define the number of clusters** beforehand.
Also, DBSCAN can handle weirdly shaped data (i.e. non-convex) much
better than K-means can. This time, you are not going to take the
outliers of the clusters and use that for fraud, but take the **smallest
clusters** in the data and label those as fraud. You again have the
scaled dataset, i.e. X_scaled available. Let\'s give it a try!

**Instructions**

-   Import `DBSCAN`.
-   Initialize a DBSCAN model setting the maximum distance between two
    samples to 0.9 and the minimum observations in the clusters to 10,
    and fit the model to the scaled data.
-   Obtain the predicted labels, these are the cluster numbers assigned
    to an observation.
-   Print the number of clusters and the rest of the performance
    metrics.
:::

::: {.cell .code execution_count="103"}
``` python
# Initialize and fit the DBscan model
db = DBSCAN(eps=0.9, min_samples=10, n_jobs=-1).fit(X_scaled)

# Obtain the predicted labels and calculate number of clusters
pred_labels = db.labels_
n_clusters = len(set(pred_labels)) - (1 if -1 in labels else 0)
```
:::

::: {.cell .code execution_count="104"}
``` python
# Print performance metrics for DBscan
print(f'Estimated number of clusters: {n_clusters}')
print(f'Homogeneity: {homogeneity_score(labels, pred_labels):0.3f}')
print(f'Silhouette Coefficient: {silhouette_score(X_scaled, pred_labels):0.3f}')
```

::: {.output .stream .stdout}
    Estimated number of clusters: 23
    Homogeneity: 0.612
    Silhouette Coefficient: 0.713
:::
:::

::: {.cell .markdown}
**The number of clusters is much higher than with K-means. For fraud
detection this is for now OK, as we are only interested in the smallest
clusters, since those are considered as abnormal. Now have a look at
those clusters and decide which one to flag as fraud.**
:::

::: {.cell .markdown}
### Assessing smallest clusters

In this exercise you\'re going to have a look at the clusters that came
out of DBscan, and flag certain clusters as fraud:

-   you first need to figure out how big the clusters are, and **filter
    out the smallest**
-   then, you\'re going to take the smallest ones and **flag those as
    fraud**
-   last, you\'ll **check with the original labels** whether this does
    actually do a good job in detecting fraud.

Available are the DBscan model predictions, so `n_clusters` is available
as well as the cluster labels, which are saved under `pred_labels`.
Let\'s give it a try!

**Instructions 1/3**

-   Count the samples within each cluster by running a bincount on the
    predicted cluster numbers under `pred_labels` and print the results.
:::

::: {.cell .code execution_count="105"}
``` python
# Count observations in each cluster number
counts = np.bincount(pred_labels[pred_labels >= 0])

# Print the result
print(counts)
```

::: {.output .stream .stdout}
    [3252  145 2714   55  174  119  122   98   54   15   76   15   43   25
       51   47   42   15   25   20   19   10]
:::
:::

::: {.cell .markdown}
**Instructions 2/3**

-   Sort the sample `counts` and take the top 3 smallest clusters, and
    print the results.
:::

::: {.cell .code execution_count="106"}
``` python
# Sort the sample counts of the clusters and take the top 3 smallest clusters
smallest_clusters = np.argsort(counts)[:3]
```
:::

::: {.cell .code execution_count="107"}
``` python
# Print the results 
print(f'The smallest clusters are clusters: {smallest_clusters}')
```

::: {.output .stream .stdout}
    The smallest clusters are clusters: [21 17  9]
:::
:::

::: {.cell .markdown}
**Instructions 3/3**

-   Within `counts`, select the smallest clusters only, to print the
    number of samples in the three smallest clusters.
:::

::: {.cell .code execution_count="108"}
``` python
# Print the counts of the smallest clusters only
print(f'Their counts are: {counts[smallest_clusters]}')
```

::: {.output .stream .stdout}
    Their counts are: [10 15 15]
:::
:::

::: {.cell .markdown}
**So now we know which smallest clusters you could flag as fraud. If you
were to take more of the smallest clusters, you cast your net wider and
catch more fraud, but most likely also more false positives. It is up to
the fraud analyst to find the right amount of cases to flag and to
investigate. In the next exercise you\'ll check the results with the
actual labels.**
:::

::: {.cell .markdown}
### Results verification

In this exercise you\'re going to **check the results** of your DBscan
fraud detection model. In reality, you often don\'t have reliable labels
and this where a fraud analyst can help you validate the results. He/She
can check your results and see whether the cases you flagged are indeed
suspicious. You can also **check historically known cases** of fraud and
see whether your model flags them.

In this case, you\'ll **use the fraud labels** to check your model
results. The predicted cluster numbers are available under `pred_labels`
as well as the original fraud `labels`.

**Instructions**

-   Create a dataframe combining the cluster numbers with the actual
    labels.
-   Create a condition that flags fraud for the three smallest clusters:
    clusters 21, 17 and 9.
-   Create a crosstab from the actual fraud labels with the newly
    created predicted fraud labels.
:::

::: {.cell .code execution_count="109"}
``` python
# Create a dataframe of the predicted cluster numbers and fraud labels 
df = pd.DataFrame({'clusternr':pred_labels,'fraud':labels})

# Create a condition flagging fraud for the smallest clusters 
df['predicted_fraud'] = np.where((df['clusternr'].isin([21, 17, 9])), 1 , 0)
```
:::

::: {.cell .code execution_count="110"}
``` python
# Run a crosstab on the results 
print(pd.crosstab(df['fraud'], df['predicted_fraud'], rownames=['Actual Fraud'], colnames=['Flagged Fraud']))
```

::: {.output .stream .stdout}
    Flagged Fraud     0   1
    Actual Fraud           
    0              6973  16
    1               176  24
:::
:::

::: {.cell .markdown}
**How does this compare to the K-means model? The good thing is: our of
all flagged cases, roughly 2/3 are actually fraud! Since you only take
the three smallest clusters, by definition you flag less cases of fraud,
so you catch less but also have less false positives. However, you are
missing quite a lot of fraud cases. Increasing the amount of smallest
clusters you flag could improve that, at the cost of more false
positives of course. In the next chapter you\'ll learn how to further
improve fraud detection models by including text analysis.**
:::

::: {.cell .markdown}
# Fraud detection using text

Use text data, text mining and topic modeling to detect fraudulent
behavior.
:::

::: {.cell .markdown}
## Using text data

-   Types of useful text data:
    1.  Emails from employees and/or clients
    2.  Transaction descriptions
    3.  Employee notes
    4.  Insurance claim form description box
    5.  Recorded telephone conversations
-   Text mining techniques for fraud detection
    1.  Word search
    2.  Sentiment analysis
    3.  Word frequencies and topic analysis
    4.  Style
-   Word search for fraud detection
    -   Flagging suspicious words:
        1.  Simple, straightforward and easy to explain
        2.  Match results can be used as a filter on top of machine
            learning model
        3.  Match results can be used as a feature in a machine learning
            model
:::

::: {.cell .markdown}
#### Word counts to flag fraud with pandas

``` python
# Using a string operator to find words
df['email_body'].str.contains('money laundering')

 # Select data that matches 
df.loc[df['email_body'].str.contains('money laundering', na=False)]

 # Create a list of words to search for
list_of_words = ['police', 'money laundering']
df.loc[df['email_body'].str.contains('|'.join(list_of_words), na=False)]

 # Create a fraud flag 
df['flag'] = np.where((df['email_body'].str.contains('|'.join(list_of_words)) == True), 1, 0)
```
:::

::: {.cell .markdown}
### Word search with dataframes

In this exercise you\'re going to work with text data, containing emails
from Enron employees. The **Enron scandal** is a famous fraud case.
Enron employees covered up the bad financial position of the company,
thereby keeping the stock price artificially high. Enron employees sold
their own stock options, and when the truth came out, Enron investors
were left with nothing. The goal is to find all emails that mention
specific words, such as \"sell enron stock\".

By using string operations on dataframes, you can easily sift through
messy email data and create flags based on word-hits. The Enron email
data has been put into a dataframe called `df` so let\'s search for
suspicious terms. Feel free to explore `df` in the Console before
getting started.

**Instructions 1/2**

-   Check the head of `df` in the console and look for any emails
    mentioning \'sell enron stock\'.
:::

::: {.cell .code execution_count="112"}
``` python
df = pd.read_csv("chapitre_4/enron_emails_clean.csv")
```
:::

::: {.cell .code execution_count="113"}
``` python
df.head()
```

::: {.output .execute_result execution_count="113"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Message-ID</th>
      <th>From</th>
      <th>To</th>
      <th>Date</th>
      <th>content</th>
      <th>clean_content</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>&lt;8345058.1075840404046.JavaMail.evans@thyme&gt;</td>
      <td>('advdfeedback@investools.com')</td>
      <td>('advdfeedback@investools.com')</td>
      <td>2002-01-29 23:20:55</td>
      <td>INVESTools Advisory\nA Free Digest of Trusted ...</td>
      <td>investools advisory free digest trusted invest...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>&lt;1512159.1075863666797.JavaMail.evans@thyme&gt;</td>
      <td>('richard.sanders@enron.com')</td>
      <td>('richard.sanders@enron.com')</td>
      <td>2000-09-20 19:07:00</td>
      <td>----- Forwarded by Richard B Sanders/HOU/ECT o...</td>
      <td>forwarded richard b sanders hou ect pm justin ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>&lt;26118676.1075862176383.JavaMail.evans@thyme&gt;</td>
      <td>('m..love@enron.com')</td>
      <td>('m..love@enron.com')</td>
      <td>2001-10-30 16:15:17</td>
      <td>hey you are not wearing your target purple shi...</td>
      <td>hey wearing target purple shirt today mine wan...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>&lt;10369289.1075860831062.JavaMail.evans@thyme&gt;</td>
      <td>('leslie.milosevich@kp.org')</td>
      <td>('leslie.milosevich@kp.org')</td>
      <td>2002-01-30 17:54:18</td>
      <td>Leslie Milosevich\n1042 Santa Clara Avenue\nAl...</td>
      <td>leslie milosevich santa clara avenue alameda c...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>&lt;26728895.1075860815046.JavaMail.evans@thyme&gt;</td>
      <td>('rtwait@graphicaljazz.com')</td>
      <td>('rtwait@graphicaljazz.com')</td>
      <td>2002-01-30 19:36:01</td>
      <td>Rini Twait\n1010 E 5th Ave\nLongmont, CO 80501...</td>
      <td>rini twait e th ave longmont co rtwait graphic...</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .code execution_count="114"}
``` python
mask = df['clean_content'].str.contains('sell enron stock', na=False)
```
:::

::: {.cell .markdown}
**Instructions 2/2**

-   Locate the data in `df` that meets the condition we created earlier.
:::

::: {.cell .code execution_count="115"}
``` python
# Select the data from df using the mask
df[mask]
```

::: {.output .execute_result execution_count="115"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Message-ID</th>
      <th>From</th>
      <th>To</th>
      <th>Date</th>
      <th>content</th>
      <th>clean_content</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>154</th>
      <td>&lt;6336501.1075841154311.JavaMail.evans@thyme&gt;</td>
      <td>('sarah.palmer@enron.com')</td>
      <td>('sarah.palmer@enron.com')</td>
      <td>2002-02-01 14:53:35</td>
      <td>\nJoint Venture: A 1997 Enron Meeting Belies O...</td>
      <td>joint venture enron meeting belies officers cl...</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown}
**You see that searching for particular string values in a dataframe can
be relatively easy, and allows you to include textual data into your
model or analysis. You can use this word search as an additional flag,
or as a feature in your fraud detection model. Let\'s look at how to
filter the data using multiple search terms.**
:::

::: {.cell .markdown}
### Using list of terms

Oftentimes you don\'t want to search on just one term. You probably can
create a full **\"fraud dictionary\"** of terms that could potentially
**flag fraudulent clients** and/or transactions. Fraud analysts often
will have an idea what should be in such a dictionary. In this exercise
you\'re going to **flag a multitude of terms**, and in the next exercise
you\'ll create a new flag variable out of it. The \'flag\' can be used
either directly in a machine learning model as a feature, or as an
additional filter on top of your machine learning model results. Let\'s
first use a list of terms to filter our data on. The dataframe
containing the cleaned emails is again available as `df`.

**Instructions**

-   Create a list to search for including \'enron stock\', \'sell
    stock\', \'stock bonus\', and \'sell enron stock\'.
-   Join the string terms in the search conditions.
-   Filter data using the emails that match with the list defined under
    `searchfor`.
:::

::: {.cell .code execution_count="116"}
``` python
# Create a list of terms to search for
searchfor = ['enron stock', 'sell stock', 'stock bonus', 'sell enron stock']

# Filter cleaned emails on searchfor list and select from df 
filtered_emails = df[df.clean_content.str.contains('|'.join(searchfor), na=False)]
filtered_emails.head()
```

::: {.output .execute_result execution_count="116"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Message-ID</th>
      <th>From</th>
      <th>To</th>
      <th>Date</th>
      <th>content</th>
      <th>clean_content</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>&lt;8345058.1075840404046.JavaMail.evans@thyme&gt;</td>
      <td>('advdfeedback@investools.com')</td>
      <td>('advdfeedback@investools.com')</td>
      <td>2002-01-29 23:20:55</td>
      <td>INVESTools Advisory\nA Free Digest of Trusted ...</td>
      <td>investools advisory free digest trusted invest...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>&lt;1512159.1075863666797.JavaMail.evans@thyme&gt;</td>
      <td>('richard.sanders@enron.com')</td>
      <td>('richard.sanders@enron.com')</td>
      <td>2000-09-20 19:07:00</td>
      <td>----- Forwarded by Richard B Sanders/HOU/ECT o...</td>
      <td>forwarded richard b sanders hou ect pm justin ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>&lt;26118676.1075862176383.JavaMail.evans@thyme&gt;</td>
      <td>('m..love@enron.com')</td>
      <td>('m..love@enron.com')</td>
      <td>2001-10-30 16:15:17</td>
      <td>hey you are not wearing your target purple shi...</td>
      <td>hey wearing target purple shirt today mine wan...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>&lt;10369289.1075860831062.JavaMail.evans@thyme&gt;</td>
      <td>('leslie.milosevich@kp.org')</td>
      <td>('leslie.milosevich@kp.org')</td>
      <td>2002-01-30 17:54:18</td>
      <td>Leslie Milosevich\n1042 Santa Clara Avenue\nAl...</td>
      <td>leslie milosevich santa clara avenue alameda c...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>&lt;26728895.1075860815046.JavaMail.evans@thyme&gt;</td>
      <td>('rtwait@graphicaljazz.com')</td>
      <td>('rtwait@graphicaljazz.com')</td>
      <td>2002-01-30 19:36:01</td>
      <td>Rini Twait\n1010 E 5th Ave\nLongmont, CO 80501...</td>
      <td>rini twait e th ave longmont co rtwait graphic...</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown}
**By joining the search terms with the \'or\' sign, i.e. \|, you can
search on a multitude of terms in your dataset very easily. Let\'s now
create a flag from this which you can use as a feature in a machine
learning model.**
:::

::: {.cell .markdown}
### Creating a flag

This time you are going to **create an actual flag** variable that gives
a **1 when the emails get a hit** on the search terms of interest, and 0
otherwise. This is the last step you need to make in order to actually
use the text data content as a feature in a machine learning model, or
as an actual flag on top of model results. You can continue working with
the dataframe `df` containing the emails, and the `searchfor` list is
the one defined in the last exercise.

**Instructions**

-   Use a numpy where condition to flag \'1\' where the cleaned email
    contains words on the `searchfor` list and 0 otherwise.
-   Join the words on the `searchfor` list with an \"or\" indicator.
-   Count the values of the newly created flag variable.
:::

::: {.cell .code execution_count="117"}
``` python
# Create flag variable where the emails match the searchfor terms
df['flag'] = np.where((df['clean_content'].str.contains('|'.join(searchfor)) == True), 1, 0)

# Count the values of the flag variable
count = df['flag'].value_counts()
print(count)
```

::: {.output .stream .stdout}
    0    1776
    1     314
    Name: flag, dtype: int64
:::
:::

::: {.cell .markdown}
**You have now managed to search for a list of strings in several lines
of text data. These skills come in handy when you want to flag certain
words based on what you discovered in your topic model, or when you know
beforehand what you want to search for. In the next exercises you\'re
going to learn how to clean text data and to create your own topic model
to further look for indications of fraud in your text data.**
:::

::: {.cell .markdown}
## Text mining to detect fraud
:::

::: {.cell .markdown}
#### Cleaning your text data

**Must dos when working with textual data:**

1.  Tokenization
    -   Split the text into sentences and the sentences in words
    -   transform everything to lowercase
    -   remove punctuation
2.  Remove all stopwords
3.  Lemmatize
    -   change from third person into first person
    -   change past and future tense verbs to present tense
    -   this makes it possible to combine all words that point to the
        same thing
4.  Stem the words
    -   reduce words to their root form
    -   e.g. walking and walked to walk

-   **Unprocessed Text**
    -   ![](vertopal_5c32587947084112be3b965be5a32bb5/c629aa54c8f240c34bf470086f5f265f1e0f63f1.JPG)
-   **Processed Text**
    -   ![](vertopal_5c32587947084112be3b965be5a32bb5/d67021cc8f60a3ec836a54ce4afe925312ec2ba0.JPG)
:::

::: {.cell .markdown}
#### Data Preprocessing I

-   Tokenizers divide strings into list of substrings
-   nltk word tokenizer can be used to find the words and punctuation in
    a string
    -   it splits the words on whitespace, and separated the punctuation
        out

``` python
from nltk import word_tokenize
from nltk.corpus import stopwords 
import string

# 1. Tokenization
text = df.apply(lambda row: word_tokenize(row["email_body"]), axis=1)
text = text.rstrip()  # remove whitespace
# replace with lowercase
# text = re.sub(r'[^a-zA-Z]', ' ', text)
text = text.lower()

 # 2. Remove all stopwords and punctuation
exclude = set(string.punctuation)
stop = set(stopwords.words('english'))
stop_free = " ".join([word for word in text if((word not in stop) and (not word.isdigit()))])
punc_free = ''.join(word for word in stop_free if word not in exclude)
```
:::

::: {.cell .markdown}
#### Data Preprocessing II

``` python
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

# Lemmatize words
lemma = WordNetLemmatizer()
normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())

# Stem words
porter= PorterStemmer()
cleaned_text = " ".join(porter.stem(token) for token in normalized.split())
print (cleaned_text)

['philip','going','street','curious','hear','perspective','may','wish',
'offer','trading','floor','enron','stock','lower','joined','company',
'business','school','imagine','quite','happy','people','day','relate',
'somewhat','stock','around','fact','broke','day','ago','knowing',
'imagine','letting','event','get','much','taken','similar',
'problem','hope','everything','else','going','well','family','knee',
'surgery','yet','give','call','chance','later']
```
:::

::: {.cell .markdown}
### Removing stopwords

In the following exercises you\'re going to **clean the Enron emails**,
in order to be able to use the data in a topic model. Text cleaning can
be challenging, so you\'ll learn some steps to do this well. The
dataframe containing the emails `df` is available. In a first step you
need to **define the list of stopwords and punctuations** that are to be
removed in the next exercise from the text data. Let\'s give it a try.

**Instructions**

-   Import the stopwords from `ntlk`.
-   Define \'english\' words to use as stopwords under the variable
    `stop`.
-   Get the punctuation set from the `string` package and assign it to
    `exclude`.
:::

::: {.cell .code execution_count="127"}
``` python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

::: {.output .stream .stderr}
    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\tNouali\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package wordnet to
    [nltk_data]     C:\Users\tNouali\AppData\Roaming\nltk_data...
:::

::: {.output .execute_result execution_count="127"}
    True
:::
:::

::: {.cell .code execution_count="122"}
``` python
# Define stopwords to exclude
stop = set(stopwords.words('english'))
stop.update(("to", "cc", "subject", "http", "from", "sent", "ect", "u", "fwd", "www", "com", 'html'))

# Define punctuations to exclude and lemmatizer
exclude = set(string.punctuation)
```
:::

::: {.cell .code execution_count="123"}
``` python
print(exclude)
```

::: {.output .stream .stdout}
    {'=', '\\', '&', ')', '`', '|', '@', '[', '%', ']', ',', '?', '+', '-', '{', ':', '}', ';', '!', '^', '$', '>', '<', '"', '_', '*', '~', "'", '/', '#', '(', '.'}
:::
:::

::: {.cell .markdown}
### Cleaning text data

Now that you\'ve defined the **stopwords and punctuations**, let\'s use
these to clean our enron emails in the dataframe `df` further. The lists
containing stopwords and punctuations are available under `stop` and
`exclude` There are a few more steps to take before you have cleaned
data, such as **\"lemmatization\"** of words, and **stemming the
verbs**. The verbs in the email data are already stemmed, and the
lemmatization is already done for you in this exercise.

**Instructions 1/2**

-   Use the previously defined variables `stop` and `exclude` to finish
    of the function: Strip the words from whitespaces using `rstrip`,
    and exclude stopwords and punctuations. Finally lemmatize the words
    and assign that to `normalized`.
:::

::: {.cell .code execution_count="128"}
``` python
# Import the lemmatizer from nltk
lemma = WordNetLemmatizer()

def clean(text, stop):
    text = str(text).rstrip()
    stop_free = " ".join([i for i in text.lower().split() if((i not in stop) and (not i.isdigit()))])
    punc_free = ''.join(i for i in stop_free if i not in exclude)
    normalized = " ".join(lemma.lemmatize(i) for i in punc_free.split())      
    return normalized
```
:::

::: {.cell .markdown}
**Instructions 2/2**

-   Apply the function `clean(text,stop)` on each line of text data in
    our dataframe, and take the column `df['clean_content']` for this.
:::

::: {.cell .code execution_count="129"}
``` python
# Clean the emails in df and print results
text_clean=[]
for text in df['clean_content']:
    text_clean.append(clean(text, stop).split())    
```
:::

::: {.cell .code execution_count="131"}
``` python
text_clean[0]
```

::: {.output .execute_result execution_count="131"}
    ['investools',
     'advisory',
     'free',
     'digest',
     'trusted',
     'investment',
     'advice',
     'unsubscribe',
     'free',
     'newsletter',
     'please',
     'see',
     'issue',
     'fried',
     'sell',
     'stock',
     'gain',
     'month',
     'km',
     'rowe',
     'january',
     'index',
     'confirms',
     'bull',
     'market',
     'aloy',
     'small',
     'cap',
     'advisor',
     'earns',
     'lbix',
     'compounding',
     'return',
     'pine',
     'tree',
     'pcl',
     'undervalued',
     'high',
     'yield',
     'bank',
     'put',
     'customer',
     'first',
     'aso',
     'word',
     'sponsor',
     'top',
     'wall',
     'street',
     'watcher',
     'ben',
     'zacks',
     'year',
     'year',
     'gain',
     'moving',
     'best',
     'brightest',
     'wall',
     'street',
     'big',
     'money',
     'machine',
     'earned',
     'ben',
     'zacks',
     'five',
     'year',
     'average',
     'annual',
     'gain',
     'start',
     'outperforming',
     'long',
     'term',
     'get',
     'zacks',
     'latest',
     'stock',
     'buylist',
     'free',
     'day',
     'trial',
     'investools',
     'c',
     'go',
     'zaks',
     'mtxtu',
     'zakstb',
     'investools',
     'advisory',
     'john',
     'brobst',
     'investools',
     'fried',
     'sell',
     'stock',
     'lock',
     'month',
     'km',
     'david',
     'fried',
     'know',
     'stock',
     'undervalued',
     'company',
     'management',
     'buy',
     'back',
     'share',
     'open',
     'market',
     'latest',
     'triumph',
     'pocketing',
     'impressive',
     'gain',
     'three',
     'short',
     'month',
     'selling',
     'four',
     'buyback',
     'stock',
     'include',
     'gain',
     'auto',
     'retailer',
     'automation',
     'incorporated',
     'gain',
     'digital',
     'phone',
     'system',
     'purveyor',
     'inter',
     'tel',
     'intl',
     'fried',
     'recent',
     'move',
     'buy',
     'kmart',
     'corporation',
     'km',
     'beleaguered',
     'discount',
     'retailer',
     'declared',
     'bankruptcy',
     'think',
     'k',
     'mart',
     'go',
     'business',
     'fried',
     'say',
     'take',
     'recovery',
     'possibility',
     'bought',
     'share',
     'another',
     'fried',
     'pick',
     'c',
     'cor',
     'net',
     'corporation',
     'ccbl',
     'provides',
     'range',
     'technology',
     'service',
     'broadband',
     'network',
     'today',
     'telecom',
     'spending',
     'slowdown',
     'hit',
     'company',
     'hard',
     'net',
     'sale',
     'fell',
     'million',
     'last',
     'quarter',
     'caused',
     'net',
     'loss',
     'million',
     'v',
     'million',
     'gain',
     'last',
     'year',
     'fried',
     'cite',
     'buyback',
     'plan',
     'million',
     'restructuring',
     'charge',
     'proof',
     'management',
     'see',
     'rosier',
     'future',
     'david',
     'fried',
     'advice',
     'see',
     'buyback',
     'index',
     'portfolio',
     'january',
     'buyback',
     'letter',
     'david',
     'fried',
     'provides',
     'wealth',
     'building',
     'opportunity',
     'company',
     'repurchasing',
     'stock',
     'free',
     'day',
     'trial',
     'go',
     'investools',
     'c',
     'go',
     'back',
     'mtxtu',
     'back',
     'rowe',
     'january',
     'index',
     'confirms',
     'bull',
     'market',
     'aloy',
     'rowe',
     'say',
     'january',
     'index',
     'confirms',
     'see',
     'bull',
     'market',
     'first',
     'five',
     'trading',
     'day',
     'provided',
     'gain',
     'nasdaq',
     'p',
     'dow',
     'industrials',
     'rowe',
     'say',
     'five',
     'day',
     'index',
     'correctly',
     'predicted',
     'market',
     'direction',
     'year',
     'since',
     'four',
     'exception',
     'include',
     'three',
     'war',
     'year',
     'fed',
     'fund',
     'rate',
     'doubled',
     'year',
     'rowe',
     'maintains',
     'sure',
     'recommendation',
     'seven',
     'company',
     'say',
     'leading',
     'market',
     'one',
     'alloy',
     'incorporated',
     'aloy',
     'medium',
     'company',
     'direct',
     'marketer',
     'provides',
     'content',
     'community',
     'commerce',
     'generation',
     'roughly',
     'million',
     'people',
     'year',
     'age',
     'rowe',
     'like',
     'market',
     'account',
     'billion',
     'disposable',
     'income',
     'grow',
     'faster',
     'overall',
     'population',
     'q',
     'saw',
     'earnings',
     'increase',
     'sale',
     'another',
     'rowe',
     'pick',
     'new',
     'century',
     'financial',
     'corporation',
     'ncen',
     'financier',
     'make',
     'buy',
     'sell',
     'service',
     'sub',
     'prime',
     'mortgage',
     'loan',
     'secured',
     'first',
     'mortgage',
     'single',
     'family',
     'home',
     'borrower',
     'typically',
     'plenty',
     'equity',
     'property',
     'secure',
     'loan',
     'suffer',
     'weak',
     'credit',
     'profile',
     'high',
     'debt',
     'income',
     'ratio',
     'q',
     'earnings',
     'grew',
     'hike',
     'sale',
     'rowe',
     'advice',
     'see',
     'investment',
     'opportunity',
     'february',
     'wall',
     'street',
     'digest',
     'momentum',
     'investor',
     'donald',
     'rowe',
     'target',
     'stock',
     'mutual',
     'fund',
     'capable',
     'generating',
     'annual',
     'return',
     'free',
     'day',
     'trial',
     'go',
     'investools',
     'c',
     'go',
     'wall',
     'mtxtu',
     'wall',
     'small',
     'cap',
     'advisor',
     'earns',
     'lbix',
     'major',
     'index',
     'suffered',
     'terrible',
     'year',
     'richard',
     'geist',
     'recommendation',
     'earned',
     'healthy',
     'list',
     'many',
     'reason',
     'selection',
     'see',
     'growth',
     'going',
     'forward',
     'include',
     'extremely',
     'bullish',
     'monetary',
     'condition',
     'high',
     'productivity',
     'inflation',
     'sight',
     'yield',
     'curve',
     'continues',
     'steepen',
     'also',
     'investor',
     'sentiment',
     'poll',
     'becoming',
     'bearish',
     'always',
     'contrary',
     'indicator',
     'say',
     'geist',
     'latest',
     'recommendation',
     'buy',
     'share',
     'leading',
     'brand',
     'lbix',
     'company',
     'canada',
     'largest',
     'independent',
     'food',
     'brand',
     'management',
     'company',
     'expanding',
     'u',
     'geist',
     'particularly',
     'like',
     'firm',
     'save',
     'money',
     'integrated',
     'distribution',
     'system',
     'system',
     'make',
     'product',
     'raw',
     'material',
     'provides',
     'packaging',
     'warehousing',
     'distribution',
     'recent',
     'financial',
     'result',
     'show',
     'leading',
     'brand',
     'roll',
     'fy',
     'saw',
     'revenue',
     'grow',
     'million',
     'net',
     'income',
     'million',
     'per',
     'share',
     'last',
     'year',
     'loss',
     'geist',
     'predicts',
     'company',
     'see',
     'revenue',
     'reach',
     'million',
     'million',
     'yield',
     'forward',
     'pe',
     'think',
     'lbix',
     'significantly',
     'undervalued',
     'geist',
     'say',
     'range',
     'leading',
     'brand',
     'strong',
     'buy',
     'richard',
     'geist',
     'advice',
     'see',
     'highlighted',
     'stock',
     'february',
     'richard',
     'geist',
     'strategic',
     'investing',
     'richard',
     'geist',
     'integrates',
     'psychological',
     'aspect',
     'investing',
     'methodology',
     'selecting',
     'small',
     'company',
     'stock',
     'free',
     'day',
     'trial',
     'go',
     'investools',
     'c',
     'go',
     'stin',
     'mtxtu',
     'stin',
     'compounding',
     'return',
     'pine',
     'tree',
     'pcl',
     'growing',
     'tree',
     'usually',
     'noisy',
     'business',
     'catch',
     'attention',
     'investment',
     'medium',
     'good',
     'business',
     'say',
     'dick',
     'young',
     'timber',
     'business',
     'le',
     'volatile',
     'capital',
     'intensive',
     'manufacturing',
     'young',
     'see',
     'demand',
     'timber',
     'increasing',
     'population',
     'increase',
     'note',
     'average',
     'return',
     'timber',
     'investment',
     'outperformed',
     'p',
     'average',
     'annual',
     'return',
     'young',
     'favorite',
     'timber',
     'play',
     'plum',
     'creek',
     'timber',
     'pcl',
     'one',
     'largest',
     'private',
     'timberland',
     'owner',
     'u',
     'reit',
     'primary',
     'goal',
     'profit',
     'acquiring',
     'managing',
     'land',
     'young',
     'say',
     'plum',
     'creek',
     'timber',
     'yield',
     'status',
     'reit',
     'make',
     'ideal',
     'tax',
     'deferred',
     'account',
     'another',
     'young',
     'timber',
     'selection',
     'deltic',
     'timber',
     'corporation',
     'del',
     'company',
     'grows',
     'harvest',
     'timber',
     'acre',
     'arkansas',
     'louisiana',
     'main',
     'company',
     'goal',
     'expand',
     'timber',
     'holding',
     'sustainable',
     'harvest',
     'level',
     'young',
     'say',
     'share',
     'good',
     'portfolio',
     'counterweight',
     'value',
     'investor',
     'appreciate',
     'intrinsic',
     'worth',
     'underlying',
     'real',
     'natural',
     'resource',
     'dick',
     'young',
     'advice',
     'see',
     'investment',
     'commentary',
     'february',
     'richard',
     'young',
     'intelligence',
     'report',
     'richard',
     'young',
     'us',
     'buy',
     'hold',
     'strategy',
     'mentor',
     'warren',
     'buffett',
     'uncover',
     'low',
     'risk',
     'high',
     'reward',
     'opportunity',
     'free',
     'day',
     'trial',
     'go',
     'investools',
     'c',
     'go',
     'inte',
     'mtxtu',
     'inte',
     'undervalued',
     'high',
     'yield',
     'bank',
     'put',
     'customer',
     'first',
     'aso',
     'amsouth',
     'bancorp',
     'aso',
     'giving',
     'investor',
     'healthy',
     'yield',
     'risk',
     'involved',
     'say',
     'jodie',
     'wei',
     'investment',
     'quality',
     'trend',
     'billion',
     'asset',
     'amsouth',
     'one',
     'largest',
     'financial',
     'institution',
     'south',
     'office',
     'credit',
     'bank',
     'success',
     'putting',
     'customer',
     'first',
     'wei',
     'like',
     'amsouth',
     'us',
     'new',
     'technology',
     'save',
     'money',
     'streamlining',
     'operation',
     'note',
     'amsouth',
     'ranked',
     'number',
     'six',
     'eweek',
     'fast',
     'track',
     'list',
     'company',
     'deploy',
     'cutting',
     'edge',
     'technology',
     'throughout',
     'operation',
     'number',
     'merrill',
     'lynch',
     'financial',
     'service',
     'firm',
     'placed',
     'higher',
     'also',
     'amsouth',
     'internet',
     'banking',
     'group',
     'quadrupled',
     'customer',
     'base',
     'last',
     'year',
     'wei',
     'say',
     'aso',
     'share',
     'undervalued',
     'stock',
     'selling',
     'near',
     'yield',
     'wei',
     'see',
     'upside',
     'potential',
     'dividend',
     'risen',
     'annually',
     'past',
     'year',
     'buyback',
     'plan',
     'million',
     'share',
     'authorized',
     'september',
     'stock',
     'pe',
     'reasonable',
     'x',
     'priced',
     'yield',
     'aso',
     'undervalued',
     'buy',
     'considered',
     'wei',
     'say',
     'jodie',
     'wei',
     'advice',
     'see',
     'investment',
     'spotlight',
     'january',
     'income',
     'digest',
     'digest',
     'excerpt',
     'investment',
     'publication',
     'highlight',
     'weather',
     'income',
     'oriented',
     'opportunity',
     'uncovered',
     'top',
     'mind',
     'wall',
     'street',
     'free',
     'day',
     'trial',
     'go',
     'investools',
     'c',
     'go',
     'indi',
     'mtxtu',
     'indi',
     'word',
     'sponsor',
     'new',
     'report',
     'top',
     'pick',
     'despite',
     'slumping',
     'economy',
     'shaky',
     'stock',
     'market',
     'frank',
     'curzio',
     'bull',
     'eye',
     'pick',
     'gained',
     'whopping',
     'curzio',
     'selected',
     'stock',
     'incredible',
     'potential',
     'get',
     'red',
     'hot',
     'pick',
     'today',
     'click',
     'investools',
     'c',
     'go',
     'fxcp',
     'fxcp',
     'mtxtu',
     'disclaimer',
     'investools',
     'advisory',
     'published',
     'solely',
     'informational',
     'purpose',
     'solicit',
     'offer',
     'buy',
     'sell',
     'stock',
     'mutual',
     'fund',
     'security',
     'attempt',
     'claim',
     'complete',
     'description',
     'security',
     'market',
     'development',
     'referred',
     'material',
     'expression',
     'opinion',
     'change',
     'without',
     'notice',
     'information',
     'obtained',
     'internal',
     'external',
     'source',
     'investools',
     'considers',
     'reliable',
     'investools',
     'independently',
     'verified',
     'information',
     'investools',
     'guarantee',
     'accurate',
     'complete',
     'investools',
     'undertake',
     'advise',
     'anyone',
     'investools',
     'employee',
     'officer',
     'director',
     'may',
     'time',
     'time',
     'position',
     'security',
     'mentioned',
     'may',
     'sell',
     'buy',
     'security',
     'remove',
     'free',
     'email',
     'list',
     'removed',
     'email',
     'distribution',
     'list',
     'free',
     'investools',
     'advisory',
     'update',
     'simply',
     'click',
     'link',
     'hit',
     'send',
     'email',
     'launched',
     'copy',
     'paste',
     'email',
     'address',
     'new',
     'outgoing',
     'email',
     'message',
     'hit',
     'send',
     'email',
     'launched',
     'mailto',
     'bonnie',
     'investools',
     'important',
     'automated',
     'system',
     'cancel',
     'paid',
     ...]
:::
:::

::: {.cell .markdown}
**Now that you have cleaned your data entirely with the necessary steps,
including splitting the text into words, removing stopwords and
punctuations, and lemmatizing your words. You are now ready to run a
topic model on this data. In the following exercises you\'re going to
explore how to do that.**
:::

::: {.cell .markdown}
## Topic modeling on fraud

1.  Discovering topics in text data
2.  \"What is the text about\"
3.  Conceptually similar to clustering data
4.  Compare topics of fraud cases to non-fraud cases and use as a
    feature or flag
5.  Or.. is there a particular topic in the data that seems to point to
    fraud?
:::

::: {.cell .markdown}
#### Latent Dirichlet Allocation (LDA)

-   With LDA you obtain:
    -   \"topics per text item\" model (i.e. probabilities)
    -   \"words per topic\" model
-   Creating your own topic model:
    -   Clean your data
    -   Create a bag of words with dictionary and corpus
        -   Dictionary contain words and word frequency from the entire
            text
        -   Corpus: word count for each line of text
    -   Feed dictionary and corpus into the LDA model
-   LDA:
    -   ![lda](vertopal_5c32587947084112be3b965be5a32bb5/42650b28b0989fdec246a6fabf61c4947daf424b.JPG)

    1.  [LDA2vec: Word Embeddings in Topic
        Models](https://www.datacamp.com/community/tutorials/lda2vec-topic-model)
    2.  see how each word in the dataset is associated with each topic
    3.  see how each text item in the data associates with topics (in
        the form of probabilities)
        1.  image on the right
:::

::: {.cell .markdown}
#### Bag of words: dictionary and corpus

-   use the `Dictionary` function in `corpora` to create a `dict` from
    the text data
    -   contains word counts
-   filter out words that appear in less than 5 emails and keep only the
    50000 most frequent words
    -   this is a way of cleaning the outlier noise
-   create the corpus, which for each email, counts the number of words
    and the count for each word (`doc2bow`)
-   `doc2bow`
    -   Document to Bag of Words
    -   converts text data into bag-of-words format
    -   each row is now a list of words with the associated word count

``` python
from gensim import corpora

 # Create dictionary number of times a word appears
dictionary = corpora.Dictionary(cleaned_emails)

# Filter out (non)frequent words 
dictionary.filter_extremes(no_below=5, keep_n=50000)

# Create corpus
corpus = [dictionary.doc2bow(text) for text in cleaned_emails]
```
:::

::: {.cell .markdown}
#### Latent Dirichlet Allocation (LDA) with gensim

-   Run the LDA model after cleaning the text date, and creating the
    dictionary and corpus
-   Pass the corpus and dictionary into the model
-   As with K-means, beforehand, pick the number of topics to obtain,
    even if there is uncertainty about what topics exist
-   The calculated LDA model, will contain the associated words for each
    topic, and topic scores per email
-   Use `print_topics` to obtain the top words from the topics

``` python
import gensim

# Define the LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 3, 
id2word=dictionary, passes=15)

# Print the three topics from the model with top words
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)

>>> (0, '0.029*"email" + 0.016*"send" + 0.016*"results" + 0.016*"invoice"')
>>> (1, '0.026*"price" + 0.026*"work" + 0.026*"management" + 0.026*"sell"')
>>> (2, '0.029*"distribute" + 0.029*"contact" + 0.016*"supply" + 0.016*"fast"')
```
:::

::: {.cell .markdown}
### Create dictionary and corpus

In order to run an LDA topic model, you first need to **define your
dictionary and corpus** first, as those need to go into the model.
You\'re going to continue working on the cleaned text data that you\'ve
done in the previous exercises. That means that `text_clean` is
available for you already to continue working with, and you\'ll use that
to create your dictionary and corpus.

This exercise will take a little longer to execute than usual.

**Instructions**

-   Import the gensim package and corpora from gensim separately.
-   Define your dictionary by running the correct function on your clean
    data `text_clean`.
-   Define the corpus by running `doc2bow` on each piece of text in
    `text_clean`.
-   Print your results so you can see `dictionary` and `corpus` look
    like.
:::

::: {.cell .code execution_count="133"}
``` python
# Define the dictionary
dictionary = corpora.Dictionary(text_clean)

# Define the corpus 
corpus = [dictionary.doc2bow(text) for text in text_clean]
```
:::

::: {.cell .code execution_count="134"}
``` python
print(dictionary)
```

::: {.output .stream .stdout}
    Dictionary<33980 unique tokens: ['account', 'accurate', 'acquiring', 'acre', 'address']...>
:::
:::

::: {.cell .code execution_count="135"}
``` python
corpus[0][:10]
```

::: {.output .execute_result execution_count="135"}
    [(0, 2),
     (1, 1),
     (2, 1),
     (3, 1),
     (4, 1),
     (5, 6),
     (6, 1),
     (7, 2),
     (8, 4),
     (9, 1)]
:::
:::

::: {.cell .markdown}
**These are the two ingredients you need to run your topic model on the
enron emails. You are now ready for the final step and create your first
fraud detection topic model.**
:::

::: {.cell .markdown}
### LDA model

Now it\'s time to **build the LDA model**. Using the `dictionary` and
`corpus`, you are ready to discover which topics are present in the
Enron emails. With a quick print of words assigned to the topics, you
can do a first exploration about whether there are any obvious topics
that jump out. Be mindful that the topic model is **heavy to calculate**
so it will take a while to run. Let\'s give it a try!

**Instructions**

-   Build the LDA model from gensim models, by inserting the `corpus`
    and `dictionary`.
-   Save the 5 topics by running `print_topics` on the model results,
    and select the top 5 words.
:::

::: {.cell .code execution_count="136"}
``` python
# Define the LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=5)

# Save the topics and top 5 words
topics = ldamodel.print_topics(num_words=5)

# Print the results
for topic in topics:
    print(topic)
```

::: {.output .stream .stdout}
    (0, '0.049*"td" + 0.036*"net" + 0.033*"money" + 0.032*"tr" + 0.028*"width"')
    (1, '0.013*"enron" + 0.010*"pm" + 0.009*"market" + 0.008*"please" + 0.008*"time"')
    (2, '0.015*"message" + 0.015*"enron" + 0.013*"original" + 0.010*"thanks" + 0.009*"e"')
    (3, '0.043*"enron" + 0.010*"company" + 0.006*"hou" + 0.006*"development" + 0.005*"employee"')
    (4, '0.032*"image" + 0.009*"click" + 0.008*"se" + 0.007*"ne" + 0.007*"sp"')
:::
:::

::: {.cell .markdown}
**You have now successfully created your first topic model on the Enron
email data. However, the print of words doesn\'t really give you enough
information to find a topic that might lead you to signs of fraud.
You\'ll therefore need to closely inspect the model results in order to
be able to detect anything that can be related to fraud in your data.
You\'ll learn more about this in the next video.**
:::

::: {.cell .markdown}
## Flagging fraud based on topic
:::

::: {.cell .markdown}
#### Using your LDA model results for fraud detection

1.  Are there any suspicious topics? (no labels)
    1.  if you don\'t have labels, first check for the frequency of
        suspicious words within topics and check whether topics seem to
        describe the fraudulent behavior
    2.  for the Enron email data, a suspicious topic would be one where
        employees are discussing stock bonuses, selling stock, stock
        price, and perhaps mentions of accounting or weak financials
    3.  Defining suspicious topics does require some pre-knowledge about
        the fraudulent behavior
    4.  If the fraudulent topic is noticeable, flag all instances that
        have a high probability for this topic
2.  Are the topics in fraud and non-fraud cases similar? (with labels)
    1.  If there a previous cases of fraud, ran a topic model on the
        fraud text only, and on the non-fraud text
    2.  Check whether the results are similar
        1.  Whether the frequency of the topics are the same in fraud vs
            non-fraud
3.  Are fraud cases associated more with certain topics? (with labels)
    1.  Check whether fraud cases have a higher probability score for
        certain topics
        1.  If so, run a topic model on new data and create a flag
            directly on the instances that score high on those topics
:::

::: {.cell .markdown}
#### To understand topics, you need to visualize

``` python
import pyLDAvis.gensim
lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=False)
```

![topics](vertopal_5c32587947084112be3b965be5a32bb5/4195bb12c6b52ba1ba404c0709285ce6067ff11d.jpg)

-   Each bubble on the left-hand side, represents a topic
-   The larger the bubble, the more prevalent that topic is
-   Click on each topic to get the details per topic in the right panel
-   The words are the most important keywords that form the selected
    topic.
-   A good topic model will have fairly big, non-overlapping bubbles,
    scattered throughout the chart
-   A model with too many topics, will typically have many overlaps, or
    small sized bubbles, clustered in one region
-   In the case of the model above, there is a slight overlap between
    topic 2 and 3, which may point to 1 topic too many
:::

::: {.cell .code execution_count="139"}
``` python
# if ipython is > 7.16.1 results in DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future
import pyLDAvis.gensim
```
:::

::: {.cell .code execution_count="140"}
``` python
lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=False)
```
:::

::: {.cell .code execution_count="141"}
``` python
pyLDAvis.display(lda_display)
```

::: {.output .execute_result execution_count="141"}
```{=html}

<link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.4.0/pyLDAvis/js/ldavis.v1.0.0.css">


<div id="ldavis_el917619086505744489079175291" style="background-color:white;"></div>
<script type="text/javascript">

var ldavis_el917619086505744489079175291_data = {"mdsDat": {"x": [-0.34089686159697874, 0.10868078874691606, 0.1052222383410191, 0.08734000007380616, 0.03965383443523754], "y": [-0.03065746596678076, -0.03752950789052951, -0.1634740277723627, 0.03444518286057542, 0.1972158187690976], "topics": [1, 2, 3, 4, 5], "cluster": [1, 1, 1, 1, 1], "Freq": [3.4006850896708922, 9.417721637058744, 10.286412121273868, 63.452675468864136, 13.44250568313236]}, "tinfo": {"Term": ["image", "enron", "money", "td", "net", "tr", "message", "pm", "original", "width", "class", "market", "e", "please", "thanks", "right", "table", "height", "time", "click", "se", "ne", "sp", "br", "bakernet", "wscc", "corp", "mail", "know", "clear", "bodydefault", "width", "td", "img", "nbsp", "cellspacing", "src", "align", "cellpadding", "coords", "rect", "stocklookup", "height", "bgcolor", "linkbarseperator", "script", "href", "valign", "colspan", "ffffff", "font", "underscore", "hourahead", "tr", "fname", "br", "span", "doubleclick", "detected", "mainheadline", "scoop", "gif", "table", "class", "border", "net", "money", "clear", "start", "center", "image", "right", "schedule", "f", "end", "thru", "simulation", "fri", "pt", "backout", "euci", "kern", "neumin", "brochure", "server", "origination", "impacted", "anc", "aggies", "ubswenergy", "blah", "wyndham", "revert", "memory", "stack", "emailed", "royalty", "auth", "unify", "srrs", "vladimir", "gorny", "emaillink", "itcapps", "dun", "curve", "nerc", "sitara", "leg", "ref", "pager", "sat", "bradford", "scheduled", "outage", "overview", "commoditylogic", "desk", "savita", "system", "ena", "application", "password", "atlanta", "design", "seller", "conference", "request", "london", "pm", "market", "operation", "purpose", "customer", "eol", "service", "id", "please", "contact", "time", "management", "product", "credit", "real", "enron", "question", "know", "process", "group", "let", "hou", "gas", "power", "e", "corp", "trading", "wj", "devon", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "sfs", "hanagriff", "darrell", "tagline", "schoolcraft", "enform", "lindberg", "lorraine", "kowalke", "xll", "breese", "whitt", "estoppel", "hereto", "agave", "exotica", "fran", "beth", "fagan", "pampa", "recipient", "vegetarian", "veggie", "mulligan", "chanley", "headcount", "dietz", "cherry", "kimberly", "barbo", "blair", "tw", "krishna", "watson", "tonight", "privileged", "jerry", "intended", "lynn", "amy", "kim", "fw", "original", "xl", "message", "thanks", "mail", "team", "november", "jason", "confidential", "thursday", "mark", "pm", "file", "copy", "e", "january", "know", "let", "enron", "go", "please", "monday", "contract", "like", "corp", "use", "get", "bankruptcy", "donate", "declared", "mr", "profit", "crisis", "div", "pocketbook", "netted", "ken", "aggressively", "wiped", "astronomical", "underhanded", "hurt", "financially", "bankrupt", "devastated", "repair", "retirement", "afford", "dealing", "partnership", "urging", "buying", "bergfelt", "lost", "writing", "feedback", "pep", "said", "development", "million", "lay", "ee", "stock", "investment", "company", "pseg", "skilling", "sold", "employee", "california", "billion", "accounting", "enron", "fund", "many", "utility", "saving", "plan", "hou", "year", "consumer", "energy", "york", "communication", "business", "state", "power", "made", "corp", "option", "new", "would", "may", "time", "could", "e", "c", "one", "price", "market", "greg", "please", "ctr", "syncrasy", "bakernet", "ecar", "npcc", "std", "frcc", "serc", "temp", "classmate", "enlarge", "sw", "hp", "mapp", "rk", "maac", "nc", "stsw", "stwbom", "matrix", "adult", "apb", "min", "wscc", "mage", "dbcaps", "sp", "engine", "labour", "borland", "delta", "nw", "deviation", "spp", "amazon", "ne", "range", "image", "se", "click", "volatility", "gift", "reg", "choice", "visit", "forecast", "mailto", "see", "email", "day", "standard", "e", "available", "p", "u", "please", "new", "information"], "Freq": [2146.0, 13285.0, 807.0, 738.0, 817.0, 492.0, 1380.0, 1280.0, 999.0, 431.0, 390.0, 1048.0, 1805.0, 1522.0, 793.0, 525.0, 291.0, 275.0, 1382.0, 572.0, 511.0, 469.0, 449.0, 253.0, 418.0, 417.0, 1532.0, 739.0, 869.0, 258.0, 112.83115359355293, 427.3088003798102, 730.3658813199314, 213.30741053292752, 160.225844534755, 93.97883125557627, 226.61601532148495, 218.85446707369104, 93.83584139612604, 46.00110911657186, 46.00110911657186, 46.00038962680521, 268.8960979983693, 52.70117245707717, 37.64654762363052, 218.5308224147732, 192.20976320801753, 59.108368535867264, 48.383385138651846, 29.926342096068954, 94.44345399673273, 25.115774140862165, 34.665491960263566, 473.4196840758281, 21.119296632281827, 240.98227006702135, 16.932061779816205, 16.926765147626263, 17.154193767263813, 16.761921659681157, 134.13743898011705, 205.3921150058713, 259.0719472230216, 304.0693960564636, 157.48627897549426, 541.817035722067, 496.064807196719, 157.97136875564385, 105.81724602269884, 115.29949246679836, 202.71977633509854, 105.43282892718027, 74.44499843115395, 82.3195973817003, 77.59906693471939, 178.34800772279974, 84.57428600764092, 155.83860299574184, 106.58743562647639, 51.84021343265203, 70.40991012087838, 51.53071897147114, 49.82466672366306, 52.047445836146075, 81.09939758992068, 68.08411449901784, 52.929017800461395, 30.585669825010775, 30.334928289240555, 31.586580570857468, 29.960118140185944, 30.583384706909236, 30.76980439982133, 32.0284377776913, 42.1937264085838, 30.455601869916947, 29.8488002440967, 25.690308469314978, 23.936938559061492, 22.66206754304143, 32.55044338608651, 32.25353521032203, 21.37983820492298, 21.379726971660894, 21.362493070380822, 62.24997826492206, 32.43584169065135, 29.213926728286225, 54.733061715938, 67.52435970683098, 73.1027803438498, 137.1909172821072, 56.90061593190997, 197.51446602588592, 216.37285655528316, 54.0786764350891, 39.02780530310675, 190.92295701893644, 56.61556860724395, 264.0837817806797, 74.09776187320445, 104.05708203564065, 73.52932605034613, 50.95026512007879, 60.3107859360116, 85.59943109593588, 221.0897465600772, 196.03454132744005, 127.68593831232685, 422.69850584172076, 356.529803934671, 139.77736145440676, 86.02381984439351, 167.24143421076715, 78.67200730994581, 248.1795725880366, 112.79297488644681, 340.9017047158968, 160.9161499260473, 312.27569897624903, 174.2182358926656, 141.29973856913898, 139.69645134681153, 107.06209213693909, 525.2195145627063, 160.67087574692002, 182.02081396737154, 128.6070851368007, 142.5964088864134, 133.7246180067115, 182.53329945940087, 150.71243262121865, 154.4867996703033, 157.7697284950367, 140.26387639785352, 134.11595596764337, 262.0397259442062, 142.18797094841293, 124.02201122780895, 86.83227959736398, 65.15841405645196, 98.16080642408568, 54.36796572353504, 87.34537798831394, 54.776727681499544, 56.14561513218911, 55.53990706408739, 44.51988371527428, 43.94369854733233, 54.927971137819924, 54.6605482815865, 39.783510421901866, 40.058912226192426, 43.36466884875234, 75.24993189471148, 45.1332669978924, 95.95318653283991, 32.65261018095816, 32.65075608577905, 193.41379998720322, 32.65074816229538, 32.65074816229538, 34.93610006951916, 32.60008540770021, 33.85216486210547, 34.36791762008537, 54.49206332479708, 126.23548393232669, 63.952634158300135, 81.31447121502472, 63.91271564756344, 43.8485507142445, 132.2634273227228, 69.6562746443468, 59.908546197946, 58.78269840307163, 188.5341806736961, 118.61262779602963, 93.93173681393819, 193.47887819975793, 162.72636958527343, 602.4184067156579, 78.0864439672816, 697.8921168399269, 437.3721573224817, 369.52368358332, 272.2689433640304, 198.09738175003406, 108.47982332551948, 78.32907688426832, 208.93520723298371, 249.2735358285269, 391.41147261871595, 179.0885429657525, 145.80793595712487, 418.41094395738793, 185.21536086772323, 258.49167447329626, 200.87303090336266, 688.8472595007294, 165.30428028009766, 257.6513203597486, 158.43910991104292, 166.81361940234711, 178.43099946747225, 195.38490917317907, 141.7001801643095, 139.98365779101937, 458.126360661515, 287.2997284638285, 289.92802600457856, 701.8721284386312, 299.9554140074884, 217.0113179696473, 272.812924925103, 141.59129020502152, 146.17047003350683, 294.689369423914, 148.00588918124078, 144.31931257714564, 141.5596832498074, 140.65803782946676, 161.71123519066415, 146.11374043657094, 155.26019046247893, 143.32993341771837, 145.13346864207392, 344.5286827076117, 146.02621849666872, 168.978537683377, 247.79049680207333, 149.5932100227887, 223.81075606396612, 214.85259551301164, 236.53863254502488, 169.55784103726828, 319.45019296968934, 212.89749353383283, 1265.2597837395858, 1630.2597699875557, 1057.770223950647, 526.2780269375058, 489.7760993109061, 1380.502261575241, 409.8226566338757, 2775.050002357846, 219.20425414069, 269.02144255573785, 218.1220906462398, 1428.0950372019915, 707.8938725122326, 346.4535462801527, 228.32227419915023, 12028.289698404558, 731.1434274159135, 517.884327505288, 446.13810322471204, 327.7639300636644, 706.3725026782655, 1736.8462432472054, 1040.4910599644672, 510.64477070573474, 1392.3696631389623, 436.659275109462, 520.1394348830818, 735.2412528671814, 618.3701370567707, 912.9338606875859, 672.2919936659679, 1194.0186190526379, 632.3546832680006, 1115.747285498308, 993.9927321848634, 790.49516342396, 915.6403935886134, 567.5842769256343, 896.1566281840026, 559.7353249957565, 625.0345124404064, 592.3723411049756, 643.2001048901382, 571.370399146409, 675.0844844513819, 251.60397200977638, 251.60397200977638, 415.9341618062955, 139.8574581823038, 139.8574581823038, 139.8574581823038, 125.88914395386972, 126.54885493662482, 130.892697769502, 128.67598934427266, 140.8559572072033, 287.92449656968876, 287.2079872656744, 140.4612406284956, 280.3124998690771, 126.03347298898025, 274.81267175689965, 55.95996271273697, 55.95996271273697, 272.0998547084005, 87.99449547663572, 45.140993904395884, 142.47566198442712, 409.0298385016754, 42.95980513911302, 43.22193822763273, 440.1467863780885, 38.67954978026111, 35.06802424309543, 31.54851295279726, 162.86236599454406, 285.7099695092525, 137.21883514898394, 130.5358099220171, 253.6541787932004, 443.58406504782255, 294.3989253367852, 1895.774771080163, 466.2817455768875, 509.3934109820899, 245.96373579667352, 175.05941201095007, 156.45565617877887, 150.93858423985057, 187.80566493540059, 156.0284262197684, 239.5518486910512, 311.5393995759494, 273.07536656633437, 308.3015639176183, 179.1040903267228, 303.809692013146, 210.71346059602735, 206.09705761285818, 209.3878811071137, 222.95862089283898, 216.2054054215157, 196.4481770213882], "Total": [2146.0, 13285.0, 807.0, 738.0, 817.0, 492.0, 1380.0, 1280.0, 999.0, 431.0, 390.0, 1048.0, 1805.0, 1522.0, 793.0, 525.0, 291.0, 275.0, 1382.0, 572.0, 511.0, 469.0, 449.0, 253.0, 418.0, 417.0, 1532.0, 739.0, 869.0, 258.0, 113.97962740902567, 431.7002658624023, 738.6813831476643, 215.91448061794216, 162.58838059417727, 95.45444402413533, 230.22903887737317, 222.9549518102618, 95.62193852963405, 47.03433999545619, 47.03433999545619, 47.03830304472369, 275.2134678012004, 54.08631821186519, 38.67064093331497, 224.7809773290428, 197.79946375583216, 60.895964659079056, 49.85422312782214, 31.055345876279535, 98.09359989714758, 26.11913740513636, 36.071944396535145, 492.79991093210697, 22.089988300891825, 253.77451854514715, 17.896821838680697, 17.917077759408947, 18.160685423002622, 17.751464865880383, 143.30235939227333, 224.99701153074489, 291.6807384470986, 390.4329705598888, 197.8885242028952, 817.6363456362141, 807.1178423472518, 258.0919365695822, 254.95826124090442, 299.28616525028, 2146.7773602059656, 525.1958515902402, 192.16279648319053, 394.4034265383991, 470.8767004817486, 179.5560414625599, 85.52766787703078, 157.87048797919957, 108.28247537334785, 52.73971783852363, 71.65886700712042, 52.48789350592465, 50.758359257836965, 53.100781787183664, 82.93064705884376, 69.70849259353919, 54.31908087911055, 31.47638227896621, 31.220582003991538, 32.51075998385671, 30.851286316932637, 31.498235742258515, 31.74177490431558, 33.07183871208869, 43.620312951683694, 31.486063807908096, 30.875463932836237, 26.633267328407566, 24.838049644877618, 23.549253217845266, 33.850778675221406, 33.58223785950067, 22.265316934073553, 22.265279323756022, 22.25771505547125, 65.1372122437185, 33.877209813361425, 30.483809839614498, 57.80230703262561, 71.82959890453135, 78.47791910489379, 151.46801867698406, 61.03881948521294, 227.0607768581768, 261.93412282451595, 58.987366561179435, 41.25825621105667, 250.11576658111832, 63.99044244573835, 385.67344700843495, 88.40033034813577, 140.55928675095197, 91.98239846084581, 58.5239488040659, 72.89843354242205, 116.39610825829257, 418.17891533522845, 362.1227862814757, 207.80442759818084, 1280.9603757847062, 1048.945236649669, 253.75230760104031, 122.49095821233426, 377.0552429655274, 106.3431400416996, 755.3809243665903, 196.6244275595391, 1522.6685989269058, 395.6924521750623, 1382.9929115324371, 462.4955642831527, 319.65446867318917, 394.98229369834013, 214.02472082441275, 13285.032734576405, 607.4261284715224, 869.661829547489, 402.9948121332382, 610.5347783848268, 526.4415138602861, 1971.6917040783076, 885.7123873695373, 1109.6706246865378, 1805.3650319970607, 1532.6460443905544, 633.3650245613981, 262.9899545433666, 143.09014493701056, 125.39703252429116, 87.79586099202307, 66.09883479882609, 99.68222723007278, 55.23934407175012, 88.77976518738313, 55.734434795888816, 57.1427530803947, 56.597840883042586, 45.397292862476725, 44.82730912111964, 56.04643365376313, 55.84215802321106, 40.683212539399285, 40.969126580803625, 44.37997031200062, 77.15985189049402, 46.306472706424195, 98.48182070324614, 33.55061118896055, 33.55453303408158, 198.8122826659812, 33.5630090760766, 33.5630090760766, 35.94059911257431, 33.54362717383564, 34.839545944328904, 35.38166336374383, 56.34124797013818, 134.20034293006162, 67.10350643160037, 85.96384249210074, 67.1844203933958, 45.58745179184227, 144.04079362166542, 74.02913061764286, 63.40157877188305, 62.20875925754477, 224.22184455040914, 137.93935234297135, 107.48846198172657, 252.36773455570454, 208.40104732855357, 999.7563701253376, 87.96943259696687, 1380.0091288062988, 793.3200963004311, 739.913637964349, 508.660776084936, 328.6308101481147, 148.73353288827016, 96.0605110805966, 451.54130445466063, 611.1155262278758, 1280.9603757847062, 361.0308153235958, 259.6166933625374, 1805.3650319970607, 432.4465933195926, 869.661829547489, 526.4415138602861, 13285.032734576405, 471.7851643248372, 1522.6685989269058, 422.367674313392, 544.892041183797, 753.259125282778, 1532.6460443905544, 388.3958897458674, 679.8101654363834, 459.0632464537264, 287.92726555976753, 290.60247387966695, 703.9057294563006, 301.0231946586681, 217.84350584507075, 273.8849773771548, 142.1589305539329, 146.75813402084256, 295.87441542271876, 148.60200184805882, 144.90226719198802, 142.13177690642058, 141.2297603766055, 162.37406953094342, 146.71642828772875, 155.90304928277592, 143.9279160719434, 145.74952952733406, 345.99644766881096, 146.65814996253576, 169.71313779977717, 248.8682495379633, 150.2856447459257, 224.8537039766049, 215.85412545144254, 237.68137084275827, 170.39383781549898, 321.0769681729411, 213.98398289517067, 1273.100427826103, 1645.082856073045, 1069.0225489838444, 529.5540803788938, 493.1079621324696, 1405.9351680884542, 413.1698871451633, 2858.667346729363, 220.32617705984111, 270.9904251602683, 219.55949370367594, 1484.7384981273726, 728.2573841158224, 352.35569390727983, 230.29115474404432, 13285.032734576405, 756.728957216541, 533.1091169684114, 460.53801572243117, 334.7161962390117, 745.5064108487567, 1971.6917040783076, 1159.2754413648443, 539.4575269626246, 1596.702252463682, 457.5844567007534, 557.0632577259221, 845.5537230775853, 700.040365961601, 1109.6706246865378, 774.5279597609989, 1532.6460443905544, 722.0351012448218, 1491.2195228658559, 1397.0725907139308, 1056.8443785901713, 1382.9929115324371, 684.7442073530589, 1805.3650319970607, 667.6546146456155, 865.1436032666562, 769.0215011781744, 1048.945236649669, 719.5711358200922, 1522.6685989269058, 252.78533987266607, 252.78533987266607, 418.2483092674617, 140.88138213280362, 140.88138213280362, 140.88138213280362, 126.8933895622971, 127.56504158254026, 132.2729678818932, 130.06732849071415, 142.52754364178105, 291.4728008813333, 290.7490132941711, 142.27374264311834, 284.33849605129643, 127.89104269868612, 279.0852020460965, 56.842037116095426, 56.842037116095426, 277.1851182137521, 89.64820307350561, 46.026596428441735, 145.29355393779764, 417.4411574615564, 43.86544792941266, 44.16129389323138, 449.7705148115246, 39.715884742021125, 36.02729582997189, 32.415303340185226, 167.37315354119278, 296.0815903648782, 141.17550530502743, 134.30499423234082, 265.22136126869566, 469.62997453985395, 311.497196963733, 2146.7773602059656, 511.17942592944866, 572.9060027767342, 272.7368726329447, 186.31086425636394, 171.15272628339383, 167.13296562777572, 236.7591444885971, 178.10111167298257, 392.45939538532707, 663.0187072137489, 582.8032899898387, 871.6152470023994, 267.7718985001423, 1805.3650319970607, 549.8952932214238, 641.4644703061416, 783.4156702424488, 1522.6685989269058, 1491.2195228658559, 785.0592152691281], "Category": ["Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5"], "logprob": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, -4.89, -3.5584, -3.0223, -4.2531, -4.5393, -5.0728, -4.1926, -4.2275, -5.0743, -5.7872, -5.7872, -5.7872, -4.0215, -5.6512, -5.9876, -4.2289, -4.3573, -5.5365, -5.7367, -6.2171, -5.0679, -6.3924, -6.0701, -3.4559, -6.5657, -4.1311, -6.7867, -6.787, -6.7736, -6.7968, -4.717, -4.2909, -4.0588, -3.8986, -4.5565, -3.3209, -3.4092, -4.5535, -4.9542, -4.8683, -4.304, -4.9578, -5.3058, -5.2053, -5.2643, -5.4507, -6.1969, -5.5857, -5.9655, -6.6863, -6.3801, -6.6923, -6.726, -6.6823, -6.2388, -6.4137, -6.6655, -7.214, -7.2222, -7.1817, -7.2346, -7.214, -7.2079, -7.1679, -6.8922, -7.2182, -7.2383, -7.3884, -7.4591, -7.5138, -7.1517, -7.1609, -7.572, -7.572, -7.5728, -6.5033, -7.1552, -7.2598, -6.632, -6.422, -6.3426, -5.7131, -6.5932, -5.3487, -5.2575, -6.644, -6.9702, -5.3826, -6.5982, -5.0582, -6.3291, -5.9895, -6.3368, -6.7036, -6.535, -6.1848, -5.2359, -5.3562, -5.7849, -4.5878, -4.7581, -5.6944, -6.1799, -5.515, -6.2692, -5.1203, -5.9089, -4.8029, -5.5536, -4.8906, -5.4742, -5.6836, -5.695, -5.9611, -4.3707, -5.5551, -5.4304, -5.7777, -5.6745, -5.7387, -5.4275, -5.6191, -5.5944, -5.5733, -5.691, -5.7358, -5.1542, -5.7656, -5.9023, -6.2587, -6.5459, -6.1361, -6.7269, -6.2528, -6.7194, -6.6948, -6.7056, -6.9268, -6.9398, -6.7167, -6.7216, -7.0393, -7.0324, -6.9531, -6.4019, -6.9131, -6.1589, -7.2368, -7.2368, -5.4579, -7.2368, -7.2368, -7.1692, -7.2384, -7.2007, -7.1856, -6.7247, -5.8846, -6.5646, -6.3244, -6.5652, -6.942, -5.8379, -6.4791, -6.6299, -6.6489, -5.4834, -5.9468, -6.1801, -5.4575, -5.6306, -4.3218, -6.3649, -4.1746, -4.6419, -4.8105, -5.1159, -5.434, -6.0361, -6.3618, -5.3807, -5.2042, -4.753, -5.5348, -5.7404, -4.6862, -5.5012, -5.1678, -5.42, -4.1877, -5.6149, -5.1711, -5.6573, -5.6058, -5.5385, -5.4477, -5.769, -5.7812, -6.415, -6.8817, -6.8726, -5.9884, -6.8385, -7.1622, -6.9334, -7.5892, -7.5574, -6.8563, -7.5449, -7.5702, -7.5895, -7.5959, -7.4564, -7.5578, -7.4971, -7.577, -7.5645, -6.7, -7.5584, -7.4124, -7.0296, -7.5343, -7.1314, -7.1722, -7.0761, -7.409, -6.7756, -7.1814, -5.3992, -5.1457, -5.5783, -6.2764, -6.3482, -5.312, -6.5265, -4.6138, -7.1522, -6.9474, -7.1571, -5.2781, -5.9799, -6.6944, -7.1114, -3.1472, -5.9476, -6.2924, -6.4416, -6.7499, -5.982, -5.0824, -5.5947, -6.3065, -5.3034, -6.463, -6.2881, -5.942, -6.1151, -5.7255, -6.0315, -5.4571, -6.0927, -5.5249, -5.6405, -5.8695, -5.7226, -6.2008, -5.7441, -6.2147, -6.1044, -6.158, -6.0757, -6.1941, -6.0273, -5.4625, -5.4625, -4.9598, -6.0497, -6.0497, -6.0497, -6.1549, -6.1497, -6.1159, -6.133, -6.0426, -5.3276, -5.3301, -6.0454, -5.3544, -6.1538, -5.3742, -6.9657, -6.9657, -5.3841, -6.513, -7.1805, -6.0311, -4.9765, -7.23, -7.224, -4.9032, -7.335, -7.433, -7.5388, -5.8974, -5.3353, -6.0687, -6.1187, -5.4543, -4.8954, -5.3054, -3.4429, -4.8455, -4.7571, -5.4851, -5.8252, -5.9375, -5.9734, -5.7549, -5.9403, -5.5115, -5.2488, -5.3806, -5.2592, -5.8023, -5.2739, -5.6398, -5.662, -5.6461, -5.5833, -5.6141, -5.7099], "loglift": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 3.3711, 3.371, 3.3699, 3.369, 3.3666, 3.3656, 3.3654, 3.3626, 3.3623, 3.359, 3.359, 3.3589, 3.358, 3.3552, 3.3544, 3.353, 3.3525, 3.3514, 3.3512, 3.3442, 3.3433, 3.342, 3.3414, 3.3411, 3.3363, 3.3295, 3.3258, 3.3243, 3.3242, 3.3238, 3.3151, 3.29, 3.2626, 3.1312, 3.1528, 2.9697, 2.8944, 2.8903, 2.5018, 2.4273, 1.0213, 1.7755, 2.4329, 1.8144, 1.5782, 2.3558, 2.3514, 2.3496, 2.3468, 2.3454, 2.345, 2.3442, 2.344, 2.3425, 2.3402, 2.339, 2.3367, 2.3339, 2.3338, 2.3337, 2.3333, 2.3331, 2.3315, 2.3305, 2.3293, 2.3293, 2.3288, 2.3265, 2.3256, 2.3242, 2.3234, 2.3222, 2.322, 2.322, 2.3215, 2.3172, 2.3191, 2.32, 2.308, 2.3008, 2.2916, 2.2636, 2.2924, 2.2232, 2.1715, 2.2757, 2.307, 2.0925, 2.2401, 1.9839, 2.1861, 2.0619, 2.1387, 2.224, 2.173, 2.0553, 1.7252, 1.7489, 1.8756, 1.2539, 1.2835, 1.7663, 2.0092, 1.5496, 2.0612, 1.2495, 1.8068, 0.866, 1.4628, 0.8745, 1.3862, 1.5462, 1.3232, 1.6699, -0.868, 1.0327, 0.7986, 1.2204, 0.9083, 0.9922, -0.0171, 0.5916, 0.3909, -0.0748, -0.0286, 0.8102, 2.2707, 2.268, 2.2633, 2.2633, 2.26, 2.259, 2.2584, 2.2581, 2.257, 2.2567, 2.2555, 2.2548, 2.2544, 2.2542, 2.253, 2.252, 2.2519, 2.2512, 2.2493, 2.2487, 2.2483, 2.2472, 2.247, 2.2468, 2.2468, 2.2468, 2.246, 2.2458, 2.2456, 2.2453, 2.241, 2.2132, 2.2263, 2.2187, 2.2244, 2.2355, 2.189, 2.2135, 2.2177, 2.2177, 2.101, 2.1234, 2.1395, 2.0086, 2.027, 1.7678, 2.1552, 1.5926, 1.6789, 1.58, 1.6494, 1.7682, 1.9588, 2.0703, 1.5037, 1.3776, 1.0887, 1.5733, 1.6974, 0.8123, 1.4264, 1.0611, 1.3109, -0.685, 1.2256, 0.4977, 1.2938, 1.0906, 0.8341, 0.2146, 1.266, 0.6941, 0.4528, 0.4527, 0.4526, 0.452, 0.4513, 0.451, 0.451, 0.4509, 0.4509, 0.4509, 0.4509, 0.4508, 0.4508, 0.4508, 0.4508, 0.4508, 0.4507, 0.4507, 0.4506, 0.4506, 0.4506, 0.4505, 0.4505, 0.4503, 0.4502, 0.4502, 0.4501, 0.45, 0.4498, 0.4498, 0.4487, 0.4458, 0.4443, 0.4487, 0.4481, 0.4366, 0.4467, 0.4252, 0.4498, 0.4476, 0.4483, 0.416, 0.4265, 0.438, 0.4463, 0.3555, 0.4205, 0.4259, 0.4231, 0.4339, 0.401, 0.3281, 0.3468, 0.4, 0.3179, 0.4081, 0.3863, 0.3151, 0.3308, 0.2597, 0.3133, 0.2052, 0.3223, 0.1648, 0.1145, 0.1645, 0.0425, 0.2672, -0.2455, 0.2786, 0.1298, 0.1939, -0.0342, 0.2243, -0.3585, 2.0021, 2.0021, 2.0012, 1.9995, 1.9995, 1.9995, 1.9988, 1.9988, 1.9963, 1.996, 1.995, 1.9945, 1.9945, 1.9939, 1.9925, 1.9921, 1.9913, 1.9911, 1.9911, 1.9882, 1.9881, 1.9873, 1.9872, 1.9864, 1.9859, 1.9852, 1.9851, 1.9803, 1.9798, 1.9796, 1.9794, 1.9711, 1.9783, 1.9783, 1.9622, 1.9497, 1.9503, 1.8824, 1.9148, 1.8892, 1.9034, 1.9445, 1.917, 1.9048, 1.7751, 1.8744, 1.5131, 1.2515, 1.2486, 0.9675, 1.6046, 0.2246, 1.0475, 0.8713, 0.6873, 0.0855, 0.0756, 0.6214]}, "token.table": {"Topic": [3, 4, 2, 4, 5, 4, 5, 4, 3, 2, 4, 1, 4, 1, 2, 4, 5, 3, 4, 2, 4, 5, 2, 3, 4, 5, 4, 2, 3, 4, 2, 4, 2, 3, 4, 5, 2, 4, 5, 4, 4, 3, 5, 2, 4, 3, 4, 1, 4, 2, 4, 5, 2, 3, 4, 5, 1, 4, 1, 4, 5, 4, 5, 1, 4, 5, 2, 4, 3, 4, 2, 4, 2, 3, 4, 5, 4, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 4, 1, 4, 1, 2, 3, 4, 5, 3, 3, 4, 4, 5, 1, 2, 4, 5, 2, 5, 1, 2, 4, 1, 2, 3, 4, 5, 1, 4, 2, 3, 4, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5, 1, 2, 4, 5, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 4, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 3, 5, 1, 2, 3, 4, 5, 5, 4, 4, 4, 5, 2, 4, 5, 2, 4, 5, 1, 4, 4, 2, 3, 4, 5, 3, 5, 3, 3, 4, 1, 4, 4, 1, 4, 2, 1, 2, 3, 4, 5, 4, 5, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 2, 4, 5, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 3, 4, 4, 5, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 2, 4, 3, 5, 1, 2, 3, 4, 5, 3, 3, 4, 1, 4, 1, 2, 3, 4, 5, 4, 1, 1, 4, 3, 4, 5, 3, 4, 4, 5, 2, 3, 4, 3, 4, 5, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 4, 5, 2, 4, 5, 1, 2, 3, 4, 5, 2, 3, 2, 3, 4, 5, 2, 3, 4, 5, 3, 3, 4, 1, 4, 3, 4, 1, 2, 3, 4, 5, 1, 4, 5, 3, 4, 5, 1, 4, 4, 1, 2, 3, 4, 5, 1, 3, 4, 5, 1, 4, 2, 4, 2, 3, 4, 5, 3, 4, 5, 4, 5, 2, 2, 3, 4, 5, 2, 3, 4, 3, 4, 4, 2, 4, 2, 3, 4, 5, 3, 4, 2, 3, 4, 5, 3, 3, 5, 4, 5, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 1, 2, 3, 4, 5, 3, 1, 4, 2, 3, 4, 5, 3, 4, 4, 5, 2, 3, 4, 5, 3, 4, 5, 2, 3, 4, 5, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 4, 2, 3, 4, 5, 1, 2, 3, 4, 5, 3, 4, 5, 2, 3, 4, 5, 1, 2, 3, 4, 5, 4, 5, 2, 3, 4, 5, 2, 4, 1, 2, 3, 4, 5, 1, 2, 4, 5, 4, 5, 2, 3, 4, 5, 1, 2, 4, 5, 3, 4, 5, 3, 1, 4, 3, 4, 5, 3, 4, 5, 2, 4, 1, 2, 3, 4, 5, 4, 2, 1, 2, 3, 4, 5, 2, 3, 4, 5, 4, 5, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 2, 4, 2, 3, 4, 2, 3, 4, 1, 2, 3, 4, 5, 2, 3, 4, 3, 4, 4, 2, 4, 5, 3, 4, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 4, 2, 4, 5, 1, 2, 3, 4, 5, 3, 4, 2, 3, 4, 5, 1, 2, 3, 4, 5, 4, 5, 2, 4, 2, 4, 2, 3, 4, 5, 1, 2, 3, 4, 5, 4, 5, 1, 2, 4, 5, 3, 4, 5, 1, 4, 2, 4, 3, 4, 5, 4, 1, 2, 3, 4, 5, 4, 5, 2, 1, 2, 3, 4, 5, 4, 5, 2, 4, 2, 3, 4, 5, 2, 3, 4, 1, 4, 5, 2, 3, 4, 1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 5, 1, 4, 1, 4, 1, 3, 4, 5, 2, 3, 4, 5, 2, 4, 5, 4, 5, 2, 3, 4, 1, 2, 3, 4, 5, 3, 2, 2, 4, 3, 4, 4, 5, 3, 4, 5, 1, 4, 5, 1, 4, 2, 2, 3, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 4, 5, 1, 2, 3, 4, 5, 1, 4, 5, 5, 3, 4, 5, 4, 5, 2, 3, 4, 5, 1, 4, 5, 3, 1, 4, 2, 3, 4, 5, 4, 5, 2, 3, 4, 5, 2, 4, 2, 3, 4, 5, 1, 2, 3, 4, 5, 3, 4, 1, 4, 1, 2, 3, 4, 5, 3, 4, 5, 1, 2, 3, 4, 5, 2, 4, 1, 4, 2, 4, 1, 2, 3, 4, 5, 2, 4, 5, 1, 4, 3, 4, 3, 4, 2, 3, 4, 5, 2, 4, 3, 4, 5, 3, 4, 5, 3, 4, 1, 4, 4, 3, 1, 2, 3, 4, 5, 4, 3, 4, 5, 2, 4, 1, 3, 4, 5, 3, 2, 3, 4, 5, 2, 4, 5], "Freq": [0.9888591261199061, 0.007974670371934726, 0.00434232917504558, 0.9900510519103922, 0.00434232917504558, 0.011154713264916932, 0.98161476731269, 0.9955123533011708, 0.9689055602719169, 0.9609045723799932, 0.9959488981267268, 0.9822612066780758, 0.017940843957590425, 0.015081741458779413, 0.0075408707293897065, 0.018852176823474266, 0.9576905826324927, 0.8745124664262136, 0.12094321344192316, 0.9848654056001681, 0.02172656849729733, 0.9776955823783798, 0.7399013071563956, 0.007114435645734573, 0.21343306937203718, 0.03557217822867287, 0.9990728540141495, 0.8714381213534385, 0.034174043974644644, 0.08543510993661162, 0.9762226946998684, 0.03754702671922571, 0.08728934506568337, 0.07819670495467469, 0.449176421483829, 0.3837094126845665, 0.985974179065795, 0.002390924189870471, 0.9946244629861158, 0.9942076227057113, 0.997683878067042, 0.9537504581111, 0.029804701815971876, 0.004632758340423542, 0.9960430431910615, 0.974799199633762, 0.02030831665903671, 0.979915101493692, 0.018488964179126263, 0.0028380412670815066, 0.9819622784102013, 0.014190206335407534, 0.9724067804438543, 0.942256623852559, 0.02326559565068047, 0.03489839347602071, 0.9914052411707736, 0.008773497709475872, 0.7933759708017621, 0.09096030238491541, 0.11117370291489662, 0.030849626471343268, 0.9871880470829846, 0.9496619336788358, 0.04334556543762321, 0.003940505948874837, 0.9338319528576176, 0.04914905015040093, 0.9813291660941772, 0.017842348474439586, 0.9792699513992966, 0.018832114449986472, 0.07569004576912916, 0.015374540546854361, 0.8692528693798427, 0.039027679849707224, 0.996203291466821, 0.05392009462723244, 0.029955608126240246, 0.8387570275347268, 0.07638680072191262, 0.021970254403153915, 0.0013731409001971197, 0.0013731409001971197, 0.9721837573395608, 0.004119422700591359, 0.9830380082795394, 0.010457851151909992, 0.9847629511752478, 0.010476201608247316, 0.3842476310384427, 0.1904531736451412, 0.0033412837481603718, 0.41766046852004646, 0.0066825674963207436, 0.9837934290463474, 0.958445223446611, 0.017748985619381683, 0.0957321611562487, 0.9034722709120971, 0.7786227673448218, 0.0307351092372956, 0.12550169605229036, 0.0614702184745912, 0.007688325820203131, 0.991794030806204, 0.6121849527732256, 0.05036964801298691, 0.3332145945474519, 0.04712814993932277, 0.02269133145226652, 0.0034909740695794646, 0.038400714765374114, 0.8884529007079738, 0.9628071001514142, 0.020058481253154462, 0.9452653500549185, 0.024237573078331246, 0.024237573078331246, 0.0017951282661905607, 0.008975641330952803, 0.9334666984190916, 0.05564897625190738, 0.0013992533984678035, 0.013642720635061084, 0.0006996266992339017, 0.9707320451870387, 0.013292907285444133, 0.5284819293742675, 0.007173962842184627, 0.4591336218998161, 0.004782641894789752, 0.8119881845575079, 0.16656167888359136, 0.02082020986044892, 0.03151313894110855, 0.02039085460895259, 0.9472478822886158, 0.0018537140553593265, 0.40688165547512234, 0.2350310183800396, 0.3411740589387672, 0.015163291508389652, 0.0018352259244371876, 0.018352259244371876, 0.30648272938101034, 0.6184711365353323, 0.05689200365755282, 0.9780088336403547, 0.021261061600877276, 0.04237015677816693, 0.11170314059698556, 0.5623675354193066, 0.23496177849710756, 0.05007382164692456, 0.0006524663692964039, 0.09134529170149654, 0.12723094201279875, 0.7790448449399062, 0.0013049327385928078, 0.005841597427252966, 0.04527238006121048, 0.055495175558903174, 0.8295068346699211, 0.06425757169978262, 0.3544462681836625, 0.03797638587682098, 0.605090414970681, 0.002531759058454732, 0.9961279275147608, 0.003955925610653385, 0.996893253884653, 0.951836866582803, 0.01535220752552908, 0.01535220752552908, 0.01535220752552908, 0.4429059219188954, 0.03182557522770506, 0.2678652581665176, 0.25460460182164046, 0.9831241006865737, 0.010031878578434426, 0.01950401861195666, 0.15259026325824918, 0.07457418881042252, 0.3992587339388775, 0.35336692544015597, 0.9737033544343371, 0.9957979811756323, 0.997926810905552, 0.023898695312659844, 0.9738718339908886, 0.8230629532674935, 0.15089487476570712, 0.013717715887791557, 0.7636463810771172, 0.20790372678539318, 0.023988891552160753, 0.9360880167258181, 0.05506400098387165, 0.9935529110872448, 0.003039360589979907, 0.005470849061963832, 0.9908315523334497, 0.0006078721179959813, 0.021250145296226302, 0.9704233018610011, 0.9923814114697385, 0.9609497340602803, 0.028263227472361187, 0.0036511677623813025, 0.9967687991300955, 0.996779514583432, 0.9488154390060983, 0.05581267288271166, 0.9434930740942301, 0.016063233465822006, 0.08751692715861645, 0.23153212374874477, 0.496298523633673, 0.1683869990899962, 0.007098169998483812, 0.9937437997877336, 0.0040559069282736825, 0.0020279534641368412, 0.9936971974270522, 0.0020279534641368412, 0.029169361758915942, 0.17844786017219164, 0.0978031541328358, 0.22820735964328354, 0.46842563295200307, 0.9528024901119951, 0.9431709443966106, 0.9617855277552686, 0.037717079519814456, 0.8371009441771905, 0.03393652476394016, 0.10180957429182047, 0.022624349842626772, 0.1656484593954194, 0.08494792789508687, 0.061587247723937984, 0.6052539862524939, 0.08494792789508687, 0.0281830879430187, 0.026930506256662316, 0.015657271079454835, 0.8717968537040451, 0.056992466729215596, 0.9868226026050417, 0.017942229138273487, 0.02517884233211999, 0.9819748509526797, 0.007016187709747748, 0.9892824670744325, 0.0003763633932934698, 0.03951815629581433, 0.05186287559584014, 0.9053797789067709, 0.0027850891103716763, 0.7428781957070506, 0.15045634343433936, 0.1034387361111083, 0.00940352146464621, 0.9832065243437293, 0.024580163108593234, 0.9768504990881925, 0.013955007129831323, 0.9720080866204965, 0.01296010782160662, 0.2079089441988316, 0.06845782308985919, 0.09381257238239962, 0.5070949858508088, 0.12170279660419411, 0.9835886390903744, 0.0031145180100908758, 0.9935312452189894, 0.9660172557573856, 0.032200575191912856, 0.04431754664947098, 0.013849233327959683, 0.4958025531409566, 0.41270715317319856, 0.033238159987103236, 0.9951169184249514, 0.950656909091807, 0.9582684303416351, 0.03058303501090325, 0.005614788086422131, 0.11791054981486475, 0.8759069414818524, 0.9717863911875338, 0.021595253137500752, 0.007880631161712798, 0.9929595263758125, 0.9881517565243352, 0.0063343061315662514, 0.0063343061315662514, 0.030393973668723318, 0.965999771818989, 0.0026429542320628974, 0.023992201882350794, 0.7821457813646359, 0.19193761505880635, 0.004798440376470159, 0.0033871040337480783, 0.17048423636531992, 0.1174196065032667, 0.521614021197204, 0.18741975653406032, 0.013238990026315251, 0.1029699224268964, 0.2059398448537928, 0.5810445622660582, 0.09855692575145797, 0.9111232127275949, 0.08000106258095956, 0.004444503476719975, 0.010734747047536627, 0.04293898819014651, 0.9392903666594548, 0.019076479466834717, 0.016956870637186417, 0.34973545689196983, 0.47267276901157135, 0.13989418275678794, 0.9528846807017346, 0.029777646271929206, 0.1375819499585265, 0.04725036665242324, 0.793528216427461, 0.020845749993716137, 0.23422089136069743, 0.03930979994864852, 0.6436979741591194, 0.0835333248908781, 0.9833758824619159, 0.9759025004036953, 0.028703014717755746, 0.9774230968751548, 0.02180125866636033, 0.9763449538302407, 0.024408623845756015, 0.00050717868210916, 0.09281369882597629, 0.024851755423348842, 0.880969370823611, 0.00101435736421832, 0.9702831545549259, 0.027722375844426453, 0.027722375844426453, 0.0034393925835553224, 0.010318177750665968, 0.9871056714803775, 0.9706800835264592, 0.025278127175168207, 0.9976962483478796, 0.050858380741995744, 0.5746997023845519, 0.025429190370997872, 0.21360519911638212, 0.13223178992918894, 0.0945603413576729, 0.0004658144894466645, 0.02189328100399323, 0.8831842719908759, 0.9865016898838791, 0.00926292666557633, 0.9757160677654649, 0.01840973712765028, 0.1222837693422782, 0.028023363807605423, 0.5986809540715704, 0.2496626957404847, 0.8429151957917702, 0.14271580034569656, 0.013379606282409051, 0.9923278843793145, 0.007260935739360838, 0.9431725375928238, 0.0023124242749230457, 0.42779849086076344, 0.4740469763592243, 0.0971218195467679, 0.2151498682145785, 0.7261308052242024, 0.05378746705364462, 0.9484194943631574, 0.04822472005236394, 0.9970446399650018, 0.9907046468559539, 0.019052012439537577, 0.10698673523988207, 0.7647570333813792, 0.031699773404409506, 0.0950993202132285, 0.9388947691859832, 0.0521608205103324, 0.20927674852039904, 0.2966670391113349, 0.4208532415300333, 0.07244195141090737, 0.9912485340550976, 0.9651778783536583, 0.021935860871674054, 0.027756732137749797, 0.9714856248212429, 0.993288541226326, 0.00377676251416854, 0.9515191144352786, 0.017300347535186885, 0.017300347535186885, 0.03460069507037377, 0.2545391966097162, 0.38180879491457426, 0.3362196850740281, 0.026593647406985273, 0.022568594829326626, 0.10620515213800764, 0.23630646350706702, 0.5535943555193649, 0.07965386410350574, 0.980001784674481, 0.9826576204291145, 0.025859411063924066, 0.6159637765154169, 0.0048122170040266945, 0.3753529263140822, 0.0048122170040266945, 0.989437037284196, 0.01766851852293207, 0.9971332593701294, 0.004207313330675652, 0.0797451910072021, 0.8626979754415499, 0.057996502550692434, 0.007249562818836554, 0.007819155891597663, 0.007819155891597663, 0.9852136423413057, 0.046479923089037034, 0.0774665384817284, 0.8676252309953579, 0.006455544873477366, 0.022796985946870497, 0.9802703957154314, 0.013515090798315338, 0.06892696307140822, 0.5000583595376675, 0.21894447093270847, 0.19867183473523548, 0.022932308681675364, 0.0025480342979639294, 0.3159562529475272, 0.04841265166131466, 0.6115282315113431, 0.9576674448245253, 0.05633337910732501, 0.3762198244424077, 0.004324365798188594, 0.6010868459482146, 0.017297463192754377, 0.009378942960932833, 0.0018757885921865665, 0.003751577184373133, 0.9716584907526414, 0.015006308737492532, 0.007028703831236207, 0.007028703831236207, 0.984018536373069, 0.014727166327376267, 0.40745160172407674, 0.5563596168119923, 0.02127257358398794, 0.016206756469287185, 0.3403418858550309, 0.002860015847521268, 0.6129967299853918, 0.027646819859372257, 0.014430789162769513, 0.9812936630683269, 0.06907355659816432, 0.11638421180238645, 0.7475083522267097, 0.066234917285911, 0.9675905920617318, 0.03023720600192912, 0.018115822191426285, 0.0550720994619359, 0.5057937555846218, 0.3521715834013269, 0.06884012432741987, 0.003741735853749938, 0.0009354339634374845, 0.9896891333168586, 0.005612603780624907, 0.013765235592325247, 0.9773317270550925, 0.08996900641549674, 0.3740816582539075, 0.45458024294145716, 0.08049858468754971, 0.614532319788072, 0.01238976451185629, 0.3716929353556887, 0.001238976451185629, 0.0014206447797667505, 0.9972926353962588, 0.0014206447797667505, 0.9738290641837066, 0.9840801625262638, 0.012301002031578297, 0.003583135159688008, 0.010749405479064023, 0.9853621689142021, 0.002129335975583342, 0.053233399389583554, 0.9454251731590039, 0.9445878269283842, 0.029518369591512005, 0.6628863832835885, 0.0317989777958917, 0.037914165833563176, 0.2164776565335704, 0.051367579516440436, 0.9948341260544039, 0.9850594213657551, 0.0033529603947183434, 0.08248282571007125, 0.02078835444725373, 0.7483807601011343, 0.14484788905183243, 0.042600996521568284, 0.6024998079478943, 0.3438509004955154, 0.00912878496890749, 0.007098169998483812, 0.9937437997877336, 0.0033774474082216427, 0.030397026673994784, 0.9659499587513899, 0.05201448618482129, 0.09709370754499974, 0.7224234192336291, 0.1283023992558925, 0.5517191206005255, 0.011822552584296975, 0.2719187094388304, 0.16551573618015764, 0.0013849742183945821, 0.01938963905752415, 0.06647876248293993, 0.8753037060253759, 0.03739430389665371, 0.0730177893148589, 0.6021467009252748, 0.3200779805582856, 0.00400097475697857, 0.9754908974505994, 0.014345454374273519, 0.8246348267679131, 0.167981168415686, 0.0038177538276292274, 0.9154502590651046, 0.03390556515055943, 0.050858347725839144, 0.048326916664932544, 0.026501857525930747, 0.03741438709543164, 0.5658926048184036, 0.3211401559024549, 0.9301979567326195, 0.0637121888173027, 0.01274243776346054, 0.9834736775052618, 0.029802232651674598, 0.9965112080806803, 0.8045017442277242, 0.010871645192266544, 0.18481796826853125, 0.004673246971432873, 0.9954016049152019, 0.017437811145312006, 0.017437811145312006, 0.947007282199252, 0.017437811145312006, 0.017075284811365644, 0.22394892771829555, 0.16943936466662832, 0.4433006633719927, 0.1464534043436361, 0.3302209873126431, 0.305239730589228, 0.36066689394430523, 0.003122657090426885, 0.9988820220206103, 0.1387799195310791, 0.8227666657913976, 0.03784906896302157, 0.033809197740462225, 0.13653714472109743, 0.02600707518497094, 0.7698094254751399, 0.033809197740462225, 0.9463486739956141, 0.047317433699780705, 0.32010337631182706, 0.06451695956672483, 0.5632826854479437, 0.05210985195773929, 0.046925669652801934, 0.4411012947363382, 0.0656959375139227, 0.27529726196310467, 0.17206078872694042, 0.9966009441238297, 0.0033220031470794324, 0.004538725326897483, 0.9939808465905486, 0.9881562056193673, 0.009235104725414648, 0.7020926381433124, 0.008163867885387354, 0.2612437723323953, 0.032655471541549415, 0.006585162890613998, 0.2650528063472134, 0.03457210517572349, 0.5547999735342294, 0.13828842070289396, 0.05136482817809384, 0.9438287177724742, 0.04205121709927919, 0.4999422477358748, 0.3083755920613807, 0.14951543857521488, 0.9707649719220505, 0.02011948128335856, 0.00502987032083964, 0.9780088336403547, 0.021261061600877276, 0.9466849465549534, 0.0556873497973502, 0.07011281830317129, 0.017528204575792822, 0.9114666379412267, 0.9948574137442173, 0.013807471358937165, 0.5412528772703369, 0.05799137970753609, 0.2264425302865695, 0.1601666677636711, 0.9971200638748616, 0.0028902030836952513, 0.9766309569470631, 0.19992541769336253, 0.01904051597079643, 0.04760128992699108, 0.672130213769114, 0.05902559950946894, 0.010550804909155815, 0.9847417915212094, 0.9716453189257126, 0.03238817729752375, 0.0015709679741566993, 0.0007854839870783496, 0.9936372436541122, 0.0031419359483133985, 0.9044813631065043, 0.08582669868893837, 0.006602053745302951, 0.011950422611589757, 0.9799346541503601, 0.005975211305794878, 0.89075802293966, 0.07813666867891755, 0.03125466747156702, 0.3850901493644383, 0.15091370718336095, 0.19254507468221915, 0.2393803631184346, 0.031223525624143645, 0.8720132236827134, 0.057253393474117543, 0.06606160785475101, 0.00880821438063347, 0.9799530311481826, 0.011263827944231983, 0.9350857904104062, 0.06280426950517654, 0.9742817323879663, 0.026692650202410035, 0.007825041066015179, 0.0019562602665037948, 0.07825041066015179, 0.9116172841907684, 0.03318156751934222, 0.1824986213563822, 0.31371663836469005, 0.4705749575470351, 0.7388563181954408, 0.12887028805734435, 0.13746164059450064, 0.00783913827483022, 0.995570560903438, 0.9767197395979093, 0.012058268390097646, 0.012058268390097646, 0.03441972011909327, 0.32831117652058195, 0.02118136622713432, 0.48849525861328524, 0.1270881973628059, 0.9909350966773323, 0.9938304423570926, 0.9513246589772959, 0.032804298585424, 0.0036901672795582475, 0.9926549982011685, 0.9928971702504439, 0.004554574175460752, 0.0022233560606324935, 0.017786848485059948, 0.9782766666782973, 0.9498893241065639, 0.022337218486530384, 0.9753918739118268, 0.9859746672569265, 0.013030502210443962, 0.9766764061363504, 0.9628541649052714, 0.022925099164411225, 0.022925099164411225, 0.04481426194165637, 0.003734521828471364, 0.011203565485414093, 0.2688855716499382, 0.6684794072963742, 0.41575432576331756, 0.12943295047348566, 0.08628863364899043, 0.3608433770775964, 0.0039222106204086565, 0.01857035769950598, 0.01857035769950598, 0.882806235253438, 0.07999538701325652, 0.007098169998483812, 0.9937437997877336, 0.006401433155866378, 0.0007112703506518197, 0.0014225407013036394, 0.9822643542501631, 0.009957784909125477, 0.9779264348942078, 0.021259270323787126, 0.9851863663088705, 0.9851863663088705, 0.0034308518564211687, 0.010292555569263506, 0.9880853346492966, 0.003955925610653385, 0.996893253884653, 0.6845169198133211, 0.005185734241010008, 0.2670653134120154, 0.041485873928080064, 0.8879571595262339, 0.10628058666144112, 0.00342840602133681, 0.9775641059361542, 0.9882474591268684, 0.009476345498476821, 0.23198171659356806, 0.5347375162156823, 0.2221519828396033, 0.009829733753964748, 0.007560123704889588, 0.990376205340536, 0.10588411965324664, 0.550849527243676, 0.2647102991331166, 0.07815256450596776, 0.9913339509498801, 0.00556929185926899, 0.15059530397141752, 0.46285909602979797, 0.3432687075819076, 0.04207809963907254, 0.011569112080459663, 0.22559768556896342, 0.04627644832183865, 0.6623316666063157, 0.05350714337212594, 0.9455737142388833, 0.05403278367079333, 0.9598216020480678, 0.03855520177360103, 0.034735104002995565, 0.2115683607455184, 0.02210415709281536, 0.705754158606319, 0.02526189382036041, 0.9526018029961478, 0.01488440317181481, 0.04465320951544443, 0.021699846768113408, 0.0612701555805555, 0.1008404643929976, 0.5488784770758097, 0.2667804690903354, 0.984289509562055, 0.9983731447536779, 0.9571525893915517, 0.03828610357566207, 0.9662594423934389, 0.9980993211533369, 0.015448155241616692, 0.12101054939266408, 0.3656063407182617, 0.4660193497887702, 0.030896310483233384, 0.028227854283879947, 0.9684325392777274, 0.002171373406452304, 0.9688655123587671, 0.016421449362013, 0.9832253098999426, 0.02979470636060432, 0.9832253098999426, 0.02979470636060432, 0.06757922712789073, 0.004223701695493171, 0.13515845425578146, 0.794055918752716, 0.9748667915918823, 0.029541417927026737, 0.040331913663921935, 0.05866460169297737, 0.9019682510295269, 0.916407058591391, 0.020827433149804342, 0.062482299449413026, 0.9849189563401003, 0.017907617388001826, 0.9891122006769844, 0.009265688062547864, 0.993773270705333, 0.9962357705065752, 0.006442041780664259, 0.08589389040885678, 0.09233593218952105, 0.7114877255533637, 0.10450423333077576, 0.9976886616291522, 0.0023955472097695434, 0.016768830468386803, 0.9797788087957433, 0.9841821063777844, 0.031747809883154336, 0.07957309480522165, 0.8866716278296126, 0.011367584972174521, 0.022735169944349043, 0.9815445286067848, 0.043130388358036585, 0.0017252155343214634, 0.897112077847161, 0.05693211263260829, 0.03278083374630348, 0.9550149564756414, 0.01092694458210116], "Term": ["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "accounting", "accounting", "accounting", "adult", "adult", "afford", "agave", "aggies", "aggressively", "align", "align", "amazon", "amazon", "amazon", "amazon", "amy", "amy", "anc", "apb", "apb", "application", "application", "application", "application", "astronomical", "atlanta", "atlanta", "atlanta", "auth", "auth", "available", "available", "available", "available", "backout", "bakernet", "bakernet", "bankrupt", "bankruptcy", "barbo", "barbo", "bergfelt", "bergfelt", "beth", "beth", "bgcolor", "bgcolor", "billion", "billion", "billion", "blah", "blair", "blair", "blair", "bodydefault", "bodydefault", "border", "border", "border", "borland", "borland", "br", "br", "br", "bradford", "bradford", "breese", "breese", "brochure", "brochure", "business", "business", "business", "business", "buying", "c", "c", "c", "c", "california", "california", "california", "california", "california", "cellpadding", "cellpadding", "cellspacing", "cellspacing", "center", "center", "center", "center", "center", "chanley", "cherry", "cherry", "choice", "choice", "class", "class", "class", "class", "classmate", "classmate", "clear", "clear", "clear", "click", "click", "click", "click", "click", "colspan", "colspan", "commoditylogic", "commoditylogic", "commoditylogic", "communication", "communication", "communication", "communication", "company", "company", "company", "company", "company", "conference", "conference", "conference", "conference", "confidential", "confidential", "confidential", "consumer", "consumer", "consumer", "consumer", "contact", "contact", "contact", "contact", "contract", "contract", "contract", "contract", "contract", "coords", "coords", "copy", "copy", "copy", "copy", "copy", "corp", "corp", "corp", "corp", "corp", "could", "could", "could", "could", "could", "credit", "credit", "credit", "credit", "crisis", "ctr", "ctr", "curve", "curve", "curve", "curve", "customer", "customer", "customer", "customer", "darrell", "darrell", "day", "day", "day", "day", "day", "dbcaps", "dealing", "declared", "delta", "delta", "design", "design", "design", "desk", "desk", "desk", "detected", "detected", "devastated", "development", "development", "development", "development", "deviation", "deviation", "devon", "dietz", "dietz", "div", "div", "donate", "doubleclick", "doubleclick", "dun", "e", "e", "e", "e", "e", "ecar", "ecar", "ee", "ee", "ee", "ee", "email", "email", "email", "email", "email", "emailed", "emaillink", "employee", "employee", "ena", "ena", "ena", "ena", "end", "end", "end", "end", "end", "energy", "energy", "energy", "energy", "energy", "enform", "enform", "engine", "engine", "enlarge", "enlarge", "enron", "enron", "enron", "enron", "enron", "eol", "eol", "eol", "eol", "estoppel", "estoppel", "euci", "euci", "exotica", "exotica", "f", "f", "f", "f", "f", "fagan", "feedback", "feedback", "ffffff", "ffffff", "file", "file", "file", "file", "file", "financially", "fname", "font", "font", "forecast", "forecast", "forecast", "fran", "fran", "frcc", "frcc", "fri", "fri", "fri", "fund", "fund", "fund", "fw", "fw", "fw", "fw", "gas", "gas", "gas", "gas", "gas", "get", "get", "get", "get", "get", "gif", "gif", "gif", "gift", "gift", "gift", "go", "go", "go", "go", "go", "gorny", "gorny", "greg", "greg", "greg", "greg", "group", "group", "group", "group", "hanagriff", "headcount", "headcount", "height", "height", "hereto", "hereto", "hou", "hou", "hou", "hou", "hou", "hourahead", "hourahead", "hourahead", "hp", "hp", "hp", "href", "href", "hurt", "id", "id", "id", "id", "id", "image", "image", "image", "image", "img", "img", "impacted", "impacted", "information", "information", "information", "information", "intended", "intended", "intended", "investment", "investment", "itcapps", "january", "january", "january", "january", "jason", "jason", "jason", "jerry", "jerry", "ken", "kern", "kern", "kim", "kim", "kim", "kim", "kimberly", "kimberly", "know", "know", "know", "know", "kowalke", "krishna", "krishna", "labour", "labour", "lay", "lay", "leg", "leg", "leg", "leg", "let", "let", "let", "let", "like", "like", "like", "like", "like", "lindberg", "linkbarseperator", "linkbarseperator", "london", "london", "london", "london", "lorraine", "lorraine", "lost", "lost", "lynn", "lynn", "lynn", "lynn", "maac", "maac", "maac", "made", "made", "made", "made", "mage", "mage", "mail", "mail", "mail", "mail", "mail", "mailto", "mailto", "mailto", "mailto", "mailto", "mainheadline", "mainheadline", "management", "management", "management", "management", "many", "many", "many", "many", "many", "mapp", "mapp", "mapp", "mark", "mark", "mark", "mark", "market", "market", "market", "market", "market", "matrix", "matrix", "may", "may", "may", "may", "memory", "memory", "message", "message", "message", "message", "message", "million", "million", "million", "million", "min", "min", "monday", "monday", "monday", "monday", "money", "money", "money", "money", "mr", "mr", "mr", "mulligan", "nbsp", "nbsp", "nc", "nc", "nc", "ne", "ne", "ne", "nerc", "nerc", "net", "net", "net", "net", "net", "netted", "neumin", "new", "new", "new", "new", "new", "november", "november", "november", "november", "npcc", "npcc", "nw", "nw", "nw", "one", "one", "one", "one", "operation", "operation", "operation", "operation", "option", "option", "option", "option", "option", "original", "original", "original", "original", "origination", "origination", "outage", "outage", "outage", "overview", "overview", "overview", "p", "p", "p", "p", "p", "pager", "pager", "pager", "pampa", "pampa", "partnership", "password", "password", "password", "pep", "pep", "plan", "plan", "plan", "plan", "please", "please", "please", "please", "please", "pm", "pm", "pm", "pm", "pocketbook", "power", "power", "power", "price", "price", "price", "price", "price", "privileged", "privileged", "process", "process", "process", "process", "product", "product", "product", "product", "product", "profit", "profit", "pseg", "pseg", "pt", "pt", "purpose", "purpose", "purpose", "purpose", "question", "question", "question", "question", "question", "range", "range", "real", "real", "real", "real", "recipient", "recipient", "recipient", "rect", "rect", "ref", "ref", "reg", "reg", "reg", "repair", "request", "request", "request", "request", "request", "retirement", "retirement", "revert", "right", "right", "right", "right", "right", "rk", "rk", "royalty", "royalty", "said", "said", "said", "said", "sat", "sat", "sat", "saving", "saving", "saving", "savita", "savita", "savita", "schedule", "schedule", "schedule", "schedule", "schedule", "scheduled", "scheduled", "scheduled", "scheduled", "schoolcraft", "schoolcraft", "scoop", "scoop", "script", "script", "se", "se", "se", "se", "see", "see", "see", "see", "seller", "seller", "seller", "serc", "serc", "server", "server", "server", "service", "service", "service", "service", "service", "sfs", "simulation", "sitara", "sitara", "skilling", "skilling", "sold", "sold", "sp", "sp", "sp", "span", "spp", "spp", "src", "src", "srrs", "stack", "stack", "stack", "standard", "standard", "standard", "standard", "standard", "start", "start", "start", "start", "start", "state", "state", "state", "state", "std", "std", "stock", "stock", "stock", "stock", "stock", "stocklookup", "stocklookup", "stsw", "stwbom", "sw", "sw", "sw", "syncrasy", "syncrasy", "system", "system", "system", "system", "table", "table", "table", "tagline", "td", "td", "team", "team", "team", "team", "temp", "temp", "thanks", "thanks", "thanks", "thanks", "thru", "thru", "thursday", "thursday", "thursday", "thursday", "time", "time", "time", "time", "time", "tonight", "tonight", "tr", "tr", "trading", "trading", "trading", "trading", "trading", "tw", "tw", "tw", "u", "u", "u", "u", "u", "ubswenergy", "underhanded", "underscore", "underscore", "unify", "urging", "use", "use", "use", "use", "use", "utility", "utility", "utility", "valign", "valign", "vegetarian", "vegetarian", "veggie", "veggie", "visit", "visit", "visit", "visit", "vladimir", "vladimir", "volatility", "volatility", "volatility", "watson", "watson", "watson", "whitt", "whitt", "width", "width", "wiped", "wj", "would", "would", "would", "would", "would", "writing", "wscc", "wscc", "wscc", "wyndham", "wyndham", "xl", "xl", "xl", "xl", "xll", "year", "year", "year", "year", "york", "york", "york"]}, "R": 30, "lambda.step": 0.01, "plot.opts": {"xlab": "PC1", "ylab": "PC2"}, "topic.order": [1, 2, 3, 4, 5]};

function LDAvis_load_lib(url, callback){
  var s = document.createElement('script');
  s.src = url;
  s.async = true;
  s.onreadystatechange = s.onload = callback;
  s.onerror = function(){console.warn("failed to load library " + url);};
  document.getElementsByTagName("head")[0].appendChild(s);
}

if(typeof(LDAvis) !== "undefined"){
   // already loaded: just create the visualization
   !function(LDAvis){
       new LDAvis("#" + "ldavis_el917619086505744489079175291", ldavis_el917619086505744489079175291_data);
   }(LDAvis);
}else if(typeof define === "function" && define.amd){
   // require.js is available: use it to load d3/LDAvis
   require.config({paths: {d3: "https://d3js.org/d3.v5"}});
   require(["d3"], function(d3){
      window.d3 = d3;
      LDAvis_load_lib("https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.4.0/pyLDAvis/js/ldavis.v3.0.0.js", function(){
        new LDAvis("#" + "ldavis_el917619086505744489079175291", ldavis_el917619086505744489079175291_data);
      });
    });
}else{
    // require.js not available: dynamically load d3 & LDAvis
    LDAvis_load_lib("https://d3js.org/d3.v5.js", function(){
         LDAvis_load_lib("https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.4.0/pyLDAvis/js/ldavis.v3.0.0.js", function(){
                 new LDAvis("#" + "ldavis_el917619086505744489079175291", ldavis_el917619086505744489079175291_data);
            })
         });
}
</script>
```
:::
:::

::: {.cell .markdown}
#### Assign topics to your original data

-   One practical application of topic modeling is to determine what
    topic a given text is about
-   To find that, find the topic number that has the highest percentage
    contribution in that text
-   The function, `get_topic_details` shown here, nicely aggregates this
    information in a presentable table
-   Combine the original text data with the output of the
    `get_topic_details` function
-   Each row contains the dominant topic number, the probability score
    with that topic and the original text data

``` python
def get_topic_details(ldamodel, corpus):
    topic_details_df = pd.DataFrame()
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_details_df = topic_details_df.append(pd.Series([topic_num, prop_topic]), ignore_index=True)
    topic_details_df.columns = ['Dominant_Topic', '% Score']
    return topic_details_df


contents = pd.DataFrame({'Original text':text_clean})
topic_details = pd.concat([get_topic_details(ldamodel,
                           corpus), contents], axis=1)
topic_details.head()


     Dominant_Topic    % Score     Original text
0    0.0              0.989108    [investools, advisory, free, ...
1    0.0              0.993513    [forwarded, richard, b, ...
2    1.0              0.964858    [hey, wearing, target, purple, ...
3    0.0              0.989241    [leslie, milosevich, santa, clara, ...
```
:::

::: {.cell .markdown}
### Interpreting the topic model

-   Use the visualization results from the pyLDAvis library shown in
    4.4.0.2.
-   Have a look at topic 1 and 3 from the LDA model on the Enron email
    data. Which one would you research further for fraud detection
    purposes?

**Possible Answers**

-   ****Topic 1.****
-   ~~Topic 3.~~
-   ~~None of these topics seem related to fraud.~~

**Topic 1 seems to discuss the employee share option program, and seems
to point to internal conversation (with \"please, may, know\" etc), so
this is more likely to be related to the internal accounting fraud and
trading stock with insider knowledge. Topic 3 seems to be more related
to general news around Enron.**
:::

::: {.cell .markdown}
### Finding fraudsters based on topic

In this exercise you\'re going to **link the results** from the topic
model **back to your original data**. You now learned that you want to
**flag everything related to topic 3**. As you will see, this is
actually not that straightforward. You\'ll be given the function
`get_topic_details()` which takes the arguments `ldamodel` and `corpus`.
It retrieves the details of the topics for each line of text. With that
function, you can append the results back to your original data. If you
want to learn more detail on how to work with the model results, which
is beyond the scope of this course, you\'re highly encouraged to read
this
[article](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/).

Available for you are the `dictionary` and `corpus`, the text data
`text_clean` as well as your model results `ldamodel`. Also defined is
`get_topic_details()`.

**Instructions 1/3**

-   Print and inspect the results from the `get_topic_details()`
    function by inserting your LDA model results and `corpus`.
:::

::: {.cell .markdown}
#### def get_topic_details
:::

::: {.cell .code execution_count="142"}
``` python
def get_topic_details(ldamodel, corpus):
    topic_details_df = pd.DataFrame()
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_details_df = topic_details_df.append(pd.Series([topic_num, prop_topic]), ignore_index=True)
    topic_details_df.columns = ['Dominant_Topic', '% Score']
    return topic_details_df
```
:::

::: {.cell .code execution_count="143"}
``` python
# Run get_topic_details function and check the results
topic_details_df = get_topic_details(ldamodel, corpus)
```
:::

::: {.cell .code execution_count="144"}
``` python
topic_details_df.head()
```

::: {.output .execute_result execution_count="144"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dominant_Topic</th>
      <th>% Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.0</td>
      <td>0.653164</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.0</td>
      <td>0.891665</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>0.685952</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>0.993495</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.0</td>
      <td>0.993390</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .code execution_count="145"}
``` python
topic_details_df.tail()
```

::: {.output .execute_result execution_count="145"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dominant_Topic</th>
      <th>% Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2085</th>
      <td>2.0</td>
      <td>0.909957</td>
    </tr>
    <tr>
      <th>2086</th>
      <td>3.0</td>
      <td>0.599637</td>
    </tr>
    <tr>
      <th>2087</th>
      <td>4.0</td>
      <td>0.999324</td>
    </tr>
    <tr>
      <th>2088</th>
      <td>1.0</td>
      <td>0.998148</td>
    </tr>
    <tr>
      <th>2089</th>
      <td>1.0</td>
      <td>0.988407</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown}
**Instructions 2/3**

-   Concatenate column-wise the results from the previously defined
    function `get_topic_details()` to the original text data contained
    under `contents` and inspect the results.
:::

::: {.cell .code execution_count="146"}
``` python
# Add original text to topic details in a dataframe
contents = pd.DataFrame({'Original text': text_clean})
topic_details = pd.concat([get_topic_details(ldamodel, corpus), contents], axis=1)
```
:::

::: {.cell .code execution_count="147"}
``` python
topic_details.sort_values(by=['% Score'], ascending=False).head(10).head()
```

::: {.output .execute_result execution_count="147"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dominant_Topic</th>
      <th>% Score</th>
      <th>Original text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>154</th>
      <td>3.0</td>
      <td>0.999957</td>
      <td>[joint, venture, enron, meeting, belies, offic...</td>
    </tr>
    <tr>
      <th>135</th>
      <td>3.0</td>
      <td>0.999953</td>
      <td>[lawyer, agree, order, safeguard, document, ho...</td>
    </tr>
    <tr>
      <th>107</th>
      <td>3.0</td>
      <td>0.999907</td>
      <td>[sample, article, original, message, schmidt, ...</td>
    </tr>
    <tr>
      <th>849</th>
      <td>2.0</td>
      <td>0.999874</td>
      <td>[original, message, received, thu, aug, cdt, e...</td>
    </tr>
    <tr>
      <th>263</th>
      <td>3.0</td>
      <td>0.999807</td>
      <td>[original, message, cook, mary, thursday, octo...</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .code execution_count="148"}
``` python
topic_details.sort_values(by=['% Score'], ascending=False).head(10).tail()
```

::: {.output .execute_result execution_count="148"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dominant_Topic</th>
      <th>% Score</th>
      <th>Original text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>285</th>
      <td>3.0</td>
      <td>0.999801</td>
      <td>[original, message, vann, suzanne, wednesday, ...</td>
    </tr>
    <tr>
      <th>2081</th>
      <td>0.0</td>
      <td>0.999631</td>
      <td>[unsubscribe, mailing, please, go, money, net,...</td>
    </tr>
    <tr>
      <th>175</th>
      <td>3.0</td>
      <td>0.999328</td>
      <td>[thanks, pretty, sure, m, mcfadden, might, int...</td>
    </tr>
    <tr>
      <th>2087</th>
      <td>4.0</td>
      <td>0.999324</td>
      <td>[image, image, image, image, image, image, ima...</td>
    </tr>
    <tr>
      <th>280</th>
      <td>3.0</td>
      <td>0.999221</td>
      <td>[financial, express, wednesday, october, anti,...</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown}
**Instructions 3/3**

-   Create a flag with the `np.where()` function to flag all content
    that has topic 3 as a dominant topic with a 1, and 0 otherwise
:::

::: {.cell .code execution_count="149"}
``` python
# Create flag for text highest associated with topic 3
topic_details['flag'] = np.where((topic_details['Dominant_Topic'] == 3.0), 1, 0)
```
:::

::: {.cell .code execution_count="150"}
``` python
topic_details_1 = topic_details[topic_details.flag == 1]
```
:::

::: {.cell .code execution_count="151"}
``` python
topic_details_1.sort_values(by=['% Score'], ascending=False).head(10)
```

::: {.output .execute_result execution_count="151"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dominant_Topic</th>
      <th>% Score</th>
      <th>Original text</th>
      <th>flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>154</th>
      <td>3.0</td>
      <td>0.999957</td>
      <td>[joint, venture, enron, meeting, belies, offic...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>135</th>
      <td>3.0</td>
      <td>0.999953</td>
      <td>[lawyer, agree, order, safeguard, document, ho...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>107</th>
      <td>3.0</td>
      <td>0.999907</td>
      <td>[sample, article, original, message, schmidt, ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>263</th>
      <td>3.0</td>
      <td>0.999807</td>
      <td>[original, message, cook, mary, thursday, octo...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>285</th>
      <td>3.0</td>
      <td>0.999801</td>
      <td>[original, message, vann, suzanne, wednesday, ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>175</th>
      <td>3.0</td>
      <td>0.999328</td>
      <td>[thanks, pretty, sure, m, mcfadden, might, int...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>280</th>
      <td>3.0</td>
      <td>0.999221</td>
      <td>[financial, express, wednesday, october, anti,...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>240</th>
      <td>3.0</td>
      <td>0.998825</td>
      <td>[wsj, heard, street, enron, ex, ceo, made, bet...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>239</th>
      <td>3.0</td>
      <td>0.998680</td>
      <td>[prolly, use, f, word, intra, company, e, mail...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1138</th>
      <td>3.0</td>
      <td>0.998678</td>
      <td>[greg, great, time, million, club, lavo, dave,...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown}
**You have now flagged all data that is highest associated with topic 3,
that seems to cover internal conversation about enron stock options. You
are a true detective. With these exercises you have demonstrated that
text mining and topic modeling can be a powerful tool for fraud
detection.**
:::

::: {.cell .markdown}
## Lesson 4: Recap
:::

::: {.cell .markdown}
### Working with imbalanced data

-   Worked with highly imbalanced fraud data
-   Learned how to resample your data
-   Learned about different resampling methods
:::

::: {.cell .markdown}
### Fraud detection with labeled data

-   Refreshed supervised learning techniques to detect fraud
-   Learned how to get reliable performance metrics and worked with the
    precision recall trade-off
-   Explored how to optimize your model parameters to handle fraud data
-   Applied ensemble methods to fraud detection
:::

::: {.cell .markdown}
### Fraud detection without labels

-   Learned about the importance of segmentation
-   Refreshed your knowledge on clustering methods
-   Learned how to detect fraud using outliers and small clusters with
    K-means clustering
-   Applied a DB-scan clustering model for fraud detection
:::

::: {.cell .markdown}
### Text mining for fraud detection

-   Know how to augment fraud detection analysis with text mining
    techniques
-   Applied word searches to flag use of certain words, and learned how
    to apply topic modeling for fraud detection
-   Learned how to effectively clean messy text data
:::

::: {.cell .markdown}
### Further learning for fraud detection

-   Network analysis to detect fraud
-   Different supervised and unsupervised learning techniques (e.g.
    Neural Networks)
-   Working with very large data
:::
