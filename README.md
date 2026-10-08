# MexEE 402: Data Preprocessing Case Study

MexEE Elective 2: Data Science and Machine Learning
Batangas State University, Alangilan Campus
1st Semester, AY 2026-2027

## Members

| Name | Student Number | Section |
|---|---|---|
| Beans, Zachary U. | 1 |MEXE - 4102|
| Sulit, Gian karl M. | 2 |MEXE - 4102|

## Notebook links

| Chapter | Links |
|---|---|
| Ch1_2_3 |https://colab.research.google.com/drive/1RIAxNAQ7HxPYDXn_e7EjG_1KfLG5u2nX?usp=sharing|
| Ch4 |https://colab.research.google.com/drive/12Ks5OK7ZAm5oW5FvvWpwfpUV6JeLTQQ8?usp=drive_link|
| Ch5 |https://colab.research.google.com/drive/1f_-z42xfyXx9l_O4cv-do5XEITr146Ng?usp=sharing|
| Ch6 |https://colab.research.google.com/drive/14iqHmap1PaqyyxHldshIERxBG3Ij8_v6?usp=drive_link|
| Ch7 |https://colab.research.google.com/drive/17G4HH2PQe9VLV-z4C58Ywdqs02rzbZyA?usp=drive_link|
| Ch8 |https://colab.research.google.com/drive/1Ka60tQQpK6HKxxUP5R2Lp05Nhk6ZJwTK?usp=drive_link|
| Ch9 |https://colab.research.google.com/drive/1_8bIt68agPEshtAWBnlHsSWmS-iFyM1K?usp=sharing|

## What we learned

**Chapter 1, 2, 3:** We learned that data preprocessing is essential for cleaning messy, real-world data before feeding it into machine learning models. It was surprising to see how much data can be lost or skewed by duplicates and missing values, and how straightforward techniques like mean imputation or deletion can drastically improve dataset quality.

**Chapter 4:** This chapter taught us how to create new, more informative variables through feature engineering, like combining temperature and sales into a single interaction feature. The biggest surprise was learning that machine learning models cannot process text directly, making categorical encoding techniques like one-hot and ordinal encoding strictly necessary for variables like weather or size.

**Chapter 5:** We explored how data scaling and normalization level the playing field for numerical features with vastly different ranges, preventing large numbers from dominating the model. It was interesting to learn that not all models require scaling, but for those that do, adjusting values to a standard range ensures fair weight distribution across all features.

**Chapter 6:** This chapter highlighted how outliers can severely distort statistical analysis and model predictions if left unchecked. We were surprised by how effectively mathematical boundaries, like the Z-score and Interquartile Range (IQR), can systematically identify these anomalies, allowing us to confidently apply strategies like capping or removal.

**Chapter 7:** We learned that keeping too many irrelevant features actually harms model accuracy, making feature selection a critical step. It was fascinating to see algorithms act like a "sculptor" through techniques like Recursive Feature Elimination (RFE) and Lasso regression, automatically chipping away unnecessary data to find the most predictive variables.

**Chapter 8:** This chapter introduced pipelines as a way to automate and streamline the entire preprocessing workflow, acting much like a conveyor belt for data. The most surprising aspect was how compactly we can bundle multiple steps—like imputation and scaling—into a single `ColumnTransformer`, ensuring consistent reproducibility without manual repetition.

**Chapter 9:** We applied everything we learned to the Titanic dataset, managing mixed data types through a comprehensive pipeline that included discretization and log transformations. We realized that putting all the preprocessing steps together requires careful planning, but it ultimately prepares complex real-world data for immediate and effective modeling.

## Chapter Questions

# Chapter 6: Outlier Detection

## Q1. What is an outlier?

An outlier is a data point that is very different from most of the other data. For example, if most students study 10–20 hours a week but one studies 100 hours, that student is an outlier. Outliers can skew results and lead to wrong conclusions.

## Q2. How does the Z-score method find outliers? What cutoff did the notebook use?

The Z-score method measures how many standard deviations a value is from the mean, using:
Z = (X – μ) / σ

A value that is too far from the mean is flagged as an outlier. The notebook used a cutoff of **3**, so any value with a Z-score below -3 or above 3 counts as an outlier.

## Q3. How does the IQR method find outliers? Write the formula for the lower and upper fence.

The IQR method looks at the middle 50% of the data, between Q1 and Q3, where `IQR = Q3 – Q1`. Any value that falls too far outside that middle range is an outlier.

- **Lower fence** = `Q1 – 1.5 × IQR`
- **Upper fence** = `Q3 + 1.5 × IQR`

Values below the lower fence or above the upper fence are outliers.

## Q4. In the sample data, which value stands out from the rest? What is its Z-score?

The value **100** stands out from the rest of the data (`10, 12, 12, 15, 20, 21, 22, 100`). Its Z-score is about **2.62**.

## Q5. Once you find an outlier, give two things you can do about it.

1. **Remove it**, if it is clearly an error or does not belong in the data.
2. **Keep it but investigate or adjust it**, for example by checking why it happened, or by replacing it with a more typical value such as the median.

# Chapter 7: Feature Selection

## Q1. What is feature selection, and why is it useful?

Feature selection is the process of choosing the most relevant features (columns) in a dataset to use for prediction. It is useful because irrelevant features can reduce the accuracy of a model, so keeping only the important ones gives better and simpler results.

## Q2. What does the filter method use to decide which features to keep?

The filter method uses statistical measures to score each feature, such as the correlation coefficient, chi-square test, or information gain. Features are ranked by their scores, and the ones with low scores are removed. In the notebook, features were kept only if their correlation with `final grade` was **above 0.5**.

## Q3. What does `RFECV` do, step by step?

`RFECV` (Recursive Feature Elimination with Cross-Validation) is a wrapper method that works like this:

1. Train a model (here, `SVR` with a linear kernel) using all the features.
2. Remove the least important feature (`step=1` removes one at a time).
3. Retrain the model and check its performance using cross-validation (`cv=5`, which splits the data into 5 parts).
4. Repeat until the features run out.
5. Keep the set of features that gave the best score.

## Q4. What does `LassoCV` do to features that are not important?

`LassoCV` is an embedded method that selects features while training the model. It shrinks the coefficients (weights) of unimportant features, setting them to **zero**, which effectively removes them. Only features with a coefficient greater than zero are kept as important.

## Q5. Which features did each of the three methods choose? Put them in a short table.

| Method | Type | Features Selected |
|---|---|---|
| Correlation (threshold > 0.5) | Filter | `study hours`, `assignments completed`, `class participation` |
| `RFECV` | Wrapper | `assignments completed` |
| `LassoCV` | Embedded | `study hours`, `class participation`, `extracurricular activities` |

# Chapter 8: Constructing a Preprocessing Pipeline

## Q1. What is a preprocessing pipeline? Explain it using the conveyor belt idea from the notebook.

A preprocessing pipeline is a series of data preparation steps that run automatically, one after another. Think of it like a conveyor belt: each station on the belt is one preprocessing step (such as cleaning, normalization, or encoding). The data enters the belt raw, passes through each station in order, and comes out the other end ready to be used by a machine learning model.

## Q2. The notebook gives three reasons for using a pipeline. Name all three.

1. **Automation**: routine preprocessing is handled automatically.
2. **Efficiency**: it streamlines the steps for a faster workflow.
3. **Reliability and reproducibility**: it reduces human error and gives consistent results every time.

## Q3. What two steps were inside the pipeline, and in what order did they run?

1. **Imputation** (`SimpleImputer`, strategy = `mean`): fills in missing values with the mean.
2. **Scaling** (`StandardScaler`): standardizes the values so they have a mean of 0 and a standard deviation of 1.

Imputation runs first, then scaling.

## Q4. What does `ColumnTransformer` do?

`ColumnTransformer` applies a preprocessing pipeline to specific columns of the dataset. In the notebook, it was used to apply the imputation and scaling pipeline only to the chosen columns, and `fit_transform` then turned `X` into the cleaned and scaled `X_transformed`.

## Q5. Which two columns of the Titanic dataset were preprocessed in this chapter?

The **`Age`** and **`Fare`** columns.

# Chapter 9: Full Pipeline and Visualization

## Q1. Which columns were handled as numerical, and which as categorical?

- **Numerical:** `Age` and `Fare`
- **Categorical:** `Embarked`, `Sex`, and `Pclass`

## Q2. How were the missing values filled in each of those two groups?

- **Numerical columns (`Age`, `Fare`):** missing values were filled with the **median** of the column (`SimpleImputer(strategy='median')`). The values were then scaled with `StandardScaler`.
- **Categorical columns (`Embarked`, `Sex`, `Pclass`):** missing values were filled with the constant text **"missing"** (`SimpleImputer(strategy='constant', fill_value='missing')`). The columns were then converted with `OneHotEncoder`.

## Q3. What is discretization? What three age labels did the notebook use, and what age ranges do they cover?

Discretization means converting continuous numbers (like exact ages) into a few groups or bins (like life stages). In the notebook, `Age` was split using `pd.cut` with the bins `[0, 12, 50, 200]`:

| Label | Age Range |
|---|---|
| Child | 0 to 12 |
| Adult | 13 to 50 |
| Elderly | Above 50 |

## Q4. Name three of the plots you produced, and say in one sentence what each one shows.

1. **Survival by Gender** (count plot): shows how many males and females survived or did not survive.
2. **Survival by Passenger Class** (count plot): shows how survival differed between 1st, 2nd, and 3rd class passengers.
3. **Correlation Heatmap**: shows how strongly the numeric columns are related to each other, using colors and correlation values.

Other plots in the notebook: Survival Count, Age Distribution by Survival, Fare vs Survival (box plot), Survival by Embarkation Port, and Survival by Family Size.

## Q5. Why is it useful to make plots after preprocessing instead of before?

Plots made after preprocessing show whether the cleaning and transformation actually worked. Missing values are filled, so the data is complete and the patterns are easier to read. Comparing the data before and after (for example, the `Age` distribution) also helps catch mistakes early, so the data is confirmed to be clean and ready before it goes into a model.

## Errors we found
In the Chapter 1, 2, 3 notebook, we found a `FutureWarning` during the imputation step. The original code used `inplace=True` when filling missing values (`df['Year'].fillna(df['Year'].mean(), inplace=True)`). The correct, future-proof version should avoid chained assignment by explicitly assigning the result back to the column: `df['Year'] = df['Year'].fillna(df['Year'].mean())`.

## Note on AI tools

We used an AI tool, specifically Gemini Pro to help complete the chapter tasks by assisting with code construction, then it was used to provide easy to digest in-depth definitions of terms and process to assists the members to answer the chapter question. The chapter questions were then answered in summary by the members. Finally, we used ai to assist format this README file according to the required template and to assist in drafting the summary paragraphs for our chapter learnings based on our notebook executions.

## References

McKinney, W. (2021). Python for Data Analysis, 3rd ed. O'Reilly.
VanderPlas, J. Python Data Science Handbook.
