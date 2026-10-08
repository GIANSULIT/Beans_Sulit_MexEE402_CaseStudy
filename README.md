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

| Chapter | Member 1 | Member 2 |
|---|---|---|
| Ch1_2_3 | [link]() |https://colab.research.google.com/drive/1RIAxNAQ7HxPYDXn_e7EjG_1KfLG5u2nX?usp=sharing|
| Ch4 | [link]() |https://colab.research.google.com/drive/12Ks5OK7ZAm5oW5FvvWpwfpUV6JeLTQQ8?usp=drive_link|
| Ch5 | [link]() |https://colab.research.google.com/drive/1f_-z42xfyXx9l_O4cv-do5XEITr146Ng?usp=sharing|
| Ch6 | [link]() |https://colab.research.google.com/drive/14iqHmap1PaqyyxHldshIERxBG3Ij8_v6?usp=drive_link|
| Ch7 | [link]() |https://colab.research.google.com/drive/17G4HH2PQe9VLV-z4C58Ywdqs02rzbZyA?usp=drive_link|
| Ch8 | [link]() |https://colab.research.google.com/drive/1Ka60tQQpK6HKxxUP5R2Lp05Nhk6ZJwTK?usp=drive_link|
| Ch9 | [link]() |https://colab.research.google.com/drive/1_8bIt68agPEshtAWBnlHsSWmS-iFyM1K?usp=sharing|

## What we learned

**Chapter 1, 2, 3:** We learned that data preprocessing is essential for cleaning messy, real-world data before feeding it into machine learning models. It was surprising to see how much data can be lost or skewed by duplicates and missing values, and how straightforward techniques like mean imputation or deletion can drastically improve dataset quality.

**Chapter 4:** This chapter taught us how to create new, more informative variables through feature engineering, like combining temperature and sales into a single interaction feature. The biggest surprise was learning that machine learning models cannot process text directly, making categorical encoding techniques like one-hot and ordinal encoding strictly necessary for variables like weather or size.

**Chapter 5:** We explored how data scaling and normalization level the playing field for numerical features with vastly different ranges, preventing large numbers from dominating the model. It was interesting to learn that not all models require scaling, but for those that do, adjusting values to a standard range ensures fair weight distribution across all features.

**Chapter 6:** This chapter highlighted how outliers can severely distort statistical analysis and model predictions if left unchecked. We were surprised by how effectively mathematical boundaries, like the Z-score and Interquartile Range (IQR), can systematically identify these anomalies, allowing us to confidently apply strategies like capping or removal.

**Chapter 7:** We learned that keeping too many irrelevant features actually harms model accuracy, making feature selection a critical step. It was fascinating to see algorithms act like a "sculptor" through techniques like Recursive Feature Elimination (RFE) and Lasso regression, automatically chipping away unnecessary data to find the most predictive variables.

**Chapter 8:** This chapter introduced pipelines as a way to automate and streamline the entire preprocessing workflow, acting much like a conveyor belt for data. The most surprising aspect was how compactly we can bundle multiple steps—like imputation and scaling—into a single `ColumnTransformer`, ensuring consistent reproducibility without manual repetition.

**Chapter 9:** We applied everything we learned to the Titanic dataset, managing mixed data types through a comprehensive pipeline that included discretization and log transformations. We realized that putting all the preprocessing steps together requires careful planning, but it ultimately prepares complex real-world data for immediate and effective modeling.

## Errors we found
In the Chapter 1, 2, 3 notebook, we found a `FutureWarning` during the imputation step. The original code used `inplace=True` when filling missing values (`df['Year'].fillna(df['Year'].mean(), inplace=True)`). The correct, future-proof version should avoid chained assignment by explicitly assigning the result back to the column: `df['Year'] = df['Year'].fillna(df['Year'].mean())`.

## Note on AI tools

We used an AI tool, specifically Gemini Pro to help format this README file according to the required template and to assist in drafting the summary paragraphs for our chapter learnings based on our notebook executions.

## References

McKinney, W. (2021). Python for Data Analysis, 3rd ed. O'Reilly.
VanderPlas, J. Python Data Science Handbook.
