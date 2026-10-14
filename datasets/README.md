# Sample datasets

Small public datasets that ship with the app, for trying the models on something harder than mushrooms. The mushroom data itself is `Mushrooms.csv` in the project root.

| File | Rows | What is predicted | Source |
|---|---|---|---|
| `titanic.csv` | 891 | `Survived` (1 or 0) | [Data Science Dojo datasets](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv), copied unchanged |
| `heart_disease.csv` | 303 | `heart_disease` (yes or no) | [UCI Heart Disease, Cleveland](https://archive.ics.uci.edu/dataset/45/heart+disease), `processed.cleveland.data` |
| `breast_cancer.csv` | 569 | `diagnosis` (malignant or benign) | Breast Cancer Wisconsin (Diagnostic), exported from scikit-learn's `load_breast_cancer` |
| `telco_churn.csv` | 7,043 | `Churn` (Yes or No) | [IBM Telco customer churn sample](https://github.com/IBM/telco-customer-churn-on-icp4d), copied unchanged |

## Changes made to the heart disease file

The published file has no header row and stores everything as number codes. For this app:

- Column names were added.
- The codes of the category columns were replaced by their meanings, taken from the dataset's own documentation (for example chest pain type 4 became "asymptomatic").
- The original target runs from 0 (no disease) to 4. It was reduced to `heart_disease`: "no" for 0 and "yes" for 1 to 4.

## Uploads

CSV files uploaded through the app are saved in `datasets/uploads/`. That folder is ignored by git.
