# Sample datasets

Small public datasets that ship with the app, for trying the models on something harder than mushrooms. The mushroom data itself is `Mushrooms.csv` in the project root.

| File | Rows | What is predicted | Source |
|---|---|---|---|
| `titanic.csv` | 891 | `Survived` (1 or 0) | [Data Science Dojo datasets](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv), copied unchanged |
| `heart_disease.csv` | 303 | `heart_disease` (yes or no) | [UCI Heart Disease, Cleveland](https://archive.ics.uci.edu/dataset/45/heart+disease), `processed.cleveland.data` |
| `breast_cancer.csv` | 569 | `diagnosis` (malignant or benign) | Breast Cancer Wisconsin (Diagnostic), exported from scikit-learn's `load_breast_cancer` |
| `telco_churn.csv` | 7,043 | `Churn` (Yes or No) | [IBM Telco customer churn sample](https://github.com/IBM/telco-customer-churn-on-icp4d), copied unchanged |
| `iris.csv` | 150 | `species` | scikit-learn's `load_iris` (Fisher's Iris data) |
| `wine.csv` | 178 | `cultivar` | scikit-learn's `load_wine` (UCI Wine) |
| `penguins.csv` | 344 | `species` | [seaborn-data](https://github.com/mwaskom/seaborn-data/blob/master/penguins.csv), copied unchanged (Palmer Station data) |
| `diabetes.csv` | 768 | `diabetes` (yes or no) | [Pima Indians Diabetes](https://archive.ics.uci.edu/dataset/34/diabetes), via the jbrownlee/Datasets copy |
| `banknotes.csv` | 1,372 | `banknote` (genuine or forged) | [UCI Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication) |
| `tictactoe.csv` | 958 | `x_wins` (yes or no) | [UCI Tic-Tac-Toe Endgame](https://archive.ics.uci.edu/dataset/101/tic+tac+toe+endgame) |
| `house_votes.csv` | 435 | `party` | [UCI Congressional Voting Records](https://archive.ics.uci.edu/dataset/105/congressional+voting+records) |
| `sonar.csv` | 208 | `object` (rock or mine) | [UCI Connectionist Bench (Sonar)](https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks) |
| `red_wine_quality.csv` | 1,599 | `good_quality` (yes or no) | [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality), red wines |

## Changes made to the heart disease file

The published file has no header row and stores everything as number codes. For this app:

- Column names were added.
- The codes of the category columns were replaced by their meanings, taken from the dataset's own documentation (for example chest pain type 4 became "asymptomatic").
- The original target runs from 0 (no disease) to 4. It was reduced to `heart_disease`: "no" for 0 and "yes" for 1 to 4.

## Changes made to the other files

- **Files published without a header** (Pima, banknotes, tic-tac-toe, votes, sonar): column names were added from each dataset's documentation, and numeric class codes were replaced by words.
- **Pima diabetes:** in the five columns glucose, blood pressure, skin thickness, insulin and BMI, a 0 is not a real measurement (the dataset uses it for "not recorded"), so it was turned into a missing value.
- **Congress votes:** the "?" in the original file (no clear position recorded) is read as a missing value by the app.
- **Red wine quality:** the 0 to 10 quality score was reduced to `good_quality` (yes for 7 or more) and the score column was removed, so that it cannot give the answer away. The file is also re-saved with commas instead of semicolons.
- **Wine cultivars:** the classes were renamed from 0, 1, 2 to cultivar_1, cultivar_2, cultivar_3.

## Uploads

CSV files uploaded through the app are saved in `datasets/uploads/`. That folder is ignored by git.
