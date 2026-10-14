# Classification Visualiser

An interactive dashboard for training and comparing seven classification algorithms. It starts with the UCI Mushroom dataset (is a mushroom edible or poisonous?), ships with four more sample datasets, and accepts CSV files of your own.

You pick the data and the column to predict, pick a model, adjust its settings, train it, and see how well it does on rows it has never seen.

- **Frontend:** React, TypeScript, Vite, Tailwind CSS, shadcn/ui, Recharts, Motion
- **Backend:** FastAPI serving scikit-learn

## What the app shows

| Section | Contents |
|---|---|
| **Dataset** | Choosing or uploading the data, the column to predict and the outcome to detect; then its size, class balance, how the columns relate, how each column is used, and a preview of the rows |
| **Results** | How often the outcome was missed, accuracy, F1, ROC-AUC, precision, recall, MCC, Cohen's kappa, a form to try the model on one example, the per-class report, and a CSV export of the predictions |
| **Charts** | Confusion matrix, ROC curve, precision-recall curve, learning curve, and feature importance |
| **Cross-validation** | Accuracy of each of the 5 folds, compared with their average |
| **Compare** | All seven models trained with default settings and ranked side by side, with a CSV export |

The model and its settings are chosen in the panel on the left. The app has a light and a dark theme.

## Run it on Windows

You need [Python](https://www.python.org/downloads/) 3.10 or newer and [Node.js](https://nodejs.org/) 20.19 or newer.

1. Double-click `setup.bat`. This is needed only once. It creates the Python environment in `backend/.venv` and installs the packages.
2. Double-click `start.bat`. It opens two windows (the API and the web page) and then the app in your browser at <http://localhost:5173>.

To stop the app, close both windows.

## Run it by hand (any system)

Run these from the project folder, in two terminals.

**Terminal 1: the API**

```bash
python -m venv backend/.venv
# Windows:      backend\.venv\Scripts\activate
# macOS/Linux:  source backend/.venv/bin/activate
pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --port 8000
```

**Terminal 2: the web page**

```bash
cd frontend
npm install
npm run dev
```

Then open <http://localhost:5173>. The web page sends every request that starts with `/api` to the API on port 8000.

## Project layout

```
├── Mushrooms.csv          The mushroom dataset
├── datasets/              The other sample datasets; uploads go in datasets/uploads/
├── backend/
│   ├── datasets.py        Reading a CSV, understanding its columns, the train/test split
│   ├── explore.py         What the Dataset page shows
│   ├── ml.py              The models, data preparation and scoring (scikit-learn)
│   ├── main.py            The HTTP API (FastAPI)
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── pages/         One file per section of the app
│       ├── components/    Layout, charts, glass buttons, UI parts
│       ├── state/         What is selected, trained and loaded
│       └── lib/           API client, formatting, feature names
├── legacy/                The earlier Streamlit version of the app
├── setup.bat              One-time install (Windows)
└── start.bat              Starts the app (Windows)
```

## How it works

1. The API reads the chosen CSV file and decides how to use each column: as a number, as a category, or not at all (row IDs, names and free text are left out).
2. The column to predict is turned into two outcomes: the one to detect, and everything else.
3. The data is split into 70% for training and 30% for testing (stratified, seed 42).
4. When you press **Train & evaluate**, the chosen model is fitted on the training part and scored on the test part.
5. The slower analyses (cross-validation, feature importance, learning curve) are computed when the page that shows them is opened.
6. Results are remembered per dataset, model and settings, so asking for the same thing twice is instant.

How the data is prepared for the models:

- **Number columns:** missing values are filled with the median. They are standardised for SVM, logistic regression and KNN.
- **Category columns:** a missing value counts as its own category, and each category becomes its own 0/1 column (one-hot encoding).
- **Naive Bayes** is the exception: it models number columns with a bell curve and category columns with category frequencies.

All of this is learned from the training rows only.

| Endpoint | Returns |
|---|---|
| `GET /api/meta` | The models and the settings each one accepts |
| `GET /api/datasets` | The sample datasets and the uploaded files |
| `POST /api/datasets` | Uploads a CSV file (the file is the request body, its name is in the `X-Filename` header) |
| `DELETE /api/datasets/{id}` | Removes an uploaded file |
| `POST /api/problem` | Everything on the Dataset page, for a dataset, target column and outcome |
| `POST /api/example` | One row of the test set, to try the model on |
| `POST /api/predict` | The model's answer for one example |
| `POST /api/train` | Metrics, confusion matrix, report, ROC and precision-recall curves, predictions |
| `POST /api/cross-validation` | The 5 fold scores |
| `POST /api/feature-importance` | The 15 most important features |
| `POST /api/learning-curve` | Training and validation accuracy at 9 training-set sizes |
| `POST /api/compare` | The scores of all seven models |

## Models and settings

| Model | Settings you can change | Features standardised first |
|---|---|---|
| Support Vector Machine | `C`, kernel, gamma | Yes |
| Logistic Regression | `C`, max iterations, solver | Yes |
| Random Forest | number of trees, max depth, min samples to split, max features | No |
| K-Nearest Neighbors | `k`, weight function, distance metric, algorithm | Yes |
| Decision Tree | max depth, split criterion, min samples to split, min samples per leaf, pruning alpha | No |
| Gradient Boosting | boosting stages, learning rate, max depth, subsample fraction | No |
| Naive Bayes | variance smoothing (number columns), category smoothing (category columns) | No |

## Datasets

| Dataset | Rows | Columns used | Predicts | Typical accuracy |
|---|---|---|---|---|
| Mushrooms (`Mushrooms.csv`) | 8,124 | 21 categories | edible or poisonous | 95 to 100% |
| Titanic passengers | 891 | 2 numbers, 5 categories | survived or died | about 80% |
| Heart disease | 303 | 5 numbers, 8 categories | heart disease or healthy | about 83% |
| Breast cancer | 569 | 30 numbers | malignant or benign | about 96% |
| Telecom customer churn | 7,043 | 3 numbers, 16 categories | left or stayed | about 80% |

Where the sample files come from is listed in `datasets/README.md`.

### Using your own data

On the Dataset page, press **Upload a CSV file**. The file needs:

- column names in the first line
- at least 50 rows, and a size of at most 10 MB
- one column with between 2 and 20 distinct values to predict, with at least 10 rows of each outcome

If the column to predict has more than two values, you choose one to detect and the rest are grouped together. Uploaded files are stored in `datasets/uploads/` on your own computer.

## The earlier Streamlit version

The first version of this app was a single Streamlit file. It is kept in `legacy/` with its own README and requirements. The models, their settings, the split and the metrics come from it. The data preparation is different: the Streamlit version turned category letters into numbers in alphabetical order, while this version uses one-hot encoding, so the scores on the mushroom data differ slightly between the two.

## License

Distributed under the MIT License.
