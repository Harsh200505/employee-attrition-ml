# Employee Attrition Prediction Using Machine Learning

A third-year undergraduate ML project using the IBM HR attrition dataset from Kaggle.

## Start here

## Deploy on Vercel

Import this repository at https://vercel.com/new. Use **Other** as the framework,
leave the root directory as the repository root, and deploy. `vercel.json` serves
the committed `web/` directory without a build step or Python server.

The hosted version displays saved experiment results and computes real predictions
in the browser using the fitted Logistic Regression coefficients, original scaling,
and one-hot encoding. Profile inputs stay in the browser. Model comparison retraining
remains available in the local Python demo; the hosted comparison button opens the
saved results instead of starting training.

After changing the training data or model, regenerate and check the hosted files:

```powershell
.\.mlenv\Scripts\python.exe build_web.py
node test_web.cjs
```

Commit the regenerated `web/` files before redeploying. The exporter deliberately
fails if the selected model is no longer Logistic Regression. The test compares
browser inference against Python for every training row and checks invalid inputs.

## Local demonstration

For a live faculty presentation, run `.\.mlenv\Scripts\python.exe demo.py` and open **http://127.0.0.1:8501**. The screen shows the dataset, runs the actual model comparison and supports predictions for hypothetical profiles. Read `FACULTY_DEMO.md` for a five-minute walkthrough. `RUN_DEMO.cmd` is a double-click launcher for this computer.

Open `Employee_Attrition_Project.ipynb`. It contains explanations, executable code, saved results and charts. Start with the problem statement and class distribution, then follow the split, exploration, model comparison and final evaluation.

**Objective:** classify the dataset's `Attrition` label and compare simple supervised learning models. The data does not establish a prediction window such as “will leave in six months.” Treat this as an educational classification experiment.

## Run on this computer

From this `employee_turnover` folder:

```powershell
.\.mlenv\Scripts\python.exe train.py
```

The prepared environment uses Python 3.12. The earlier `.venv` environment is an incomplete Python 3.9 setup and is not used.

## Set up on a teammate's computer

Download or clone this repository and open its root folder in VS Code. Install Python 3.12, then use the VS Code terminal to run:

```powershell
python -m venv .mlenv
.\.mlenv\Scripts\python.exe -m pip install -r requirements.txt
.\.mlenv\Scripts\python.exe demo.py
```

Open **http://127.0.0.1:8501** in a browser and keep the terminal running. Use **Run model comparison** to reproduce training, or try a hypothetical employee profile. The dataset and recorded results are already included. No API key or online model service is required.

For macOS/Linux, use `python3 -m venv .mlenv`, then `.mlenv/bin/python -m pip install -r requirements.txt` and `.mlenv/bin/python demo.py`. To run only the training script, replace `demo.py` with `train.py`.

The Python environments are local installations and are intentionally excluded from Git. `requirements-lock.txt` records the complete package versions from the original Windows run; `requirements.txt` lists the project's direct dependencies.

To interact with the notebook, open it in a Jupyter-capable editor and select a Python environment with these dependencies and a notebook kernel. The saved notebook can be read without rerunning it. `build_notebook.py` regenerates and executes its cells without requiring a Jupyter server.

## Method

1. Audit missing values, duplicate rows and class balance.
2. Reserve a stratified 20% test set with random seed 42.
3. Explore training data only, using within-group attrition proportions.
4. Fit imputation, scaling and one-hot encoding within each training fold.
5. Compare a majority baseline, Logistic Regression, Decision Tree and Random Forest using five-fold stratified cross-validation.
6. Select by mean average precision before inspecting test metrics.
7. Evaluate the selected model and baseline on the held-out test set at threshold 0.5.

Class weights address imbalance in model training. We do not apply SMOTE or tune thresholds in this first experiment. Accuracy, precision, recall, F1, ROC-AUC, average precision and a confusion matrix are reported. The test set is not used to select the winner.

## Files

- `Employee_Attrition_Project.ipynb`: main learning and presentation notebook.
- `train.py`: reproducible experiment.
- `RESULTS.md`: measured findings and interpretation.
- `data/`: original CSV, source information and archive.
- `results/`: audit, saved split, cross-validation results, test predictions and figures.

## What you should be able to explain

- Classification versus regression; feature versus target.
- Class imbalance and the limitations of accuracy.
- Train/test split and cross-validation.
- One-hot encoding, scaling and data leakage.
- How Logistic Regression, a tree and a forest differ.
- False positives, false negatives, precision, recall and F1.
- Why correlation is not causation and why test results do not prove real-world readiness.

## Scope and limitations

This is an educational dataset with a small sample and one random test split. There is no external validation, time-based validation, calibration assessment or fairness audit. Employee identifiers, constant fields, gender and marital status are excluded. Other attributes can still act as proxies, so exclusion does not establish fairness. Model scores are not validated individual resignation probabilities. Results should not guide employment decisions.

Future student extensions include validation-only threshold selection, nested model tuning, and a simple demonstration interface after understanding the model. Preserve this first experiment when making later changes; a previously inspected test set is no longer an unseen benchmark for new decisions.

Dataset source: [Kaggle IBM HR Analytics](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset). Method reference: [scikit-learn data leakage guidance](https://scikit-learn.org/1.8/common_pitfalls.html).

Code and explanations were prepared with Codex assistance. Review, adapt and understand them, and follow your college's attribution requirements.
