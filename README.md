# Employee Attrition Prediction Using Machine Learning

An educational classification project using the IBM HR attrition dataset from Kaggle. It compares three classifiers with a majority baseline and provides both a local Python demo and a static Vercel demo.

## Start here

- **See the findings:** [Results and interpretation](RESULTS.md), including the confusion matrix and limitations.
- **Explore the hosted demo:** deploy this repository using the instructions below. A verified public deployment URL is not recorded in this repository; add the actual URL here only after opening and checking it.
- **Run the full experiment locally:** follow [Local demonstration](#local-demonstration) or the [five-minute faculty walkthrough](FACULTY_DEMO.md). The hosted comparison shows saved results; local Python can retrain.
- **Read the method and code:** [notebook](Employee_Attrition_Project.ipynb) and [train.py](train.py).

**At a glance (recorded 16 September 2026):** 1,470 records; 237 labelled “left” (16.1%). Logistic Regression was selected by training-fold average precision. On 294 held-out records it detected 29 of 47 leavers (61.7% recall), with 35.8% precision and 52 false alarms. The majority baseline reached 84.0% accuracy while detecting no leavers. See [RESULTS.md](RESULTS.md) for the full comparison and metric definitions.

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

Commit the regenerated `web/` files before redeploying. After deployment, open the assigned Vercel URL, check the comparison and a hypothetical prediction, and place that verified URL in the Start here section and your project profile. The exporter deliberately
fails if the selected model is no longer Logistic Regression. The test compares
browser inference against Python for every training row and checks invalid inputs.

## Local demonstration

For a live faculty presentation, run `.\.mlenv\Scripts\python.exe demo.py` and open **http://127.0.0.1:8501**. The screen shows the dataset, runs the actual model comparison and supports predictions for hypothetical profiles. Read `FACULTY_DEMO.md` for a five-minute walkthrough. `RUN_DEMO.cmd` is a double-click launcher for this computer.

Open `Employee_Attrition_Project.ipynb`. It contains explanations, executable code, saved results and charts. Start with the problem statement and class distribution, then follow the split, exploration, model comparison and final evaluation.

**Objective:** classify the dataset's `Attrition` label and compare simple supervised learning models. The data does not establish a prediction window such as “will leave in six months.” Treat this as an educational classification experiment.

## Run on this computer

From the repository root:

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

## Screenshots and project proof

Capture screenshots from a running demo; do not present the saved chart as proof that a new training run occurred. For a portfolio or Havloc submission, use this order:

1. **Overview:** the hosted or local demo with the dataset summary visible. Include the browser address only if it is a verified public deployment or the local address is clearly labelled “local demo.”
2. **Model comparison:** show the table and its “saved experiment” label. For proof of a fresh run, use the local demo, click **Run model comparison**, and capture the “Live run complete” status.
3. **Held-out evaluation:** show the confusion matrix and precision-recall curve, with the 29 detected leavers, 18 misses and 52 false alarms explained in the caption.
4. **Hypothetical prediction:** submit a sample profile and show the model output. Label the score as an uncalibrated model score, not an employee's resignation probability.

Save readable images in a `screenshots/` folder if you choose to commit them, and link them here with descriptive alt text. Crop personal browser details and any real employee information. The committed `results/test_evaluation.png` is a generated experiment figure, not a screenshot of the hosted application. The repository currently does not document a verified public demo URL or committed interface screenshots; avoid claiming either until added.

## Internship-facing project summary

**Short description:** Built a reproducible Python machine-learning experiment to classify the “left” versus “stayed” labels in the IBM HR educational dataset. I compared Logistic Regression, Decision Tree and Random Forest against a majority baseline, kept preprocessing inside five-fold cross-validation, and selected by average precision on training data. The selected Logistic Regression detected 29 of 47 leavers on the held-out test set (61.7% recall, 35.8% precision), while the baseline detected none despite 84.0% accuracy. I also provide a local retraining demo and a static browser demo that runs the exported fitted model. This is an academic demonstration, not a validated HR decision tool. Implementation and explanations were prepared with Codex assistance; describe your own contribution and understanding accurately.

**What to discuss in an interview:** why accuracy hid the minority class, why preprocessing belongs inside each validation fold, how the test set was held out for selection, and what the 52 false alarms imply. Do not claim a future resignation horizon, causal factors, calibrated probabilities, fairness, production readiness, or a deployed Python backend.

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

Future student extensions include validation-only threshold selection, nested model tuning, calibration checks, and external or time-based validation. Preserve this first experiment when making later changes; a previously inspected test set is no longer an unseen benchmark for new decisions.

Dataset source: [Kaggle IBM HR Analytics](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset). Method reference: [scikit-learn data leakage guidance](https://scikit-learn.org/1.8/common_pitfalls.html).

Code and explanations were prepared with Codex assistance. Review, adapt and understand them, and follow your college's attribution requirements.
