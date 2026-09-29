# How to run and present the project

## Open the demonstration

Clone or download the repository, open its root folder in VS Code, and complete the environment setup in [README.md](README.md#set-up-on-a-teammates-computer). In a PowerShell terminal at the repository root run:

```powershell
.\.mlenv\Scripts\python.exe demo.py
```

Open **http://127.0.0.1:8501** in your browser. Keep the terminal running during the presentation. On Windows, you can instead double-click `RUN_DEMO.cmd` after setting up the environment and then open the same address. If the demo is already running, simply open the address; do not start a second copy.

This demo runs entirely on the laptop using the downloaded dataset and installed libraries. Internet access is not needed for training, charts or predictions. Press Ctrl+C in the server terminal when you finish.

## Hosted versus local demo

The Vercel site displays the saved September 16 experiment and computes predictions in the browser with exported Logistic Regression coefficients and preprocessing. Its **View model comparison** button does not retrain. It has no Python server. Use the local demo for the live training step below; if presenting from Vercel, say “saved comparison” and skip the retraining claim. The [PR preview](https://employee-attrition-ml-git-docs-demo-and-resu-e270b6-harsh200505.vercel.app) is Ready in Vercel, but a signed-out visit on 29 September 2026 redirected to Vercel login. Public access and the comparison and prediction controls have not been verified.

## Five-minute presentation

### 1. State the problem — 30 seconds

“My project compares machine-learning models for classifying employee attrition. The target has two labels: left and stayed. I used the IBM HR educational dataset from Kaggle.”

Show the record count and dataset preview. Mention that the dataset does not specify a future prediction window.

### 2. Explain the method — 45 seconds

Show the four-step workflow. Explain the 80/20 stratified split, encoding categories and scaling numeric inputs. Preprocessing is learned only from training folds. The comparison uses Logistic Regression, Decision Tree, Random Forest and a majority baseline.

### 3. Run the actual code — about 30–60 seconds

Click **Run model comparison**. The browser invokes the existing `train.py` using the project environment. The running status is real; completion time depends on the laptop. Previously saved results remain visible and are labelled accordingly until the new run finishes.

While it runs, explain that model selection uses mean average precision across five training folds. Re-running the fixed experiment demonstrates reproducibility; it does not create a new independent test.

### 4. Discuss results — 60 seconds

Explain why Logistic Regression was selected. In the recorded run, it detects 29 of 47 actual leavers, misses 18 and incorrectly flags 52 stayers. The baseline has 84% accuracy but detects zero leavers. Point to the confusion matrix and explain one false positive and one false negative.

### 5. Demonstrate inference — 60 seconds

Go to **Try a hypothetical employee**. Click **Predict this profile** with the defaults. Then change overtime or another input and predict again. In the local demo, Python refits the fixed selected pipeline on the original training split for inference. In the hosted demo, JavaScript evaluates exported coefficients and preprocessing; both compute a score from the entered values.

Defaults are medians and modes from training data, not a real employee. All 28 model inputs are visible or expandable. Keep years-related inputs consistent. Changing one field need not change the predicted label; the model combines all inputs. The score is not a calibrated real-world resignation probability.

Numeric inputs such as income, age, distance and experience are not capped by the training dataset. Income and measured quantities allow decimals. Values outside the training range are accepted and identified in the prediction output. Defined rating scales (such as satisfaction 1–4), whole-number counts, nonnegative values and consistency between experience fields still apply.

### 6. Finish with limitations — 30 seconds

“This is an educational experiment on a small dataset. I have not shown real-world validity, causality, calibration or fairness. False alarms remain substantial. Future work could choose thresholds using training validation and test on another suitable dataset.”

## Screenshots for project proof

Capture the overview, saved model-comparison table, evaluation figure and a submitted hypothetical prediction from an app session you can actually open. The current PR preview prompts for Vercel sign-in in a signed-out browser, so use the local demo for proof until public access and both hosted interactions are checked. If a screenshot claims live retraining, use the local demo after the status says **Live run complete**. Label the hosted comparison as saved results. Show the confusion-matrix counts in a caption: 195 stayed correctly, 52 false alarms, 18 leavers missed and 29 detected. Avoid displaying real employee details or describing the output score as a calibrated probability. See the README for the suggested screenshot order.

## If faculty ask to see code

- `train.py`: data preparation, models, cross-validation and evaluation.
- `Employee_Attrition_Project.ipynb`: the same experiment with explanations and saved outputs.
- `demo.py`: local web server, live training action and prediction endpoint.
- `demo/index.html`, `demo/style.css`, `demo/app.js`: presentation interface.

Suggested walkthrough: `split_data` → `make_models` → `compare_models` → `final_evaluation` in `train.py`. Explain why the test set is excluded from preprocessing and model selection.

## Troubleshooting

- **Address already in use:** open the existing demo at the address above.
- **Browser cannot connect:** keep the `demo.py` terminal open and check its startup message.
- **Prediction rejects a profile:** correct the input message, especially years at company versus total working years.
- **Training fails:** previous results remain visible; read `results/demo_run.log`.
- **`iprint` warning in the training log:** a library compatibility warning from the existing experiment. It did not prevent completion. The demo preserves the model settings.

Implementation was prepared with Codex assistance. Understand the code and describe your contribution accurately according to your college's rules.
