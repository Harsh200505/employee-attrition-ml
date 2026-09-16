"""Build and execute a readable notebook from the same experiment code."""
import ast
import base64
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
source = (ROOT / 'train.py').read_text(encoding='utf-8')
tree = ast.parse(source)
functions = {node.name: ast.get_source_segment(source, node)
             for node in tree.body if isinstance(node, ast.FunctionDef)}
header = source[:source.index('def load_data')]
header = header.replace("ROOT = Path(__file__).resolve().parent", "ROOT = Path.cwd()\nif not (ROOT / 'data' / 'WA_Fn-UseC_-HR-Employee-Attrition.csv').exists():\n    ROOT = ROOT / 'employee_turnover'")
cells = []
def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))
def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md('''# Employee Attrition Prediction Using Machine Learning
### Third-year student project

**Question:** Can a model distinguish employees labelled as having left from those labelled as having stayed, using the available HR attributes?

**Data:** [IBM HR Analytics on Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset), 1,470 records and 35 columns. This is an educational sample, not a prospective study of future resignations. The dataset has no defined prediction time horizon.

**Workflow:** audit → stratified train/test split → training-only exploration → preprocessing pipelines → five-fold model comparison → final test evaluation → limitations.

Run cells from top to bottom. The final test is for reporting; do not change models based on its results and claim it remains unseen.''')
md('## 1. Libraries and reproducibility\nSeed 42 fixes the split and randomized model settings. The project requirements record library versions.')
code(header)
md('## 2. Load and check the data\nThe target is `Attrition`: Yes = 1, No = 0. Check missing values, duplicate records, employee IDs and class balance before modelling.')
code(functions['load_data'] + "\n\ndf = load_data()\nprint('Shape:', df.shape)\nprint('Missing cells:', df.isna().sum().sum())\nprint('Duplicate rows:', df.duplicated().sum())\nprint(df['Attrition'].value_counts())\nprint(df.head().to_string(index=False))")
md('''## 3. Hold out 20% before exploration
Stratification preserves approximately the same class proportions in each subset. `EmployeeNumber` is an identifier; `EmployeeCount`, `StandardHours` and `Over18` are documented constant fields. Gender and marital status are excluded as a project choice, but other inputs can still act as proxies; fairness is not established.

We retain age and work-history inputs for this educational comparison and explicitly acknowledge the limitation. Ordinal numeric survey responses are treated as numeric, which assumes equal spacing for the linear model.''')
code(functions['split_data'] + "\n\nX_train, X_test, y_train, y_test = split_data(df)\nOUT.mkdir(exist_ok=True)\nassert set(X_train.index).isdisjoint(X_test.index)\nprint('Training:', len(X_train), 'Testing:', len(X_test))\nprint('Training attrition fraction:', round(y_train.mean(), 4))")
md('## 4. Explore training data\nCompare the proportion leaving **within each group**, not raw counts across unequal groups. These are associations and do not prove that overtime or satisfaction causes attrition.')
code(functions['exploratory_analysis'] + '\n\nexploratory_analysis(X_train, y_train)\nprint(pd.read_csv(OUT / "training_group_rates.csv").round(3).to_string(index=False))')
md('''## 5. Preprocessing and models
Numeric inputs are imputed with the training median and standardized. Categories are imputed with the training mode and one-hot encoded. All transformations stay inside each pipeline to prevent validation data influencing preprocessing.

- **Majority baseline:** always predicts the common class.
- **Logistic Regression:** a simple, explainable linear classifier.
- **Decision Tree:** bounded depth to limit overfitting.
- **Random Forest:** averages multiple trees.

The learned models use class weights to give the minority class more influence. Their output scores are not validated as calibrated personal resignation probabilities.''')
code(functions['make_models'] + '\n\nmodels = make_models(X_train)')
md('''## 6. Select using training cross-validation
Five stratified folds provide repeated validation within training data. Choose the highest mean **average precision (AP)**, a ranking metric emphasizing the minority class. AP is not ordinary precision and is not the same calculation as trapezoidal PR area. Fold standard deviations describe variation, not confidence intervals. No hyperparameter search or threshold tuning is performed.''')
code(functions['compare_models'] + '\n\ncomparison = compare_models(models, X_train, y_train)\ncomparison.to_csv(OUT / "cv_comparison.csv", index=False)\nselected = comparison.iloc[0]["model"]\nprint(comparison.round(3).to_string(index=False))\nprint("Selected:", selected)')
md('''## 7. Final held-out evaluation
Fit the selected model on all training rows and compare it with the baseline. Use the predefined 0.5 threshold.

**Precision:** among predicted leavers, how many left? **Recall:** among actual leavers, how many were detected? **F1:** balances precision and recall. **ROC-AUC/AP:** evaluate ranking across thresholds. In the confusion matrix, rows are actual labels and columns are predictions; the order is stayed, left.''')
code(functions['final_evaluation'] + '\n\nresults, fitted = final_evaluation(models, selected, X_train, X_test, y_train, y_test)\nprint(json.dumps(results, indent=2))')
md('''## 8. What to explain in your viva
1. Why can accuracy be misleading when most employees stayed?
2. Why must scaling and encoding happen inside cross-validation?
3. What is the difference between precision and recall for the left class?
4. Why was the model selected using training folds instead of test accuracy?
5. What does a false positive mean in this project?

**Limitations:** small educational dataset; one random holdout; no external or time-based validation; no causal findings; no fairness or calibration assessment. The model classifies dataset labels rather than establishing when someone will resign. It should not be used to make employment decisions.

**Future work:** nested cross-validation, validation-only threshold selection, calibration assessment, and external validation if an appropriate dataset becomes available.

**Attribution:** dataset from Kaggle; implementation and explanations prepared with Codex assistance. Review and understand the code before presenting it as your project.

Method reference: [scikit-learn guidance on avoiding data leakage](https://scikit-learn.org/1.8/common_pitfalls.html).''')

notebook = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {
    'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
namespace = {}
count = 0
for cell in notebook.cells:
    if cell.cell_type != 'code':
        continue
    count += 1
    capture = io.StringIO()
    with redirect_stdout(capture):
        exec(compile(cell.source, f'notebook-cell-{count}', 'exec'), namespace)
    cell.execution_count = count
    cell.outputs = []
    if capture.getvalue():
        cell.outputs.append(nbf.v4.new_output('stream', name='stdout', text=capture.getvalue()))
    image_name = ('training_eda.png' if 'exploratory_analysis(X_train, y_train)\nprint' in cell.source
                  else 'test_evaluation.png' if 'results, fitted = final_evaluation' in cell.source else None)
    if image_name:
        cell.outputs.append(nbf.v4.new_output('display_data', data={
            'image/png': base64.b64encode((ROOT / 'results' / image_name).read_bytes()).decode('ascii'),
            'text/plain': image_name}))
nbf.validate(notebook)
nbf.write(notebook, ROOT / 'Employee_Attrition_Project.ipynb')
print('Executed and validated', count, 'notebook code cells.')
