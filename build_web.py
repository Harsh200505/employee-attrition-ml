"""Export the fitted logistic model and saved report for static Vercel hosting."""
import csv
import json
import shutil
from pathlib import Path

import demo

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'web'


def main():
    train, X, y, fields, _ = demo.prediction_setup()
    selected = demo.SAVED['metrics']['selected_model']
    if selected != 'Logistic Regression':
        raise ValueError('The browser exporter supports Logistic Regression only.')
    pipeline = train.make_models(X)[selected].fit(X, y)
    prep = pipeline.named_steps['prepare']
    numeric = prep.transformers_[0][2]
    categorical = prep.transformers_[1][2]
    scaler = prep.named_transformers_['numeric'].named_steps['scale']
    encoder = prep.named_transformers_['categorical'].named_steps['encode']
    model = pipeline.named_steps['model']
    export = dict(name=selected, numeric=numeric, categorical=categorical,
                  mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                  categories=[c.tolist() for c in encoder.categories_],
                  coefficients=model.coef_[0].tolist(), intercept=float(model.intercept_[0]))
    with train.DATA.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))[:8]
    columns = ['Age', 'Department', 'JobRole', 'MonthlyIncome', 'OverTime', 'YearsAtCompany', 'Attrition']
    bundle = dict(model=export, fields=fields, status={**demo.snapshot(), 'status': 'Saved experiment results. Predictions run in your browser.'},
                  data=dict(columns=columns, rows=[{k: r[k] for k in columns} for r in rows]))
    OUT.mkdir(exist_ok=True)
    for name in ['app.js', 'style.css']:
        shutil.copyfile(ROOT / 'demo' / name, OUT / name)
    shutil.copyfile(ROOT / 'browser_model.js', OUT / 'browser_model.js')
    html = (ROOT / 'demo/index.html').read_text(encoding='utf-8')
    html = html.replace('<script src="/app.js" defer>', '<script src="/browser_model.js" defer></script><script src="/app.js" defer>')
    html = html.replace('A local student demonstration', 'A student demonstration')
    html = html.replace('Run the experiment live', 'Explore the saved experiment').replace('Re-run the same split and settings used in the report.', 'View the measured results and try the trained model below.')
    html = html.replace('Run model comparison <span aria-hidden="true">→</span>', 'View model comparison <span aria-hidden="true">→</span>')
    html = html.replace('The previous results stay visible during training. Re-running this experiment is a demonstration, not a new independent test.', 'This hosted demo uses the original fitted model. To retrain the models, run demo.py from the GitHub project on your computer.')
    (OUT / 'index.html').write_text(html, encoding='utf-8')
    (OUT / 'model-data.json').write_text(json.dumps(bundle, allow_nan=False), encoding='utf-8')
    for source, dest in [('training_eda.png', 'eda.png'), ('test_evaluation.png', 'evaluation.png')]:
        shutil.copyfile(ROOT / 'results' / source, OUT / dest)
    # Reference cases are kept outside the published directory for parity testing.
    cases = X.to_dict(orient='records')
    expected = pipeline.predict_proba(X)[:, 1].tolist()
    (ROOT / 'web_test_cases.json').write_text(json.dumps(dict(cases=cases, expected=expected)), encoding='utf-8')
    print(f'Exported {selected} and {len(cases)} parity test cases to {OUT}')


if __name__ == '__main__':
    main()
