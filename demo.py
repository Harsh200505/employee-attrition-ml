"""Local faculty demonstration. Run with the project Python: python demo.py."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
import csv
import json
import math
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
PORT = 8501
LOCK = threading.RLock()
STATE = {'running': False, 'status': 'Showing saved experiment results.', 'started': None,
         'error': None, 'completed': None}
PREDICTOR = None

# These are defined scales, not the smallest/largest values in the sample.
RATING_SCALES = {
    'Education': (1, 5), 'EnvironmentSatisfaction': (1, 4),
    'JobInvolvement': (1, 4), 'JobLevel': (1, 5),
    'JobSatisfaction': (1, 4), 'PerformanceRating': (1, 4),
    'RelationshipSatisfaction': (1, 4), 'StockOptionLevel': (0, 3),
    'WorkLifeBalance': (1, 4),
}
COUNT_FIELDS = {'NumCompaniesWorked', 'TrainingTimesLastYear'}


def read_json(name):
    return json.loads((ROOT / 'results' / name).read_text(encoding='utf-8'))


def snapshot():
    with LOCK:
        state = dict(STATE)
        state['elapsed'] = round(time.time() - state['started']) if state['running'] else None
        return {**state, **SAVED}


def load_saved():
    if not (ROOT / 'results' / 'metrics.json').exists():
        return {'metrics': None, 'audit': None, 'comparison': []}
    with (ROOT / 'results' / 'cv_comparison.csv').open(encoding='utf-8', newline='') as handle:
        comparison = list(csv.DictReader(handle))
    return {'metrics': read_json('metrics.json'), 'audit': read_json('data_audit.json'),
            'comparison': comparison}


SAVED = load_saved()


def run_experiment():
    global SAVED, PREDICTOR
    try:
        result = subprocess.run([sys.executable, '-u', str(ROOT / 'train.py')],
                                cwd=ROOT, capture_output=True, text=True, timeout=300)
        (ROOT / 'results' / 'demo_run.log').write_text(result.stdout + result.stderr, encoding='utf-8')
        if result.returncode:
            raise RuntimeError('Training failed. Details are saved in results/demo_run.log.')
        updated = load_saved()
        with LOCK:
            SAVED = updated
            PREDICTOR = None
            STATE.update(status='Live run complete. Results below are from this run.',
                         completed=time.strftime('%H:%M:%S'), error=None)
    except Exception as exc:
        with LOCK:
            STATE.update(error=str(exc), status='The live run could not finish. Previous results remain visible.')
    finally:
        with LOCK:
            STATE['running'] = False


def prediction_setup():
    import train
    df = train.load_data()
    X_train, _, y_train, _ = train.split_data(df)
    fields = []
    defaults = {}
    for name in X_train.columns:
        series = X_train[name]
        if series.dtype.kind in 'biufc':
            default = int(series.median())
            field = {'name': name, 'kind': 'number', 'min': 1 if name == 'Age' else 0,
                     'step': 1 if name in COUNT_FIELDS or name in RATING_SCALES else 'any',
                     'training_min': float(series.min()), 'training_max': float(series.max()),
                     'value': default}
            if name in RATING_SCALES:
                field['min'], field['max'] = RATING_SCALES[name]
        else:
            default = str(series.mode().iloc[0])
            field = {'name': name, 'kind': 'category', 'options': sorted(series.unique().tolist()),
                     'value': default}
        fields.append(field)
        defaults[name] = default
    return train, X_train, y_train, fields, defaults


def validate_profile(values, fields):
    if not isinstance(values, dict) or set(values) != {field['name'] for field in fields}:
        raise ValueError('Supply all displayed model inputs.')
    outside_training = []
    for field in fields:
        name = field['name']
        value = values[name]
        if field['kind'] == 'number':
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{name} must be a finite number.')
            if value < field['min']:
                raise ValueError(f"{name} must be at least {field['min']}.")
            if 'max' in field and value > field['max']:
                raise ValueError(f"{name} uses a rating scale from {field['min']} to {field['max']}.")
            if field['step'] == 1 and int(value) != value:
                raise ValueError(f'{name} must be a whole number.')
            if not field['training_min'] <= value <= field['training_max']:
                outside_training.append(name)
        elif value not in field['options']:
            raise ValueError(f'Invalid value for {name}.')
    if values['YearsAtCompany'] > values['TotalWorkingYears']:
        raise ValueError('Years at company cannot exceed total working years.')
    for column in ['YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']:
        if values[column] > values['YearsAtCompany']:
            raise ValueError(f'{column} cannot exceed YearsAtCompany.')
    return outside_training


def predict(values):
    global PREDICTOR
    import pandas as pd
    with LOCK:
        if STATE['running']:
            raise ValueError('Wait for the experiment to finish before predicting.')
        train, X_train, y_train, fields, defaults = prediction_setup()
        outside_training = validate_profile(values, fields)
        selected = SAVED['metrics']['selected_model'] if SAVED['metrics'] else None
        if selected is None:
            raise ValueError('Run the experiment first.')
        if PREDICTOR is None:
            # Refit the fixed selected model using only the original training split.
            PREDICTOR = train.make_models(X_train)[selected].fit(X_train, y_train)
        row = pd.DataFrame([values], columns=X_train.columns)
        score = float(PREDICTOR.predict_proba(row)[0, 1])
        return {'model': selected, 'score': score,
                'prediction': 'Left' if score >= 0.5 else 'Stayed', 'threshold': 0.5,
                'outside_training': outside_training}


class Handler(BaseHTTPRequestHandler):
    def allowed(self):
        return self.headers.get('Host') in {f'127.0.0.1:{PORT}', f'localhost:{PORT}'}

    def send(self, content, mime='application/json', status=200):
        body = json.dumps(content).encode() if mime == 'application/json' else content
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.allowed():
            return self.send({'error': 'Invalid host'}, status=403)
        path = urlparse(self.path).path
        assets = {'/': ('demo/index.html', 'text/html; charset=utf-8'),
                  '/app.js': ('demo/app.js', 'text/javascript; charset=utf-8'),
                  '/style.css': ('demo/style.css', 'text/css; charset=utf-8'),
                  '/eda.png': ('results/training_eda.png', 'image/png'),
                  '/evaluation.png': ('results/test_evaluation.png', 'image/png')}
        try:
            if path == '/api/status':
                return self.send(snapshot())
            if path == '/api/data':
                with (ROOT / 'data' / 'WA_Fn-UseC_-HR-Employee-Attrition.csv').open(newline='', encoding='utf-8') as handle:
                    rows = list(csv.DictReader(handle))
                columns = ['Age', 'Department', 'JobRole', 'MonthlyIncome', 'OverTime', 'YearsAtCompany', 'Attrition']
                return self.send({'columns': columns, 'rows': [{k: row[k] for k in columns} for row in rows[:8]]})
            if path == '/api/fields':
                return self.send({'fields': prediction_setup()[3]})
            if path in assets:
                file, mime = assets[path]
                return self.send((ROOT / file).read_bytes(), mime)
            return self.send({'error': 'Not found'}, status=404)
        except Exception as exc:
            self.send({'error': str(exc)}, status=500)

    def do_POST(self):
        if not self.allowed() or self.headers.get('Origin') not in {f'http://127.0.0.1:{PORT}', f'http://localhost:{PORT}'}:
            return self.send({'error': 'Local requests only'}, status=403)
        if self.headers.get('Content-Type') != 'application/json':
            return self.send({'error': 'Expected JSON'}, status=415)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 <= length <= 20000:
                raise ValueError('Request too large.')
            payload = json.loads(self.rfile.read(length) or b'{}')
            if self.path == '/api/run':
                with LOCK:
                    if STATE['running']:
                        return self.send({'error': 'An experiment is already running.'}, status=409)
                    STATE.update(running=True, started=time.time(), completed=None, error=None,
                                 status='Training and comparing four models with five-fold validation…')
                    threading.Thread(target=run_experiment, daemon=True).start()
                return self.send({'started': True})
            if self.path == '/api/predict':
                return self.send(predict(payload))
            return self.send({'error': 'Not found'}, status=404)
        except (ValueError, TypeError) as exc:
            self.send({'error': str(exc)}, status=400)
        except Exception as exc:
            self.send({'error': str(exc)}, status=500)

    def log_message(self, *_):
        pass


if __name__ == '__main__':
    try:
        server = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
        print(f'Faculty demo: http://127.0.0.1:{PORT}', flush=True)
        print('Keep this terminal open. Press Ctrl+C to stop the demo.', flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    except OSError as exc:
        print(f'Could not start demo: {exc}. If the demo is already running, open http://127.0.0.1:{PORT}', flush=True)
        sys.exit(1)
