const $ = id => document.getElementById(id);
const pct = x => (Number(x) * 100).toFixed(1) + '%';
let lastStamp = null;
let fields = [];
function text(tag, value) { const e = document.createElement(tag); e.textContent = value; return e; }
function cards(id, rows) {
  $(id).replaceChildren(...rows.map(([label, value, note]) => {
    const card = document.createElement('div'); card.className = 'stat';
    card.append(text('span', label), text('strong', value), text('small', note)); return card;
  }));
}
function table(id, columns, rows, selected) {
  const element = $(id); element.querySelectorAll('thead,tbody').forEach(e => e.remove());
  const head = document.createElement('thead'), hrow = document.createElement('tr');
  columns.forEach(c => {const th = text('th', c); th.scope = 'col'; hrow.append(th);}); head.append(hrow);
  const body = document.createElement('tbody');
  rows.forEach(row => { const tr = document.createElement('tr'); if (row[0] === selected) tr.className = 'chosen'; row.forEach(v => tr.append(text('td', v))); body.append(tr); });
  element.append(head, body);
}
async function api(path, data) {
  if (window.hostedApi) return window.hostedApi(path, data);
  const options = data === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)};
  const response = await fetch(path, options); const value = await response.json();
  if (!response.ok) throw new Error(value.error || 'Request failed'); return value;
}
function showError(message) { $('error').textContent = message || ''; $('error').hidden = !message; }
async function refresh() {
  try {
    const state = await api('/api/status');
    $('run').disabled = state.running; $('predict-button').disabled = state.running;
    $('status').textContent = state.status + (state.running ? ` (${state.elapsed}s)` : '');
    showError(state.error);
    if (!state.metrics) return;
    const a = state.audit, m = state.metrics, score = m.test_metrics[m.selected_model];
    cards('stats', [['Employee records',a.rows,'IBM HR dataset'],['Labelled left',a.left,pct(a.left/a.rows)+' of the dataset'],['Training records',a.train_rows,'Five-fold model comparison'],['Test records',a.test_rows,a.test_left+' actual leavers']]);
    table('comparison',['Model','Average precision','Recall','F1','Accuracy'],state.comparison.map(r=>[r.model,Number(r.average_precision_mean).toFixed(3),pct(r.recall_mean),Number(r.f1_mean).toFixed(3),pct(r.accuracy_mean)]),m.selected_model);
    $('selected').textContent = m.selected_model + ' · held-out test';
    $('run-time').textContent = state.completed ? 'Live run completed at ' + state.completed : 'Saved experiment';
    cards('test-stats',[['Recall',pct(score.recall),'Actual leavers detected'],['Precision',pct(score.precision),'Predicted leavers who left'],['F1',score.f1.toFixed(3),'Precision / recall balance'],['Accuracy',pct(score.accuracy),'All correct classifications']]);
    const [[tn,fp],[fn,tp]] = score.confusion_matrix;
    $('interpretation').textContent = `Detected ${tp} of ${tp+fn} actual leavers. Missed ${fn}. Incorrectly flagged ${fp} employees who stayed; correctly classified ${tn} stayers. The baseline detects no leavers despite ${pct(m.test_metrics['Majority baseline'].accuracy)} accuracy.`;
    if (lastStamp !== state.completed) { lastStamp = state.completed; if (!state.running) { const stamp = Date.now(); $('eda').src='/eda.png?v='+stamp; $('evaluation').src='/evaluation.png?v='+stamp; } }
  } catch (e) { showError((window.hostedApi ? 'Cannot load the hosted demo. ' : 'Cannot reach the local demo. Keep the demo.py terminal running. ') + e.message); }
}
$('run').addEventListener('click', async () => { $('run').disabled=true; showError(''); try { await api('/api/run',{}); await refresh(); } catch(e){showError(e.message); $('run').disabled=false;} });
async function loadData() { const d=await api('/api/data'); table('data-table',d.columns,d.rows.map(r=>d.columns.map(c=>r[c]))); }
function pretty(name) {return name.replace(/([a-z])([A-Z])/g,'$1 $2');}
async function loadFields() {
  fields=(await api('/api/fields')).fields;
  const main=['Age','MonthlyIncome','OverTime','JobSatisfaction','YearsAtCompany','TotalWorkingYears'];
  for (const field of fields) {
    const label=text('label',pretty(field.name)), input=document.createElement(field.kind==='number'?'input':'select');
    input.name=field.name; input.id='field-'+field.name; label.htmlFor=input.id; input.required=true;
    if(field.kind==='number'){
      input.type='number';input.min=field.min;input.step=field.step ?? 'any';
      if(field.max !== undefined) input.max=field.max;
      if(field.max !== undefined) label.append(text('small',` (scale ${field.min}–${field.max})`));
    }
    else field.options.forEach(value=>{const o=text('option',value);o.value=value;input.append(o);});
    input.value=field.value;label.append(input);$(main.includes(field.name)?'main-fields':'extra-fields').append(label);
  }
}
$('predict-form').addEventListener('submit', async event => {
  event.preventDefault(); const button=$('predict-button');button.disabled=true;button.textContent='Predicting…';
  try {
    const values={};fields.forEach(f=>{const v=$('field-'+f.name).value;values[f.name]=f.kind==='number'?Number(v):v;});
    const result=await api('/api/predict',values), panel=$('prediction-result');panel.replaceChildren();
    const label=text('p','PREDICTED DATASET LABEL');label.className='eyebrow';
    panel.append(label,text('h3',result.prediction),text('p',result.model),text('strong',result.score.toFixed(3)),text('p','Model score for “left” · threshold 0.500'),text('small','This score is not a calibrated resignation probability. This hypothetical profile is for demonstrating inference only.'));
    if(result.outside_training?.length) panel.append(text('p',`Accepted values outside the training range: ${result.outside_training.map(pretty).join(', ')}. Predictions beyond the training examples may be less reliable.`));
  } catch(e) { const panel=$('prediction-result');panel.replaceChildren(text('h3','Check the inputs'),text('p',e.message)); }
  finally{button.disabled=false;button.textContent='Predict this profile';}
});
Promise.all([refresh(),loadData(),loadFields()]).catch(e=>showError(e.message));
if (!window.hostedApi) setInterval(refresh,2500);

// Optional structured actions for browsers that support WebMCP.
if (!window.hostedApi && document.modelContext?.registerTool) {
  const lifecycle = new AbortController();
  const schema = {type:'object',properties:{},additionalProperties:false};
  const check = input => { if (!input || typeof input !== 'object' || Array.isArray(input) || Object.keys(input).length) throw new Error('Expected an empty object.'); };
  for (const tool of [
    {name:'read_experiment_results',description:'Read the saved or completed experiment results and current training status.',inputSchema:schema,annotations:{readOnlyHint:true},async execute(input){check(input);return api('/api/status');}},
    {name:'start_model_comparison',description:'Start a live rerun of the fixed training experiment and update the visible status. Completion is asynchronous.',inputSchema:schema,annotations:{readOnlyHint:false},async execute(input){check(input);const value=await api('/api/run',{});await refresh();return value;}}
  ]) {
    try { Promise.resolve(document.modelContext.registerTool(tool,{signal:lifecycle.signal})).catch(()=>{}); } catch (_) { /* Ordinary browser controls remain available. */ }
  }
  addEventListener('pagehide',()=>lifecycle.abort(),{once:true});
}
