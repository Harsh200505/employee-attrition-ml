/* Inference uses the original fitted scaler, encoder, and logistic coefficients. */
function predictExported(model, fields, values) {
  if (!values || Object.keys(values).length !== fields.length || fields.some(f => !(f.name in values))) throw new Error('Supply all displayed model inputs.');
  const outside_training = [];
  for (const f of fields) {
    const v = values[f.name];
    if (f.kind === 'number') {
      if (typeof v !== 'number' || !Number.isFinite(v) || v < f.min || (f.max !== undefined && v > f.max) || (f.step === 1 && !Number.isInteger(v))) throw new Error(`Check the value for ${f.name}.`);
      if (v < f.training_min || v > f.training_max) outside_training.push(f.name);
    } else if (!f.options.includes(v)) throw new Error(`Invalid value for ${f.name}.`);
  }
  if (values.YearsAtCompany > values.TotalWorkingYears) throw new Error('Years at company cannot exceed total working years.');
  for (const name of ['YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']) if (values[name] > values.YearsAtCompany) throw new Error(`${name} cannot exceed YearsAtCompany.`);
  const features = model.numeric.map((n,i) => (values[n]-model.mean[i])/model.scale[i]);
  model.categorical.forEach((n,i) => model.categories[i].forEach(c => features.push(values[n] === c ? 1 : 0)));
  const z = model.intercept + features.reduce((sum,x,i) => sum+x*model.coefficients[i],0);
  const score = z >= 0 ? 1/(1+Math.exp(-z)) : Math.exp(z)/(1+Math.exp(z));
  return {model:model.name, score, prediction:score>=0.5?'Left':'Stayed', threshold:0.5, outside_training};
}
if (typeof window !== 'undefined') {
  const ready = fetch('/model-data.json').then(r => {if (!r.ok) throw new Error('Cannot load the saved model.'); return r.json();});
  window.hostedApi = async (path, values) => {
    const bundle = await ready;
    if (path === '/api/status') return bundle.status;
    if (path === '/api/data') return bundle.data;
    if (path === '/api/fields') return {fields:bundle.fields};
    if (path === '/api/predict') return predictExported(bundle.model,bundle.fields,values);
    if (path === '/api/run') {document.getElementById('experiment').scrollIntoView({behavior:'smooth'});return {started:false};}
    throw new Error('Unknown action.');
  };
}
if (typeof module !== 'undefined') module.exports = {predictExported};
