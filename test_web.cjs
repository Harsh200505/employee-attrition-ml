const assert = require('node:assert/strict');
const fs = require('node:fs');
const {predictExported} = require('./browser_model.js');
const bundle = JSON.parse(fs.readFileSync('web/model-data.json','utf8'));
const {cases,expected} = JSON.parse(fs.readFileSync('web_test_cases.json','utf8'));
let maxError = 0;
for (let i=0;i<cases.length;i++) {
  const result = predictExported(bundle.model,bundle.fields,cases[i]);
  maxError = Math.max(maxError,Math.abs(result.score-expected[i]));
  assert.equal(result.prediction,expected[i]>=0.5?'Left':'Stayed');
}
assert.ok(maxError < 1e-12, `Score mismatch: ${maxError}`);
const defaults = Object.fromEntries(bundle.fields.map(f=>[f.name,f.value]));
assert.throws(()=>predictExported(bundle.model,bundle.fields,{...defaults,Age:NaN}));
assert.throws(()=>predictExported(bundle.model,bundle.fields,{...defaults,OverTime:'invalid'}));
assert.throws(()=>predictExported(bundle.model,bundle.fields,{...defaults,YearsAtCompany:100,TotalWorkingYears:2}));
assert.throws(()=>predictExported(bundle.model,bundle.fields,{}));
console.log(`PASS: ${cases.length} Python/JavaScript predictions match; maximum error ${maxError}. Invalid inputs rejected.`);
