const FIELDS = [
  ["Pregnancies", "Pregnancies", "0 - 20", 6],
  ["Glucose", "Glucose (mg/dL)", "0 - 300", 148],
  ["BloodPressure", "Blood Pressure (mm Hg)", "0 - 200", 72],
  ["SkinThickness", "Skin Thickness (mm)", "0 - 100", 35],
  ["Insulin", "Insulin (mu U/ml)", "0 - 900", 0],
  ["BMI", "BMI", "0 - 70", 33.6],
  ["DiabetesPedigreeFunction", "Diabetes Pedigree Function", "0 - 3", 0.627],
  ["Age", "Age (years)", "1 - 120", 50],
];
const $ = (id) => document.getElementById(id);

// build the form inputs
$("fields").innerHTML = FIELDS.map(([key, label, hint]) =>
  `<label>${label}<input type="number" step="any" id="f-${key}" placeholder="${hint}" required></label>`
).join("");

function showError(msg) { $("error").textContent = msg; $("error").hidden = !msg; }

async function api(url, options) {
  const res = await fetch(url, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

$("sample-btn").addEventListener("click", () => {
  FIELDS.forEach(([key, , , sample]) => { $("f-" + key).value = sample; });
});

$("predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  showError("");
  const body = { model: $("model").value };
  for (const [key, label] of FIELDS) {
    const v = $("f-" + key).value;
    if (v === "") { showError(`Please enter ${label}.`); return; }
    body[key] = Number(v);
  }
  try {
    const r = await api("/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    const pct = (r.probability * 100).toFixed(1);
    $("result").hidden = false;
    $("result").className = "result " + (r.prediction ? "high" : "low");
    $("result-label").textContent = r.label;
    $("result-bar").style.width = pct + "%";
    $("result-detail").textContent = `Probability of diabetes: ${pct}% (${r.model})`;
    loadHistory();
  } catch (err) { showError(err.message); }
});

function table(el, headers, rows) {
  if (!rows.length) { el.innerHTML = ""; el.insertAdjacentHTML("afterend", ""); return false; }
  el.innerHTML = "<tr>" + headers.map((h) => `<th>${h}</th>`).join("") + "</tr>" +
    rows.map((r) => "<tr>" + r.map((c) => `<td>${c}</td>`).join("") + "</tr>").join("");
  return true;
}

async function loadMetrics() {
  try {
    const m = await api("/api/metrics");
    const p = (v) => (v * 100).toFixed(1) + "%";
    if (!table($("metrics-table"), ["Model", "Accuracy", "Precision", "Recall", "F1", "CV Acc."],
      m.map((r) => [r.name, p(r.accuracy), p(r.precision), p(r.recall), p(r.f1), p(r.cv_accuracy)])))
      $("metrics-table").innerHTML = '<tr><td class="empty">Run train_model.py to generate metrics.</td></tr>';
  } catch (err) { showError(err.message); }
}

async function loadHistory() {
  try {
    const h = await api("/api/history");
    if (!table($("history-table"),
      ["#", "Time", "Preg.", "Glucose", "BP", "Skin", "Insulin", "BMI", "DPF", "Age", "Model", "Result", "Prob."],
      h.map((r) => [r.id, r.created_at, r.pregnancies, r.glucose, r.blood_pressure, r.skin_thickness,
        r.insulin, r.bmi, r.pedigree, r.age, r.model.replace("_", " "),
        r.prediction ? "Diabetic" : "Non-diabetic", (r.probability * 100).toFixed(1) + "%"])))
      $("history-table").innerHTML = '<tr><td class="empty">No predictions yet.</td></tr>';
  } catch (err) { showError(err.message); }
}

$("clear-btn").addEventListener("click", async () => {
  if (!confirm("Delete all saved predictions?")) return;
  try { await api("/api/history", { method: "DELETE" }); loadHistory(); } catch (err) { showError(err.message); }
});

loadMetrics();
loadHistory();
