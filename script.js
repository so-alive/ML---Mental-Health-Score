/* ============================================================
   Config
   ============================================================ */
const API_BASE_URL = "http://127.0.0.1:8000";
const PREDICT_ENDPOINT = `${API_BASE_URL}/predict`;

// The backend returns a raw regression value. We don't know the
// exact training range, so the gauge assumes a 0–10 scale (the
// common range for "mental health score" datasets). The band
// thresholds below are only a rough guide, not a clinical scale.
const GAUGE_MIN = 0;
const GAUGE_MAX = 10;

const SCORE_BANDS = [
  { max: 3, label: "Signs of strain" },
  { max: 5.5, label: "Room to recover" },
  { max: 7.5, label: "Holding steady" },
  { max: Infinity, label: "Thriving" },
];

/* ============================================================
   Element refs
   ============================================================ */
const form = document.getElementById("predictForm");
const submitBtn = document.getElementById("submitBtn");
const formError = document.getElementById("formError");
const panelInner = document.getElementById("panelInner");

const resultView = document.getElementById("resultView");
const resultNumber = document.getElementById("resultNumber");
const resultBand = document.getElementById("resultBand");
const resultGaugeFill = document.getElementById("resultGaugeFill");
const recalculateBtn = document.getElementById("recalculateBtn");

// Numeric fields and the type they should be cast to before sending.
const NUMERIC_FIELDS = {
  age: "int",
  avg_daily_usage_hours: "float",
  daily_unlocks: "int",
  study_hours: "float",
  physical_activity_hours: "float",
  sleep_hours_per_night: "float",
};

/* ============================================================
   Helpers
   ============================================================ */
function clearFieldErrors() {
  form.querySelectorAll(".field").forEach((el) => el.classList.remove("has-error"));
  form.querySelectorAll(".field__error").forEach((el) => (el.textContent = ""));
}

function setFieldError(name, message) {
  const input = form.elements[name];
  if (!input) return;
  const fieldWrap = input.closest(".field");
  const errorEl = form.querySelector(`[data-error-for="${name}"]`);
  if (fieldWrap) fieldWrap.classList.add("has-error");
  if (errorEl) errorEl.textContent = message;
}

function showFormError(message) {
  formError.textContent = message;
  formError.hidden = false;
}

function hideFormError() {
  formError.hidden = true;
  formError.textContent = "";
}

function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  submitBtn.classList.toggle("is-loading", isLoading);
  submitBtn.querySelector(".btn__label").textContent = isLoading
    ? "Analyzing your answers…"
    : "Estimate my score";
}

function scoreBand(score) {
  const found = SCORE_BANDS.find((b) => score <= b.max);
  return found ? found.label : "—";
}

function collectFormData() {
  const formData = new FormData(form);
  const payload = {};

  for (const [key, rawValue] of formData.entries()) {
    if (key in NUMERIC_FIELDS) {
      const num = Number(rawValue);
      payload[key] = NUMERIC_FIELDS[key] === "int" ? Math.trunc(num) : num;
    } else {
      payload[key] = rawValue.trim();
    }
  }

  return payload;
}

/**
 * Basic client-side validation mirroring the backend's Pydantic
 * constraints, so the user gets instant feedback before we ever
 * hit the network.
 */
function validatePayload(payload) {
  const errors = {};

  const numberChecks = [
    ["age", 10, 100],
    ["avg_daily_usage_hours", 0, 24],
    ["daily_unlocks", 0, Infinity],
    ["study_hours", 0, 24],
    ["physical_activity_hours", 0, 2],
    ["sleep_hours_per_night", 0, 24],
  ];

  for (const [field, min, max] of numberChecks) {
    const value = payload[field];
    if (value === "" || value === null || Number.isNaN(value)) {
      errors[field] = "This field is required.";
    } else if (value < min || value > max) {
      errors[field] = `Must be between ${min} and ${max === Infinity ? "∞" : max}.`;
    }
  }

  const requiredSelects = [
    "gender",
    "academic_level",
    "most_used_platform",
    "purpose_of_use",
    "stress_level",
  ];
  for (const field of requiredSelects) {
    if (!payload[field]) errors[field] = "Please choose an option.";
  }

  if (!payload.country) errors.country = "Please enter a country.";

  return errors;
}

/* ============================================================
   Rendering the result
   ============================================================ */
function renderResult(score) {
  form.hidden = true;
  resultView.hidden = false;

  resultNumber.textContent = score.toFixed(2);
  resultBand.textContent = scoreBand(score);

  const pct = Math.max(
    0,
    Math.min(100, ((score - GAUGE_MIN) / (GAUGE_MAX - GAUGE_MIN)) * 100)
  );
  // Reset then animate on next frame for a clean transition.
  resultGaugeFill.style.width = "0%";
  requestAnimationFrame(() => {
    resultGaugeFill.style.width = `${pct}%`;
  });

  resultView.scrollIntoView({ behavior: "smooth", block: "start" });
}

function resetToForm() {
  resultView.hidden = true;
  form.hidden = false;
  hideFormError();
}

/* ============================================================
   API call
   ============================================================ */
async function submitPrediction(payload) {
  let response;
  try {
    response = await fetch(PREDICT_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (networkErr) {
    throw new Error(
      "Couldn't reach the prediction server. Make sure the FastAPI backend is running at " +
        API_BASE_URL +
        "."
    );
  }

  let data = null;
  try {
    data = await response.json();
  } catch (_) {
    // no JSON body — fall through to status-based error below
  }

  if (!response.ok) {
    if (response.status === 422 && data && Array.isArray(data.detail)) {
      // FastAPI validation error shape: [{ loc: ["body","age"], msg: "..." }, ...]
      data.detail.forEach((issue) => {
        const field = issue.loc?.[issue.loc.length - 1];
        if (field) setFieldError(field, issue.msg);
      });
      throw new Error("Please fix the highlighted fields and try again.");
    }

    const detailMsg =
      (data && (data.detail || data.message)) ||
      `Request failed with status ${response.status}.`;
    throw new Error(typeof detailMsg === "string" ? detailMsg : "Something went wrong.");
  }

  if (!data || typeof data.predicted_mental_health_score !== "number") {
    throw new Error("The server responded, but no score was returned.");
  }

  return data.predicted_mental_health_score;
}

/* ============================================================
   Events
   ============================================================ */
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  hideFormError();
  clearFieldErrors();

  const payload = collectFormData();
  const clientErrors = validatePayload(payload);

  if (Object.keys(clientErrors).length > 0) {
    Object.entries(clientErrors).forEach(([field, msg]) => setFieldError(field, msg));
    showFormError("A few fields need your attention before we can run the estimate.");
    return;
  }

  setLoading(true);
  try {
    const score = await submitPrediction(payload);
    renderResult(score);
  } catch (err) {
    showFormError(err.message || "Something went wrong. Please try again.");
  } finally {
    setLoading(false);
  }
});

recalculateBtn.addEventListener("click", () => {
  resetToForm();
  panelInner.scrollIntoView({ behavior: "smooth", block: "start" });
});
