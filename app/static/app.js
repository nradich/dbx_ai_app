'use strict';

/*
 * tailspot. — browser client for the Tailspot Databricks App.
 * Implements the client side of app/API.md.
 *
 * Pure helpers come first so they can be tested in Node (e.g. via `vm`);
 * all DOM wiring lives in initApp(), which only runs when a document exists.
 * Server- and user-provided data is only ever written with textContent /
 * createElement — never parsed as HTML.
 */

const API_CONFIG_URL = '/api/config';
const API_SUBMISSIONS_URL = '/api/submissions';
const DEFAULT_CONFIG = Object.freeze({
  airport_iata: 'SAN',
  max_upload_bytes: 20971520,
  window_minutes_before: 60,
  window_minutes_after: 10,
});
const POLL_INTERVAL_MS = 3000;
const POLL_GIVE_UP_MS = 30 * 60 * 1000;
const MAX_FUTURE_MINUTES = 10;
const ACCEPTED_EXTENSIONS = /\.(jpe?g|png)$/i;
const LOOKUP_FAILED_TEXT = 'Possible departures could not be loaded. The airline prediction above is still shown.';
const NOT_FOUND_WHILE_POLLING_TEXT =
  'This submission is no longer available. The app may have restarted. Please submit the photo again.';
const POLL_REQUEST_TIMEOUT_MS = 30000;
const CONFIG_TIMEOUT_MS = 10000;
const MAX_CONSECUTIVE_POLL_FAILURES = 5;
const ACCEPTED_TYPES = ['image/jpeg', 'image/png'];
const PROGRESS_STEPS = ['upload', 'model', 'schedule', 'record'];
const WEEKDAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
const WEEKDAYS_SHORT = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

/* ------------------------------------------------------------------ */
/* Pure helpers (no DOM)                                               */
/* ------------------------------------------------------------------ */

function pad2(value) {
  return String(value).padStart(2, '0');
}

/** Parse 'YYYY-MM-DDTHH:MM[:SS]' as plain wall-clock parts (no time zone math). */
function parseLocalDateTime(value) {
  const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?$/.exec(String(value == null ? '' : value).trim());
  if (!match) return null;
  const [year, month, day, hour, minute] = match.slice(1, 6).map(Number);
  const second = match[6] === undefined ? 0 : Number(match[6]);
  if (month < 1 || month > 12 || day < 1 || day > 31 || hour > 23 || minute > 59 || second > 59) return null;
  return { year, month, day, hour, minute, date: `${match[1]}-${match[2]}-${match[3]}` };
}

/** Parse 'YYYY-MM-DD'. */
function parseLocalDate(value) {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(value == null ? '' : value).trim());
  if (!match) return null;
  const [year, month, day] = match.slice(1, 4).map(Number);
  if (month < 1 || month > 12 || day < 1 || day > 31) return null;
  return { year, month, day };
}

/** Weekday index (0 = Sunday) of a calendar date, independent of the browser's time zone. */
function weekdayIndex(year, month, day) {
  return new Date(Date.UTC(year, month - 1, day)).getUTCDay();
}

/** 8, 20 -> '8:20 AM'; 0, 5 -> '12:05 AM'; 12, 0 -> '12:00 PM'. */
function formatClock12(hour, minute) {
  const suffix = hour < 12 ? 'AM' : 'PM';
  const displayHour = hour % 12 === 0 ? 12 : hour % 12;
  return `${displayHour}:${pad2(minute)} ${suffix}`;
}

/** 'HH:MM[:SS]' -> 12-hour clock, e.g. '13:00:00' -> '1:00 PM'. Unknown formats are returned as-is. */
function formatTime12(value) {
  const text = String(value == null ? '' : value).trim();
  const match = /^(\d{1,2}):(\d{2})(?::\d{2}(?:\.\d+)?)?$/.exec(text);
  if (!match) return text;
  const hour = Number(match[1]);
  const minute = Number(match[2]);
  if (hour > 23 || minute > 59) return text;
  return formatClock12(hour, minute);
}

/** '2026-10-27T08:20' -> 'Tuesday · 8:20 AM' (from the string itself, not the browser zone). */
function formatSightingLine(observedLocal) {
  const parts = parseLocalDateTime(observedLocal);
  if (!parts) return '';
  const weekday = WEEKDAYS[weekdayIndex(parts.year, parts.month, parts.day)];
  return `${weekday} · ${formatClock12(parts.hour, parts.minute)}`;
}

/**
 * minutes_after_scheduled_gate_departure -> display parts.
 * Negative: the departure is scheduled after the sighting; positive: before it.
 */
function offsetParts(minutes) {
  const value = Number(minutes);
  if (minutes === null || minutes === undefined || !Number.isFinite(value)) return { value: '', suffix: '' };
  const rounded = Math.round(Math.abs(value));
  if (rounded === 0) return { value: 'At sighting time', suffix: '' };
  return { value: `${rounded} min`, suffix: value < 0 ? 'after sighting' : 'before sighting' };
}

function formatOffset(minutes) {
  const parts = offsetParts(minutes);
  return parts.suffix ? `${parts.value} ${parts.suffix}` : parts.value;
}

/** 0.9312 -> '93.1%' (digits = 1) or '93%' (digits = 0). */
function formatPercent(probability, digits = 1) {
  if (probability === null || probability === undefined || probability === '') return '—';
  const value = Number(probability);
  if (!Number.isFinite(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}

/** Short weekday ('Mon') when a candidate's local date differs from the sighting date, else ''. */
function candidateDayLabel(candidateLocalDate, observedLocal) {
  const candidate = parseLocalDate(candidateLocalDate);
  if (!candidate) return '';
  const observed = parseLocalDateTime(observedLocal);
  if (observed && observed.date === String(candidateLocalDate).trim()) return '';
  return WEEKDAYS_SHORT[weekdayIndex(candidate.year, candidate.month, candidate.day)];
}

/** Milliseconds -> 'mm:ss'. */
function formatElapsed(ms) {
  const total = Math.max(0, Math.floor(Number(ms) / 1000) || 0);
  return `${pad2(Math.floor(total / 60))}:${pad2(total % 60)}`;
}

/** 'southwest' -> 'Southwest', 'jet_blue' -> 'Jet Blue'. */
function titleCase(label) {
  return String(label == null ? '' : label)
    .split(/[\s_-]+/)
    .filter(Boolean)
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

/** {label: p} -> [[label, p], ...] sorted by probability descending (ties by label). */
function sortedProbabilities(probabilities) {
  if (!probabilities || typeof probabilities !== 'object' || Array.isArray(probabilities)) return [];
  return Object.entries(probabilities)
    .map(([label, value]) => [label, value === null || value === '' ? NaN : Number(value)])
    .filter(([, value]) => Number.isFinite(value))
    .sort((a, b) => (b[1] - a[1]) || a[0].localeCompare(b[0]));
}

/** 20971520 -> '20 MB'; 25500000 -> '24.3 MB'. */
function formatBytes(bytes) {
  const megabytes = Math.round((Number(bytes) / 1048576) * 10) / 10;
  return `${megabytes} MB`;
}

/** Returns an error message for an unacceptable file, or null if it may be used. */
function validateUploadFile(file, maxBytes) {
  if (!file) return 'Please choose a photo.';
  // Some platforms report an empty MIME type; fall back to the extension (the server checks the real format).
  const typeOk = file.type
    ? ACCEPTED_TYPES.includes(file.type)
    : ACCEPTED_EXTENSIONS.test(String(file.name || ''));
  if (!typeOk) return 'Please choose a JPEG or PNG image.';
  if (!(file.size > 0)) return 'That file is empty. Please choose another photo.';
  if (file.size > maxBytes) {
    return `That photo is ${formatBytes(file.size)}. Please choose one up to ${formatBytes(maxBytes)}.`;
  }
  return null;
}

/** Browser-local Date -> 'YYYY-MM-DDTHH:MM' for a datetime-local input. */
function formatLocalInputValue(date) {
  return `${date.getFullYear()}-${pad2(date.getMonth() + 1)}-${pad2(date.getDate())}` +
    `T${pad2(date.getHours())}:${pad2(date.getMinutes())}`;
}

/** Server now_local -> 'YYYY-MM-DDTHH:MM', or null if malformed. */
function normalizeNowLocal(value) {
  if (!parseLocalDateTime(value)) return null;
  return String(value).trim().slice(0, 16);
}

/**
 * Wall-clock 'YYYY-MM-DDTHH:MM[:SS]' + minutes -> 'YYYY-MM-DDTHH:MM'.
 * Arithmetic is done on the parsed parts via Date.UTC, so the browser's time zone never applies.
 */
function addMinutesToLocal(value, minutes) {
  const parts = parseLocalDateTime(value);
  const delta = Number(minutes);
  if (!parts || !Number.isFinite(delta)) return null;
  const shifted = new Date(Date.UTC(parts.year, parts.month - 1, parts.day, parts.hour, parts.minute) + delta * 60000);
  return `${shifted.getUTCFullYear()}-${pad2(shifted.getUTCMonth() + 1)}-${pad2(shifted.getUTCDate())}` +
    `T${pad2(shifted.getUTCHours())}:${pad2(shifted.getUTCMinutes())}`;
}

/** Latest allowed sighting: reference now_local + whole minutes elapsed since it was read + 10 min. */
function maxObservedLocal(referenceNowLocal, elapsedMs) {
  const elapsedMinutes = Math.max(0, Math.floor(Number(elapsedMs) / 60000) || 0);
  return addMinutesToLocal(referenceNowLocal, elapsedMinutes + MAX_FUTURE_MINUTES);
}

/** True when a valid observed value is later than the (minute-precision) max. */
function isAfterMax(value, max) {
  if (!max || !isValidObservedLocal(value)) return false;
  return String(value).trim().slice(0, 16) > max;
}

/** True for 'YYYY-MM-DDTHH:MM' or 'YYYY-MM-DDTHH:MM:SS' with plausible parts. */
function isValidObservedLocal(value) {
  return parseLocalDateTime(value) !== null;
}

/** Human-readable error from a non-2xx response body ({"detail": ...}) with status fallbacks. */
function errorMessageFromResponse(status, bodyText) {
  let detail = '';
  try {
    const body = JSON.parse(bodyText);
    if (body && typeof body.detail === 'string') {
      detail = body.detail;
    } else if (body && Array.isArray(body.detail)) {
      detail = body.detail
        .map(item => (item && typeof item.msg === 'string' ? item.msg : ''))
        .filter(Boolean)
        .join(' ');
    }
  } catch (_) {
    // Not JSON (e.g. an HTML error page from a proxy); fall through.
  }
  if (detail.trim()) return detail.trim();
  if (status === 413) return 'That photo is too large for the server. Please choose a smaller one.';
  if (status === 401 || status === 403) return 'Your session may have expired. Reload the page and try again.';
  if (status === 404) return 'That submission could not be found.';
  if (status === 502 || status === 503 || status === 504) {
    return `The server is busy or not responding right now (HTTP ${status}). Please try again in a minute.`;
  }
  return `Something went wrong (HTTP ${status}). Please try again.`;
}

/** Same-origin polling URL from a 202 body; only accepts the documented shape. */
function statusUrlFor(body) {
  if (!body || typeof body !== 'object') return null;
  if (typeof body.status_url === 'string' && /^\/api\/submissions\/[A-Za-z0-9-]+$/.test(body.status_url)) {
    return body.status_url;
  }
  if (typeof body.submission_id === 'string' && /^[A-Za-z0-9-]+$/.test(body.submission_id)) {
    return `${API_SUBMISSIONS_URL}/${body.submission_id}`;
  }
  return null;
}

/** Working-state copy for a status payload: which progress step, which message, attempt label. */
function workingCopy(status) {
  const data = status && typeof status === 'object' ? status : {};
  const stage = data.stage || data.status || '';
  const serverMessage = typeof data.message === 'string' ? data.message.trim() : '';
  const attempts = Number(data.inference_attempts);
  switch (stage) {
    case 'uploading':
      return { step: 'upload', message: 'Uploading your photo…', attempt: '' };
    case 'queued':
      return { step: 'model', message: 'Photo saved. Starting…', attempt: '' };
    case 'waiting_for_model':
      return {
        step: 'model',
        message: serverMessage || 'Waiting for the model to respond…',
        attempt: Number.isFinite(attempts) && attempts > 1 ? `Attempt ${attempts}` : '',
      };
    case 'looking_up_schedule':
      return { step: 'schedule', message: 'Checking the San Diego schedule…', attempt: '' };
    case 'recording':
      return { step: 'record', message: 'Saving the result…', attempt: '' };
    case 'done':
      return { step: 'record', message: 'Finishing up…', attempt: '' };
    default:
      return { step: 'model', message: serverMessage || 'Working on it…', attempt: '' };
  }
}

/** Short reference for a submission id. */
function shortId(id) {
  return typeof id === 'string' ? id.slice(0, 8) : '';
}

/** True when the warning text would only repeat what the candidates area already says. */
function isDuplicateWarning(warningText, shownText) {
  const warning = String(warningText || '').trim();
  return Boolean(warning) && Boolean(shownText) && String(shownText).includes(warning);
}

/** Airlines a user can name when the prediction was wrong (values match the API's REVIEW_CHOICES). */
const REVIEW_OPTIONS = [
  ['alaska', 'Alaska Airlines'], ['american', 'American Airlines'], ['delta', 'Delta Air Lines'],
  ['southwest', 'Southwest Airlines'], ['united', 'United Airlines'], ['other', 'Another airline'],
];

/** Choices offered after "No": every option except the predicted airline. */
function reviewChoices(predicted) {
  return REVIEW_OPTIONS.filter(([value]) => value !== predicted);
}

/** Display name for a review label. */
function reviewLabelName(label) {
  const option = REVIEW_OPTIONS.find(([value]) => value === label);
  return option ? option[1] : titleCase(label);
}

/** Text describing a saved review ({label, correct, source}), or '' when there is none. */
function reviewSummary(review) {
  if (!review || typeof review !== 'object' || !review.label) return '';
  const name = reviewLabelName(review.label);
  if (review.source && review.source !== 'app_user') return `A reviewer confirmed this was ${name}.`;
  return review.correct
    ? `Thanks — recorded as correct (${name}).`
    : `Thanks — recorded as wrong. It was ${name === 'Another airline' ? 'another airline' : name}.`;
}

/* ------------------------------------------------------------------ */
/* DOM wiring                                                          */
/* ------------------------------------------------------------------ */

function initApp() {
  const $ = selector => document.querySelector(selector);

  // Input card
  const input = $('#photo');
  const upload = $('#upload-zone');
  const uploadLimit = $('#upload-limit');
  const preview = $('#preview');
  const fileName = $('#file-name');
  const changePhoto = $('#change-photo');
  const useWholePhoto = $('#use-whole-photo');
  const canvas = $('#crop-canvas');
  const context = canvas.getContext('2d');
  const fields = ['left', 'top', 'right', 'bottom'].map(name => $(`#crop-${name}`));
  const cropError = $('#crop-error');
  const apply = $('#apply-crop');
  const cropResult = $('#cropped-result');
  const croppedImage = $('#cropped-image');
  const downloadCrop = $('#download-crop');
  const photoError = $('#photo-error');
  const observed = $('#observed');
  const submitButton = $('#submit-button');
  const submitNote = $('#submit-note');

  // Result card
  const resultCard = $('#result-card');
  const resultsTitle = $('#results-title');
  const badge = $('#result-badge');
  const corner = $('#result-corner');
  const emptyState = $('#empty-state');
  const workingState = $('#working-state');
  const workingMessage = $('#working-message');
  const workingAttempt = $('#working-attempt');
  const workingElapsed = $('#working-elapsed');
  const progressItems = Array.from(document.querySelectorAll('#progress-steps li'));
  const cancelWorking = $('#cancel-working');
  const resultState = $('#result-state');
  const airlineEmblem = $('#airline-emblem');
  const airlineName = $('#airline-name');
  const airlineConfidence = $('#airline-confidence');
  const probabilitiesBlock = $('#probabilities');
  const probabilityList = $('#probability-list');
  const resultWarning = $('#result-warning');
  const sightingLine = $('#sighting-line');
  const candidateCount = $('#candidate-count');
  const candidateList = $('#candidate-list');
  const noCandidates = $('#no-candidates');
  const submissionFootnote = $('#submission-footnote');
  const resetResult = $('#reset-result');
  const failedState = $('#failed-state');
  const failedMessage = $('#failed-message');
  const retrySubmission = $('#retry-submission');
  const resetFailed = $('#reset-failed');
  const feedback = $('#feedback');
  const feedbackActions = $('#feedback-actions');
  const feedbackYes = $('#feedback-yes');
  const feedbackNo = $('#feedback-no');
  const feedbackCorrection = $('#feedback-correction');
  const feedbackAirline = $('#feedback-airline');
  const feedbackSave = $('#feedback-save');
  const feedbackCancel = $('#feedback-cancel');
  const feedbackStatus = $('#feedback-status');
  const feedbackChange = $('#feedback-change');

  const reducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const config = Object.assign({}, DEFAULT_CONFIG);
  let photoFile = null;
  let bitmap = null;
  let box = null;
  let dragStart = null;
  let croppedUrl = null;
  let cropBlob = null;
  let generation = 0;
  let busy = false;
  let userEditedTime = false;
  let currentRun = null;
  let lastSnapshot = null;
  // Reference wall clock for the datetime-local max (server now_local, or browser time as a fallback).
  let referenceNowLocal = null;
  let referenceLoadedAt = 0;
  // Submission the "Was this right?" block refers to: {id, predicted, predictedName}.
  let feedbackTarget = null;

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  /* ---------- Crop: normalized [left, top, right, bottom], outward rounding ---------- */

  function releaseCropPreview() {
    cropResult.hidden = true;
    cropBlob = null;
    if (croppedUrl) URL.revokeObjectURL(croppedUrl);
    croppedUrl = null;
    croppedImage.removeAttribute('src');
  }

  function clearCrop() {
    box = null;
    fields.forEach(field => { field.value = ''; });
    apply.disabled = true;
    cropError.textContent = '';
    releaseCropPreview();
    updateSubmitState();
  }

  function draw() {
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (!bitmap) return;
    context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    if (!box) return;
    const [l, t, r, b] = box;
    const x = l * canvas.width;
    const y = t * canvas.height;
    const w = (r - l) * canvas.width;
    const h = (b - t) * canvas.height;
    context.fillStyle = 'rgba(11, 30, 20, .48)';
    context.fillRect(0, 0, canvas.width, y);
    context.fillRect(0, y + h, canvas.width, canvas.height - y - h);
    context.fillRect(0, y, x, h);
    context.fillRect(x + w, y, canvas.width - x - w, h);
    context.strokeStyle = '#fff';
    context.lineWidth = Math.max(2, canvas.width / 400);
    context.strokeRect(x, y, w, h);
  }

  function validateBox(values) {
    return Boolean(bitmap) && values.length === 4 && values.every(Number.isFinite) &&
      values[0] >= 0 && values[1] >= 0 && values[2] <= 1 && values[3] <= 1 &&
      values[0] < values[2] && values[1] < values[3] &&
      Math.ceil(values[2] * bitmap.width) - Math.floor(values[0] * bitmap.width) >= 2 &&
      Math.ceil(values[3] * bitmap.height) - Math.floor(values[1] * bitmap.height) >= 2;
  }

  function setBox(values, updateFields = true) {
    if (updateFields) values = values.map(value => Math.round(value * 10000) / 10000);
    releaseCropPreview();
    if (!validateBox(values)) {
      box = null;
      apply.disabled = true;
      cropError.textContent = 'Choose a box within the image, at least two pixels wide and tall.';
    } else {
      box = values;
      apply.disabled = false;
      cropError.textContent = '';
      if (updateFields) fields.forEach((field, i) => { field.value = (values[i] * 100).toFixed(2); });
    }
    draw();
    updateSubmitState();
  }

  function point(event) {
    const rect = canvas.getBoundingClientRect();
    return [
      Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
      Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)),
    ];
  }

  function boxFromDrag(end) {
    return [
      Math.min(dragStart[0], end[0]), Math.min(dragStart[1], end[1]),
      Math.max(dragStart[0], end[0]), Math.max(dragStart[1], end[1]),
    ];
  }

  /* ---------- Photo selection ---------- */

  function showPhotoError(message) {
    photoError.hidden = !message;
    photoError.textContent = message || '';
  }

  async function loadPhoto(file) {
    if (!file || busy) return;
    const problem = validateUploadFile(file, config.max_upload_bytes);
    if (problem) {
      showPhotoError(problem);
      return;
    }
    if (typeof createImageBitmap !== 'function') {
      showPhotoError('This browser cannot open photos here. Please use a recent version of Chrome, Edge, Firefox or Safari.');
      return;
    }
    const request = ++generation;
    let decoded;
    try {
      decoded = await createImageBitmap(file, { imageOrientation: 'from-image' });
    } catch (_) {
      if (request === generation) showPhotoError('The image could not be opened. Please choose another JPEG or PNG.');
      return;
    }
    if (request !== generation || busy) {
      decoded.close();
      return;
    }
    if (bitmap) bitmap.close();
    bitmap = decoded;
    photoFile = file;
    const scale = Math.min(1, 1200 / Math.max(bitmap.width, bitmap.height));
    canvas.width = Math.max(1, Math.round(bitmap.width * scale));
    canvas.height = Math.max(1, Math.round(bitmap.height * scale));
    fileName.textContent = file.name;
    upload.hidden = true;
    preview.hidden = false;
    showPhotoError('');
    clearCrop();
    draw();
  }

  function clearPhoto() {
    generation += 1;
    dragStart = null;
    if (bitmap) bitmap.close();
    bitmap = null;
    photoFile = null;
    input.value = '';
    fileName.textContent = '';
    clearCrop();
    context.clearRect(0, 0, canvas.width, canvas.height);
    preview.hidden = true;
    upload.hidden = false;
    showPhotoError('');
  }

  /* ---------- Submit button state ---------- */

  function updateObservedMax() {
    const max = referenceNowLocal ? maxObservedLocal(referenceNowLocal, Date.now() - referenceLoadedAt) : null;
    if (max) observed.max = max;
    else observed.removeAttribute('max');
    return max;
  }

  function updateSubmitState() {
    const hasPhoto = Boolean(photoFile && bitmap);
    const max = updateObservedMax();
    const hasTime = isValidObservedLocal(observed.value);
    const tooLate = hasTime && isAfterMax(observed.value, max);
    const ready = hasPhoto && Boolean(box) && hasTime && !tooLate;
    submitButton.disabled = busy || !ready;
    let note;
    if (busy) note = 'Working on it — this can take a few minutes.';
    else if (!hasPhoto) note = 'Choose a photo to get started.';
    else if (!box) note = 'Outline the tail, or use the whole photo.';
    else if (!hasTime) note = 'Enter when you saw it.';
    else if (tooLate) note = `That time is more than ${MAX_FUTURE_MINUTES} minutes in the future.`;
    else note = 'Your photo, tail outline and time are saved with the result.';
    submitNote.textContent = note;
  }

  function setBusy(value) {
    busy = value;
    input.disabled = value;
    changePhoto.disabled = value;
    useWholePhoto.disabled = value;
    fields.forEach(field => { field.disabled = value; });
    observed.disabled = value;
    apply.disabled = value || !box;
    downloadCrop.disabled = value;
    canvas.classList.toggle('is-locked', value);
    canvas.setAttribute('aria-disabled', String(value));
    submitButton.setAttribute('aria-busy', String(value));
    if (value) dragStart = null;
    updateSubmitState();
  }

  /* ---------- Result card phases ---------- */

  function showPhase(phase) {
    emptyState.hidden = phase !== 'empty';
    workingState.hidden = phase !== 'working';
    resultState.hidden = phase !== 'result';
    failedState.hidden = phase !== 'failed';
    const badges = { working: 'WORKING', result: 'RESULT', failed: 'FAILED' };
    badge.hidden = !badges[phase];
    badge.textContent = badges[phase] || '';
    badge.classList.toggle('is-working', phase === 'working');
    badge.classList.toggle('is-failed', phase === 'failed');
    corner.hidden = Boolean(badges[phase]);
  }

  function renderWorking(status, run) {
    if (status) run.lastStatus = status;
    const copy = workingCopy(run.lastStatus);
    if (workingMessage.textContent !== copy.message) workingMessage.textContent = copy.message;
    const attemptText = run.failures > 0 ? 'RECONNECTING…' : copy.attempt.toUpperCase();
    if (workingAttempt.textContent !== attemptText) workingAttempt.textContent = attemptText;
    workingAttempt.hidden = !attemptText;
    const activeIndex = PROGRESS_STEPS.indexOf(copy.step);
    progressItems.forEach(item => {
      const index = PROGRESS_STEPS.indexOf(item.dataset.step);
      item.classList.toggle('is-done', index < activeIndex);
      item.classList.toggle('is-active', index === activeIndex);
      if (index === activeIndex) item.setAttribute('aria-current', 'step');
      else item.removeAttribute('aria-current');
    });
  }

  function tickClock(run) {
    if (!isCurrent(run)) return;
    workingElapsed.textContent = formatElapsed(Date.now() - run.startedAt);
    run.clockTimer = setTimeout(() => tickClock(run), 1000);
  }

  function renderProbabilities(prediction) {
    const rows = sortedProbabilities(prediction && prediction.probabilities);
    probabilityList.replaceChildren();
    probabilitiesBlock.hidden = rows.length === 0;
    rows.forEach(([label, probability], index) => {
      const item = el('li', index === 0 ? 'probability-row is-top' : 'probability-row');
      const percent = formatPercent(probability, 1);
      item.appendChild(el('span', 'probability-name', titleCase(label)));
      const track = el('span', 'probability-track');
      track.setAttribute('aria-hidden', 'true');
      const fill = el('span', 'probability-fill');
      fill.style.width = `${Math.max(0, Math.min(100, probability * 100))}%`;
      track.appendChild(fill);
      item.appendChild(track);
      item.appendChild(el('span', 'probability-value', percent));
      probabilityList.appendChild(item);
    });
  }

  function renderCandidate(candidate, observedLocal) {
    const row = el('div', 'flight');
    const icon = el('div', 'flight-icon', '↗');
    icon.setAttribute('aria-hidden', 'true');
    row.appendChild(icon);

    const main = el('div', 'flight-main');
    const title = el('strong');
    const city = candidate.destination_city || candidate.destination_iata || 'Unknown destination';
    title.appendChild(document.createTextNode(String(city)));
    if (candidate.destination_city && candidate.destination_iata) {
      title.appendChild(document.createTextNode(' '));
      title.appendChild(el('span', '', String(candidate.destination_iata)));
    }
    main.appendChild(title);
    const day = candidateDayLabel(candidate.local_date, observedLocal);
    const time = formatTime12(candidate.scheduled_departure_local);
    const details = [];
    if (candidate.flight_number) details.push(String(candidate.flight_number));
    if (time) details.push(`Scheduled ${day ? `${day} ` : ''}${time}`);
    main.appendChild(el('small', '', details.join(' · ')));
    row.appendChild(main);

    const offset = offsetParts(candidate.minutes_after_scheduled_gate_departure);
    const timing = el('div', 'flight-time', offset.value);
    if (offset.suffix) {
      timing.appendChild(document.createTextNode(' '));
      timing.appendChild(el('span', '', offset.suffix));
    }
    row.appendChild(timing);
    return row;
  }

  function renderResult(data, run) {
    const prediction = data.prediction && typeof data.prediction === 'object' ? data.prediction : null;
    const label = prediction && typeof prediction.airline === 'string' ? prediction.airline : '';
    const name = prediction && prediction.airline_name
      ? String(prediction.airline_name)
      : (label ? titleCase(label) : 'No airline identified');

    // Airline block
    airlineEmblem.textContent = (prediction ? name.trim().charAt(0).toUpperCase() : '') || '?';
    airlineName.replaceChildren(document.createTextNode(name));
    if (prediction && prediction.iata_code) {
      airlineName.appendChild(document.createTextNode(' '));
      airlineName.appendChild(el('span', 'airline-code', String(prediction.iata_code)));
    }
    const hasConfidence = prediction && prediction.confidence !== null && prediction.confidence !== undefined &&
      Number.isFinite(Number(prediction.confidence));
    airlineConfidence.textContent = hasConfidence
      ? `Model confidence ${formatPercent(prediction.confidence, 0)} (uncalibrated)`
      : 'Model confidence unavailable';
    renderProbabilities(prediction);

    // Candidates
    const observedLocal = data.observed_local || run.snapshot.observed;
    const airport = data.airport_iata || config.airport_iata;
    sightingLine.textContent = `${formatSightingLine(observedLocal) || observedLocal} sighting · ${airport}`;
    const lookupFailed = data.schedule_lookup === 'failed';
    const candidates = !lookupFailed && Array.isArray(data.candidates)
      ? data.candidates.filter(item => item && typeof item === 'object')
      : [];
    candidateCount.textContent = lookupFailed
      ? 'UNAVAILABLE'
      : `${candidates.length} CANDIDATE${candidates.length === 1 ? '' : 'S'}`;
    candidateList.replaceChildren(...candidates.map(candidate => renderCandidate(candidate, observedLocal)));
    let emptyText = '';
    if (lookupFailed) {
      emptyText = LOOKUP_FAILED_TEXT;
    } else if (!candidates.length) {
      emptyText = prediction
        ? `No ${name} scheduled gate departures were found from ${config.window_minutes_before} min before ` +
          `to ${config.window_minutes_after} min after your sighting. Runway timing varies, so the flight ` +
          'may fall outside this window — or the airline prediction may be wrong.'
        : 'No airline was identified, so no departures could be looked up.';
    }
    noCandidates.textContent = emptyText;
    noCandidates.hidden = !emptyText;

    // Warning (error / not recorded), unless it only repeats the candidates-area text
    const errorText = data.error ? String(data.error).trim() : '';
    const warningText = errorText || (data.recorded === false ? 'This result could not be saved.' : '');
    const showWarning = Boolean(warningText) && !isDuplicateWarning(warningText, emptyText);
    resultWarning.textContent = showWarning ? warningText : '';
    resultWarning.hidden = !showWarning;

    // Footnote
    const submissionId = data.submission_id || run.submissionId;
    const parts = [];
    if (submissionId) parts.push(`Submission ${shortId(submissionId)}${data.recorded === true ? ' · saved to Unity Catalog' : ''}`);
    if (data.model && data.model.model_version) parts.push(`Model v${data.model.model_version}`);
    submissionFootnote.textContent = parts.join(' · ');
    submissionFootnote.hidden = parts.length === 0;

    renderFeedback(data, prediction, name);
    showPhase('result');
    focusResults();
  }

  /* ---------- "Did we get the airline right?" ---------- */

  function setFeedbackBusy(value) {
    [feedbackYes, feedbackNo, feedbackSave, feedbackCancel, feedbackChange, feedbackAirline]
      .forEach(control => { control.disabled = value; });
  }

  function showFeedbackQuestion() {
    feedbackActions.hidden = false;
    feedbackCorrection.hidden = true;
    feedbackChange.hidden = true;
    feedbackStatus.textContent = '';
    feedbackStatus.classList.remove('is-error');
  }

  function showSavedReview(review) {
    feedbackActions.hidden = true;
    feedbackCorrection.hidden = true;
    feedbackStatus.classList.remove('is-error');
    feedbackStatus.textContent = reviewSummary(review);
    // Only the user's own answer can be changed; a reviewer's label is final.
    feedbackChange.hidden = review.source !== 'app_user';
  }

  function renderFeedback(data, prediction, predictedName) {
    const submissionId = typeof data.submission_id === 'string' ? data.submission_id : '';
    const canReview = Boolean(prediction && prediction.airline && submissionId && data.recorded === true);
    feedback.hidden = !canReview;
    feedbackTarget = canReview ? { id: submissionId, predicted: prediction.airline, predictedName } : null;
    setFeedbackBusy(false);
    if (!canReview) return;
    feedbackYes.textContent = `✓ Yes, it was ${predictedName}`;
    feedbackAirline.replaceChildren(el('option', '', 'Choose an airline…'),
      ...reviewChoices(prediction.airline).map(([value, label]) => {
        const option = el('option', '', label);
        option.value = value;
        return option;
      }));
    feedbackAirline.firstChild.value = '';
    if (data.review && data.review.label) showSavedReview(data.review);
    else showFeedbackQuestion();
  }

  async function sendReview(correct, airline) {
    const target = feedbackTarget;
    if (!target) return;
    setFeedbackBusy(true);
    feedbackStatus.classList.remove('is-error');
    feedbackStatus.textContent = 'Saving your answer…';
    let response;
    let text;
    try {
      response = await fetch(`${API_SUBMISSIONS_URL}/${encodeURIComponent(target.id)}/review`, {
        method: 'POST',
        headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify(correct ? { correct: true } : { correct: false, airline }),
      });
      text = await response.text();
    } catch (_) {
      text = null;
    }
    if (feedbackTarget !== target) return; // the user started over meanwhile
    setFeedbackBusy(false);
    let data = null;
    try {
      data = response && response.ok ? JSON.parse(text) : null;
    } catch (_) {
      data = null;
    }
    if (data && data.review) {
      showSavedReview(data.review);
      feedbackChange.focus();
      return;
    }
    feedbackStatus.classList.add('is-error');
    feedbackStatus.textContent = response
      ? errorMessageFromResponse(response.status, text || '')
      : "We couldn't reach the server. Please try again.";
  }

  function focusResults() {
    resultsTitle.focus({ preventScroll: false });
  }

  function showFailed(message) {
    retrySubmission.hidden = !(currentSnapshot() || lastSnapshot);
    showPhase('failed');
    failedMessage.textContent = message;
    focusResults();
  }

  /* ---------- Submission lifecycle ---------- */

  function isCurrent(run) {
    return Boolean(run) && run === currentRun && !run.cancelled;
  }

  function stopRun(run) {
    if (!run) return;
    run.cancelled = true;
    clearTimeout(run.pollTimer);
    clearTimeout(run.clockTimer);
    if (run.controller) run.controller.abort();
    if (currentRun === run) currentRun = null;
  }

  function failRun(run, message) {
    stopRun(run);
    setBusy(false);
    showFailed(message);
  }

  function completeRun(run, data) {
    stopRun(run);
    setBusy(false);
    renderResult(data, run);
  }

  function schedulePoll(run) {
    if (!isCurrent(run)) return;
    run.pollTimer = setTimeout(() => { poll(run); }, POLL_INTERVAL_MS);
  }

  function transientPollFailure(run, message) {
    run.failures += 1;
    if (run.failures >= MAX_CONSECUTIVE_POLL_FAILURES) {
      failRun(run, `${message} We stopped checking after several tries. Please try again.`);
      return;
    }
    renderWorking(null, run);
    schedulePoll(run);
  }

  /** fetch with a per-request timeout; also aborts when the optional external (run) signal aborts. */
  async function fetchWithTimeout(url, options, timeoutMs, externalSignal) {
    if (typeof AbortController !== 'function') return fetch(url, options);
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const onExternalAbort = () => controller.abort();
    if (externalSignal) {
      if (externalSignal.aborted) controller.abort();
      else externalSignal.addEventListener('abort', onExternalAbort, { once: true });
    }
    try {
      return await fetch(url, Object.assign({}, options, { signal: controller.signal }));
    } finally {
      clearTimeout(timer);
      if (externalSignal) externalSignal.removeEventListener('abort', onExternalAbort);
    }
  }

  async function poll(run) {
    if (!isCurrent(run)) return;
    if (Date.now() - run.startedAt > POLL_GIVE_UP_MS) {
      failRun(run, 'This is taking much longer than expected, so we stopped checking. ' +
        'The model may still be waking up — please try again in a few minutes.');
      return;
    }
    let response;
    let text;
    try {
      response = await fetchWithTimeout(run.statusUrl, {
        headers: { Accept: 'application/json' },
        cache: 'no-store',
        credentials: 'same-origin',
      }, POLL_REQUEST_TIMEOUT_MS, run.controller && run.controller.signal);
      text = await response.text();
    } catch (_) {
      if (isCurrent(run)) transientPollFailure(run, 'We lost touch with the server.');
      return;
    }
    if (!isCurrent(run)) return;
    if (response.status >= 500 || response.status === 408 || response.status === 429) {
      transientPollFailure(run, errorMessageFromResponse(response.status, text));
      return;
    }
    if (response.status === 404) {
      failRun(run, NOT_FOUND_WHILE_POLLING_TEXT);
      return;
    }
    if (!response.ok) {
      failRun(run, errorMessageFromResponse(response.status, text));
      return;
    }
    let data = null;
    try {
      data = JSON.parse(text);
    } catch (_) {
      data = null;
    }
    if (!data || typeof data !== 'object') {
      transientPollFailure(run, 'The server sent an unexpected response.');
      return;
    }
    run.failures = 0;
    if (data.status === 'succeeded') {
      completeRun(run, data);
    } else if (data.status === 'failed') {
      failRun(run, String(data.error || data.message || 'The submission could not be completed.'));
    } else {
      renderWorking(data, run);
      schedulePoll(run);
    }
  }

  async function submit(snapshot) {
    if (busy || !snapshot) return;
    feedbackTarget = null;
    const run = {
      snapshot,
      startedAt: Date.now(),
      cancelled: false,
      failures: 0,
      pollTimer: null,
      clockTimer: null,
      statusUrl: null,
      submissionId: null,
      lastStatus: null,
      controller: typeof AbortController === 'function' ? new AbortController() : null,
    };
    currentRun = run;
    lastSnapshot = snapshot;
    setBusy(true);
    showPhase('working');
    renderWorking({ stage: 'uploading' }, run);
    tickClock(run);
    if (window.matchMedia && window.matchMedia('(max-width: 760px)').matches) {
      resultCard.scrollIntoView({ behavior: reducedMotion ? 'auto' : 'smooth', block: 'start' });
    }

    const form = new FormData();
    form.append('photo', snapshot.file, snapshot.file.name || 'photo');
    ['crop_left', 'crop_top', 'crop_right', 'crop_bottom'].forEach((key, i) => {
      form.append(key, String(snapshot.box[i]));
    });
    form.append('observed_local', snapshot.observed);
    // Upright (EXIF-oriented) size the browser decoded; the server rejects a mismatch.
    form.append('image_width', String(snapshot.imageWidth));
    form.append('image_height', String(snapshot.imageHeight));

    let response;
    let text;
    try {
      response = await fetch(API_SUBMISSIONS_URL, {
        method: 'POST',
        body: form,
        headers: { Accept: 'application/json' },
        credentials: 'same-origin',
        signal: run.controller ? run.controller.signal : undefined,
      });
      text = await response.text();
    } catch (_) {
      // Aborted by "Start over" / "Stop waiting": the run is no longer current, so stay silent.
      if (isCurrent(run)) failRun(run, "We couldn't reach the server. Check your connection and try again.");
      return;
    }
    if (!isCurrent(run)) return;
    if (!response.ok) {
      failRun(run, errorMessageFromResponse(response.status, text));
      return;
    }
    let body = null;
    try {
      body = JSON.parse(text);
    } catch (_) {
      body = null;
    }
    const statusUrl = statusUrlFor(body);
    if (!statusUrl) {
      failRun(run, 'The server accepted the photo but did not say where to follow its progress. Please try again.');
      return;
    }
    run.statusUrl = statusUrl;
    run.submissionId = typeof body.submission_id === 'string' ? body.submission_id : null;
    renderWorking(body, run);
    schedulePoll(run);
  }

  function currentSnapshot() {
    if (!photoFile || !bitmap || !box || !isValidObservedLocal(observed.value)) return null;
    if (isAfterMax(observed.value, updateObservedMax())) return null;
    return {
      file: photoFile,
      box: box.slice(),
      observed: observed.value,
      imageWidth: Math.round(bitmap.width),
      imageHeight: Math.round(bitmap.height),
    };
  }

  function startOver() {
    stopRun(currentRun);
    feedbackTarget = null;
    lastSnapshot = null;
    clearPhoto();
    setBusy(false); // also refreshes the datetime-local max
    showPhase('empty');
    input.focus();
  }

  /* ---------- Config ---------- */

  async function loadConfig() {
    let data = null;
    try {
      const response = await fetchWithTimeout(API_CONFIG_URL, {
        headers: { Accept: 'application/json' },
        cache: 'no-store',
        credentials: 'same-origin',
      }, CONFIG_TIMEOUT_MS);
      if (response.ok) data = await response.json();
    } catch (_) {
      data = null;
    }
    if (data && typeof data === 'object') {
      if (typeof data.airport_iata === 'string' && data.airport_iata) config.airport_iata = data.airport_iata;
      ['max_upload_bytes', 'window_minutes_before', 'window_minutes_after'].forEach(key => {
        const value = Number(data[key]);
        if (data[key] !== null && data[key] !== undefined && Number.isFinite(value) && value >= 0) config[key] = value;
      });
    }
    uploadLimit.textContent = formatBytes(config.max_upload_bytes);
    const serverNow = normalizeNowLocal(data && data.now_local);
    referenceNowLocal = serverNow || formatLocalInputValue(new Date());
    referenceLoadedAt = Date.now();
    if (!userEditedTime && !observed.value) observed.value = referenceNowLocal;
    updateSubmitState(); // sets the datetime-local max (now + 10 min)
  }

  /* ---------- Events ---------- */

  input.addEventListener('change', () => {
    const file = input.files && input.files[0];
    input.value = '';
    loadPhoto(file);
  });
  changePhoto.addEventListener('click', () => { if (!busy) input.click(); });
  useWholePhoto.addEventListener('click', () => { if (!busy && bitmap) setBox([0, 0, 1, 1]); });

  ['dragenter', 'dragover'].forEach(name => upload.addEventListener(name, event => {
    event.preventDefault();
    if (!busy) upload.classList.add('dragging');
  }));
  ['dragleave', 'drop'].forEach(name => upload.addEventListener(name, event => {
    event.preventDefault();
    upload.classList.remove('dragging');
  }));
  upload.addEventListener('drop', event => {
    if (busy) return;
    const file = event.dataTransfer && event.dataTransfer.files[0];
    loadPhoto(file);
  });

  canvas.addEventListener('pointerdown', event => {
    if (busy || !bitmap || event.button !== 0) return;
    event.preventDefault();
    clearCrop();
    draw();
    dragStart = point(event);
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointermove', event => {
    if (!dragStart || busy) return;
    setBox(boxFromDrag(point(event)));
  });
  canvas.addEventListener('pointerup', event => {
    if (!dragStart || busy) return;
    setBox(boxFromDrag(point(event)));
    dragStart = null;
  });
  canvas.addEventListener('pointercancel', () => { dragStart = null; });

  fields.forEach(field => field.addEventListener('input', () => {
    if (busy || !bitmap) return;
    if (fields.some(item => item.value.trim() === '')) {
      box = null;
      apply.disabled = true;
      releaseCropPreview();
      cropError.textContent = '';
      draw();
      updateSubmitState();
    } else {
      setBox(fields.map(item => Number(item.value) / 100), false);
    }
  }));

  apply.addEventListener('click', () => {
    if (busy || !box || !bitmap) return;
    const [l, t, r, b] = box;
    const x = Math.floor(l * bitmap.width);
    const y = Math.floor(t * bitmap.height);
    const width = Math.ceil(r * bitmap.width) - x;
    const height = Math.ceil(b * bitmap.height) - y;
    const result = document.createElement('canvas');
    result.width = width;
    result.height = height;
    result.getContext('2d').drawImage(bitmap, x, y, width, height, 0, 0, width, height);
    const appliedBox = box;
    result.toBlob(blob => {
      if (!blob || appliedBox !== box) return;
      releaseCropPreview();
      cropBlob = blob;
      croppedUrl = URL.createObjectURL(blob);
      croppedImage.src = croppedUrl;
      cropResult.hidden = false;
    }, 'image/png');
  });

  downloadCrop.addEventListener('click', () => {
    if (busy || !cropBlob || !croppedUrl) return;
    const link = document.createElement('a');
    link.href = croppedUrl;
    link.download = 'tail-crop.png';
    link.click();
  });

  ['input', 'change'].forEach(name => observed.addEventListener(name, () => {
    userEditedTime = true;
    updateSubmitState();
  }));

  submitButton.addEventListener('click', () => {
    const snapshot = currentSnapshot();
    if (snapshot) submit(snapshot);
  });
  retrySubmission.addEventListener('click', () => {
    // Prefer what the form shows now (the user may have fixed the box or time); else resend the last attempt.
    const snapshot = currentSnapshot() || lastSnapshot;
    if (snapshot) submit(snapshot);
  });
  cancelWorking.addEventListener('click', startOver);
  resetResult.addEventListener('click', startOver);
  resetFailed.addEventListener('click', startOver);

  feedbackYes.addEventListener('click', () => sendReview(true, null));
  feedbackNo.addEventListener('click', () => {
    feedbackActions.hidden = true;
    feedbackCorrection.hidden = false;
    feedbackStatus.textContent = '';
    feedbackStatus.classList.remove('is-error');
    feedbackAirline.value = '';
    feedbackAirline.focus();
  });
  feedbackSave.addEventListener('click', () => {
    if (!feedbackAirline.value) {
      feedbackStatus.classList.add('is-error');
      feedbackStatus.textContent = 'Choose the airline it actually was.';
      feedbackAirline.focus();
      return;
    }
    sendReview(false, feedbackAirline.value);
  });
  feedbackCancel.addEventListener('click', () => {
    showFeedbackQuestion();
    feedbackNo.focus();
  });
  feedbackChange.addEventListener('click', () => {
    showFeedbackQuestion();
    feedbackYes.focus();
  });

  showPhase('empty');
  updateSubmitState();
  loadConfig();
}

if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initApp);
  else initApp();
}
