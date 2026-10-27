const input = document.querySelector('#photo');
const upload = document.querySelector('#upload-zone');
const preview = document.querySelector('#preview');
const fileName = document.querySelector('#file-name');
const canvas = document.querySelector('#crop-canvas');
const context = canvas.getContext('2d');
const fields = ['left', 'top', 'right', 'bottom'].map(name => document.querySelector(`#crop-${name}`));
const error = document.querySelector('#crop-error');
const apply = document.querySelector('#apply-crop');
const cropResult = document.querySelector('#cropped-result');
const croppedImage = document.querySelector('#cropped-image');
let bitmap, box, dragStart, croppedUrl, cropBlob, generation = 0;

function clearCrop() {
  box = null;
  cropBlob = null;
  fields.forEach(field => { field.value = ''; });
  apply.disabled = true;
  error.textContent = '';
  cropResult.hidden = true;
  if (croppedUrl) URL.revokeObjectURL(croppedUrl);
  croppedUrl = null;
}

function draw() {
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  if (!box) return;
  const [l, t, r, b] = box;
  const x = l * canvas.width, y = t * canvas.height;
  const w = (r - l) * canvas.width, h = (b - t) * canvas.height;
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
  return values.length === 4 && values.every(Number.isFinite) &&
    values[0] >= 0 && values[1] >= 0 && values[2] <= 1 && values[3] <= 1 &&
    values[0] < values[2] && values[1] < values[3] &&
    Math.ceil(values[2] * bitmap.width) - Math.floor(values[0] * bitmap.width) >= 2 &&
    Math.ceil(values[3] * bitmap.height) - Math.floor(values[1] * bitmap.height) >= 2;
}

function setBox(values, updateFields = true) {
  if (updateFields) values = values.map(value => Math.round(value * 10000) / 10000);
  cropResult.hidden = true;
  cropBlob = null;
  if (croppedUrl) URL.revokeObjectURL(croppedUrl);
  croppedUrl = null;
  if (!validateBox(values)) {
    box = null;
    apply.disabled = true;
    error.textContent = 'Choose a box within the image, at least two pixels wide and tall.';
  } else {
    box = values;
    apply.disabled = false;
    error.textContent = '';
    if (updateFields) fields.forEach((field, i) => { field.value = (values[i] * 100).toFixed(2); });
  }
  draw();
}

async function previewFile(file) {
  if (!file) return;
  if (!['image/jpeg', 'image/png'].includes(file.type)) {
    window.alert('Please choose a JPEG or PNG image.');
    return;
  }
  const request = ++generation;
  let decoded;
  try {
    decoded = await createImageBitmap(file, { imageOrientation: 'from-image' });
  } catch (_) {
    window.alert('The image could not be opened. Please choose another JPEG or PNG.');
    return;
  }
  if (request !== generation) { decoded.close(); return; }
  if (bitmap) bitmap.close();
  bitmap = decoded;
  const scale = Math.min(1, 1200 / Math.max(bitmap.width, bitmap.height));
  canvas.width = Math.max(1, Math.round(bitmap.width * scale));
  canvas.height = Math.max(1, Math.round(bitmap.height * scale));
  fileName.textContent = file.name;
  upload.hidden = true;
  preview.hidden = false;
  clearCrop();
  draw();
}

function point(event) {
  const rect = canvas.getBoundingClientRect();
  return [Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
          Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height))];
}

input.addEventListener('change', () => previewFile(input.files[0]));
document.querySelector('#change-photo').addEventListener('click', () => input.click());
for (const name of ['dragenter', 'dragover']) upload.addEventListener(name, event => {
  event.preventDefault(); upload.classList.add('dragging');
});
for (const name of ['dragleave', 'drop']) upload.addEventListener(name, event => {
  event.preventDefault(); upload.classList.remove('dragging');
});
upload.addEventListener('drop', event => previewFile(event.dataTransfer.files[0]));
canvas.addEventListener('pointerdown', event => {
  if (event.button !== 0) return;
  event.preventDefault();
  clearCrop();
  draw();
  dragStart = point(event);
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener('pointermove', event => {
  if (!dragStart) return;
  const end = point(event);
  setBox([Math.min(dragStart[0], end[0]), Math.min(dragStart[1], end[1]),
          Math.max(dragStart[0], end[0]), Math.max(dragStart[1], end[1])]);
});
canvas.addEventListener('pointerup', event => {
  if (!dragStart) return;
  const end = point(event);
  setBox([Math.min(dragStart[0], end[0]), Math.min(dragStart[1], end[1]),
          Math.max(dragStart[0], end[0]), Math.max(dragStart[1], end[1])]);
  dragStart = null;
});
canvas.addEventListener('pointercancel', () => { dragStart = null; });
fields.forEach(field => field.addEventListener('input', () => {
  if (fields.some(item => item.value.trim() === '')) {
    box = null; apply.disabled = true; cropResult.hidden = true; cropBlob = null;
    if (croppedUrl) URL.revokeObjectURL(croppedUrl);
    croppedUrl = null; error.textContent = ''; draw();
  } else setBox(fields.map(item => Number(item.value) / 100), false);
}));
apply.addEventListener('click', () => {
  if (!box) return;
  const [l, t, r, b] = box;
  const x = Math.floor(l * bitmap.width), y = Math.floor(t * bitmap.height);
  const width = Math.ceil(r * bitmap.width) - x, height = Math.ceil(b * bitmap.height) - y;
  const result = document.createElement('canvas');
  result.width = width; result.height = height;
  result.getContext('2d').drawImage(bitmap, x, y, width, height, 0, 0, width, height);
  const appliedBox = box;
  result.toBlob(blob => {
    if (!blob || appliedBox !== box) return;
    cropBlob = blob;
    croppedUrl = URL.createObjectURL(blob);
    croppedImage.src = croppedUrl;
    cropResult.hidden = false;
  }, 'image/png');
});
document.querySelector('#download-crop').addEventListener('click', () => {
  if (!cropBlob) return;
  const link = document.createElement('a');
  link.href = croppedUrl; link.download = 'tail-crop.png'; link.click();
});
document.querySelector('#show-result').addEventListener('click', () => {
  document.querySelector('#empty-state').hidden = true;
  document.querySelector('#result-state').hidden = false;
});
document.querySelector('#reset-result').addEventListener('click', () => {
  document.querySelector('#result-state').hidden = true;
  document.querySelector('#empty-state').hidden = false;
});
