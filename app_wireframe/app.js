const input = document.querySelector('#photo');
const upload = document.querySelector('#upload-zone');
const preview = document.querySelector('#preview');
const previewImage = document.querySelector('#preview-image');
const fileName = document.querySelector('#file-name');
let currentObjectUrl;

function previewFile(file) {
  if (!file) return;
  if (!['image/jpeg', 'image/png'].includes(file.type)) {
    window.alert('Please choose a JPEG or PNG image.');
    return;
  }
  if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl);
  currentObjectUrl = URL.createObjectURL(file);
  previewImage.src = currentObjectUrl;
  fileName.textContent = file.name;
  upload.hidden = true;
  preview.hidden = false;
}

input.addEventListener('change', () => previewFile(input.files[0]));
document.querySelector('#change-photo').addEventListener('click', () => input.click());
for (const name of ['dragenter', 'dragover']) upload.addEventListener(name, event => {
  event.preventDefault();
  upload.classList.add('dragging');
});
for (const name of ['dragleave', 'drop']) upload.addEventListener(name, event => {
  event.preventDefault();
  upload.classList.remove('dragging');
});
upload.addEventListener('drop', event => previewFile(event.dataTransfer.files[0]));
document.querySelector('#show-result').addEventListener('click', () => {
  document.querySelector('#empty-state').hidden = true;
  document.querySelector('#result-state').hidden = false;
});
document.querySelector('#reset-result').addEventListener('click', () => {
  document.querySelector('#result-state').hidden = true;
  document.querySelector('#empty-state').hidden = false;
});
