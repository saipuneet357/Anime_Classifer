
function removeUpload() {
  $('.file-upload-input').replaceWith($('.file-upload-input').clone());
  $('.file-upload-content').hide();
  $('.image-upload-wrap').show();
  location.reload(true);
  window.location = window.location.href+'?eraseCache=true';
}



