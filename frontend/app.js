const uploadBtn = document.getElementById('upload-btn')
const pdfFile = document.getElementById('pdf-file')
const uploadResult = document.getElementById('upload-result')

const queryBtn = document.getElementById('query-btn')
const queryInput = document.getElementById('query-input')
const sourceInput = document.getElementById('source-input')
const queryResult = document.getElementById('query-result')

uploadBtn.addEventListener('click', async () => {
  uploadResult.textContent = 'Uploading...'
  if (!pdfFile.files.length) {
    uploadResult.textContent = 'Please select a PDF file.'
    return
  }
  const fd = new FormData()
  fd.append('file', pdfFile.files[0])

  try {
    const res = await fetch('/upload_pdf', { method: 'POST', body: fd })
    const data = await res.json()
    if (!res.ok) {
      uploadResult.textContent = JSON.stringify(data, null, 2)
      return
    }
    uploadResult.textContent = `Indexed ${data.ids.length} chunks for source: ${data.source}`
  } catch (err) {
    uploadResult.textContent = String(err)
  }
})

queryBtn.addEventListener('click', async () => {
  queryResult.textContent = 'Querying...'
  const payload = { query: queryInput.value || '', top_k: 5 }
  if (sourceInput.value) payload.source = sourceInput.value

  try {
    const res = await fetch('/query_pdf', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    const data = await res.json()
    if (!res.ok) {
      queryResult.textContent = JSON.stringify(data, null, 2)
      return
    }
    queryResult.textContent = JSON.stringify(data, null, 2)
  } catch (err) {
    queryResult.textContent = String(err)
  }
})
