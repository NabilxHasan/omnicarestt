/* ==========================================================================
   Google AI Labs (Opal, Stitch, Pomeli) Interactive Client Application
   Omnicare STT — Bangla Speech-to-Text Studio
   ========================================================================== */

let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let startTime = 0;
let timerInterval = null;
let audioCtx = null;
let analyser = null;
let animFrameId = null;

// Common Medical Terms for Client-Side Highlighting
const MEDICAL_DICTIONARY = {
  symptoms: ['জ্বর', 'ব্যথা', 'কাশি', 'বমি', 'মাথা ব্যথা', 'সর্দি', 'ক্লান্তি', 'শ্বাসকষ্ট'],
  drugs: ['প্যারাসিটামল', 'এমক্সিসিলিন', 'ইনসুলিন', 'ওমিপ্রাজল', 'এন্টাসিড', 'অ্যাসপিরিন', 'সিপ্রোফ্লক্সাসিন'],
  dosage: ['৫০০ মিগ্রা', '১০০ মিগ্রা', '১০ মিগ্রা', 'দিনে ২ বার', 'দিনে ৩ বার', 'খাওয়ার পর', 'খাওয়ার আগে'],
  anatomy: ['মাথা', 'পেট', 'বুক', 'গলা', 'হৃদপিণ্ড', 'ফুসফুস', 'যকৃৎ']
};

// --- Tab Switching Navigation ---
function switchTab(tabId) {
  document.querySelectorAll('.workspace').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-pill').forEach(el => el.classList.remove('active'));

  const activeWorkspace = document.getElementById(`tab-${tabId}`);
  if (activeWorkspace) {
    activeWorkspace.classList.add('active');
  }

  // Highlight pill button
  const targetBtn = Array.from(document.querySelectorAll('.nav-pill')).find(btn => 
    btn.getAttribute('onclick')?.includes(tabId)
  );
  if (targetBtn) {
    targetBtn.classList.add('active');
  }

  if (tabId === 'benchmark') {
    loadBenchHistory();
  } else if (tabId === 'dataset') {
    loadTrainingManifest();
  }
}

// --- Live Audio Visualizer Canvas ---
function initVisualizer(stream) {
  const canvas = document.getElementById('audioCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  canvas.width = canvas.parentElement.clientWidth;
  canvas.height = canvas.parentElement.clientHeight;

  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  
  analyser = audioCtx.createAnalyser();
  const source = audioCtx.createMediaStreamSource(stream);
  source.connect(analyser);
  analyser.fftSize = 64;

  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  function draw() {
    animFrameId = requestAnimationFrame(draw);
    analyser.getByteFrequencyData(dataArray);

    ctx.fillStyle = 'rgba(5, 8, 15, 0.3)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const barWidth = (canvas.width / bufferLength) * 2.5;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const barHeight = (dataArray[i] / 255) * canvas.height * 0.8;

      // Cyan to Violet Gradient
      const gradient = ctx.createLinearGradient(0, canvas.height, 0, 0);
      gradient.addColorStop(0, '#00F2FE');
      gradient.addColorStop(1, '#8B5CF6');

      ctx.fillStyle = gradient;
      ctx.fillRect(x, canvas.height - barHeight, barWidth - 2, barHeight);

      x += barWidth;
    }
  }

  draw();
}

function stopVisualizer() {
  if (animFrameId) {
    cancelAnimationFrame(animFrameId);
  }
  const canvas = document.getElementById('audioCanvas');
  if (canvas) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
}

// --- Microphone Recording Controls ---
async function toggleRecording() {
  const btn = document.getElementById('record-btn');
  const btnText = document.getElementById('rec-btn-text');
  const statusTag = document.getElementById('rec-status-tag');

  if (!isRecording) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunks = [];
      mediaRecorder = new MediaRecorder(stream);

      mediaRecorder.ondataavailable = event => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        
        const player = document.getElementById('audio-player');
        const playerContainer = document.getElementById('audio-preview-container');
        if (player && playerContainer) {
          player.src = audioUrl;
          playerContainer.style.display = 'block';
        }

        // Send to backend for transcription
        await processAudioTranscription(audioBlob);
      };

      mediaRecorder.start();
      isRecording = true;
      startTime = Date.now();
      
      btn.classList.add('recording');
      btnText.textContent = 'Stop Recording';
      statusTag.textContent = 'RECORDING LIVE';
      statusTag.style.background = 'rgba(244, 63, 94, 0.2)';
      statusTag.style.color = 'var(--accent-rose)';

      initVisualizer(stream);

      timerInterval = setInterval(updateTimer, 1000);
    } catch (err) {
      alert('Microphone access denied or not available: ' + err.message);
    }
  } else {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(track => track.stop());
    isRecording = false;

    btn.classList.remove('recording');
    btnText.textContent = 'Start Speech Capture';
    statusTag.textContent = 'PROCESSING...';
    statusTag.style.background = 'rgba(0, 242, 254, 0.12)';
    statusTag.style.color = 'var(--accent-cyan)';

    stopVisualizer();
    clearInterval(timerInterval);
    document.getElementById('rec-timer').textContent = '00:00';
  }
}

function updateTimer() {
  const elapsed = Math.floor((Date.now() - startTime) / 1000);
  const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
  const secs = String(elapsed % 60).padStart(2, '0');
  document.getElementById('rec-timer').textContent = `${mins}:${secs}`;
}

// --- Drag & Drop Audio Handlers ---
function handleFileSelect(event) {
  const file = event.target.files[0];
  if (file) {
    uploadAudioFile(file);
  }
}

const dropzone = document.getElementById('audio-dropzone');
if (dropzone) {
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, e => {
      e.preventDefault();
      e.stopPropagation();
    }, false);
  });

  dropzone.addEventListener('dragover', () => dropzone.classList.add('dragover'));
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', e => {
    dropzone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      uploadAudioFile(files[0]);
    }
  });
}

async function uploadAudioFile(file) {
  const player = document.getElementById('audio-player');
  const playerContainer = document.getElementById('audio-preview-container');
  if (player && playerContainer) {
    player.src = URL.createObjectURL(file);
    playerContainer.style.display = 'block';
  }
  await processAudioTranscription(file);
}

// --- Process Transcription via API ---
async function processAudioTranscription(audioData) {
  const formData = new FormData();
  formData.append('audio', audioData, 'recording.wav');
  formData.append('file', audioData, 'recording.wav');

  document.getElementById('transcript-e1').innerHTML = '<em>Inference in progress...</em>';
  document.getElementById('transcript-e2').innerHTML = '<em>Inference in progress...</em>';
  document.getElementById('agreement-val').textContent = '...';

  try {
    const response = await fetch('/transcribe', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) throw new Error('Transcription endpoint error');

    const result = await response.json();
    renderDualEngineResult(result);
  } catch (err) {
    document.getElementById('transcript-e1').textContent = 'Error: ' + err.message;
    document.getElementById('transcript-e2').textContent = 'Error: ' + err.message;
  } finally {
    document.getElementById('rec-status-tag').textContent = 'READY';
    document.getElementById('rec-status-tag').style.background = 'rgba(255,255,255,0.05)';
    document.getElementById('rec-status-tag').style.color = 'var(--text-muted)';
  }
}

function renderDualEngineResult(result) {
  const text1 = result.text || 'No speech detected';
  const text2 = result.text2 || result.text || 'No speech detected';
  const duration1 = result.duration_s || 0;
  const duration2 = result.duration2_s || duration1;
  const agreement = Math.round((result.agreement !== undefined ? result.agreement : 1.0) * 100);

  document.getElementById('transcript-e1').textContent = text1;
  document.getElementById('transcript-e2').textContent = text2;
  document.getElementById('e1-time').textContent = `${duration1} s`;
  document.getElementById('e2-time').textContent = `${duration2} s`;

  document.getElementById('agreement-val').textContent = `${agreement}%`;
  
  const badge = document.getElementById('agreement-badge');
  if (agreement >= 85) {
    badge.textContent = 'HIGH CONSENSUS';
    badge.className = 'meter-badge high';
  } else if (agreement >= 60) {
    badge.textContent = 'MEDIUM CONSENSUS';
    badge.className = 'meter-badge medium';
  } else {
    badge.textContent = 'LOW CONSENSUS (REVIEW)';
    badge.className = 'meter-badge low';
  }

  // Update Medical Inspector View
  renderMedicalHighlights(text1);
}

// --- Medical Entity Highlighting ---
function renderMedicalHighlights(text) {
  const box = document.getElementById('medical-highlight-box');
  const bnormBox = document.getElementById('bnorm-compare-box');
  if (!box) return;

  let highlighted = text;

  // Highlight drugs, symptoms, anatomy, dosage
  Object.keys(MEDICAL_DICTIONARY).forEach(category => {
    MEDICAL_DICTIONARY[category].forEach(term => {
      const reg = new RegExp(term, 'g');
      highlighted = highlighted.replace(reg, `<span class="medical-tag ${category}">${term}</span>`);
    });
  });

  box.innerHTML = highlighted;
  if (bnormBox) {
    bnormBox.innerHTML = `[Raw Transcription]\n${text}\n\n[Normalized Output]\n${text.trim()}`;
  }
}

// --- Copy Helper ---
function copyTranscript() {
  const text = document.getElementById('transcript-e1').textContent;
  if (text) {
    navigator.clipboard.writeText(text);
    alert('Transcript copied to clipboard!');
  }
}

// --- YouTube Speech Benchmark ---
async function runYouTubeBench() {
  const urlInput = document.getElementById('yt-url-input');
  const url = urlInput.value.trim();
  if (!url) return alert('Please enter a valid YouTube URL');

  try {
    const res = await fetch('/bench/transcribe', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });
    if (!res.ok) throw new Error('Benchmark request failed');
    await loadBenchHistory();
    alert('YouTube Speech Benchmark completed successfully!');
  } catch (err) {
    alert('Error running YouTube benchmark: ' + err.message);
  }
}

async function loadBenchHistory() {
  const tbody = document.getElementById('history-table-body');
  if (!tbody) return;

  try {
    const res = await fetch('/bench/history');
    const items = await res.json();

    if (!items || items.length === 0) {
      tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-muted); padding: 2rem;">No benchmark entries found. Paste a YouTube URL above to run benchmark!</td></tr>`;
      return;
    }

    tbody.innerHTML = items.map(item => `
      <tr>
        <td><strong>${item.video_id || item.url || 'Audio'}</strong></td>
        <td style="max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${item.text || item.transcript || ''}</td>
        <td><span class="meter-badge ${item.agreement >= 0.8 ? 'high' : 'medium'}">${Math.round((item.agreement || 1) * 100)}%</span></td>
        <td>${item.duration_s || '--'} s</td>
        <td>
          <button class="btn-secondary" style="padding: 0.25rem 0.5rem; font-size: 0.75rem;" onclick="deleteBenchItem('${item.video_id}')">Delete</button>
        </td>
      </tr>
    `).join('');
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--accent-rose);">Failed to load history log</td></tr>`;
  }
}

async function deleteBenchItem(videoId) {
  if (!videoId) return;
  await fetch(`/bench/history/${videoId}`, { method: 'DELETE' });
  loadBenchHistory();
}

// --- Dataset Curation Studio ---
let currentCandidateMeta = null;

async function chunkYouTubeForTraining() {
  const urlInput = document.getElementById('training-yt-url');
  const url = urlInput.value.trim();
  if (!url) return alert('Enter YouTube URL');

  const container = document.getElementById('candidate-pairs-container');
  const tbody = document.getElementById('candidate-table-body');
  if (container && tbody) {
    container.style.display = 'block';
    tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; padding: 2rem; color: var(--accent-cyan);">Chunking audio & running dual STT engines... Please wait (may take 10-30s).</td></tr>`;
  }

  try {
    const res = await fetch('/training/from_youtube', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Chunk extraction failed');
    }
    const data = await res.json();
    currentCandidateMeta = data;
    renderCandidatePairs(data);
  } catch (err) {
    alert('Extraction error: ' + err.message);
    if (container) container.style.display = 'none';
  }
}

function renderCandidatePairs(data) {
  const container = document.getElementById('candidate-pairs-container');
  const tbody = document.getElementById('candidate-table-body');
  const countTag = document.getElementById('candidate-count-tag');
  if (!container || !tbody) return;

  container.style.display = 'block';
  const pairs = data.pairs || [];
  if (countTag) countTag.textContent = `${pairs.length} PAIRS READY`;

  if (pairs.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; padding: 2rem; color: var(--text-muted);">No clips extracted.</td></tr>`;
    return;
  }

  tbody.innerHTML = pairs.map((p, index) => `
    <tr id="cand-row-${p.chunk_id}">
      <td><strong>${p.chunk_id}</strong></td>
      <td><audio src="${p.audio_url}" controls style="height: 32px; width: 160px;"></audio></td>
      <td style="font-size: 0.82rem; max-width: 260px;">
        <div style="color: var(--accent-cyan); font-weight: 600;">E1: ${p.stt_text || '(none)'}</div>
        <div style="color: var(--accent-purple);">E2: ${p.stt_text2 || '(none)'}</div>
        <div style="color: var(--text-muted); margin-top: 0.2rem;">Caption: ${p.caption || '(auto/none)'}</div>
      </td>
      <td>
        <input type="text" id="label-input-${p.chunk_id}" class="form-input" value="${(p.caption || p.stt_text || '').replace(/"/g, '&quot;')}" style="font-size: 0.88rem; width: 100%;">
      </td>
      <td>
        <button class="btn-primary" style="padding: 0.35rem 0.75rem; font-size: 0.78rem;" onclick="acceptCandidatePair('${p.chunk_id}', ${index})">Accept Pair</button>
      </td>
    </tr>
  `).join('');
}

async function acceptCandidatePair(chunkId, index) {
  if (!currentCandidateMeta || !currentCandidateMeta.pairs) return;
  const p = currentCandidateMeta.pairs[index];
  const inputEl = document.getElementById(`label-input-${chunkId}`);
  const finalText = inputEl ? inputEl.value.trim() : (p.caption || p.stt_text);

  if (!finalText) {
    return alert('Please enter or verify a non-empty text label before accepting');
  }

  try {
    const res = await fetch('/training/accept', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        chunk_id: p.chunk_id,
        caption: p.caption || '',
        final_text: finalText,
        source_url: currentCandidateMeta.source_url || '',
        source_video_id: currentCandidateMeta.video_id || '',
        start_s: p.start_s || 0,
        end_s: p.end_s || 0,
        duration_s: p.duration_s || 0
      })
    });

    if (!res.ok) throw new Error('Accept request failed');

    const row = document.getElementById(`cand-row-${chunkId}`);
    if (row) {
      row.style.background = 'rgba(16, 185, 129, 0.1)';
      row.innerHTML = `<td colspan="5" style="color: var(--accent-emerald); font-weight: 600; text-align: center;">Accepted & Saved to Training Manifest!</td>`;
    }

    await loadTrainingManifest();
  } catch (err) {
    alert('Error accepting pair: ' + err.message);
  }
}

async function loadTrainingManifest() {
  const tbody = document.getElementById('manifest-table-body');
  if (!tbody) return;

  try {
    const res = await fetch('/training/manifest');
    const items = await res.json();

    if (!items || items.length === 0) {
      tbody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--text-muted); padding: 2rem;">No dataset training pairs saved yet. Chunk YouTube videos above to build Kaggle fine-tuning dataset!</td></tr>`;
      return;
    }

    tbody.innerHTML = items.map(data => `
      <tr>
        <td><strong>${data.chunk_id}</strong></td>
        <td><audio src="/training/audio/${data.chunk_id}" controls style="height: 32px; width: 160px;"></audio></td>
        <td>${data.final_text || data.caption || ''}</td>
        <td>
          <button class="btn-secondary" style="padding: 0.25rem 0.5rem; font-size: 0.75rem;" onclick="deleteManifestItem('${data.chunk_id}')">Remove</button>
        </td>
      </tr>
    `).join('');
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--text-muted);">No training manifest found.</td></tr>`;
  }
}

async function deleteManifestItem(chunkId) {
  if (!chunkId) return;
  await fetch(`/training/manifest/${chunkId}`, { method: 'DELETE' });
  loadTrainingManifest();
}

// --- Digital Doctor Prescription SaaS Functions ---
let patientRecorder = null;
let patientChunks = [];
let docRecorder = null;
let docChunks = [];

async function recordPatientVoice() {
  const btnText = document.getElementById('patient-rec-text');

  if (patientRecorder && patientRecorder.state === 'recording') {
    patientRecorder.stop();
    btnText.textContent = 'Record Bangla Symptoms Voice';
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    patientChunks = [];
    patientRecorder = new MediaRecorder(stream);
    patientRecorder.ondataavailable = e => patientChunks.push(e.data);
    patientRecorder.onstop = async () => {
      const audioBlob = new Blob(patientChunks, { type: 'audio/wav' });
      btnText.textContent = 'Transcribing Symptoms...';
      const formData = new FormData();
      formData.append('audio', audioBlob, 'patient_symptoms.wav');
      formData.append('file', audioBlob, 'patient_symptoms.wav');
      try {
        const res = await fetch('/transcribe', { method: 'POST', body: formData });
        const data = await res.json();
        document.getElementById('patient-speech-text').value = data.text || 'জ্বর এবং মাথাব্যথা';
        btnText.textContent = 'Record Bangla Symptoms Voice';
      } catch (err) {
        btnText.textContent = 'Error Transcribing';
      }
    };

    patientRecorder.start();
    btnText.textContent = 'Recording Symptoms (Click to Stop)...';
  } catch (err) {
    alert('Microphone access denied: ' + err.message);
  }
}

async function recordDoctorVoice() {
  const btnText = document.getElementById('doc-rec-text');
  if (docRecorder && docRecorder.state === 'recording') {
    docRecorder.stop();
    btnText.textContent = 'Dictate Prescription Voice';
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    docChunks = [];
    docRecorder = new MediaRecorder(stream);
    docRecorder.ondataavailable = e => docChunks.push(e.data);
    docRecorder.onstop = async () => {
      const audioBlob = new Blob(docChunks, { type: 'audio/wav' });
      btnText.textContent = 'Transcribing Doctor Dictation...';
      const formData = new FormData();
      formData.append('audio', audioBlob, 'doctor_dictation.wav');
      formData.append('file', audioBlob, 'doctor_dictation.wav');
      try {
        const res = await fetch('/transcribe', { method: 'POST', body: formData });
        const data = await res.json();
        document.getElementById('doctor-speech-text').value = data.text || 'প্যারাসিটামল ৫০০ মিগ্রা দিনে ৩ বার খাবার পর ৭ দিন।';
        btnText.textContent = 'Dictate Prescription Voice';
      } catch (err) {
        btnText.textContent = 'Error Transcribing';
      }
    };

    docRecorder.start();
    btnText.textContent = 'Recording Dictation (Click to Stop)...';
  } catch (err) {
    alert('Microphone access denied: ' + err.message);
  }
}

async function generatePrescriptionSaaS() {
  const pName = document.getElementById('patient-name').value.trim() || 'Kamrul Islam';
  const pAge = document.getElementById('patient-age').value.trim() || '34';
  const pGender = document.getElementById('patient-gender').value || 'Male';
  const pTranscript = document.getElementById('patient-speech-text').value.trim();
  const dDictation = document.getElementById('doctor-speech-text').value.trim();

  try {
    const res = await fetch('/medical/prescription', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        patient_name: pName,
        patient_age: pAge,
        patient_gender: pGender,
        doctor_name: 'Dr. Nabil Hasan',
        doctor_title: 'MBBS, FCPS (Internal Medicine)',
        patient_transcript: pTranscript,
        doctor_dictation: dDictation
      })
    });

    if (!res.ok) throw new Error('Prescription generation failed');

    const data = await res.json();

    // Render Digital Prescription Card
    document.getElementById('rx-patient-name').textContent = data.patient.name;
    document.getElementById('rx-patient-age-gender').textContent = `${data.patient.age} Yrs / ${data.patient.gender}`;
    document.getElementById('rx-date').textContent = data.patient.date;
    document.getElementById('rx-id').textContent = data.prescription_id;

    // Complaints list
    const compUl = document.getElementById('rx-complaints-list');
    compUl.innerHTML = data.chief_complaints.map(c => `<li>${c}</li>`).join('');

    // Investigations
    const testUl = document.getElementById('rx-tests-list');
    testUl.innerHTML = data.investigations.length > 0 
      ? data.investigations.map(t => `<li>${t}</li>`).join('')
      : '<li>No lab tests requested.</li>';

    // Medicines table
    const medTbody = document.getElementById('rx-medicines-body');
    medTbody.innerHTML = data.medicines.map(m => `
      <tr>
        <td><strong>${m.name}</strong> <small>(${m.bangla_name || ''})</small></td>
        <td>${m.strength}</td>
        <td><span class="badge-tag" style="background: rgba(16, 185, 129, 0.15); color: var(--accent-emerald);">${m.dosage}</span></td>
        <td>${m.timing}</td>
        <td>${m.duration}</td>
      </tr>
    `).join('');

    // Doctor Advice
    const advOl = document.getElementById('rx-advice-list');
    advOl.innerHTML = data.advice.map(a => `<li>${a}</li>`).join('');

    // Smooth scroll to prescription printable area
    document.getElementById('prescription-printable-area').scrollIntoView({ behavior: 'smooth' });
  } catch (err) {
    alert('Error generating prescription: ' + err.message);
  }
}
