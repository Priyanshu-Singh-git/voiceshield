// VoiceShield browser demo: AASIST anti-spoofing with ONNX Runtime Web, fully client-side.
const $ = (id) => document.getElementById(id);
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
const SR = 16000, N = 64600;   // 4.04 s windows, the model's training input length

const S = { meta: null, session: null, results: {} };
window.voiceshield = S; // exposed for automated checks

// decode any audio file / blob to 16 kHz mono Float32Array
async function toMono16k(arrayBuffer) {
  const ctx = new AudioContext({ sampleRate: SR });   // decode straight to 16 kHz: no double resampling
  const buf = await ctx.decodeAudioData(arrayBuffer);
  ctx.close();
  const len = Math.ceil(buf.duration * SR);
  const off = new OfflineAudioContext(1, len, SR);
  const src = off.createBufferSource(); src.buffer = buf; src.connect(off.destination); src.start();
  return (await off.startRendering()).getChannelData(0);
}

// same rule as training: repeat-pad short clips; split long clips into up to 3 evenly spaced windows
function windows(x) {
  if (x.length <= N) {
    const w = new Float32Array(N);
    for (let i = 0; i < N; i++) w[i] = x[i % x.length];
    return [{ start: 0, data: w }];
  }
  const k = Math.min(3, Math.ceil(x.length / N));
  const step = (x.length - N) / Math.max(k - 1, 1);
  return Array.from({ length: k }, (_, i) => { const s = Math.round(i * step); return { start: s, data: x.slice(s, s + N) }; });
}

async function scoreWindow(w) {
  const out = await S.session.run({ wave: new ort.Tensor("float32", w, [1, N]) });
  const l = out.logits.data;
  return l[1] - l[0];   // > threshold = bona fide
}

// log-magnitude spectrogram (naive DFT on 512-sample frames; fast enough for a few seconds)
function drawSpectrogram(x, wins) {
  const cv = $("spec"), ctx = cv.getContext("2d"), W = cv.width, H = cv.height;
  ctx.fillStyle = "#111"; ctx.fillRect(0, 0, W, H);
  const n = 256, hop = Math.max(128, Math.floor(x.length / W)), bins = 96;
  const cols = Math.min(W, Math.floor((x.length - n) / hop));
  const img = ctx.createImageData(cols, bins);
  const hann = Float32Array.from({ length: n }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1)));
  for (let c = 0; c < cols; c++) {
    const o = c * hop;
    for (let b = 0; b < bins; b++) {
      const k = Math.floor((b / bins) * (n / 2));
      let re = 0, im = 0;
      for (let i = 0; i < n; i++) { const v = x[o + i] * hann[i], a = 2 * Math.PI * k * i / n; re += v * Math.cos(a); im -= v * Math.sin(a); }
      const db = Math.max(0, Math.min(1, (10 * Math.log10(re * re + im * im + 1e-9) + 60) / 70));
      const p = ((bins - 1 - b) * cols + c) * 4;
      img.data[p] = 232 * db; img.data[p + 1] = 245 * db; img.data[p + 2] = 66 * db; img.data[p + 3] = 255;
    }
  }
  const t = document.createElement("canvas"); t.width = cols; t.height = bins; t.getContext("2d").putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true; ctx.drawImage(t, 0, 0, W, H);
  // mark the analysed windows
  wins.forEach((w) => {
    const x0 = (w.start / x.length) * W, x1 = Math.min(W, ((w.start + N) / x.length) * W);
    ctx.strokeStyle = w.score > S.meta.threshold ? "#7dd99a" : "#ff8a80"; ctx.lineWidth = 3;
    ctx.strokeRect(x0 + 2, 2, Math.max(x1 - x0 - 4, 4), H - 4);
  });
}

async function analyse(x, key) {
  $("badge").className = "badge wait"; $("badge").textContent = "Listening…";
  try {
    const wins = windows(x);
    const t0 = performance.now();
    for (const w of wins) w.score = await scoreWindow(w.data);
    const ms = performance.now() - t0;
    const thr = S.meta.threshold;
    const mean = wins.reduce((a, w) => a + w.score, 0) / wins.length;
    const real = mean > thr;
    $("badge").className = "badge " + (real ? "real" : "fake");
    $("badge").textContent = real ? "Real voice" : "AI-generated";
    const pos = 50 + 50 * Math.tanh((mean - thr) / S.meta.scale);
    $("needle").style.left = Math.max(2, Math.min(98, pos)) + "%";
    $("detail").textContent = `${(x.length / SR).toFixed(1)} s analysed in ${wins.length} window${wins.length > 1 ? "s" : ""} · ${Math.round(ms)} ms in your browser · score ${(mean - thr).toFixed(2)} vs threshold`;
    $("windows").innerHTML = wins.length > 1 ? wins.map((w, i) => `<div class="win">Window ${i + 1}<b style="color:${w.score > thr ? "var(--real)" : "var(--fake)"}">${w.score > thr ? "real" : "AI"}</b></div>`).join("") : "";
    drawSpectrogram(x, wins);
    if (key) S.results[key] = { score: mean, real, ms };
  } catch (e) { $("badge").className = "badge wait"; $("badge").innerHTML = `<span class="err">Error: ${e.message}</span>`; console.error(e); }
}

async function record() {
  const btn = $("rec");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    const rec = new MediaRecorder(stream), chunks = [];
    rec.ondataavailable = (e) => chunks.push(e.data);
    const done = new Promise((r) => (rec.onstop = r));
    rec.start(); btn.classList.add("on"); btn.disabled = true;
    for (let s = 4; s > 0; s--) { btn.textContent = `● Recording… ${s}`; await new Promise((r) => setTimeout(r, 1000)); }
    rec.stop(); await done; stream.getTracks().forEach((t) => t.stop());
    btn.classList.remove("on"); btn.disabled = false; btn.textContent = "● Record 4 seconds";
    document.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", "false"));
    await analyse(await toMono16k(await new Blob(chunks).arrayBuffer()), null);
  } catch (e) { btn.classList.remove("on"); btn.disabled = false; btn.textContent = "● Record 4 seconds"; $("detail").innerHTML = `<span class="err">${e.message}</span>`; }
}

async function main() {
  S.meta = await (await fetch("meta.json")).json();
  const m = S.meta;
  $("stats").innerHTML = m.stats.rows.map((r) => `<tr><td>${r.label}</td><td class="n"><b>${r.value}</b></td></tr>`).join("");
  $("statsNote").textContent = m.stats.note;
  const box = $("samples");
  m.samples.forEach((s) => {
    const b = document.createElement("button"); b.className = "chip" + (s.attack === "-" ? " real" : ""); b.textContent = s.label; b.title = s.title || s.label; b.type = "button";
    b.onclick = async () => {
      box.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", "false")); b.setAttribute("aria-pressed", "true");
      await analyse(await toMono16k(await (await fetch(s.file)).arrayBuffer()), s.file);
    };
    box.appendChild(b);
  });
  S.session = await ort.InferenceSession.create("models/" + m.model, { executionProviders: ["wasm"] });
  $("badge").textContent = "Ready";
  $("rec").onclick = record;
  $("file").onchange = async (e) => {
    const f = e.target.files[0]; if (!f) return;
    box.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", "false"));
    await analyse(await toMono16k(await f.arrayBuffer()), null);
  };
  const q = new URLSearchParams(location.search).get("sample");
  if (q !== null && box.children[+q]) box.children[+q].click();
}
main().catch((e) => { $("badge").innerHTML = `<span class="err">Failed to load: ${e.message}</span>`; });
