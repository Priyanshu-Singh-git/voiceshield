// VoiceShield deck (16:9) from deck_data.json. Style: design_reference/DECK_STYLE.md
// off-white background, near-black ink, neon-yellow accent, huge grotesk titles, big rounded cards, ✳ badges.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf8"));
const COL = { bg: "F2F2EF", card: "FAFAF8", ink: "111111", ink2: "3D3D3D", muted: "6B6B6B", line: "E2E2DD", y: "E8F542", white: "FFFFFF", fake: "B42318", real: "1F7A3A" };
const THEME = { name: "VoiceShield", colors: { dk1: COL.ink, lt1: COL.white, dk2: COL.ink2, lt2: COL.bg, accent1: COL.y, accent2: COL.ink,
  accent3: "9CA3AF", accent4: COL.fake, accent5: COL.muted, accent6: COL.line, hlink: COL.ink, folHlink: COL.ink2 } };
const BLACK = "Segoe UI Black", LIGHT = "Segoe UI Light", SANS = "Segoe UI";

async function applyTheme(file) {
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  const part = "ppt/theme/theme1.xml";
  let xml = await zip.file(part).async("string");
  for (const [slot, hex] of Object.entries(THEME.colors))
    xml = xml.replace(new RegExp("<a:" + slot + ">[\\s\\S]*?</a:" + slot + ">"), `<a:${slot}><a:srgbClr val="${hex}"/></a:${slot}>`);
  zip.file(part, xml.replace(/<a:clrScheme name="[^"]*">/, `<a:clrScheme name="${THEME.name}">`));
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "VoiceShield: AI voice detection";
pres.author = "Priyanshu Singh";
const W = 13.333, H = 7.5, M = 0.55;
const R = 0.35; // card corner radius (in)

function png(file) { const b = fs.readFileSync(file); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }
function cover(file, x, y, w, h) { // crop-to-fill a box
  const s = png(file), r = Math.max(w / s.w, h / s.h), iw = s.w * r, ih = s.h * r;
  return { path: file, x, y, w, h, sizing: { type: "crop", x: (iw - w) / 2, y: (ih - h) / 2, w, h } };
}
function fit(file, x, y, w, h) {
  const s = png(file), r = Math.min(w / s.w, h / s.h), iw = s.w * r, ih = s.h * r;
  return { path: file, x: x + (w - iw) / 2, y: y + (h - ih) / 2, w: iw, h: ih };
}

pres.defineSlideMaster({ title: "BASE", background: { color: COL.bg }, objects: [
  { text: { text: "VoiceShield", options: { x: M, y: 0.28, w: 3, h: 0.3, fontFace: BLACK, fontSize: 11, color: COL.ink, margin: 0 } } },
  { text: { text: "AI voice detection · 2026", options: { x: W - M - 4, y: 0.28, w: 4, h: 0.3, fontFace: SANS, fontSize: 10, color: COL.muted, align: "right", margin: 0 } } },
], slideNumber: { x: W - M - 0.5, y: H - 0.42, w: 0.5, h: 0.25, fontFace: SANS, fontSize: 10, color: COL.muted, align: "right" } });

function slide() { return pres.addSlide({ masterName: "BASE" }); }
function card(s, x, y, w, h, fill = COL.card, name) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: R, fill: { color: fill }, line: { type: "none" }, objectName: name });
}
function star(s, x, y, size = 0.55, dark = false) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: size, h: size, rectRadius: 0.1, fill: { color: dark ? COL.ink : COL.y }, line: { type: "none" } });
  s.addText("✳", { x, y, w: size, h: size, fontFace: BLACK, fontSize: size * 34, color: dark ? COL.y : COL.ink, align: "center", valign: "middle", margin: 0 });
}
function title(s, l1, l2, x, y, w, size = 60, opts = {}) {  // two-line title, light + black weights
  s.addText([{ text: l1, options: { fontFace: opts.firstBold ? BLACK : LIGHT, breakLine: true } },
             { text: l2, options: { fontFace: opts.firstBold ? LIGHT : BLACK } }],
    { x, y, w, h: size / 72 * 2.3, fontSize: size, color: COL.ink, margin: 0, valign: "top", lineSpacingMultiple: 0.88, charSpacing: -1.5 });
}
function pill(s, text, x, y, w, yellow = true) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.42, rectRadius: 0.21, fill: { color: yellow ? COL.y : COL.white }, line: { color: COL.ink, width: yellow ? 0 : 1 } });
  s.addText(text, { x, y, w, h: 0.42, fontFace: SANS, fontSize: 12, bold: true, color: COL.ink, align: "center", valign: "middle", margin: 0 });
}
function meta(s, text, x, y, w) { s.addText(text, { x, y, w, h: 0.3, fontFace: SANS, fontSize: 10.5, color: COL.muted, margin: 0 }); }

// ================= 1. cover / thumbnail =================
{
  const s = slide();
  s.addText("Detects AI-cloned and synthetic voices from a few seconds of audio.", { x: M, y: 0.8, w: 3.4, h: 0.7, fontFace: SANS, fontSize: 11, color: COL.ink2, margin: 0 });
  const st = D.cover_stats;
  st.forEach(([v, l], i) => {
    s.addText(v, { x: M + 3.9 + i * 2.1, y: 0.75, w: 2, h: 0.5, fontFace: BLACK, fontSize: 30, color: COL.ink, margin: 0 });
    s.addText(l, { x: M + 3.9 + i * 2.1, y: 1.25, w: 2, h: 0.3, fontFace: SANS, fontSize: 10, color: COL.muted, margin: 0 });
  });
  s.addText([{ text: "Voice", options: { breakLine: true } }, { text: "-Shield", options: {} }],
    { x: W - M - 6.4, y: 0.55, w: 6.4, h: 3.1, fontFace: BLACK, fontSize: 104, color: COL.ink, align: "right", margin: 0, lineSpacingMultiple: 0.84, charSpacing: -3, objectName: "Product name" });
  s.addText([{ text: "Real voice or AI clone?  ", options: { color: COL.ink } }, { text: "Know in a second.", options: { color: COL.ink, highlight: COL.y } }],
    { x: M, y: 3.55, w: 8, h: 0.6, fontFace: SANS, fontSize: 24, bold: true, margin: 0, objectName: "USP" });
  card(s, M, 4.35, W - 2 * M, 2.6, COL.ink, "Render strip");
  s.addImage({ ...fit(D.assets.render_wave, M + 4.6, 4.4, W - 2 * M - 5.9, 2.5), objectName: "3D speech render" });
  s.addText([{ text: "Graph-attention network · runs in the browser · ", options: { color: "BBBBBB" } }, { text: "audio never leaves your device", options: { color: COL.y } }],
    { x: M + 0.45, y: 5.85, w: 4.2, h: 0.9, fontFace: SANS, fontSize: 13, margin: 0, valign: "bottom" });
  star(s, W - M - 0.95, 6.15, 0.6);
}

// ================= 2. problem: numbered cards =================
{
  const s = slide();
  title(s, "Anyone's voice", "can be cloned", M, 0.75, 8, 52);
  const items = [
    ["01", "Seconds of audio", "Free tools clone a voice from a short recording.", COL.card, COL.ink],
    ["02", "Convincing calls", "Fraudsters use cloned voices to pose as a boss, a relative or a customer.", COL.y, COL.ink],
    ["03", "Ears aren't enough", "Good synthetic speech is hard to spot by listening alone.", COL.ink, COL.white],
    ["04", "Screen automatically", "Call centres, banks and platforms need a fast, automatic check.", COL.card, COL.ink],
  ];
  const cw = (W - 2 * M - 3 * 0.25) / 4;
  items.forEach(([n, h, p, bg, fg], i) => {
    const x = M + i * (cw + 0.25), y = 2.75;
    card(s, x, y, cw, 4.1, bg, `Card ${n}`);
    s.addText(h, { x: x + 0.3, y: y + 0.3, w: cw - 0.6, h: 0.8, fontFace: SANS, fontSize: 16, bold: true, color: fg, margin: 0, valign: "top" });
    s.addText(p, { x: x + 0.3, y: y + 1.05, w: cw - 0.6, h: 1.5, fontFace: SANS, fontSize: 12.5, color: fg === COL.white ? "CCCCCC" : COL.ink2, margin: 0, valign: "top" });
    s.addText(n, { x: x + 0.3, y: y + 2.75, w: cw - 0.6, h: 1.1, fontFace: BLACK, fontSize: 64, color: fg, margin: 0, valign: "bottom", charSpacing: -2 });
  });
}

// ================= 3. solution intro =================
{
  const s = slide();
  card(s, M, 0.8, 5.6, 6.1, "E9E9E6", "Render card");
  s.addImage({ ...fit(D.assets.render_ribbon, M + 0.2, 1.0, 5.2, 5.7), objectName: "3D speech ribbon" });
  title(s, "Real voice", "or AI clone?", M + 6.1, 1.0, 6.6, 64, { firstBold: true });
  s.addText("VoiceShield listens to a few seconds of speech and flags synthetic or cloned voices, the tiny vocoder artefacts that ears miss. It runs in the browser, so recordings never leave the device.",
    { x: M + 6.1, y: 3.7, w: 6.0, h: 1.5, fontFace: SANS, fontSize: 15, color: COL.ink2, margin: 0, valign: "top" });
  pill(s, "Try the live demo  →", M + 6.1, 5.4, 2.6, true);
  pill(s, "Record · upload · test clips", M + 8.9, 5.4, 2.9, false);
  star(s, W - M - 0.6, 6.3, 0.55, true);
}

// ================= 4. results =================
{
  const s = slide();
  title(s, "Catches AI voices", "it has never heard", M, 0.75, 9, 50);
  const st = D.result_stats;  // [value, label, sub]
  const cw = (W - 2 * M - 3 * 0.25) / 4;
  st.forEach(([v, l, sub], i) => {
    const x = M + i * (cw + 0.25), y = 2.75, yellow = i === 0;
    card(s, x, y, cw, 2.2, yellow ? COL.y : COL.card, `Stat ${i + 1}`);
    s.addText(v, { x: x + 0.3, y: y + 0.25, w: cw - 0.6, h: 0.9, fontFace: BLACK, fontSize: 40, color: COL.ink, margin: 0, charSpacing: -1 });
    s.addText([{ text: l, options: { bold: true, breakLine: true } }, { text: sub, options: { color: COL.muted } }],
      { x: x + 0.3, y: y + 1.2, w: cw - 0.6, h: 0.9, fontFace: SANS, fontSize: 12, color: COL.ink, margin: 0, valign: "top" });
  });
  card(s, M, 5.2, W - 2 * M, 1.7, COL.ink, "Note");
  s.addText([{ text: "How it was measured  ", options: { bold: true, color: COL.y } },
             { text: D.measure_note, options: { color: "DDDDDD" } }],
    { x: M + 0.4, y: 5.35, w: W - 2 * M - 1.6, h: 1.4, fontFace: SANS, fontSize: 13, margin: 0, valign: "middle" });
  star(s, W - M - 0.95, 5.75, 0.6);
}

// ================= 5. product / demo =================
{
  const s = slide();
  title(s, "Product", "overview", M, 0.75, 5, 56, { firstBold: true });
  const feats = [["Record or upload", "4-second mic recording, or any audio file"], ["Instant verdict", "Real voice vs AI-generated, with a confidence needle"],
                 ["See the evidence", "Spectrogram with each analysed window marked"], ["Private by design", "The model runs in the browser; nothing is uploaded"]];
  feats.forEach(([h, p], i) => {
    const y = 2.6 + i * 1.05;
    pill(s, `0${i + 1}`, M, y + 0.05, 0.75, i === 0);
    s.addText([{ text: h, options: { bold: true, breakLine: true } }, { text: p, options: { color: COL.muted } }],
      { x: M + 0.95, y, w: 3.9, h: 0.9, fontFace: SANS, fontSize: 13, color: COL.ink, margin: 0, valign: "top" });
  });
  card(s, M + 5.1, 0.8, W - 2 * M - 5.1, 6.1, COL.ink, "Screenshot frame");
  s.addImage({ ...fit(D.assets.shot, M + 5.3, 1.0, W - 2 * M - 5.5, 5.7), objectName: "Demo screenshot" });
}

// ================= 6. how it works =================
{
  const s = slide();
  const steps = [["Raw waveform in", "No hand-made features: 4 s of 16 kHz audio."],
                 ["Learnable band-pass filters", "70 sinc filters split the sound into frequency bands."],
                 ["Two graphs", "One graph over frequency bands, one over time frames, each with attention."],
                 ["Fused by a master node", "A heterogeneous graph links both views and scores real vs AI."]];
  steps.forEach(([h, p], i) => {
    const y = 0.8 + i * 1.55;
    card(s, M, y, 6.2, 1.35, i === 2 ? COL.y : COL.card, `Step ${i + 1}`);
    s.addText(`0${i + 1}`, { x: M + 0.3, y, w: 0.9, h: 1.35, fontFace: BLACK, fontSize: 28, color: COL.ink, valign: "middle", margin: 0 });
    s.addText([{ text: h, options: { bold: true, breakLine: true } }, { text: p, options: { color: COL.ink2 } }],
      { x: M + 1.25, y: y + 0.15, w: 4.7, h: 1.05, fontFace: SANS, fontSize: 13, color: COL.ink, margin: 0, valign: "middle" });
  });
  title(s, "Graph attention", "on raw audio", M + 6.7, 0.8, 5.6, 50);
  s.addText(D.how_note, { x: M + 6.7, y: 3.1, w: 5.4, h: 2.2, fontFace: SANS, fontSize: 14, color: COL.ink2, margin: 0, valign: "top" });
  pill(s, "AASIST · ICASSP 2022 · built from scratch", M + 6.7, 5.5, 4.3, false);
}

// ================= 7. per-attack =================
{
  const s = slide();
  title(s, "Every attack", "type, tested", M, 0.75, 6, 50);
  s.addText(D.attack_note, { x: M, y: 2.55, w: 4.3, h: 2.6, fontFace: SANS, fontSize: 13, color: COL.ink2, margin: 0, valign: "top" });
  pill(s, D.attack_chart.legend, M, 5.3, 3.6, true);
  card(s, M + 4.7, 0.8, W - 2 * M - 4.7, 6.1, COL.card, "Chart card");
  const a = D.per_attack;
  s.addChart(pres.charts.BAR, [{ name: D.attack_chart.series, labels: a.map((r) => r.attack), values: a.map((r) => +r.value.toFixed(2)) }], {
    x: M + 4.9, y: 1.0, w: W - 2 * M - 5.1, h: 5.7, barDir: "col", barGapWidthPct: 45,
    chartColors: a.map((r) => (r.seen ? "9CA3AF" : COL.y)), valAxisLabelFormatCode: "0.0", showValue: true, dataLabelFormatCode: D.attack_chart.fmt,
    dataLabelFontSize: 9, dataLabelColor: COL.ink, catAxisLabelColor: COL.ink, valAxisLabelColor: COL.muted, catAxisLabelFontSize: 10,
    valGridLine: { color: COL.line, size: 0.75 }, catGridLine: { style: "none" }, showLegend: false,
    showTitle: true, title: D.attack_chart.title, titleColor: COL.ink, titleFontSize: 12, titleFontFace: SANS,
  });
}

// ================= 8. benchmark =================
{
  const s = slide();
  title(s, "Benchmarked", "against the field", M, 0.75, 9, 50);
  const hdr = ["Model", "Parameters", "EER ↓", "min t-DCF ↓", "Source"].map((t) => ({ text: t, options: { bold: true, color: COL.ink, fill: { color: COL.y } } }));
  const rows = D.bench.map((r) => [r.model, r.params, r.eer, r.tdcf, r.source].map((t) => ({ text: String(t),
    options: { color: COL.ink, bold: r.mine, fill: { color: r.mine ? COL.card : COL.bg } } })));
  card(s, M, 2.6, W - 2 * M, 0.42 * (rows.length + 1) + 0.4, COL.card, "Table card");
  s.addTable([hdr, ...rows], { x: M + 0.2, y: 2.8, w: W - 2 * M - 0.4, colW: [3.6, 1.7, 1.4, 1.7, 3.43], fontFace: SANS, fontSize: 13,
    rowH: 0.42, margin: 0.08, border: { type: "solid", pt: 0.75, color: COL.line } });
  s.addText(D.bench_note, { x: M, y: 6.25, w: W - 2 * M, h: 0.7, fontFace: SANS, fontSize: 11.5, color: COL.muted, margin: 0, valign: "top" });
}

// ================= 9. working together =================
{
  const s = slide();
  title(s, "A vision backed", "by clear steps", M, 0.75, 9, 50);
  const steps = [["Send sample calls", "A few minutes of your real recordings (and any known fakes)."],
                 ["Calibrated pilot", "I tune the threshold on your audio and report catch rate vs false alarms."],
                 ["Deploy", "Browser, API or on-premise service: your audio stays with you."]];
  const cw = (W - 2 * M - 2 * 0.25) / 3;
  steps.forEach(([h, p], i) => {
    const x = M + i * (cw + 0.25), y = 2.75, dark = i === 1;
    card(s, x, y, cw, 3.2, dark ? COL.ink : COL.card, `Work ${i + 1}`);
    s.addText(`0${i + 1}`, { x: x + 0.3, y: y + 0.25, w: 1.5, h: 0.8, fontFace: BLACK, fontSize: 40, color: dark ? COL.y : COL.ink, margin: 0 });
    s.addText([{ text: h, options: { bold: true, breakLine: true } }, { text: p, options: { color: dark ? "CCCCCC" : COL.ink2 } }],
      { x: x + 0.3, y: y + 1.2, w: cw - 0.6, h: 1.8, fontFace: SANS, fontSize: 14, color: dark ? COL.white : COL.ink, margin: 0, valign: "top" });
  });
  star(s, W - M - 0.6, 6.35, 0.55);
}

// ================= 10. CTA =================
{
  const s = slide();
  title(s, "Screen your calls", "for AI voices", M, 0.85, 9, 64);
  card(s, M, 3.6, 3.0, 3.3, COL.white, "QR card");
  s.addImage({ path: D.assets.qr, x: M + 0.25, y: 3.75, w: 2.5, h: 2.5 });
  s.addText("Scan for the live demo", { x: M, y: 6.3, w: 3.0, h: 0.4, fontFace: SANS, fontSize: 11, color: COL.muted, align: "center", margin: 0 });
  const links = [["Live demo", D.demo_url], ["Code, benchmark and limits", D.repo_url]];
  s.addText(links.flatMap(([l, u], i) => [{ text: l, options: { bold: true, color: COL.ink, breakLine: true } },
    { text: u.replace(/^https:\/\//, ""), options: { color: COL.ink, hyperlink: { url: u }, breakLine: i < links.length - 1 } },
    ...(i < links.length - 1 ? [{ text: " ", options: { fontSize: 8, breakLine: true } }] : [])]),
    { x: M + 3.4, y: 3.7, w: 7.5, h: 2.2, fontFace: SANS, fontSize: 18, margin: 0, valign: "top" });
  pill(s, "Send me sample calls · I'll run a pilot", M + 3.4, 6.0, 4.4, true);
  s.addText("Priyanshu Singh · Freelance AI / ML Engineer · Data: ASVspoof 2019 (ODC-By)", { x: M + 3.4, y: 6.6, w: 8, h: 0.3, fontFace: SANS, fontSize: 10, color: COL.muted, margin: 0 });
  star(s, W - M - 1.6, 4.2, 1.4, true);
}

(async () => {
  const out = path.join(__dirname, "deck.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out);
  console.log("wrote", out);
})();
