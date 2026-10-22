// Build the presentation deck: node build_slides.js
// Optional: set PPTX_SKILL to a folder providing scripts/apply_theme.js to write the colour theme into the file;
// without it the deck is identical except that theme-based colours (e.g. slide titles) use Office defaults.
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const applyTheme = process.env.PPTX_SKILL ? require(process.env.PPTX_SKILL + "/scripts/apply_theme.js").applyTheme : null;

const ROOT = path.resolve(__dirname, "..");
const PROJ = path.join(ROOT, "MedTwinIDS");
const FIG = path.join(PROJ, "results", "figures");
const D = JSON.parse(fs.readFileSync(path.join(__dirname, "slide_data.json")));
const INS = JSON.parse(fs.readFileSync(path.join(__dirname, "slide_insights.json")));
const GROUP = "[Group ID]";
const MEMBERS = "[Member 1] · [Member 2] · [Member 3] · [Member 4] · [Member 5]";

const THEME = {
  name: "MedTwin",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "14213D", lt1: "FFFFFF", dk2: "0B2545", lt2: "EEF3F8",
    accent1: "13838F", accent2: "EB6834", accent3: "2A78D6", accent4: "1BAF7A",
    accent5: "D03B3B", accent6: "6250D6", hlink: "2A78D6", folHlink: "6250D6",
  },
};
const HEX = { navy: "0B2545", teal: "13838F", orange: "EB6834", blue: "2A78D6", aqua: "1BAF7A", red: "D03B3B",
  ink: "14213D", muted: "5B6573", tint: "EEF3F8", white: "FFFFFF", grid: "D9DEE5" };

const fmt = (x, d = 3) => Number(x).toFixed(d);
const pct = (x, d = 1) => (100 * x).toFixed(d) + "%";
const int = (x) => Number(x).toLocaleString("en-GB");

async function icon(Comp, color = "FFFFFF", size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: String(size) }));
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";
  pres.title = "MedTwin-IDS";
  pres.author = "ICT3214 Coursework 2 Group";
  pres.company = "ICT3214 Coursework 2 Group";
  pres.subject = "ICT3214 Security Analytics, Coursework 2: Presentation";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;

  pres.defineSlideMaster({
    title: "TITLE_DARK",
    background: { color: HEX.navy },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.9, w: 11.7, h: 1.5, fontFace: "Cambria",
        fontSize: 44, bold: true, color: C.background1, valign: "bottom" }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.55, w: 11.7, h: 2.6, fontFace: "Calibri",
        fontSize: 20, color: "CADCFC", valign: "top" }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "CONTENT",
    background: { color: HEX.white },
    margin: [0.5, 0.6, 0.6, 0.6],
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.9, fontFace: "Cambria",
        fontSize: 32, bold: true, color: C.text2, valign: "middle", margin: 0 }, text: "" } },
      { text: { text: "MedTwin-IDS | ICT3214 Coursework 2", options: { x: 0.6, y: 7.0, w: 6, h: 0.3, fontSize: 10,
        color: HEX.muted, margin: 0 } } },
    ],
    slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: HEX.muted, align: "right" },
  });

  const add = (section, master = "CONTENT") => pres.addSlide({ masterName: master, sectionTitle: section });
  const T = (s, text) => s.addText(text, { placeholder: "title" });
  const box = (s, x, y, w, h, fill = HEX.tint, name = "card") =>
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill },
      line: { color: fill }, objectName: name });
  const txt = (s, text, o) => s.addText(text, Object.assign({ isTextBox: true, fontSize: 15, color: HEX.ink,
    valign: "top", margin: 0 }, o));
  const badge = async (s, Comp, x, y, d = 0.62, fill = HEX.teal) => {
    s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill }, objectName: "icon-circle" });
    s.addImage({ data: await icon(Comp), x: x + d * 0.22, y: y + d * 0.22, w: d * 0.56, h: d * 0.56, objectName: "icon" });
  };
  const stat = (s, x, y, w, big, label, colour = HEX.teal) => {
    txt(s, big, { x, y, w, h: 0.85, fontSize: 40, bold: true, color: colour, fontFace: "Cambria" });
    txt(s, label, { x, y: y + 0.85, w, h: 0.6, fontSize: 13, color: HEX.muted });
  };
  // place an image inside a box, keeping its true aspect ratio and centring it horizontally
  const fit = (s, file, x, y, maxW, maxH, name) => {
    const sz = D.sizes[path.basename(file)];
    let w = maxW, h = maxW * sz.h / sz.w;
    if (h > maxH) { h = maxH; w = maxH * sz.w / sz.h; }
    s.addImage({ path: file, x: x + (maxW - w) / 2, y, w, h, objectName: name });
    return h;
  };
  const best = D.best;
  const M = D.models;

  // 1 ------------------------------------------------------------------ title
  pres.addSection({ title: "Introduction" });
  let s = add("Introduction", "TITLE_DARK");
  s.addText("MedTwin-IDS", { placeholder: "title" });
  s.addText([
    { text: "AI-driven intrusion detection and device risk scoring for a medical IoT digital twin", options: { breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 10 } },
    { text: `ICT3214 Security Analytics · Coursework 2 · Group ${GROUP}`, options: { breakLine: true, fontSize: 16 } },
    { text: MEMBERS, options: { fontSize: 16 } },
  ], { placeholder: "body" });
  s.addNotes("Introduce the team and the project in one sentence: we built a digital twin of a hospital ward, used it to generate labelled medical IoT traffic, and trained machine learning models that detect attacks in real time and score every device's risk.");

  // 2 ------------------------------------------------------------------ problem
  s = add("Introduction");
  T(s, "Attacks on ward devices are patient-safety events");
  const probs = [
    [fa.FaHeartbeat, "Spoofed monitor", "An impostor publishes normal vitals while the patient deteriorates."],
    [fa.FaEdit, "Altered vitals", "A man-in-the-middle changes heart rate or SpO2 on the way to the nurse station."],
    [fa.FaSyringe, "Rogue pump command", "Guessed credentials let an attacker change an infusion rate."],
  ];
  for (let i = 0; i < 3; i++) {
    const x = 0.6 + i * 4.1;
    box(s, x, 1.55, 3.8, 2.75);
    await badge(s, probs[i][0], x + 0.3, 1.8);
    txt(s, probs[i][1], { x: x + 0.3, y: 2.6, w: 3.2, h: 0.45, fontSize: 19, bold: true, color: HEX.navy });
    txt(s, probs[i][2], { x: x + 0.3, y: 3.1, w: 3.2, h: 1.1, fontSize: 15 });
  }
  box(s, 0.6, 4.6, 12.1, 1.95, "FFF4EE");
  await badge(s, fa.FaDatabase, 0.9, 4.85, 0.62, HEX.orange);
  txt(s, "The data gap", { x: 1.75, y: 4.85, w: 10.5, h: 0.45, fontSize: 19, bold: true, color: HEX.navy });
  txt(s, "Public IoMT datasets (e.g. WUSTL-EHMS-2020, about 16,000 records) are small and fixed: attack type, intensity and timing cannot be varied, so unseen attacks and real-time behaviour cannot be tested.",
    { x: 1.75, y: 5.35, w: 10.6, h: 1.0, fontSize: 15 });
  s.addNotes("Motivation. Medical devices are hard to patch and an attack can directly harm a patient. Research is held back by small, fixed datasets. That is why we generate our own data with a digital twin.");

  // 3 ------------------------------------------------------------------ RQs
  s = add("Introduction");
  T(s, "Five research questions");
  const rqs = [
    ["RQ1", "Can a ward digital twin generate realistic, exactly labelled IoMT data?"],
    ["RQ2", "Which of Random Forest, XGBoost and an MLP balances accuracy and speed best?"],
    ["RQ3", "Do twin-only features improve detection of integrity attacks?"],
    ["RQ4", "How well do detectors handle attack families never seen in training?"],
    ["RQ5", "Can the model run in real time with explainable device risk scores?"],
  ];
  rqs.forEach((r, i) => {
    const y = 1.5 + i * 0.98;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y, w: 1.15, h: 0.72, rectRadius: 0.1,
      fill: { color: HEX.teal }, line: { color: HEX.teal }, objectName: "rq-badge" });
    txt(s, r[0], { x: 0.6, y, w: 1.15, h: 0.72, fontSize: 18, bold: true, color: HEX.white, align: "center", valign: "middle" });
    txt(s, r[1], { x: 2.0, y, w: 6.7, h: 0.72, fontSize: 17, valign: "middle" });
  });
  box(s, 9.1, 1.5, 3.6, 4.65);
  await badge(s, fa.FaLightbulb, 9.4, 1.75);
  txt(s, "Project scope", { x: 9.4, y: 2.55, w: 3.0, h: 0.45, fontSize: 19, bold: true, color: HEX.navy });
  txt(s, [
    { text: "Idea 1: AI-driven IDS for medical IoT digital twins", options: { bullet: true, breakLine: true } },
    { text: "Idea 7: per-device risk score", options: { bullet: true, breakLine: true } },
    { text: "Idea 12: explainable alerts", options: { bullet: true, breakLine: true } },
    { text: "Bonus: simulated data, real time", options: { bullet: true } },
  ], { x: 9.4, y: 3.1, w: 3.1, h: 2.8, fontSize: 15, paraSpaceAfter: 6 });
  s.addNotes("These five questions structure the rest of the talk. Each result slide answers one of them.");

  // 4 ------------------------------------------------------------------ twin
  pres.addSection({ title: "Data" });
  s = add("Data");
  T(s, "A digital twin of an eight-bed ward");
  const layers = [
    [fa.FaHeartbeat, "Physiology", "Vitals per patient follow an Ornstein-Uhlenbeck process; clinical episodes ramp in and out"],
    [fa.FaProcedures, "Devices", "8 monitors (1 Hz), 8 infusion pumps (5 s), 8 wearables (2 s), gateway, nurse station, EHR"],
    [fa.FaNetworkWired, "Network", "MQTT, HL7, HTTPS and ARP packet events with real timing, sizes, flags and sequence numbers"],
  ];
  for (let i = 0; i < 3; i++) {
    const y = 1.5 + i * 1.7;
    box(s, 0.6, y, 7.4, 1.4);
    await badge(s, layers[i][0], 0.85, y + 0.38);
    txt(s, layers[i][1], { x: 1.75, y: y + 0.18, w: 6.0, h: 0.4, fontSize: 19, bold: true, color: HEX.navy });
    txt(s, layers[i][2], { x: 1.75, y: y + 0.6, w: 6.0, h: 0.75, fontSize: 14 });
  }
  box(s, 8.4, 1.5, 4.3, 4.8, "FFF4EE");
  txt(s, "Benign look-alikes (hard negatives)", { x: 8.7, y: 1.7, w: 3.8, h: 0.8, fontSize: 18, bold: true, color: HEX.navy });
  txt(s, [
    "Clinical episodes (tachycardia, desaturation, fever...)", "Network congestion", "2 to 6 MB firmware updates",
    "Device reboots", "Nurse pump changes and pauses", "Wearable motion artefacts", "Capture glitches for the cleaner",
  ].map((t, i, a) => ({ text: t, options: { bullet: true, breakLine: i < a.length - 1 } })),
    { x: 8.7, y: 2.55, w: 3.8, h: 3.6, fontSize: 14, paraSpaceAfter: 5 });
  s.addNotes("The twin has three synchronised layers. Monitors and wearables sample the same simulated patient, which later lets us check that the two sensors agree. We deliberately add benign events that look suspicious so models cannot rely on 'anything unusual is an attack'.");

  // 5 ------------------------------------------------------------------ attacks
  s = add("Data");
  T(s, "Six attack families, each at three intensities");
  const hdr = ["Attack", "Footprint", "Low / medium / high"].map((t) => ({ text: t, options: { bold: true, color: HEX.white, fill: { color: HEX.navy } } }));
  const arows = [
    ["DoS flood", "SYN or MQTT PUBLISH flood on the gateway", "25 / 80 / 250 pkt/s"],
    ["Spoofing", "Client-ID takeover or IP/MAC clone with false vitals", "1x / 2x / 4x rate"],
    ["Data alteration", "ARP poisoning; vitals changed in transit", "0.2 / 1 / 3 ARP/s"],
    ["Unauthorised access", "Credential guessing, then rogue pump command", "0.5 / 2 / 8 tries/s"],
    ["Reconnaissance", "Host and port scanning of the ward", "0.5 / 4 / 20 probes/s"],
    ["Exfiltration", "EHR records staged, then uploaded externally", "15 / 120 / 400 kB/s"],
  ].map((r, i) => r.map((c) => ({ text: c, options: { fill: { color: i % 2 ? HEX.white : HEX.tint } } })));
  s.addTable([hdr, ...arows], { x: 0.6, y: 1.5, w: 12.1, colW: [2.6, 6.4, 3.1], fontSize: 15, color: HEX.ink,
    border: { type: "solid", pt: 0.5, color: HEX.grid }, rowH: 0.62, valign: "middle", margin: 0.08 });
  txt(s, "Sources vary: rogue host, compromised workstation or compromised device. Every packet carries ground truth.",
    { x: 0.6, y: 6.15, w: 12.1, h: 0.5, fontSize: 14, color: HEX.muted, italic: true });
  s.addNotes("Each campaign lasts 60 to 240 seconds and uses a random source, so the same attack family appears with different footprints. Low-intensity campaigns are deliberately subtle.");

  // 6 ------------------------------------------------------------------ dataset
  s = add("Data");
  T(s, "Two independent, exactly labelled runs");
  stat(s, 0.6, 1.5, 3.0, int(D.train.packets), `packets, training run (${D.train.hours} h, seed 7)`);
  stat(s, 0.6, 3.05, 3.0, int(D.test.packets), `packets, test run (${D.test.hours} h, seed 2026)`, HEX.orange);
  stat(s, 0.6, 4.6, 3.0, int(D.train.flows + D.test.flows), "flow windows with 55 features");
  const cls = ["normal", "dos_flood", "spoofing", "data_alteration", "unauthorised_access", "reconnaissance", "exfiltration"];
  const clsN = ["Normal", "DoS", "Spoofing", "Data alt.", "Unauth.", "Recon.", "Exfil."];
  s.addChart(pres.charts.BAR, [
    { name: "Training run", labels: clsN, values: cls.map((c) => D.train.class_counts[c] || 0) },
    { name: "Test run", labels: clsN, values: cls.map((c) => D.test.class_counts[c] || 0) },
  ], { x: 4.0, y: 1.4, w: 8.7, h: 5.3, barDir: "col", barGrouping: "clustered", chartColors: [HEX.blue, HEX.orange],
    valAxisLogScaleBase: 10, valAxisLabelFormatCode: "#,##0", valAxisTitle: "Flow windows (log scale)", showValAxisTitle: true,
    valAxisTitleFontSize: 12, valAxisLabelFontSize: 12, catAxisLabelFontSize: 12, valAxisTitleColor: HEX.muted,
    catAxisLabelColor: HEX.muted, valAxisLabelColor: HEX.muted, valGridLine: { color: "E1E0D9", size: 0.5 },
    catGridLine: { style: "none" }, showLegend: true, legendPos: "t", legendFontSize: 12,
    showTitle: true, title: "Class distribution", titleFontSize: 14, titleColor: HEX.ink,
    catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", titleFontFace: "+mn-lt" });
  s.addNotes("The two runs differ in seed, patients, time of day and attack timing, so the test run is genuinely unseen. Normal traffic dominates, as in a real ward, which is why we report macro F1 as well as accuracy.");

  // 7 ------------------------------------------------------------------ architecture
  pres.addSection({ title: "Method" });
  s = add("Method");
  T(s, "End-to-end pipeline");
  fit(s, path.join(PROJ, "docs", "diagrams", "architecture.png"), 0.6, 1.3, 12.1, 5.55, "architecture");
  s.addNotes("Walk through the numbers: (1) the twin generates packets, (2) the parser cleans them and builds flow windows, (3) three models are trained and one is selected by cross-validation, (4) the live detector and risk engine score devices, (5) the dashboard shows it all.");

  // 8 ------------------------------------------------------------------ features
  s = add("Method");
  T(s, "55 features in three groups");
  const groups = [
    [fa.FaNetworkWired, "Network (34)", HEX.blue, ["Packets, bytes, loads, rates", "Packet sizes, inter-arrival, jitter", "SYN, RST, FIN, ARP counts", "MQTT CONNECT, refused, PUBLISH, CMD", "Host fan-out and port spread"]],
    [fa.FaHeartbeat, "Biometric (10)", HEX.orange, ["Temperature, SpO2, pulse", "Blood pressure, heart rate", "Respiration rate, ST segment", "Infusion rate", "Telemetry present"]],
    [fa.FaCubes, "Digital twin (11)", HEX.teal, ["Unknown source, MAC or TTL mismatch", "Duplicated or missing sequence numbers", "Monitor vs wearable residual (HR, SpO2)", "Vitals change and spread", "Requested pump-rate change"]],
  ];
  for (let i = 0; i < 3; i++) {
    const x = 0.6 + i * 4.1;
    box(s, x, 1.5, 3.8, 4.4);
    await badge(s, groups[i][0], x + 0.3, 1.75, 0.62, groups[i][2]);
    txt(s, groups[i][1], { x: x + 1.1, y: 1.85, w: 2.6, h: 0.45, fontSize: 19, bold: true, color: HEX.navy });
    txt(s, groups[i][3].map((t, k, a) => ({ text: t, options: { bullet: true, breakLine: k < a.length - 1 } })),
      { x: x + 0.3, y: 2.6, w: 3.3, h: 3.1, fontSize: 14, paraSpaceAfter: 6 });
  }
  txt(s, "5 s service-level bidirectional flow windows, oriented by the TCP SYN. No addresses, ports or time stamps are used as features.",
    { x: 0.6, y: 6.15, w: 12.1, h: 0.5, fontSize: 14, color: HEX.muted, italic: true });
  s.addNotes("Network and biometric features mirror WUSTL-EHMS-2020. The twin features are new: they compare traffic with what the twin expects, for example whether the monitor and the wearable on the same patient agree.");

  // 9 ------------------------------------------------------------------ models & protocol
  s = add("Method");
  T(s, "Three models, one leakage-aware protocol");
  const mods = [
    ["Random Forest", HEX.blue, "Bagged trees; robust, little tuning; class weights"],
    ["XGBoost", HEX.orange, "Boosted trees fix earlier errors; suits rare, hard classes"],
    ["MLP (deep network)", HEX.aqua, `${D.mlp_layers} ReLU layers; scaled features; oversampling`],
  ];
  mods.forEach((m, i) => {
    const x = 0.6 + i * 4.1;
    box(s, x, 1.5, 3.8, 1.9);
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: 1.78, w: 0.32, h: 0.32, fill: { color: m[1] }, line: { color: m[1] }, objectName: "model-dot" });
    txt(s, m[0], { x: x + 0.8, y: 1.72, w: 2.9, h: 0.45, fontSize: 19, bold: true, color: HEX.navy });
    txt(s, m[2], { x: x + 0.3, y: 2.3, w: 3.3, h: 1.0, fontSize: 15 });
  });
  txt(s, "+ Isolation Forest and label-free twin rules, both built from normal traffic only, as safety nets for unseen attacks",
    { x: 0.6, y: 3.6, w: 12.1, h: 0.4, fontSize: 15, color: HEX.teal, bold: true });
  const steps = ["Tune on temporal hold-out (last 25%)", "5-fold CV with 30-min time blocks", "Select on CV macro F1 only", "Test once on the independent run"];
  steps.forEach((t, i) => {
    const x = 0.6 + i * 3.05;
    s.addShape(pres.shapes.CHEVRON, { x, y: 4.4, w: 2.95, h: 1.3, fill: { color: i === 3 ? HEX.navy : HEX.teal },
      line: { color: HEX.white }, objectName: "protocol-step" });
    txt(s, t, { x: x + 0.45, y: 4.4, w: 2.1, h: 1.3, fontSize: 14, bold: true, color: HEX.white, valign: "middle", align: "center" });
  });
  txt(s, "Follows Arp et al. (USENIX Security 2022) to avoid data snooping and temporal leakage.",
    { x: 0.6, y: 6.0, w: 12.1, h: 0.45, fontSize: 14, color: HEX.muted, italic: true });
  s.addNotes("The three models cover bagging, boosting and deep learning. The protocol is the important part: the test run is touched once, after the model has been chosen on cross-validation alone.");

  // 10 ----------------------------------------------------------------- E1
  pres.addSection({ title: "Results" });
  s = add("Results");
  T(s, `${best} is the most accurate and stable, and fast`);
  const order = ["Random Forest", "XGBoost", "MLP"];
  const head = ["Model", "Accuracy", "Binary F1", "Macro F1", "95% interval", "CV macro F1", "Extra wards", "Latency (ms)"].map((t) => ({ text: t, options: { bold: true, color: HEX.white, fill: { color: HEX.navy } } }));
  const mrows = order.map((m) => [m, fmt(M[m].accuracy, 4), fmt(M[m].f1, 4), fmt(M[m].macro_f1),
    `${fmt(D.boot[m].low)} to ${fmt(D.boot[m].high)}`, fmt(M[m].cv_macro_f1), fmt(D.extra_mean[m]), fmt(M[m].latency_ms, 2)]
    .map((c, j) => ({ text: c, options: { bold: m === best, fill: { color: m === best ? "FFF4EE" : HEX.white }, align: j ? "center" : "left" } })));
  s.addTable([head, ...mrows], { x: 0.6, y: 1.5, w: 12.1, colW: [2.0, 1.25, 1.3, 1.25, 1.95, 1.5, 1.4, 1.45], fontSize: 14,
    color: HEX.ink, border: { type: "solid", pt: 0.5, color: HEX.grid }, rowH: 0.55, valign: "middle", margin: 0.06 });
  stat(s, 0.6, 4.0, 3.8, fmt(M[best].macro_f1), `macro F1 over 7 classes (${best})`, HEX.orange);
  stat(s, 4.75, 4.0, 3.8, pct(M[best].accuracy, 2), "accuracy on the unseen test run", HEX.teal);
  stat(s, 8.9, 4.0, 3.8, fmt(M[best].latency_ms, 2) + " ms", "to score one flow (one thread)", HEX.blue);
  txt(s, INS.e1, { x: 0.6, y: 5.75, w: 12.1, h: 0.9, fontSize: 15 });
  s.addNotes(INS.e1_notes);

  // 11 ----------------------------------------------------------------- confusion
  s = add("Results");
  T(s, "The difference is in the integrity attacks");
  fit(s, path.join(FIG, "fig_confusion.png"), 0.6, 1.4, 12.1, 4.2, "confusion");
  txt(s, INS.confusion, { x: 0.6, y: 5.75, w: 12.1, h: 1.0, fontSize: 15 });
  s.addNotes(INS.confusion_notes);

  // 12 ----------------------------------------------------------------- ablation + SHAP
  s = add("Results");
  T(s, "Twin features make integrity attacks visible");
  const fsets = ["network_only", "biometric_only", "network_biometric", "full_twin"];
  const fnames = ["Network", "Biometric", "Network + biometric", "Full twin"];
  s.addChart(pres.charts.BAR, order.map((m) => ({ name: m, labels: fnames,
    values: fsets.map((f) => D.ablation.find((r) => r.model === m && r.feature_set === f).macro_f1) })),
  { x: 0.6, y: 1.4, w: 7.2, h: 5.2, barDir: "col", barGrouping: "clustered", chartColors: [HEX.blue, HEX.orange, HEX.aqua],
    valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFormatCode: "0.0", showValue: false,
    showTitle: true, title: "Macro F1 by feature set (test run)", titleFontSize: 14, titleColor: HEX.ink,
    valAxisLabelFontSize: 12, catAxisLabelFontSize: 12, catAxisLabelColor: HEX.muted, valAxisLabelColor: HEX.muted,
    valGridLine: { color: "E1E0D9", size: 0.5 }, catGridLine: { style: "none" }, showLegend: true, legendPos: "t",
    legendFontSize: 12, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", titleFontFace: "+mn-lt" });
  box(s, 8.2, 1.5, 4.5, 4.95);
  txt(s, "What the model relies on (SHAP)", { x: 8.5, y: 1.7, w: 4.0, h: 0.45, fontSize: 18, bold: true, color: HEX.navy });
  const sg = D.shap_group;
  stat(s, 8.5, 2.3, 1.95, pct(sg.twin || 0, 0), "twin features", HEX.teal);
  stat(s, 10.5, 2.3, 1.95, pct(sg.network || 0, 0), "network features", HEX.blue);
  txt(s, INS.ablation, { x: 8.5, y: 4.0, w: 4.0, h: 2.3, fontSize: 14 });
  s.addNotes(INS.ablation_notes);

  // 13 ----------------------------------------------------------------- zero-day
  s = add("Results");
  T(s, "Unseen attacks: three layers close the gap");
  fit(s, path.join(FIG, "fig_zero_day.png"), 0.6, 1.4, 7.5, 5.2, "zero-day");
  box(s, 8.4, 1.5, 4.3, 4.95);
  txt(s, INS.zero_title, { x: 8.7, y: 1.7, w: 3.8, h: 0.8, fontSize: 18, bold: true, color: HEX.navy });
  txt(s, INS.zero_day, { x: 8.7, y: 2.6, w: 3.8, h: 3.7, fontSize: 14 });
  s.addNotes(INS.zero_day_notes);

  // 14 ----------------------------------------------------------------- twin rules
  s = add("Results");
  T(s, "Designed fix: twin consistency rules");
  const R = INS.rules;
  const ruleList = [
    ["R1", "Sender is in the asset registry"],
    ["R2", "MAC address and TTL match the registry"],
    ["R3", "Sequence numbers neither duplicated nor missing"],
    ["R4", "Monitor and wearable on one patient agree"],
    ["R5", "Pump rate does not change unexpectedly"],
  ];
  box(s, 0.6, 1.5, 6.3, 4.95);
  ruleList.forEach((r, i) => {
    const y = 1.75 + i * 0.78;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.9, y, w: 0.75, h: 0.55, rectRadius: 0.08,
      fill: { color: HEX.teal }, line: { color: HEX.teal }, objectName: "rule-badge" });
    txt(s, r[0], { x: 0.9, y, w: 0.75, h: 0.55, fontSize: 16, bold: true, color: HEX.white, align: "center", valign: "middle" });
    txt(s, r[1], { x: 1.85, y, w: 4.9, h: 0.55, fontSize: 15, valign: "middle" });
  });
  txt(s, "Thresholds: 99.9th percentile of normal training traffic. No attack labels needed.",
    { x: 0.9, y: 5.7, w: 5.8, h: 0.6, fontSize: 13, color: HEX.muted, italic: true });
  stat(s, 7.3, 1.5, 2.6, pct(R.spoof, 0), "of unseen spoofing caught", HEX.teal);
  stat(s, 10.1, 1.5, 2.6, pct(R.alt, 0), "of unseen data alteration caught", HEX.teal);
  stat(s, 7.3, 3.2, 5.4, `${fmt(R.hyb, 2)} \u2192 ${fmt(R.hyb3, 2)}`, `mean recall on unseen families (${best} + Isolation Forest, then + rules)`, HEX.orange);
  stat(s, 7.3, 4.9, 5.4, `${R.live_fa} \u2192 ${R.rules_fa}`, "false alerts in the live run: the cost, so the rules are an option", HEX.red);
  s.addNotes(INS.rules_notes);

  // 15 ----------------------------------------------------------------- sensitivity
  s = add("Results");
  T(s, "Sensitivity to attack intensity and window length");
  fit(s, path.join(FIG, "fig_intensity.png"), 0.6, 1.5, 6.0, 3.6, "intensity");
  fit(s, path.join(FIG, "fig_window.png"), 6.9, 1.5, 5.8, 3.6, "window");
  txt(s, INS.sensitivity, { x: 0.6, y: 5.3, w: 12.1, h: 1.4, fontSize: 15 });
  s.addNotes(INS.sensitivity_notes);

  // 16 ----------------------------------------------------------------- real time
  s = add("Results");
  T(s, "Real-time detection on a live twin");
  fit(s, path.join(FIG, "fig_realtime_wide.png"), 0.6, 1.25, 12.1, 4.2, "realtime");
  const L = D.live, RP = D.replay;
  stat(s, 0.6, 5.55, 2.9, `${RP.detected}/${RP.campaigns}`, "test-run campaigns detected (streamed)", HEX.teal);
  stat(s, 3.65, 5.55, 2.9, `${L.detected}/${L.campaigns}`, "campaigns on a new live twin", HEX.blue);
  stat(s, 6.7, 5.55, 2.9, `${RP.median_delay_s} s`, "median time-to-detect", HEX.orange);
  stat(s, 9.75, 5.55, 2.95, `${Math.round(L.avg_window_latency_ms)} ms`, "processing per 5 s window", HEX.navy);
  s.addNotes(INS.realtime_notes);

  // 17 ----------------------------------------------------------------- dashboard
  s = add("Results");
  T(s, "Analyst dashboard (live demo)");
  fit(s, path.join(FIG, "dash_live_ward_monitor.png"), 0.6, 1.3, 12.1, 5.5, "dashboard");
  s.addNotes("Live demo: start 'python -m medtwin.realtime --minutes 20 --speed 10' and the dashboard. Point out the ward map, a device turning red, the incident with its explanation and recommended action, and the risk timeline. Then show the Model Evaluation and Experiments pages.");

  // 18 ----------------------------------------------------------------- testing
  s = add("Results");
  T(s, "Tested and reproducible");
  fit(s, path.join(FIG, "term_pytest.png"), 0.6, 1.5, 7.1, 4.95, "pytest");
  box(s, 8.0, 1.5, 4.7, 4.95);
  await badge(s, fa.FaCheckCircle, 8.3, 1.75, 0.62, HEX.aqua);
  txt(s, `${INS.n_tests} automated tests pass`, { x: 9.1, y: 1.85, w: 3.5, h: 0.45, fontSize: 18, bold: true, color: HEX.navy });
  txt(s, [
    "Live, window-by-window features equal offline features for all 55",
    "Twin is deterministic; every attack family is labelled",
    "Each twin rule fires only on its violation",
    "Risk engine protects impersonated devices",
    "Real-time detector runs end to end",
    "Seeded runs: re-running gives the same data and results",
  ].map((t, k, a) => ({ text: t, options: { bullet: true, breakLine: k < a.length - 1 } })),
    { x: 8.3, y: 2.6, w: 4.2, h: 3.7, fontSize: 14, paraSpaceAfter: 6 });
  s.addNotes("Run 'python -m pytest -v' from the MedTwinIDS folder. The most important test checks that the live detector sees exactly the same 55 feature values as the offline training data, so the deployed model gets the inputs it was trained on. The user manual reproduces every table and figure in the report.");

  // 19 ----------------------------------------------------------------- limitations
  pres.addSection({ title: "Conclusion" });
  s = add("Conclusion");
  T(s, "Limitations and next steps");
  box(s, 0.6, 1.5, 5.9, 4.9);
  await badge(s, fa.FaExclamationTriangle, 0.9, 1.75, 0.62, HEX.orange);
  txt(s, "Limitations", { x: 1.75, y: 1.85, w: 4.5, h: 0.45, fontSize: 19, bold: true, color: HEX.navy });
  txt(s, INS.limitations.map((t, k, a) => ({ text: t, options: { bullet: true, breakLine: k < a.length - 1 } })),
    { x: 0.9, y: 2.6, w: 5.3, h: 3.6, fontSize: 14, paraSpaceAfter: 6 });
  box(s, 6.8, 1.5, 5.9, 4.9);
  await badge(s, fa.FaRocket, 7.1, 1.75);
  txt(s, "Next steps", { x: 7.95, y: 1.85, w: 4.5, h: 0.45, fontSize: 19, bold: true, color: HEX.navy });
  txt(s, INS.future.map((t, k, a) => ({ text: t, options: { bullet: true, breakLine: k < a.length - 1 } })),
    { x: 7.1, y: 2.6, w: 5.3, h: 3.6, fontSize: 14, paraSpaceAfter: 6 });
  s.addNotes("Be honest about limitations: the data is simulated, so the next step is validation on real captures and the public WUSTL-EHMS-2020 benchmark.");

  // 20 ----------------------------------------------------------------- conclusion
  s = add("Conclusion", "TITLE_DARK");
  s.addText("Conclusion", { placeholder: "title" });
  s.addText(INS.conclusion.map((t, k, a) => ({ text: t, options: { bullet: true, breakLine: k < a.length - 1 } })),
    { placeholder: "body", fontSize: 18, paraSpaceAfter: 8 });
  s.addNotes("Answer the five research questions in one line each, then invite questions.");

  // 21 ----------------------------------------------------------------- references
  s = add("Conclusion");
  T(s, "Key references");
  txt(s, [
    "A. A. Hady et al., \"Intrusion detection system for healthcare systems using medical and network data,\" IEEE Access, 2020.",
    "D. Arp et al., \"Dos and don'ts of machine learning in computer security,\" USENIX Security, 2022.",
    "T. Chen and C. Guestrin, \"XGBoost: A scalable tree boosting system,\" ACM SIGKDD, 2016.",
    "L. Breiman, \"Random forests,\" Machine Learning, 2001.",
    "F. T. Liu, K. M. Ting and Z.-H. Zhou, \"Isolation forest,\" IEEE ICDM, 2008.",
    "S. M. Lundberg and S.-I. Lee, \"A unified approach to interpreting model predictions,\" NIPS, 2017.",
    "Y. Mirsky et al., \"Kitsune: An ensemble of autoencoders for online network intrusion detection,\" NDSS, 2018.",
    "The full reference list is in the final report.",
  ].map((t, k, a) => ({ text: t, options: { bullet: { type: "number" }, breakLine: k < a.length - 1 } })),
    { x: 0.6, y: 1.5, w: 12.1, h: 5.0, fontSize: 15, paraSpaceAfter: 8 });
  s.addNotes("References for the slides; the report has the complete list.");

  const out = path.join(ROOT, "deliverables", "ICT3214_CW2_Presentation.pptx");
  await pres.writeFile({ fileName: out });
  if (applyTheme) await applyTheme(out, THEME);
  else console.log("PPTX_SKILL not set: colour theme left at the Office defaults");
  // own metadata and British English proofing (shared with the Word builders)
  require("child_process").execFileSync("python3", [path.join(__dirname, "finalise.py"), out, "--title",
    "MedTwin-IDS: AI-Driven Intrusion Detection and Device Risk Scoring for a Medical IoT Digital Twin",
    "--subject", pres.subject], { stdio: "inherit" });
  console.log("saved", out);
})();
