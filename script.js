//  Setup Canvas

const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");

function resize() {
  const maxW = Math.min(window.innerWidth - 24, 640);
  const maxH = Math.min(window.innerHeight - 230, 520);
  canvas.width = Math.floor(maxW);
  canvas.height = Math.floor(maxH);
}
resize();
window.addEventListener("resize", () => {
  resize();
  if (!running) drawBg();
});

const W = () => canvas.width;
const H = () => canvas.height;

//  Sintesis Audio (Web Audio API)

let audioCtx = null;

function initAudio() {
  if (!audioCtx)
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === "suspended") audioCtx.resume();
}

function playLaserSound() {
  if (!audioCtx) return;
  const osc = audioCtx.createOscillator(),
    g = audioCtx.createGain();
  osc.type = "triangle";
  osc.frequency.setValueAtTime(900, audioCtx.currentTime);
  osc.frequency.exponentialRampToValueAtTime(150, audioCtx.currentTime + 0.12);
  g.gain.setValueAtTime(0.1, audioCtx.currentTime);
  g.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.12);
  osc.connect(g);
  g.connect(audioCtx.destination);
  osc.start();
  osc.stop(audioCtx.currentTime + 0.12);
}

function playExplosionSound() {
  if (!audioCtx) return;
  const osc = audioCtx.createOscillator(),
    g = audioCtx.createGain();
  osc.type = "sawtooth";
  osc.frequency.setValueAtTime(120, audioCtx.currentTime);
  osc.frequency.exponentialRampToValueAtTime(10, audioCtx.currentTime + 0.2);
  g.gain.setValueAtTime(0.2, audioCtx.currentTime);
  g.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.2);
  osc.connect(g);
  g.connect(audioCtx.destination);
  osc.start();
  osc.stop(audioCtx.currentTime + 0.2);
}

function playHitSound() {
  if (!audioCtx) return;
  const osc = audioCtx.createOscillator(),
    g = audioCtx.createGain();
  osc.type = "square";
  osc.frequency.setValueAtTime(250, audioCtx.currentTime);
  osc.frequency.setValueAtTime(100, audioCtx.currentTime + 0.08);
  g.gain.setValueAtTime(0.2, audioCtx.currentTime);
  g.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.2);
  osc.connect(g);
  g.connect(audioCtx.destination);
  osc.start();
  osc.stop(audioCtx.currentTime + 0.2);
}

function playGameOverSound() {
  if (!audioCtx) return;
  const now = audioCtx.currentTime;
  [180, 150, 120, 90].forEach((freq, i) => {
    const d = 0.18;
    const osc = audioCtx.createOscillator(),
      g = audioCtx.createGain();
    osc.type = "sawtooth";
    osc.frequency.setValueAtTime(freq, now + i * d);
    g.gain.setValueAtTime(0.15, now + i * d);
    g.gain.exponentialRampToValueAtTime(0.01, now + i * d + d - 0.02);
    osc.connect(g);
    g.connect(audioCtx.destination);
    osc.start(now + i * d);
    osc.stop(now + i * d + d);
  });
}

//  Jaringan Saraf Tiruan Feed-Forward (Perceptron) — Arsitektur Ditingkatkan
//  Arsitektur   : 8 input → 12 hidden (tanh) → 8 hidden (tanh) → 4 output (softmax)
//  Pembelajaran : REINFORCE Policy Gradient  (online, per-langkah)
//  Otak bersama : satu instance NeuralNetwork per tier alien
//  Otak persisten antar gelombang → alien bertambah pintar setiap level!

class NeuralNetwork {
  /**
   * @param {number} inp   Neuron input  (default 8)
   * @param {number} hid1  Neuron layer  pertama  (default 12)
   * @param {number} hid2  Neuron layer hidden kedua   (default 8)
   * @param {number} out   Neuron output (default 4: KIRI/KANAN/BOM/TURUN)
   */
  constructor(inp = 8, hid1 = 12, hid2 = 8, out = 4) {
    this.inp = inp;
    this.hid1 = hid1;
    this.hid2 = hid2;
    this.out = out;
    this.lr = 0.05; // laju pembelajaran awal (disesuaikan per level)
    this.trainCount = 0; // total langkah gradien (untuk % kecerdasan HUD)
    this.totalReward = 0; // reward kumulatif (diagnostik)
    this._cache = null; // cache forward-pass terakhir untuk backprop
    this._initWeights();
  }

  // Inisialisasi Xavier / He (mencegah gradien menghilang atau meledak di awal)
  _initWeights() {
    const r = (fanIn) => (Math.random() * 2 - 1) * Math.sqrt(2 / fanIn);

    // Layer 1: inp → hid1
    this.W1 = Array.from({ length: this.hid1 }, () =>
      Array.from({ length: this.inp }, () => r(this.inp)),
    );
    this.b1 = new Array(this.hid1).fill(0);

    // Layer 2: hid1 → hid2
    this.W2 = Array.from({ length: this.hid2 }, () =>
      Array.from({ length: this.hid1 }, () => r(this.hid1)),
    );
    this.b2 = new Array(this.hid2).fill(0);

    // Layer output: hid2 → out
    this.W3 = Array.from({ length: this.out }, () =>
      Array.from({ length: this.hid2 }, () => r(this.hid2)),
    );
    this.b3 = new Array(this.out).fill(0);
  }

  // Softmax stabil secara numeris
  _softmax(arr) {
    const max = Math.max(...arr);
    const exps = arr.map((x) => Math.exp(Math.min(30, x - max)));
    const sum = exps.reduce((a, b) => a + b, 0) || 1e-10;
    return exps.map((x) => x / sum);
  }

  /**
   * Forward pass: inputs[] → distribusi probabilitas aksi.
   * Menyimpan semua aktivasi antara ke cache untuk backpropagation.
   */
  forward(inputs) {
    // Layer hidden 1 (aktivasi tanh)
    const h1 = this.W1.map((row, i) => {
      const z = row.reduce((s, w, j) => s + w * inputs[j], 0) + this.b1[i];
      return Math.tanh(z);
    });

    // Layer hidden 2 (aktivasi tanh)
    const h2 = this.W2.map((row, i) => {
      const z = row.reduce((s, w, j) => s + w * h1[j], 0) + this.b2[i];
      return Math.tanh(z);
    });

    // Layer output (linear → softmax)
    const raw = this.W3.map(
      (row, i) => row.reduce((s, w, j) => s + w * h2[j], 0) + this.b3[i],
    );
    const probs = this._softmax(raw);

    this._cache = {
      inputs: [...inputs],
      h1: [...h1],
      h2: [...h2],
      probs: [...probs],
    };
    return probs;
  }

  /**
   * Simpan snapshot cache terkini untuk digunakan pada delayed reward
   * (misal: status saat bom dijatuhkan, latih saat bom kena/meleset).
   */
  saveSnapshot() {
    if (!this._cache) return null;
    return {
      inputs: [...this._cache.inputs],
      h1: [...this._cache.h1],
      h2: [...this._cache.h2],
      probs: [...this._cache.probs],
    };
  }

  // Update gradien inti (REINFORCE): δ = reward × (1[a=i] − p[i])
  _applyGradient(inputs, h1, h2, probs, actionIdx, reward) {
    const lr = this.lr;
    // Clip reward agar tidak ada satu sinyal pun yang terlalu mendominasi
    const rClip = Math.max(-3, Math.min(3, reward));
    const dOut = probs.map((p, i) => rClip * ((i === actionIdx ? 1 : 0) - p));

    // --- Perbarui bobot output W3, b3 ---
    for (let i = 0; i < this.out; i++) {
      for (let j = 0; j < this.hid2; j++) this.W3[i][j] += lr * dOut[i] * h2[j];
      this.b3[i] += lr * dOut[i];
    }

    // --- Backpropagasi ke layer hidden 2 (turunan tanh = 1 − h²) ---
    const dH2 = h2.map((h, j) => {
      const up = this.W3.reduce((s, row, i) => s + row[j] * dOut[i], 0);
      return up * (1 - h * h);
    });

    // Perbarui bobot W2, b2
    for (let i = 0; i < this.hid2; i++) {
      for (let j = 0; j < this.hid1; j++) this.W2[i][j] += lr * dH2[i] * h1[j];
      this.b2[i] += lr * dH2[i];
    }

    // --- Backpropagasi ke layer hidden 1 ---
    const dH1 = h1.map((h, j) => {
      const up = this.W2.reduce((s, row, i) => s + row[j] * dH2[i], 0);
      return up * (1 - h * h);
    });

    // Perbarui bobot W1, b1
    for (let i = 0; i < this.hid1; i++) {
      for (let j = 0; j < this.inp; j++)
        this.W1[i][j] += lr * dH1[i] * inputs[j];
      this.b1[i] += lr * dH1[i];
    }

    this.trainCount++;
    this.totalReward += reward;
  }

  /** Latih berdasarkan pass forward() terbaru (immediate reward). */
  train(actionIdx, reward) {
    if (!this._cache) return;
    const { inputs, h1, h2, probs } = this._cache;
    this._applyGradient(inputs, h1, h2, probs, actionIdx, reward);
  }

  /** Latih menggunakan snapshot yang disimpan sebelumnya (delayed reward). */
  trainFromSnapshot(snap, actionIdx, reward) {
    if (!snap) return;
    this._applyGradient(
      snap.inputs,
      snap.h1,
      snap.h2,
      snap.probs,
      actionIdx,
      reward,
    );
  }

  /**
   * Kecerdasan: 0–100.
   * Meningkat seiring langkah latihan; mendatar di 100%.
   * Digunakan untuk tampilan HUD dan peluruhan epsilon-greedy.
   */
  get intelligence() {
    return Math.min(100, Math.floor(this.trainCount / 40));
  }

  setLR(lr) {
    this.lr = Math.max(0.008, Math.min(0.15, lr));
  }
}

//  Otak Bersama (satu per tier alien, persisten antar gelombang)

let alienBrains = null; // [otakTier0, otakTier1, otakTier2]

function initBrains() {
  // Otak baru di awal game — mereka akan belajar selama sesi bermain
  alienBrains = [
    new NeuralNetwork(8, 12, 8, 4), // Tier 0: Cumi  — dasar
    new NeuralNetwork(8, 12, 8, 4), // Tier 1: Kepiting — menengah
    new NeuralNetwork(8, 12, 8, 4), // Tier 2: UFO   — mahir
  ];
}

// Konstanta indeks aksi (pemetaan neuron output)
const ACT_LEFT = 0; // bergerak ke kiri
const ACT_RIGHT = 1; // bergerak ke kanan
const ACT_BOMB = 2; // menjatuhkan bom
const ACT_DESCEND = 3; // menghindari peluru ATAU turun mendekati pemain

//  Status

let score, lives, level, running, paused, gameOverFlag, raf;
let player, bullets, aliens, bombs, particles, stars;
let frameCount = 0;
const keys = {};

//  Konstanta

const PLAYER_W = 36,
  PLAYER_H = 28;
const BULLET_W = 3,
  BULLET_H = 14;
const BOMB_W = 4,
  BOMB_H = 10;
const SHOOT_COOLDOWN = 160; // ms — lebih cepat dari sebelumnya (220ms) agar pemain lebih mudah
const ALIEN_DECISION_FRAMES = 8; // NN memutuskan kembali setiap N frame game
const MAX_LIVES = 5; // nyawa awal — ditambah agar permainan lebih mudah

//  Inisialisasi / Reset

function init() {
  score = 0;
  lives = MAX_LIVES;
  level = 1;
  paused = false;
  gameOverFlag = false;
  frameCount = 0;

  player = {
    x: W() / 2 - PLAYER_W / 2,
    y: H() - 56,
    speed: 4.5, // sedikit lebih cepat agar mudah menghindar
    lastShot: 0,
    iframes: 0,
  };

  bullets = [];
  bombs = [];
  particles = [];
  stars = Array.from({ length: 80 }, () => ({
    x: Math.random() * W(),
    y: Math.random() * H(),
    r: Math.random() * 1.4 + 0.3,
    spd: Math.random() * 0.4 + 0.1,
    a: Math.random(),
  }));

  initBrains(); // otak baru untuk game baru
  spawnWave();
  updateHUD();
}

//  Spawn Gelombang (otak sengaja TIDAK direset di sini)

function spawnWave() {
  aliens = [];
  let cols = Math.min(6 + level, 12); // mulai lebih sedikit alien di level awal
  const rows = Math.min(1 + Math.floor(level / 2), 4); // baris lebih sedikit di awal

  const maxAvailableWidth = W() - 32;
  let spacing = Math.min(44, maxAvailableWidth / cols);
  if (spacing < 30) cols = Math.max(2, Math.floor(maxAvailableWidth / 30));

  const actualSpacing = Math.min(44, maxAvailableWidth / cols);
  const sx = (W() - (cols - 1) * actualSpacing - 28) / 2;

  // Sesuaikan laju pembelajaran dengan level → adaptasi lebih cepat di gelombang berikutnya
  if (alienBrains) {
    const lr = Math.min(0.12, 0.05 + level * 0.006);
    alienBrains.forEach((b) => b.setLR(lr));
  }

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const tier = r < 2 ? 0 : r < 3 ? 1 : 2;
      aliens.push({
        x: sx + c * actualSpacing,
        y: 28 + r * 40,
        w: 28,
        h: 22,
        tier,
        vx: 0,
        vy: 0,
        alive: true,

        //  Status NN
        decisionTimer: Math.floor(Math.random() * ALIEN_DECISION_FRAMES), // jeda waktu (stagger)
        bombCooldown: 90 + Math.floor(Math.random() * 140), // lebih lama → bom lebih jarang
        lastAction: -1,
        lastSnapshot: null,
        pendingReward: 0,
        lastDist: Infinity,
      });
    }
  }
}

//  HUD

function updateHUD() {
  document.getElementById("scoreVal").textContent = score;
  document.getElementById("livesVal").textContent =
    "♥".repeat(Math.max(0, lives)) + "♡".repeat(Math.max(0, MAX_LIVES - lives));
  document.getElementById("levelVal").textContent = level;

  // Tampilan kecerdasan AI
  if (alienBrains) {
    const avgIntel = Math.floor(
      alienBrains.reduce((s, b) => s + b.intelligence, 0) / alienBrains.length,
    );
    const aiEl = document.getElementById("aiVal");
    if (aiEl) {
      aiEl.textContent = avgIntel + "%";
      // Hue: 0°=merah (bodoh) → 120°=hijau (pintar)
      const hue = Math.floor(avgIntel * 1.2);
      aiEl.style.color = `hsl(${hue},100%,60%)`;
      aiEl.style.textShadow = `0 0 10px hsl(${hue},100%,60%)`;
    }
  }
}

//  Partikel

function burst(x, y, color, n = 14) {
  for (let i = 0; i < n; i++) {
    const a = Math.random() * Math.PI * 2,
      s = 1 + Math.random() * 4;
    particles.push({
      x,
      y,
      vx: Math.cos(a) * s,
      vy: Math.sin(a) * s,
      life: 1,
      decay: 0.03 + Math.random() * 0.04,
      r: 1.5 + Math.random() * 3,
      color,
    });
  }
}

//  Fungsi bantu gambar

function drawBg() {
  ctx.fillStyle = "#050505";
  ctx.fillRect(0, 0, W(), H());
  stars.forEach((s) => {
    ctx.globalAlpha = 0.3 + s.a * 0.6;
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.globalAlpha = 1;
}

function drawPlayer() {
  const p = player;
  if (p.iframes > 0 && Math.floor(Date.now() / 80) % 2) return;
  ctx.save();
  ctx.translate(p.x + PLAYER_W / 2, p.y + PLAYER_H / 2);
  ctx.shadowColor = "#00f5ff";
  ctx.shadowBlur = 14;
  ctx.fillStyle = "#1e293b";
  ctx.beginPath();
  ctx.moveTo(0, -PLAYER_H / 2);
  ctx.lineTo(PLAYER_W / 2, PLAYER_H / 2);
  ctx.lineTo(-PLAYER_W / 2, PLAYER_H / 2);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = "#ffffff";
  ctx.beginPath();
  ctx.ellipse(0, -4, 7, 9, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#00f5ff";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(-PLAYER_W / 2, PLAYER_H / 2);
  ctx.lineTo(-8, 2);
  ctx.moveTo(PLAYER_W / 2, PLAYER_H / 2);
  ctx.lineTo(8, 2);
  ctx.stroke();
  if (Math.random() > 0.4) {
    ctx.fillStyle = `rgba(255,${(100 + Math.random() * 100) | 0},0,0.9)`;
    ctx.beginPath();
    ctx.moveTo(-5, PLAYER_H / 2);
    ctx.lineTo(5, PLAYER_H / 2);
    ctx.lineTo(0, PLAYER_H / 2 + 8 + Math.random() * 6);
    ctx.closePath();
    ctx.fill();
  }
  ctx.shadowBlur = 0;
  ctx.restore();
}

// Renderer sprite alien per tier
const ALIEN_SHAPES = [
  (c, w, h) => {
    // tier 0 — cumi-cumi
    c.fillStyle = "#39ff14";
    c.beginPath();
    c.arc(0, -2, w * 0.38, 0, Math.PI * 2);
    c.fill();
    c.fillStyle = "#ffffff";
    c.beginPath();
    c.arc(-w * 0.16, -4, 3, 0, Math.PI * 2);
    c.fill();
    c.beginPath();
    c.arc(w * 0.16, -4, 3, 0, Math.PI * 2);
    c.fill();
    c.strokeStyle = "#39ff14";
    c.lineWidth = 2;
    for (let t = -1; t <= 1; t++) {
      c.beginPath();
      c.moveTo(t * 5, w * 0.3);
      c.lineTo(t * 8, w * 0.55);
      c.stroke();
    }
  },
  (c, w, h) => {
    // tier 1 — kepiting
    c.fillStyle = "#00ff66";
    c.fillRect(-w * 0.4, -h * 0.3, w * 0.8, h * 0.6);
    c.fillStyle = "#ffffff";
    c.beginPath();
    c.arc(-w * 0.4, -h * 0.1, 5, 0, Math.PI * 2);
    c.fill();
    c.beginPath();
    c.arc(w * 0.4, -h * 0.1, 5, 0, Math.PI * 2);
    c.fill();
    c.beginPath();
    c.arc(-w * 0.2, -h * 0.1, 3, 0, Math.PI * 2);
    c.fill();
    c.beginPath();
    c.arc(w * 0.2, -h * 0.1, 3, 0, Math.PI * 2);
    c.fill();
  },
  (c, w, h) => {
    // tier 2 — UFO
    c.fillStyle = "#00ffcc";
    c.beginPath();
    c.ellipse(0, 0, w * 0.48, h * 0.28, 0, 0, Math.PI * 2);
    c.fill();
    c.fillStyle = "#ffffff";
    c.beginPath();
    c.arc(0, -h * 0.18, w * 0.22, 0, Math.PI * 2);
    c.fill();
    c.strokeStyle = "rgba(0,255,102,0.5)";
    c.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
      c.beginPath();
      c.arc(0, 0, w * (0.18 + i * 0.06), 0, Math.PI * 2);
      c.stroke();
    }
  },
];

// Aksi → warna indikator untuk umpan balik visual
const ACT_COLORS = ["#ff4444", "#4488ff", "#ffff00", "#ff8800"];
// Aksi → panah/simbol kecil untuk umpan balik visual
const ACT_SYMBOLS = ["◀", "▶", "●", "▼"];

function drawAliens() {
  const BASE_COLORS = ["#39ff14", "#00ff66", "#00ffcc"];
  aliens.forEach((a) => {
    if (!a.alive) return;
    const brain = alienBrains ? alienBrains[a.tier] : null;
    const intel = brain ? brain.intelligence : 0;

    ctx.save();
    ctx.translate(a.x + a.w / 2, a.y + a.h / 2);

    // Efek pendaran meningkat saat AI semakin pintar (6–22px)
    const glow = 6 + (intel / 100) * 16;
    ctx.shadowColor = BASE_COLORS[a.tier];
    ctx.shadowBlur = glow;

    ALIEN_SHAPES[a.tier](ctx, a.w, a.h);
    ctx.shadowBlur = 0;

    // Tampilkan aksi terakhir sebagai indikator kecil di atas alien
    if (a.lastAction >= 0) {
      ctx.font = "7px monospace";
      ctx.textAlign = "center";
      ctx.fillStyle = ACT_COLORS[a.lastAction];
      ctx.globalAlpha = 0.75;
      ctx.fillText(ACT_SYMBOLS[a.lastAction], 0, -a.h / 2 - 4);
      ctx.globalAlpha = 1;
    }

    ctx.restore();
  });
}

function drawBullets() {
  bullets.forEach((b) => {
    ctx.save();
    const grad = ctx.createLinearGradient(b.x, b.y, b.x, b.y + BULLET_H);
    grad.addColorStop(0, "#ffe600");
    grad.addColorStop(0.5, "#ff5500");
    grad.addColorStop(1, "#ff0000");
    ctx.fillStyle = grad;
    ctx.shadowColor = "#ff5500";
    ctx.shadowBlur = 12;
    ctx.fillRect(b.x, b.y, BULLET_W, BULLET_H);
    ctx.restore();
  });

  bombs.forEach((b) => {
    ctx.save();
    ctx.shadowColor = "#39ff14";
    ctx.shadowBlur = 8;
    ctx.fillStyle = "#39ff14";
    ctx.fillRect(b.x, b.y, BOMB_W, BOMB_H);
    ctx.restore();
  });
}

//  Gambar partikel

function drawParticles() {
  particles.forEach((p) => {
    ctx.globalAlpha = p.life;
    ctx.fillStyle = p.color;
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.globalAlpha = 1;
}

function drawGroundLine() {
  ctx.strokeStyle = "rgba(255,255,255,0.15)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, H() - 8);
  ctx.lineTo(W(), H() - 8);
  ctx.stroke();
}

//  AI: Buat vektor input untuk satu alien (8 nilai, semuanya di rentang [0,1])

/**
 * Input yang dikirim ke NN setiap langkah keputusan:
 *  [0] Tengah-X pemain / lebar canvas
 *  [1] Tengah-Y pemain / tinggi canvas
 *  [2] Tengah-X alien  / lebar canvas
 *  [3] Tengah-Y alien  / tinggi canvas
 *  [4] (playerX − alienX) / W + 0.5  → 0.5 = kolom yang sama
 *  [5] (playerY − alienY) / H + 0.5  → 0.5 = baris yang sama  ← BARU
 *  [6] 1 jika ada peluru mendekat, 0 jika tidak
 *  [7] X / W peluru terdekat yang mengancam (0.5 jika tidak ada)
 */
function getAlienInputs(alien) {
  const px = (player.x + PLAYER_W / 2) / W();
  const py = (player.y + PLAYER_H / 2) / H();
  const ax = (alien.x + alien.w / 2) / W();
  const ay = (alien.y + alien.h / 2) / H();

  const deltaX = Math.max(0, Math.min(1, px - ax + 0.5));
  const deltaY = Math.max(0, Math.min(1, py - ay + 0.5)); // info vertikal tambahan

  // Deteksi peluru pemain yang mendekat (bergerak KE ATAS; ancaman = di bawah alien)
  let bulletNear = 0,
    bulletX = 0.5,
    minD = Infinity;
  for (const b of bullets) {
    const ddx = Math.abs(b.x + BULLET_W / 2 - (alien.x + alien.w / 2));
    const ddy = b.y - (alien.y + alien.h); // positif = peluru di bawah alien
    if (ddy > -BULLET_H && ddy < H() * 0.4 && ddx < alien.w * 3.5) {
      const d = Math.hypot(ddx, ddy);
      if (d < minD) {
        minD = d;
        bulletNear = 1;
        bulletX = (b.x + BULLET_W / 2) / W();
      }
    }
  }

  return [px, py, ax, ay, deltaX, deltaY, bulletNear, bulletX];
}

//  AI: Update Jaringan Saraf per alien (dipanggil setiap frame)

/**
 * Setiap alien memiliki referensi ke otak bersama sesuai tier-nya.
 * Setiap ALIEN_DECISION_FRAMES frame, alien akan:
 *   1. Melatih aksi SEBELUMNYA menggunakan reward yang terakumulasi
 *      (reward mendekat + reward delayed bom kena/meleset).
 *   2. Melakukan forward pass → memilih aksi dengan ε-greedy.
 *   3. Menerapkan aksi ke kecepatan / menjatuhkan bom.
 *
 * Reward mendekat (shaped): bernilai positif saat alien memperkecil
 * jarak horizontal ke pemain, negatif jika menjauh.
 * Ini memastikan otak belajar mengejar pemain bahkan sebelum
 * mendapat sinyal eksplisit dari bom yang kena.
 */
function updateAlienNN(alien) {
  if (!alien.alive || !alienBrains) return;
  const brain = alienBrains[alien.tier];

  //  1. Fisika
  alien.x += alien.vx;
  alien.y += alien.vy;

  // Batas dinding
  if (alien.x <= 0) {
    alien.x = 0;
    alien.vx = Math.abs(alien.vx) * 0.5;
  }
  if (alien.x + alien.w >= W()) {
    alien.x = W() - alien.w;
    alien.vx = -Math.abs(alien.vx) * 0.5;
  }
  // Batasan langit-langit
  if (alien.y < 8) {
    alien.y = 8;
    alien.vy = Math.abs(alien.vy) * 0.2;
  }

  if (alien.bombCooldown > 0) alien.bombCooldown--;

  //  2. Timer keputusan
  alien.decisionTimer--;
  if (alien.decisionTimer > 0) return; // belum waktunya mengambil keputusan
  alien.decisionTimer = ALIEN_DECISION_FRAMES;

  const inputs = getAlienInputs(alien);
  const distNow = Math.abs(player.x + PLAYER_W / 2 - (alien.x + alien.w / 2));

  //  3. Latih aksi SEBELUMNYA (delayed + shaped reward)
  if (alien.lastAction >= 0 && alien.lastSnapshot) {
    // Reward mendekat: positif jika jarak horizontal ke pemain berkurang
    const approach =
      alien.lastDist !== Infinity
        ? ((alien.lastDist - distNow) / W()) * 1.2 // dikalibrasi agar tidak terlalu dominan
        : 0;
    const reward = alien.pendingReward + approach;
    brain.trainFromSnapshot(alien.lastSnapshot, alien.lastAction, reward);
    alien.pendingReward = 0;
  }

  //  4. Forward pass + pemilihan ε-greedy
  const intel = brain.intelligence; // 0–100
  // Epsilon turun dari 80% ke 5% seiring bertambahnya kecerdasan
  const epsilon = Math.max(0.05, 0.8 - intel * 0.0075);
  const probs = brain.forward(inputs);

  const action =
    Math.random() < epsilon
      ? Math.floor(Math.random() * 4) // eksplorasi
      : probs.indexOf(Math.max(...probs)); // eksploitasi

  // Simpan untuk latihan langkah berikutnya
  alien.lastAction = action;
  alien.lastSnapshot = brain.saveSnapshot();
  alien.lastDist = distNow;

  //  5. Terapkan aksi
  // Kecepatan alien lebih lambat di awal, meningkat lebih perlahan per level
  const speed = 1.0 + level * 0.1 + alien.tier * 0.2; // sebelumnya: 1.2 + 0.15 + 0.25

  switch (action) {
    case ACT_LEFT:
      alien.vx = -speed;
      alien.vy *= 0.7;
      break;

    case ACT_RIGHT:
      alien.vx = speed;
      alien.vy *= 0.7;
      break;

    case ACT_BOMB:
      if (alien.bombCooldown <= 0) {
        // Ambil snapshot status otak saat bom dijatuhkan untuk pelatihan delayed
        const dropSnap = brain.saveSnapshot();
        bombs.push({
          x: alien.x + alien.w / 2 - BOMB_W / 2,
          y: alien.y + alien.h,
          tier: alien.tier,
          dropAction: ACT_BOMB,
          dropSnap,
          srcAlien: alien, // referensi untuk pendingReward saat kena
        });
        alien.bombCooldown = 80 + Math.floor(Math.random() * 80); // cooldown lebih lama
      }
      // Simultan memperkecil jarak horizontal saat menunggu bom
      {
        const signDx = Math.sign(
          player.x + PLAYER_W / 2 - (alien.x + alien.w / 2),
        );
        alien.vx = signDx * speed * 0.5;
        alien.vy = 0.25; // turun perlahan saat menjatuhkan bom
      }
      break;

    case ACT_DESCEND:
      if (inputs[6] > 0.5) {
        // Peluru mendekat! Hindari ke samping menjauhi kolom peluru
        alien.vx = inputs[7] < 0.5 ? speed : -speed;
        alien.vy *= 0.4;
      } else {
        // Tidak ada peluru: turun mendekati pemain
        alien.vy = Math.min(2.0, alien.vy + 0.35); // lebih lambat dari sebelumnya
        alien.vx *= 0.55;
      }
      break;
  }

  // Batasi agar kecepatan tidak tak terkendali
  const maxSpd = speed * 1.6;
  alien.vx = Math.max(-maxSpd, Math.min(maxSpd, alien.vx));
  alien.vy = Math.max(-0.5, Math.min(2.5, alien.vy)); // batas vy turun dari 3.0 → 2.5
}

//  Update Utama (dipanggil setiap frame)

function update(now) {
  frameCount++;

  //  Paralaks bintang
  stars.forEach((s) => {
    s.y += s.spd;
    if (s.y > H()) {
      s.y = 0;
      s.x = Math.random() * W();
    }
  });

  //  Pergerakan pemain
  const spd = player.speed + level * 0.2; // scaling lebih landai dari sebelumnya (0.3)
  if ((keys["ArrowLeft"] || keys["a"]) && player.x > 0)
    player.x = Math.max(0, player.x - spd);
  if ((keys["ArrowRight"] || keys["d"]) && player.x < W() - PLAYER_W)
    player.x = Math.min(W() - PLAYER_W, player.x + spd);

  //  Tembakan pemain
  if (
    (keys[" "] || keys["touchFire"]) &&
    now - player.lastShot > SHOOT_COOLDOWN
  ) {
    bullets.push({ x: player.x + PLAYER_W / 2 - BULLET_W / 2, y: player.y });
    player.lastShot = now;
    playLaserSound();
  }
  if (player.iframes > 0) player.iframes--;

  //  Peluru pemain (bergerak naik)
  for (let i = bullets.length - 1; i >= 0; i--) {
    bullets[i].y -= 10; // sedikit lebih cepat dari sebelumnya (9)
    if (bullets[i].y < -BULLET_H) bullets.splice(i, 1);
  }

  //  Update Jaringan Saraf Alien
  aliens.forEach((a) => updateAlienNN(a));

  //  Bom
  for (let i = bombs.length - 1; i >= 0; i--) {
    const b = bombs[i];
    // Kecepatan bom dikurangi dari 4.5 → 3.0, scaling per level juga lebih landai
    b.y += 3.0 + level * 0.2;

    // Meleset — keluar layar → reward negatif untuk menjatuhkan bom itu
    if (b.y > H()) {
      if (alienBrains && b.dropSnap)
        alienBrains[b.tier].trainFromSnapshot(b.dropSnap, b.dropAction, -0.6);
      bombs.splice(i, 1);
      continue;
    }

    // Kena pemain → reward positif yang besar!
    if (
      player.iframes === 0 &&
      b.x < player.x + PLAYER_W &&
      b.x + BOMB_W > player.x &&
      b.y < player.y + PLAYER_H &&
      b.y + BOMB_H > player.y
    ) {
      lives--;
      burst(player.x + PLAYER_W / 2, player.y + PLAYER_H / 2, "#ffffff", 20);
      player.iframes = 130; // invincibility lebih lama sedikit

      // Reward: bom berhasil mengenai pemain
      if (alienBrains && b.dropSnap)
        alienBrains[b.tier].trainFromSnapshot(b.dropSnap, b.dropAction, 2.5);
      // Tambahkan reward ke alien yang menjatuhkan bom (jika masih hidup)
      if (b.srcAlien && b.srcAlien.alive) b.srcAlien.pendingReward += 1.0;

      bombs.splice(i, 1);
      updateHUD();
      playHitSound();
    }
  }

  //  Tabrakan Peluru ↔ Alien
  for (let bi = bullets.length - 1; bi >= 0; bi--) {
    let hit = false;
    for (let ai = aliens.length - 1; ai >= 0; ai--) {
      const a = aliens[ai];
      if (!a.alive) continue;
      if (
        bullets[bi].x < a.x + a.w &&
        bullets[bi].x + BULLET_W > a.x &&
        bullets[bi].y < a.y + a.h &&
        bullets[bi].y + BULLET_H > a.y
      ) {
        a.alive = false;

        // Penalti: tertembak itu BURUK — otak harus menghindari kondisi ini
        if (alienBrains && a.lastSnapshot)
          alienBrains[a.tier].trainFromSnapshot(
            a.lastSnapshot,
            a.lastAction,
            -2.0,
          );

        const pts = [10, 25, 50][a.tier];
        score += pts;
        burst(
          a.x + a.w / 2,
          a.y + a.h / 2,
          ["#888888", "#aaaaaa", "#dddddd"][a.tier],
        );
        bullets.splice(bi, 1);
        hit = true;
        updateHUD();
        playExplosionSound();
        break;
      }
    }
    if (hit) continue;
  }

  //  Partikel
  for (let i = particles.length - 1; i >= 0; i--) {
    const p = particles[i];
    p.x += p.vx;
    p.y += p.vy;
    p.vy += 0.07;
    p.life -= p.decay;
    if (p.life <= 0) particles.splice(i, 1);
  }

  //  Alien menyentuh tanah atau menabrak pemain
  aliens.forEach((a) => {
    if (!a.alive) return;
    if (a.y + a.h >= H() - 10) {
      lives = 0;
      updateHUD();
    }
    if (
      player.iframes === 0 &&
      a.x < player.x + PLAYER_W &&
      a.x + a.w > player.x &&
      a.y < player.y + PLAYER_H &&
      a.y + a.h > player.y
    ) {
      lives--;
      burst(player.x + PLAYER_W / 2, player.y + PLAYER_H / 2, "#ffffff", 20);
      player.iframes = 130;
      updateHUD();
      playHitSound();
    }
  });

  //  Gelombang selesai → level berikutnya
  if (aliens.every((a) => !a.alive)) {
    level++;
    updateHUD();
    bombs = [];
    spawnWave(); // otak terus belajar dari status sebelumnya
  }

  //  Penyegaran HUD berkala untuk pembacaan kecerdasan AI
  if (frameCount % 30 === 0) updateHUD();
}

//  Loop Game

function loop(now) {
  if (!running) return;
  if (!paused) {
    update(now);
    if (lives <= 0) {
      endGame();
      return;
    }
  }

  drawBg();
  drawGroundLine();
  drawBullets();
  drawAliens();
  drawPlayer();
  drawParticles();

  if (paused) {
    ctx.fillStyle = "rgba(5,5,5,0.65)";
    ctx.fillRect(0, 0, W(), H());
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 28px Orbitron, sans-serif";
    ctx.textAlign = "center";
    ctx.shadowColor = "#ffffff";
    ctx.shadowBlur = 16;
    ctx.fillText("PAUSE", W() / 2, H() / 2);
    ctx.shadowBlur = 0;
    ctx.font = "12px Share Tech Mono, monospace";
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    ctx.fillText("Tekan P untuk lanjut", W() / 2, H() / 2 + 30);
  }

  raf = requestAnimationFrame(loop);
}

//  Akhiri Game

function endGame() {
  running = false;
  gameOverFlag = true;
  playGameOverSound();
  const ov = document.getElementById("overlay");
  document.getElementById("overlayTitle").textContent =
    lives <= 0 ? "GAME OVER" : "MENANG!";
  document.getElementById("overlayTitle").style.color =
    lives <= 0 ? "#ff3333" : "#ffffff";
  document.getElementById("overlayTitle").style.textShadow =
    lives <= 0 ? "0 0 20px #888888" : "0 0 20px #ffffff";
  document.getElementById("overlayMsg").textContent =
    `Kamu mencapai Level ${level}`;
  document.getElementById("finalScore").textContent = score;
  document.getElementById("finalScore").classList.remove("hidden");
  document.getElementById("startBtn").textContent = "MAIN LAGI";
  ov.classList.remove("hidden");
}

//  Input

document.addEventListener("keydown", (e) => {
  keys[e.key] = true;
  if (e.key === " ") e.preventDefault();
  if ((e.key === "p" || e.key === "P") && running) paused = !paused;
});
document.addEventListener("keyup", (e) => {
  keys[e.key] = false;
});

const touchMap = {
  leftBtn: () => {
    keys["ArrowLeft"] = true;
  },
  rightBtn: () => {
    keys["ArrowRight"] = true;
  },
  fireBtn: () => {
    keys["touchFire"] = true;
  },
};
const touchRelease = {
  leftBtn: () => {
    keys["ArrowLeft"] = false;
  },
  rightBtn: () => {
    keys["ArrowRight"] = false;
  },
  fireBtn: () => {
    keys["touchFire"] = false;
  },
};
Object.keys(touchMap).forEach((id) => {
  const el = document.getElementById(id);
  el.addEventListener(
    "touchstart",
    (e) => {
      e.preventDefault();
      touchMap[id]();
    },
    { passive: false },
  );
  el.addEventListener(
    "touchend",
    (e) => {
      e.preventDefault();
      touchRelease[id]();
    },
    { passive: false },
  );
  el.addEventListener("mousedown", () => touchMap[id]());
  el.addEventListener("mouseup", () => touchRelease[id]());
});

//  Tombol Mulai

document.getElementById("startBtn").addEventListener("click", () => {
  initAudio();
  document.getElementById("overlay").classList.add("hidden");
  document.getElementById("finalScore").classList.add("hidden");
  document.getElementById("startBtn").textContent = "MULAI";
  if (raf) cancelAnimationFrame(raf);
  running = true;
  init();
  requestAnimationFrame(loop);
});
