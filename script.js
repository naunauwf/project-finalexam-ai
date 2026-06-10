//  Canvas setup 
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

//  Audio Synthesis using Web Audio API
let audioCtx = null;

function initAudio() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (audioCtx.state === "suspended") {
    audioCtx.resume();
  }
}

function playLaserSound() {
  if (!audioCtx) return;
  const osc = audioCtx.createOscillator();
  const gainNode = audioCtx.createGain();
  
  osc.type = "triangle";
  osc.frequency.setValueAtTime(900, audioCtx.currentTime);
  osc.frequency.exponentialRampToValueAtTime(150, audioCtx.currentTime + 0.12);
  
  gainNode.gain.setValueAtTime(0.1, audioCtx.currentTime);
  gainNode.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.12);
  
  osc.connect(gainNode);
  gainNode.connect(audioCtx.destination);
  
  osc.start();
  osc.stop(audioCtx.currentTime + 0.12);
}

function playExplosionSound() {
  if (!audioCtx) return;
  const osc = audioCtx.createOscillator();
  const gainNode = audioCtx.createGain();
  
  osc.type = "sawtooth";
  osc.frequency.setValueAtTime(120, audioCtx.currentTime);
  osc.frequency.exponentialRampToValueAtTime(10, audioCtx.currentTime + 0.2);
  
  gainNode.gain.setValueAtTime(0.2, audioCtx.currentTime);
  gainNode.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.2);
  
  osc.connect(gainNode);
  gainNode.connect(audioCtx.destination);
  
  osc.start();
  osc.stop(audioCtx.currentTime + 0.2);
}

function playHitSound() {
  if (!audioCtx) return;
  const osc = audioCtx.createOscillator();
  const gainNode = audioCtx.createGain();
  
  osc.type = "square";
  osc.frequency.setValueAtTime(250, audioCtx.currentTime);
  osc.frequency.setValueAtTime(100, audioCtx.currentTime + 0.08);
  
  gainNode.gain.setValueAtTime(0.2, audioCtx.currentTime);
  gainNode.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.2);
  
  osc.connect(gainNode);
  gainNode.connect(audioCtx.destination);
  
  osc.start();
  osc.stop(audioCtx.currentTime + 0.2);
}

function playGameOverSound() {
  if (!audioCtx) return;
  const now = audioCtx.currentTime;
  const notes = [180, 150, 120, 90];
  const duration = 0.18;
  
  notes.forEach((freq, index) => {
    const osc = audioCtx.createOscillator();
    const gainNode = audioCtx.createGain();
    
    osc.type = "sawtooth";
    osc.frequency.setValueAtTime(freq, now + index * duration);
    
    gainNode.gain.setValueAtTime(0.15, now + index * duration);
    gainNode.gain.exponentialRampToValueAtTime(0.01, now + index * duration + duration - 0.02);
    
    osc.connect(gainNode);
    gainNode.connect(audioCtx.destination);
    
    osc.start(now + index * duration);
    osc.stop(now + index * duration + duration);
  });
}

//  State 
let score, lives, level, running, paused, gameOverFlag, raf;
let player, bullets, aliens, bombs, particles, stars;
const keys = {};

//  Constants 
const PLAYER_W = 36,
  PLAYER_H = 28;
const BULLET_W = 3,
  BULLET_H = 14;
const BOMB_W = 4,
  BOMB_H = 10;
const SHOOT_COOLDOWN = 220; // ms

//  Init / reset 
function init() {
  score = 0;
  lives = 3;
  level = 1;
  paused = false;
  gameOverFlag = false;
  player = {
    x: W() / 2 - PLAYER_W / 2,
    y: H() - 56,
    speed: 4,
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
  spawnWave();
  updateHUD();
}

function spawnWave() {
  aliens = [];
  const cols = Math.min(8 + level, 14);
  const rows = Math.min(2 + Math.floor(level / 2), 5);
  const sx = (W() - cols * 44) / 2;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const tier = r < 2 ? 0 : r < 4 ? 1 : 2;
      aliens.push({
        x: sx + c * 44,
        y: 28 + r * 38,
        w: 28,
        h: 22,
        tier,
        vx: 0.6 + level * 0.18,
        vy: 0,
        alive: true,
        bombTimer: 180 + Math.floor(Math.random() * 300),
      });
    }
  }
}

//  HUD 
function updateHUD() {
  document.getElementById("scoreVal").textContent = score;
  document.getElementById("livesVal").textContent =
    "♥".repeat(lives) + "♡".repeat(Math.max(0, 3 - lives));
  document.getElementById("levelVal").textContent = level;
}

//  Particles 
function burst(x, y, color, n = 14) {
  for (let i = 0; i < n; i++) {
    const a = Math.random() * Math.PI * 2;
    const s = 1 + Math.random() * 4;
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

//  Draw helpers 
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
  if (p.iframes > 0 && Math.floor(Date.now() / 80) % 2) return; // blink when hit
  ctx.save();
  ctx.translate(p.x + PLAYER_W / 2, p.y + PLAYER_H / 2);

  // engine glow
  ctx.shadowColor = "#ffffff";
  ctx.shadowBlur = 14;

  // body
  ctx.fillStyle = "#333333";
  ctx.beginPath();
  ctx.moveTo(0, -PLAYER_H / 2);
  ctx.lineTo(PLAYER_W / 2, PLAYER_H / 2);
  ctx.lineTo(-PLAYER_W / 2, PLAYER_H / 2);
  ctx.closePath();
  ctx.fill();

  // cockpit
  ctx.fillStyle = "#ffffff";
  ctx.beginPath();
  ctx.ellipse(0, -4, 7, 9, 0, 0, Math.PI * 2);
  ctx.fill();

  // wings accent
  ctx.strokeStyle = "#888888";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(-PLAYER_W / 2, PLAYER_H / 2);
  ctx.lineTo(-8, 2);
  ctx.moveTo(PLAYER_W / 2, PLAYER_H / 2);
  ctx.lineTo(8, 2);
  ctx.stroke();

  // thruster
  if (Math.random() > 0.4) {
    ctx.fillStyle = `rgba(${(200 + Math.random() * 55) | 0}, ${(200 + Math.random() * 55) | 0}, ${(200 + Math.random() * 55) | 0}, 0.9)`;
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

// Alien emoji-style sprites per tier
const ALIEN_SHAPES = [
  (c, w, h) => {
    // tier 0 — squid
    c.fillStyle = "#888888";
    c.beginPath();
    c.arc(0, -2, w * 0.38, 0, Math.PI * 2);
    c.fill();
    c.fillStyle = "#ffffff";
    // eyes
    c.beginPath();
    c.arc(-w * 0.16, -4, 3, 0, Math.PI * 2);
    c.fill();
    c.beginPath();
    c.arc(w * 0.16, -4, 3, 0, Math.PI * 2);
    c.fill();
    // tentacles
    c.strokeStyle = "#888888";
    c.lineWidth = 2;
    for (let t = -1; t <= 1; t += 1) {
      c.beginPath();
      c.moveTo(t * 5, w * 0.3);
      c.lineTo(t * 8, w * 0.55);
      c.stroke();
    }
  },
  (c, w, h) => {
    // tier 1 — crab
    c.fillStyle = "#aaaaaa";
    c.fillRect(-w * 0.4, -h * 0.3, w * 0.8, h * 0.6);
    c.fillStyle = "#ffffff";
    c.beginPath();
    c.arc(-w * 0.4, -h * 0.1, 5, 0, Math.PI * 2);
    c.fill();
    c.beginPath();
    c.arc(w * 0.4, -h * 0.1, 5, 0, Math.PI * 2);
    c.fill();
    c.fillStyle = "#cccccc";
    c.beginPath();
    c.arc(-w * 0.2, -h * 0.1, 3, 0, Math.PI * 2);
    c.fill();
    c.beginPath();
    c.arc(w * 0.2, -h * 0.1, 3, 0, Math.PI * 2);
    c.fill();
  },
  (c, w, h) => {
    // tier 2 — UFO
    c.fillStyle = "#dddddd";
    c.beginPath();
    c.ellipse(0, 0, w * 0.48, h * 0.28, 0, 0, Math.PI * 2);
    c.fill();
    c.fillStyle = "#ffffff";
    c.beginPath();
    c.arc(0, -h * 0.18, w * 0.22, 0, Math.PI * 2);
    c.fill();
    c.strokeStyle = "rgba(255,255,255,0.4)";
    c.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
      c.beginPath();
      c.arc(0, 0, w * (0.18 + i * 0.06), 0, Math.PI * 2);
      c.stroke();
    }
  },
];

function drawAliens() {
  const COLORS = ["#888888", "#aaaaaa", "#dddddd"];
  aliens.forEach((a) => {
    if (!a.alive) return;
    ctx.save();
    ctx.translate(a.x + a.w / 2, a.y + a.h / 2);
    ctx.shadowColor = COLORS[a.tier];
    ctx.shadowBlur = 8;
    ALIEN_SHAPES[a.tier](ctx, a.w, a.h);
    ctx.shadowBlur = 0;
    ctx.restore();
  });
}

function drawBullets() {
  bullets.forEach((b) => {
    ctx.save();
    ctx.shadowColor = "#ffffff";
    ctx.shadowBlur = 10;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(b.x, b.y, BULLET_W, BULLET_H);
    ctx.restore();
  });
  bombs.forEach((b) => {
    ctx.save();
    ctx.shadowColor = "#888888";
    ctx.shadowBlur = 8;
    ctx.fillStyle = "#888888";
    ctx.fillRect(b.x, b.y, BOMB_W, BOMB_H);
    ctx.restore();
  });
}

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

//  Update 
function update(now) {
  // stars parallax
  stars.forEach((s) => {
    s.y += s.spd;
    if (s.y > H()) {
      s.y = 0;
      s.x = Math.random() * W();
    }
  });

  // player movement
  const spd = player.speed + level * 0.3;
  if ((keys["ArrowLeft"] || keys["a"]) && player.x > 0)
    player.x = Math.max(0, player.x - spd);
  if ((keys["ArrowRight"] || keys["d"]) && player.x < W() - PLAYER_W)
    player.x = Math.min(W() - PLAYER_W, player.x + spd);

  // shoot
  if (
    (keys[" "] || keys["touchFire"]) &&
    now - player.lastShot > SHOOT_COOLDOWN
  ) {
    bullets.push({
      x: player.x + PLAYER_W / 2 - BULLET_W / 2,
      y: player.y,
    });
    player.lastShot = now;
    playLaserSound();
  }

  if (player.iframes > 0) player.iframes--;

  // bullets
  for (let i = bullets.length - 1; i >= 0; i--) {
    bullets[i].y -= 9;
    if (bullets[i].y < -BULLET_H) bullets.splice(i, 1);
  }

  // aliens march
  let hitEdge = false;
  aliens.forEach((a) => {
    if (!a.alive) return;
    a.x += a.vx;
    if (a.x + a.w >= W() || a.x <= 0) hitEdge = true;
  });
  if (hitEdge) {
    aliens.forEach((a) => {
      if (!a.alive) return;
      a.vx *= -1;
      a.y += 14;
    });
  }

  // alien bombs
  aliens.forEach((a) => {
    if (!a.alive) return;
    a.bombTimer--;
    if (a.bombTimer <= 0) {
      bombs.push({ x: a.x + a.w / 2 - BOMB_W / 2, y: a.y + a.h });
      a.bombTimer = 120 + Math.floor(Math.random() * (220 - level * 10));
    }
  });
  for (let i = bombs.length - 1; i >= 0; i--) {
    bombs[i].y += 5 + level * 0.4;
    if (bombs[i].y > H()) {
      bombs.splice(i, 1);
      continue;
    }
    // hit player
    if (
      player.iframes === 0 &&
      bombs[i].x < player.x + PLAYER_W &&
      bombs[i].x + BOMB_W > player.x &&
      bombs[i].y < player.y + PLAYER_H &&
      bombs[i].y + BOMB_H > player.y
    ) {
      lives--;
      burst(player.x + PLAYER_W / 2, player.y + PLAYER_H / 2, "#ffffff", 20);
      player.iframes = 120;
      bombs.splice(i, 1);
      updateHUD();
      playHitSound();
    }
  }

  // bullet-alien collision
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

  // particles
  for (let i = particles.length - 1; i >= 0; i--) {
    const p = particles[i];
    p.x += p.vx;
    p.y += p.vy;
    p.vy += 0.07;
    p.life -= p.decay;
    if (p.life <= 0) particles.splice(i, 1);
  }

  // alien touches ground or player
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
      player.iframes = 120;
      updateHUD();
      playHitSound();
    }
  });

  // wave clear
  if (aliens.every((a) => !a.alive)) {
    level++;
    updateHUD();
    bombs = [];
    spawnWave();
    // level-up flash message handled via canvas
  }
}

//  Game loop 
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

function endGame() {
  running = false;
  gameOverFlag = true;
  playGameOverSound();
  const ov = document.getElementById("overlay");
  document.getElementById("overlayTitle").textContent =
    lives <= 0 ? "GAME OVER" : "MENANG!";
  document.getElementById("overlayTitle").style.color =
    lives <= 0 ? "#888888" : "#ffffff";
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

// Touch buttons
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

//  Start button 
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
