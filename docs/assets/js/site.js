'use strict';

// Resolve media relative to this script so the site works from both published entry points.
const assetsRoot = new URL('../', document.currentScript.src);

const videos = [...document.querySelectorAll('video')];
// Playback is always initiated by the visitor. Stop other players to avoid competing audio.
videos.forEach(video => video.addEventListener('play', () => {
  videos.forEach(other => { if (other !== video) other.pause(); });
}));

// Each replay contains camera and curve in the same encoded frames, sharing all controls.
const monitoringLayout = matchMedia('(max-width: 780px)');
const pendingRestores = new WeakMap();
function setMonitoringSource(video, trial, preservePlayback = false) {
  const layout = monitoringLayout.matches ? 'stacked' : 'wide';
  if (video.dataset.trial === trial && video.dataset.layout === layout) return;
  const time = preservePlayback ? video.currentTime : 0;
  const resume = preservePlayback && !video.paused;
  const rate = video.playbackRate;
  video.pause();
  const previous = pendingRestores.get(video);
  if (previous) video.removeEventListener('loadedmetadata', previous);
  video.dataset.trial = trial;
  video.dataset.layout = layout;
  video.poster = new URL(`monitoring/${trial}-${layout}.jpg`, assetsRoot).href;
  video.querySelector('source').src = new URL(`monitoring/${trial}-${layout}.mp4`, assetsRoot).href;
  const restore = () => {
    pendingRestores.delete(video);
    video.playbackRate = rate;
    if (time > 0) video.currentTime = Math.min(time, video.duration);
    if (resume) video.play().catch(() => {});
  };
  pendingRestores.set(video, restore);
  video.addEventListener('loadedmetadata', restore, { once: true });
  video.load();
}
document.querySelectorAll('.monitor-video').forEach(video => {
  setMonitoringSource(video, video.dataset.trial);
  const player = video.closest('.monitoring-player');
  const controls = player.querySelector('.replay-controls');
  const toggle = player.querySelector('.replay-toggle');
  const seek = player.querySelector('.replay-seek');
  const time = player.querySelector('.replay-time');
  const speed = player.querySelector('.replay-speed');
  const fullscreen = player.querySelector('.replay-fullscreen');
  const status = player.querySelector('.replay-status');
  const formatTime = seconds => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
  const updateControls = () => {
    const duration = Number.isFinite(video.duration) ? video.duration : 0;
    seek.max = duration || 1;
    seek.disabled = !duration;
    seek.value = video.currentTime;
    seek.setAttribute('aria-valuetext', `${formatTime(video.currentTime)} of ${formatTime(duration)}`);
    time.textContent = `${formatTime(video.currentTime)} / ${formatTime(duration)}`;
    toggle.textContent = video.paused ? 'Play' : 'Pause';
    toggle.setAttribute('aria-label', video.paused ? 'Play monitoring replay' : 'Pause monitoring replay');
    speed.value = String(video.playbackRate);
  };
  toggle.addEventListener('click', () => {
    if (!video.paused) video.pause();
    else video.play().catch(() => {
      status.textContent = 'Unable to play this replay. Please try again.';
      status.hidden = false;
    });
  });
  seek.addEventListener('input', () => { video.currentTime = Number(seek.value); });
  speed.addEventListener('change', () => { video.playbackRate = Number(speed.value); });
  fullscreen.hidden = !player.requestFullscreen;
  fullscreen.addEventListener('click', () => {
    const action = document.fullscreenElement === player ? document.exitFullscreen() : player.requestFullscreen();
    action.catch(() => {});
  });
  document.addEventListener('fullscreenchange', () => {
    fullscreen.textContent = document.fullscreenElement === player ? 'Exit fullscreen' : 'Fullscreen';
  });
  ['loadedmetadata', 'durationchange', 'timeupdate', 'play', 'pause', 'ended', 'ratechange', 'emptied'].forEach(event => {
    video.addEventListener(event, updateControls);
  });
  video.addEventListener('playing', () => { status.hidden = true; });
  video.addEventListener('error', () => {
    status.textContent = 'Unable to load this replay. Please refresh to try again.';
    status.hidden = false;
  });
  video.controls = false;
  controls.hidden = false;
  updateControls();
});
monitoringLayout.addEventListener('change', () => {
  document.querySelectorAll('.monitor-video').forEach(video => {
    setMonitoringSource(video, video.dataset.trial, true);
  });
});

const overview = document.querySelector('#overview-player');
document.querySelectorAll('[data-seek]').forEach(button => {
  button.addEventListener('click', () => {
    const seek = () => {
      overview.currentTime = Number(button.dataset.seek);
      overview.play().catch(() => {});
    };
    if (overview.readyState >= 1) seek();
    else {
      overview.addEventListener('loadedmetadata', seek, { once: true });
      overview.load();
    }
    overview.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'center' });
    overview.focus({ preventScroll: true });
  });
});

document.querySelectorAll('[data-filter]').forEach(button => {
  button.addEventListener('click', () => {
    document.querySelectorAll('[data-filter]').forEach(item => {
      const active = item === button;
      item.classList.toggle('is-active', active);
      item.setAttribute('aria-pressed', String(active));
    });
    document.querySelectorAll('.experiment-card').forEach(card => {
      card.hidden = button.dataset.filter !== 'all' && card.dataset.task !== button.dataset.filter;
      if (card.hidden) card.querySelector('video').pause();
    });
  });
});

document.querySelectorAll('[data-clip]').forEach(button => {
  button.addEventListener('click', () => {
    if (button.getAttribute('aria-pressed') === 'true') return;
    const card = button.closest('.experiment-card');
    const video = card.querySelector('.monitor-video');
    setMonitoringSource(video, button.dataset.clip);
    const curveUrl = new URL(`curves/${button.dataset.clip}.png`, assetsRoot).href;
    card.querySelector('.curve-open').href = curveUrl;
    card.querySelectorAll('[data-clip]').forEach(item => {
      const active = item === button;
      item.classList.toggle('is-active', active);
      item.setAttribute('aria-pressed', String(active));
    });
  });
});

const slider = document.querySelector('#lambda-slider');
const distribution = document.querySelector('#distribution');
const bars = Array.from({ length: 20 }, () => {
  const bar = document.createElement('span');
  bar.setAttribute('aria-hidden', 'true');
  distribution.append(bar);
  return bar;
});
function updateHorizon() {
  const lambda = Number(slider.value) / 100;
  const expected = 1 / (1 - lambda);
  const mean = expected >= 10 ? expected.toFixed(0) : expected.toFixed(1).replace(/\.0$/, '');
  document.querySelector('#lambda-value').textContent = `λ = ${lambda.toFixed(2)}`;
  document.querySelector('#horizon-value').textContent = mean;
  slider.setAttribute('aria-valuetext', `lambda ${lambda.toFixed(2)}, mean horizon ${mean} steps`);
  const masses = bars.map((_, i) => lambda ** (5 * i) - lambda ** (5 * (i + 1)));
  const maxMass = Math.max(...masses);
  const scale = [0.05, 0.1, 0.25, 0.5, 1].find(limit => limit >= maxMass);
  const tail = lambda ** 100;
  document.querySelector('#distribution-scale').textContent = `Scale: 0–${Math.round(scale * 100)}%`;
  document.querySelector('#tail-probability').textContent = tail > 0 && tail < 0.001 ? '<0.1%' : `${(tail * 100).toFixed(1)}%`;
  bars.forEach((bar, i) => {
    bar.style.height = `${masses[i] / scale * 100}%`;
    bar.title = `${i * 5 + 1}–${i * 5 + 5} steps: ${(masses[i] * 100).toFixed(1)}% probability`;
  });
  distribution.setAttribute('aria-label', `Geometric horizon probabilities at lambda ${lambda.toFixed(2)} in equal five-step groups from 1 to 100 steps. Vertical scale 0 to ${Math.round(scale * 100)} percent. ${(tail * 100).toFixed(1)} percent of probability lies beyond 100 steps and is reported separately.`);
}
slider.addEventListener('input', updateHorizon);
updateHorizon();

document.querySelector('#copy-citation').addEventListener('click', async () => {
  const text = document.querySelector('#citation').innerText;
  const status = document.querySelector('#copy-status');
  try {
    await navigator.clipboard.writeText(text);
    status.textContent = 'Citation copied to clipboard.';
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(document.querySelector('#citation'));
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = 'Citation selected. Press Ctrl+C or ⌘C to copy.';
  }
});

const links = [...document.querySelectorAll('.site-header nav a')];
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      links.forEach(link => {
        const active = link.hash === `#${entry.target.id}`;
        link.classList.toggle('is-active', active);
        if (active) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      });
    }
  }, { rootMargin: '-15% 0px -70% 0px', threshold: 0 });
  links.forEach(link => {
    const section = document.querySelector(link.hash);
    if (section) observer.observe(section);
  });
}
