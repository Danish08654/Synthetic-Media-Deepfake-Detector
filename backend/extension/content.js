const API_URL = 'http://localhost:5000';

// Badge styles
const BADGE_STYLES = {
  real:      { bg: '#27ae60', text: '🟢 Real',         emoji: '✅' },
  fake:      { bg: '#e74c3c', text: '🔴 AI-Generated', emoji: '⚠️' },
  uncertain: { bg: '#f39c12', text: '🟡 Uncertain',    emoji: '❓' },
  loading:   { bg: 'rgb(52, 152, 219)', text: '⏳ Scanning...',  emoji: '🔍' }
};

function createBadge(style) {
  const badge       = document.createElement('div');
  badge.className   = 'deepfake-badge';
  badge.textContent = style.text;
  badge.style.cssText = `
    position: absolute;
    top: 4px;
    left: 4px;
    background: ${style.bg};
    color: white;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 4px;
    z-index: 9999;
    pointer-events: none;
    font-family: -apple-system, sans-serif;
    box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    transition: all 0.2s ease;
  `;
  return badge;
}

async function analyseImage(imgElement) {
  const src = imgElement.src || imgElement.currentSrc;

  // Skip tiny images, icons, data URIs
  if (!src || src.startsWith('data:') ||
      imgElement.naturalWidth < 100 ||
      imgElement.naturalHeight < 100) return;

  // Skip already processed
  if (imgElement.dataset.deepfakeChecked) return;
  imgElement.dataset.deepfakeChecked = 'true';

  // Wrap image in relative container
  const wrapper = document.createElement('div');
  wrapper.style.cssText = `
    position: relative;
    display: inline-block;
    width: ${imgElement.offsetWidth}px;
  `;
  imgElement.parentNode.insertBefore(wrapper, imgElement);
  wrapper.appendChild(imgElement);

  // Add loading badge
  const badge = createBadge(BADGE_STYLES.loading);
  wrapper.appendChild(badge);

  try {
    const resp   = await fetch(`${API_URL}/detect/url`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ url: src }),
      signal:  AbortSignal.timeout(8000)
    });

    const result = await resp.json();
    const score  = result.fake_probability;

    let style;
    if (score >= 0.65)       style = BADGE_STYLES.fake;
    else if (score <= 0.35)  style = BADGE_STYLES.real;
    else                     style = BADGE_STYLES.uncertain;

    badge.textContent   = style.text;
    badge.style.background = style.bg;
    badge.title = `Score: ${(score * 100).toFixed(1)}% fake | Confidence: ${(result.confidence * 100).toFixed(0)}%`;

  } catch (err) {
    badge.remove();
  }
}

// Scan all images on page
function scanImages() {
  const images = document.querySelectorAll('img');
  images.forEach(img => {
    if (img.complete) {
      analyseImage(img);
    } else {
      img.addEventListener('load', () => analyseImage(img));
    }
  });
}

// Initial scan
scanImages();

// Watch for dynamically loaded images
const observer = new MutationObserver(mutations => {
  mutations.forEach(mutation => {
    mutation.addedNodes.forEach(node => {
      if (node.tagName === 'IMG') analyseImage(node);
      if (node.querySelectorAll) {
        node.querySelectorAll('img').forEach(img => analyseImage(img));
      }
    });
  });
});

observer.observe(document.body, { childList: true, subtree: true });

// Listen for messages from popup
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.action === 'scan_page') {
    document.querySelectorAll('[data-deepfake-checked]').forEach(
      el => delete el.dataset.deepfakeChecked
    );
    scanImages();
    sendResponse({ status: 'scanning', count: document.querySelectorAll('img').length });
  }
});