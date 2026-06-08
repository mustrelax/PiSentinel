import { Device } from "https://esm.sh/mediasoup-client@3";
import { Minimap } from "./minimap.js";

const socket = window.io();

// UI Elements
const gridContainer = document.getElementById('video-grid');
const gridSelect = document.getElementById('grid-select');
const statusInd = document.getElementById('connection-status');
const reconnectBtn = document.getElementById('reconnect-btn');
const logArea = document.getElementById('log-area');

// State
let device;
let consumerTransport;
let transportReady = false;

// We store active producer IDs in an ordered list to maintain slot order.
// When a new one joins, push. When one leaves, splice.
let producerIds = [];
let consumers = new Map(); // producerId -> { consumer, track, stream, deviceName }

const DEVICE_BADGE_PALETTE = [
    '#3a6ea5', '#a53a6e', '#3aa56e', '#a5853a',
    '#6e3aa5', '#3aa5a5', '#a55c3a', '#5c3aa5',
];

function colorForDevice(deviceName) {
    let hash = 0;
    const s = String(deviceName || '');
    for (let i = 0; i < s.length; i++) hash = (hash * 31 + s.charCodeAt(i)) | 0;
    return DEVICE_BADGE_PALETTE[Math.abs(hash) % DEVICE_BADGE_PALETTE.length];
}

// Logging
const log = (msg) => {
    console.log(msg);
    const p = document.createElement('p');
    p.textContent = `> ${msg}`;
    logArea.appendChild(p);
    logArea.scrollTop = logArea.scrollHeight;
};

const viewerCountEl = document.getElementById('viewer-count');
const modalOverlay = document.getElementById('modal-overlay');
const maxVideo = document.getElementById('max-video');
const closeModalBtn = document.getElementById('close-modal');

// --- Initialization ---

// Auto-start on load
init();

gridSelect.onchange = renderGrid;
reconnectBtn.onclick = () => window.location.reload();
closeModalBtn.onclick = closeMaximizedView;

async function init() {
    log('Connecting to server...');

    // Wait for socket connect
    if (!socket.connected) {
        await new Promise(r => socket.once('connect', r));
    }

    // Identify as viewer (not producer)
    socket.emit('identify', { role: 'viewer' });

    statusInd.textContent = 'Connected';
    statusInd.classList.add('connected');

    // Render grid immediately so user sees placeholders
    renderGrid();

    try {
        await setupMediasoup();
        // After setup, fetch existing
        fetchProducers();
        fetchAlertHistory();
    } catch (e) {
        log('Setup failed: ' + e);
        statusInd.textContent = 'Error';
    }
}

async function setupMediasoup() {
    // 1. Get Caps
    const caps = await new Promise(resolve => socket.emit('getRouterRtpCapabilities', resolve));
    device = new Device();
    await device.load({ routerRtpCapabilities: caps });

    // 2. Create Recv Transport
    const transParams = await new Promise((resolve, reject) => {
        socket.emit('createTransport', { sender: false }, (resp) => {
            if (resp.error) reject(resp.error); else resolve(resp);
        });
    });

    consumerTransport = device.createRecvTransport(transParams);

    consumerTransport.on('connect', ({ dtlsParameters }, cb, errback) => {
        socket.emit('connectTransport', { dtlsParameters }, (r) => {
            if (r.error) errback(r.error); else cb();
        });
    });

    consumerTransport.on('connectionstatechange', (state) => {
        log(`Transport State: ${state}`);
        if (state === 'failed') transportReady = false;
    });

    transportReady = true;
    log('Mediasoup Ready.');
}

function fetchProducers() {
    socket.emit('getProducers', (list) => {
        log(`Existing producers: ${list.length}`);
        list.forEach(p => handleNewProducer(p.producerId, p.deviceName));
    });
}

// --- Logic ---

function handleNewProducer(id, name) {
    if (producerIds.includes(id)) return;
    log(`New Stream: ${name}`);
    producerIds.push(id);
    consumeStream(id, name); // Start consuming immediately
    renderGrid(); // Update slots
}

function handleProducerClosed(id) {
    log(`Stream Closed: ${id}`);
    const idx = producerIds.indexOf(id);
    if (idx > -1) {
        producerIds.splice(idx, 1);
    }

    // Cleanup consumer and drop unit from minimap
    if (consumers.has(id)) {
        const c = consumers.get(id);
        const deviceName = c.deviceName;
        c.consumer.close();
        consumers.delete(id);
        if (deviceName && minimap) minimap.onUnitDisconnected(deviceName);
    }

    renderGrid(); // Slots will shift left
}

async function consumeStream(producerId, deviceName) {
    log(`[CONSUME] Starting for ${deviceName} (${producerId})`);

    if (!transportReady) {
        log('[CONSUME] Transport not ready!');
        return;
    }

    try {
        const { rtpCapabilities } = device;
        log('[CONSUME] Sending consume request...');

        const data = await new Promise((resolve, reject) => {
            socket.emit('consume', { producerId, rtpCapabilities }, (response) => {
                if (!response) {
                    reject(new Error('No response from server'));
                } else {
                    resolve(response);
                }
            });
            // Timeout after 5 seconds
            setTimeout(() => reject(new Error('Consume timeout')), 5000);
        });

        log(`[CONSUME] Response: ${JSON.stringify(data)}`);

        if (data.error) {
            log(`[CONSUME] Failed: ${data.error}`);
            return;
        }

        const { id, kind, rtpParameters } = data;
        log(`[CONSUME] Creating consumer... kind=${kind}`);

        const consumer = await consumerTransport.consume({
            id, producerId, kind, rtpParameters
        });

        log(`[CONSUME] Consumer created, track: ${consumer.track?.kind}`);

        // CRITICAL: Resume consumer to start receiving frames
        await consumer.resume();
        log('[CONSUME] Client consumer resumed');

        // Notify Server to Resume (This triggers Keyframe Request!)
        await new Promise(resolve => {
            socket.emit('resumeConsumer', { consumerId: id }, resolve);
        });
        log('[CONSUME] Server consumer resumed');

        const stream = new MediaStream([consumer.track]);

        log(`[CONSUME] Stream created with ${stream.getTracks().length} track(s)`);

        consumers.set(producerId, {
            consumer,
            stream,
            deviceName
        });

        log('[CONSUME] Success! Rendering grid...');
        renderGrid();

    } catch (e) {
        log(`[CONSUME] Error: ${e.message}`);
        console.error(e);
    }
}

// --- Rendering ---
// This is the key "Slot Logic".
// We always render N slots (fixed).
// We fill them with active streams in order.
function renderGrid() {
    const gridSize = parseInt(gridSelect.value);
    const totalSlots = gridSize * gridSize;

    // Update Grid CSS
    gridContainer.className = `video-grid grid-${gridSize}x${gridSize}`;
    gridContainer.innerHTML = '';

    for (let i = 0; i < totalSlots; i++) {
        const slot = document.createElement('div');
        slot.className = 'video-slot';

        // Is there a stream for this slot?
        if (i < producerIds.length) {
            // Yes, fill it
            const pid = producerIds[i];
            const data = consumers.get(pid);

            if (data) {
                slot.classList.add('active');
                if (data.deviceName) slot.setAttribute('data-device', data.deviceName);

                const video = document.createElement('video');
                video.autoplay = true;
                video.playsInline = true;
                video.muted = true;
                video.srcObject = data.stream;

                // Low latency optimizations
                video.style.transform = 'translateZ(0)';  // GPU acceleration
                video.controls = false;
                video.disablePictureInPicture = true;
                if ('latencyHint' in video) video.latencyHint = 'interactive';

                // Explicitly try to play
                video.play().catch(e => console.warn('Video play failed:', e));

                const label = document.createElement('div');
                label.className = 'overlay-name';
                label.textContent = data.deviceName || `Cam-${i + 1}`;

                slot.appendChild(video);
                slot.appendChild(label);

                // Health info button (top-right of tile). Hover to see Sensor
                // Health for this Pi. Populated by the unitHealth handler.
                const infoWrap = document.createElement('div');
                infoWrap.className = 'health-info-wrap';
                infoWrap.innerHTML = `
                    <button class="health-info-btn" title="Sensor health" aria-label="Sensor health">i</button>
                    <span class="health-pill" data-pill="">—</span>
                    <div class="health-popover">
                        <div class="health-popover-title">
                            <span class="health-popover-name"></span>
                            <span class="health-pill" data-pill=""></span>
                        </div>
                        <div class="health-popover-body">no telemetry yet</div>
                    </div>`;
                // Stop clicks on the info chip from triggering the slot's
                // maximize handler — let the user explore the popover.
                infoWrap.addEventListener('click', (e) => e.stopPropagation());
                slot.appendChild(infoWrap);

                // Click to maximize
                slot.onclick = () => openMaximizedView(data.stream);

                // Paint the popover with whatever we already know about this Pi.
                renderHealthForDevice(data.deviceName, slot);
            } else {
                // We have the ID but maybe consumer isn't ready yet?
                // It will rerender when consume finishes.
                slot.classList.add('loading');
                slot.textContent = "Loading...";
            }
        } else {
            // No stream, empty slot
            slot.classList.add('empty');
        }

        gridContainer.appendChild(slot);
    }
}

// --- Maximize Logic ---
function openMaximizedView(stream) {
    maxVideo.srcObject = stream;
    maxVideo.style.transform = 'translateZ(0)';  // GPU
    modalOverlay.classList.add('open');
}

function closeMaximizedView() {
    modalOverlay.classList.remove('open');
    maxVideo.srcObject = null;
}

// --- Socket Events ---
socket.on('newProducer', ({ producerId, deviceName }) => {
    handleNewProducer(producerId, deviceName);
});

socket.on('producer-closed', ({ producerId }) => {
    handleProducerClosed(producerId);
});

socket.on('stats', (data) => {
    viewerCountEl.textContent = data.viewers;
});

// --- Drone Alert Handler ---
const alertList = document.getElementById('alert-list');
let alertSound = null;

// Track alert states: "deviceName_trackId" -> { status, element }
const alertStates = new Map();
const lockedAlerts = new Map(); // Track currently locked trackId per device
const activeRecordings = new Map(); // Track active MediaRecorders per device

// Start recording for a device's video stream
function startRecordingForDevice(deviceName, trackId) {
    // Find the video element for this device
    const videoElement = findVideoElementForDevice(deviceName);
    if (!videoElement || !videoElement.srcObject) {
        console.log(`[RECORD] No video stream for ${deviceName}`);
        return;
    }

    // Don't start if already recording
    if (activeRecordings.has(deviceName)) {
        console.log(`[RECORD] Already recording ${deviceName}`);
        return;
    }

    try {
        const stream = videoElement.srcObject;
        const recorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
        const chunks = [];

        recorder.ondataavailable = (e) => {
            if (e.data.size > 0) chunks.push(e.data);
        };

        recorder.onstop = () => {
            const blob = new Blob(chunks, { type: 'video/webm' });
            uploadRecording(blob, deviceName, trackId);
        };

        recorder.start(1000); // Collect data every second
        activeRecordings.set(deviceName, { recorder, trackId, startTime: Date.now() });

        // Add recording indicator to video slot
        const slot = findVideoSlotForDevice(deviceName);
        if (slot) slot.classList.add('recording');

        console.log(`[RECORD] Started recording ${deviceName} track #${trackId}`);
    } catch (err) {
        console.error(`[RECORD] Failed to start: ${err}`);
    }
}

// Stop recording for a device
function stopRecordingForDevice(deviceName, trackId) {
    const recording = activeRecordings.get(deviceName);
    if (!recording) return;

    // Only stop if it's the same track
    if (recording.trackId !== trackId) return;

    try {
        recording.recorder.stop();
        activeRecordings.delete(deviceName);

        // Remove recording indicator
        const slot = findVideoSlotForDevice(deviceName);
        if (slot) slot.classList.remove('recording');

        console.log(`[RECORD] Stopped recording ${deviceName} track #${trackId}`);
    } catch (err) {
        console.error(`[RECORD] Failed to stop: ${err}`);
    }
}

// Upload recorded video to server
function uploadRecording(blob, deviceName, trackId) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `${deviceName}_track${trackId}_${timestamp}.webm`;

    const formData = new FormData();
    formData.append('video', blob, filename);

    fetch('/upload-video', { method: 'POST', body: formData })
        .then(res => res.json())
        .then(data => {
            console.log(`[RECORD] Uploaded: ${filename}`);
            // Update track log with video path
            socket.emit('updateTrackLogVideo', {
                deviceName,
                trackId,
                video_path: `/vlogs/${filename}`
            });
        })
        .catch(err => console.error(`[RECORD] Upload failed: ${err}`));
}

// Find video element for a device
function findVideoElementForDevice(deviceName) {
    const slots = document.querySelectorAll('.video-slot');
    for (const slot of slots) {
        const label = slot.querySelector('.overlay-name');
        if (label && label.textContent === deviceName) {
            return slot.querySelector('video');
        }
    }
    return null;
}

// Find video slot for a device
function findVideoSlotForDevice(deviceName) {
    const slots = document.querySelectorAll('.video-slot');
    for (const slot of slots) {
        const label = slot.querySelector('.overlay-name');
        if (label && label.textContent === deviceName) {
            return slot;
        }
    }
    return null;
}

socket.on('droneAlert', (data) => {
    handleAlert(data, true);
});

function fetchAlertHistory() {
    socket.emit('getAlertHistory', (history) => {
        // Render history (reverse so newest ends up on top via insertBefore)
        if (history && history.length) {
            [...history].reverse().forEach(alert => {
                handleAlert(alert, false);
            });
        }
    });
}

function handleAlert(data, playSound = true) {
    if (playSound) {
        log(`[ALERT] Drone detected by ${data.deviceName}`);
        playAlertSound();
    }

    // Unique key for this track
    const alertKey = `${data.deviceName}_${data.trackId}`;

    // Add to alert list
    const alertItem = document.createElement('div');
    alertItem.className = 'alert-item';
    alertItem.dataset.deviceName = data.deviceName;
    alertItem.dataset.trackId = data.trackId;

    const time = new Date(data.timestamp).toLocaleTimeString('tr-TR');

    // Default status if not provided
    const status = data.status || 'PASSIVE';
    const badgeClass = status.toLowerCase();

    const deviceColor = colorForDevice(data.deviceName);
    alertItem.innerHTML = `
        <span class="alert-device-badge" style="background:${deviceColor}">${data.deviceName}</span>
        <span class="alert-badge ${badgeClass}">${status}</span>
        <br>
        <span class="alert-time">${time} - Track #${data.trackId}</span>
    `;

    // Store reference by unique key
    alertStates.set(alertKey, {
        status: status,
        element: alertItem
    });

    // Add to list (will be reordered immediately after)
    alertList.appendChild(alertItem);

    // Limit to 50 alerts in UI
    while (alertList.children.length > 50) {
        alertList.removeChild(alertList.lastChild);
    }

    // Reorder alerts by status (LOCKED at top, PASSIVE next, LOST at bottom)
    reorderAlertsByStatus();

    // Update video slot border (show only for LOCKED - active tracking)
    if (status === 'LOCKED') {
        updateActiveVideoSlot(data.deviceName, true);
    }
}

// Handle tracking status updates
socket.on('trackingStatus', (data) => {
    const alertKey = `${data.deviceName}_${data.trackId}`;
    const state = alertStates.get(alertKey);

    if (state && state.element) {
        // Update badge
        const badge = state.element.querySelector('.alert-badge');
        if (badge) {
            badge.className = `alert-badge ${data.status.toLowerCase()}`;
            badge.textContent = data.status;
        }
        state.status = data.status;

        // Reorder alerts: LOCKED at top, PASSIVE below, LOST at bottom
        reorderAlertsByStatus();
    }

    // Update video slot border 
    if (data.status === 'LOCKED') {
        // Track which alert is currently locked for this device
        lockedAlerts.set(data.deviceName, data.trackId);
        updateActiveVideoSlot(data.deviceName, true);
    } else if (data.status === 'LOST') {
        // Only remove border if THIS was the locked track
        const lockedTrackId = lockedAlerts.get(data.deviceName);
        if (lockedTrackId === data.trackId) {
            lockedAlerts.delete(data.deviceName);
            updateActiveVideoSlot(data.deviceName, false);
        }
    }
    // PASSIVE doesn't change the border state
});

// Reorder all alerts by status priority
function reorderAlertsByStatus() {
    const alertItems = Array.from(alertList.children);

    if (alertItems.length <= 1) return;

    // Sort by status: LOCKED first, then PASSIVE, then LOST
    alertItems.sort((a, b) => {
        const getStatusPriority = (el) => {
            const badge = el.querySelector('.alert-badge');
            if (!badge) return 3;
            const status = badge.textContent.toUpperCase();
            if (status === 'LOCKED') return 0;
            if (status === 'PASSIVE') return 1;
            return 2; // LOST or other
        };
        return getStatusPriority(a) - getStatusPriority(b);
    });

    // Clear and rebuild in sorted order
    alertList.innerHTML = '';
    alertItems.forEach(item => alertList.appendChild(item));
}

// Update video slot with active tracking border
function updateActiveVideoSlot(deviceName, isActive) {
    // Find video slot for this device
    const slots = document.querySelectorAll('.video-slot');
    slots.forEach(slot => {
        const label = slot.querySelector('.overlay-name');
        if (label && label.textContent === deviceName) {
            if (isActive) {
                slot.classList.add('tracking-active');
            } else {
                slot.classList.remove('tracking-active');
            }
        }
    });
}

function playAlertSound() {
    try {
        if (!alertSound) {
            alertSound = new Audio('alert.mp3');
        }
        alertSound.currentTime = 0;
        alertSound.play().catch(() => {
            // Browser might block autoplay
        });
    } catch (e) {
        console.warn('Could not play alert sound');
    }
}

// --- Track Log Handlers ---
const trackLogsBtn = document.getElementById('track-logs-btn');
const trackLogsModal = document.getElementById('track-logs-modal');
const closeLogsModal = document.getElementById('close-logs-modal');
const trackLogModalList = document.getElementById('track-log-modal-list');
const logCountEl = document.getElementById('log-count');
let trackLogCount = 0;
const trackLogs = []; // Store all logs

// Open modal on button click
trackLogsBtn.onclick = () => {
    trackLogsModal.classList.add('open');
};

// Close modal
closeLogsModal.onclick = () => {
    trackLogsModal.classList.remove('open');
};

// Close on outside click
trackLogsModal.onclick = (e) => {
    if (e.target === trackLogsModal) {
        trackLogsModal.classList.remove('open');
    }
};

socket.on('newTrackLog', (data) => {
    log(`[TRACKLOG] ${data.sentinel_name} Track #${data.track_id} completed`);
    trackLogs.unshift(data);
    renderTrackLogCard(data, trackLogModalList);
    trackLogCount++;
    logCountEl.textContent = trackLogCount;
});

socket.on('trackLogUpdated', (data) => {
    // 1. Update local array
    const idx = trackLogs.findIndex(l => l.sentinel_name === data.sentinel_name && l.track_id === data.track_id);
    if (idx !== -1) trackLogs[idx] = data;

    // 2. Re-render list (simple approach to ensure buttons appear)
    if (trackLogModalList) {
        trackLogModalList.innerHTML = '';
        // Iterate reverse so that prepending results in Newest at Top
        for (let i = trackLogs.length - 1; i >= 0; i--) {
            renderTrackLogCard(trackLogs[i], trackLogModalList);
        }
    }
});

function fetchTrackLogHistory() {
    socket.emit('getTrackLogHistory', (history) => {
        if (history && history.length) {
            // Clear existing
            trackLogModalList.innerHTML = '';
            trackLogs.length = 0;

            // Add in order (newest first)
            history.forEach(logData => {
                trackLogs.push(logData);
                renderTrackLogCard(logData, trackLogModalList);
            });

            trackLogCount = history.length;
            logCountEl.textContent = trackLogCount;
        }
    });
}

function renderTrackLogCard(data, container) {
    const card = document.createElement('div');
    card.className = 'track-log-card';

    // Format timestamp
    const startTime = new Date(data.start_time).toLocaleString('tr-TR');
    const endTime = new Date(data.end_time).toLocaleString('tr-TR');

    // Track type badge class
    const trackType = data.track_type || 'PASSIVE';
    const typeBadgeClass = trackType === 'ACTIVE' ? 'locked' : 'passive';

    // False detection badge
    const possibleFalse = data.possible_false || false;
    const falseBadge = possibleFalse ? '<span class="alert-badge false">FALSE?</span>' : '';

    // Video button (only for ACTIVE tracks with video)
    // Video button (if video exists)
    const hasVideo = !!data.video_path;
    const hasSnapshot = !!data.snapshot_path;

    let mediaBtn = '';
    if (hasVideo) {
        mediaBtn = '<button class="video-btn" title="Play Video">▶</button>';
    } else if (hasSnapshot) {
        mediaBtn = '<button class="snapshot-btn" title="View Snapshot">📷</button>';
    }

    // Main visible content
    card.innerHTML = `
        <div class="log-header">
            <span class="log-device">${data.sentinel_name} #${data.track_id}</span>
            <span class="alert-badge ${typeBadgeClass}">${trackType}</span>
            ${falseBadge}
            ${mediaBtn}
            <span class="log-duration">${data.duration_sec}s</span>
            <span class="expand-icon">▼</span>
        </div>
        <div class="log-summary">
            <span class="direction-badge">${data.entry_direction} → ${data.exit_direction}</span>
            <span class="log-time">${startTime}</span>
        </div>
        <div class="log-details" style="display: none;">
            <div class="detail-row"><span>Track Type:</span> <span>${trackType}</span></div>
            <div class="detail-row"><span>Possible False:</span> <span>${possibleFalse ? 'Yes' : 'No'}</span></div>
            <div class="detail-row"><span>Start:</span> <span>${startTime}</span></div>
            <div class="detail-row"><span>End:</span> <span>${endTime}</span></div>
            <div class="detail-row"><span>Duration:</span> <span>${data.duration_sec} seconds</span></div>
            <div class="detail-row"><span>Entry Bearing:</span> <span>${data.entry_bearing}° (${data.entry_direction})</span></div>
            <div class="detail-row"><span>Exit Bearing:</span> <span>${data.exit_bearing}° (${data.exit_direction})</span></div>
            <div class="detail-row"><span>Last Position:</span> <span>X: ${data.last_pos_x}, Y: ${data.last_pos_y}</span></div>
            <div class="detail-row"><span>Avg Speed:</span> <span>${data.avg_speed_px} px/frame</span></div>
            <div class="detail-row"><span>Max Speed:</span> <span>${data.max_speed_px} px/frame</span></div>
            <div class="detail-row"><span>Avg Distance:</span> <span>${data.avg_distance_m} m</span></div>
            <div class="detail-row"><span>Min Distance:</span> <span>${data.min_distance_m} m</span></div>
            <div class="detail-row"><span>Max Distance:</span> <span>${data.max_distance_m || 'N/A'} m</span></div>
            <div class="detail-row"><span>Camera Facing:</span> <span>${data.camera_facing}°</span></div>
            <div class="detail-row"><span>Reason:</span> <span>${data.reason}</span></div>
            ${hasVideo ? `<div class="detail-row video-row"><span>Video:</span> <button class="video-play-btn" onclick="playTrackVideo('${data.video_path}')">▶ Play Recording</button></div>` : ''}
            ${hasSnapshot ? `<div class="detail-row video-row"><span>Snapshot:</span> <button class="video-play-btn" onclick="viewTrackSnapshot('${data.snapshot_path}')">📷 View Snapshot</button></div>` : ''}
        </div>
    `;

    // Button click handlers
    if (hasVideo) {
        const videoBtnEl = card.querySelector('.video-btn');
        if (videoBtnEl) {
            videoBtnEl.onclick = (e) => {
                e.stopPropagation();
                playTrackVideo(data.video_path);
            };
        }
    } else if (hasSnapshot) {
        const snapBtnEl = card.querySelector('.snapshot-btn');
        if (snapBtnEl) {
            snapBtnEl.onclick = (e) => {
                e.stopPropagation();
                viewTrackSnapshot(data.snapshot_path);
            };
        }
    }

    // Click to expand/collapse
    card.onclick = (e) => {
        if (e.target.classList.contains('video-btn') ||
            e.target.classList.contains('snapshot-btn') ||
            e.target.classList.contains('video-play-btn')) return;

        const details = card.querySelector('.log-details');
        const icon = card.querySelector('.expand-icon');
        if (details.style.display === 'none') {
            details.style.display = 'block';
            icon.textContent = '▲';
            card.classList.add('expanded');
        } else {
            details.style.display = 'none';
            icon.textContent = '▼';
            card.classList.remove('expanded');
        }
    };

    container.insertBefore(card, container.firstChild);

    // Limit to 50 cards in modal
    while (container.children.length > 50) {
        container.removeChild(container.lastChild);
    }
}

// Fetch history on init
fetchTrackLogHistory();

// Video player modal
window.playTrackVideo = function playTrackVideo(videoPath) {
    showModal(videoPath, 'video');
}

window.viewTrackSnapshot = function viewTrackSnapshot(imagePath) {
    showModal(imagePath, 'image');
}

function showModal(path, type) {
    // Extract just the filename from the path
    const filename = path.split('/').pop();

    // Create modal if it doesn't exist
    let modal = document.getElementById('video-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'video-modal';
        modal.className = 'video-modal';
        modal.innerHTML = `
            <div class="video-modal-content">
                <button class="video-modal-close">&times;</button>
                <h3 class="video-modal-title">Track Log</h3>
                <div id="modal-media-container"></div>
                <p class="video-path"></p>
            </div>
        `;
        document.body.appendChild(modal);

        // Close button handler
        modal.querySelector('.video-modal-close').onclick = () => {
            closeMediaModal();
        };

        // Click outside to close
        modal.onclick = (e) => {
            if (e.target === modal) {
                closeMediaModal();
            }
        };
    }

    const container = modal.querySelector('#modal-media-container');
    const pathDisplay = modal.querySelector('.video-path');
    const title = modal.querySelector('.video-modal-title');

    container.innerHTML = ''; // Clear previous

    if (type === 'video') {
        title.textContent = 'Track Recording';
        const video = document.createElement('video');
        video.id = 'video-player';
        video.controls = true;
        video.autoplay = true;
        video.innerHTML = '<source src="">Your browser does not support video playback.';
        video.src = `/vlogs/${filename}`;
        container.appendChild(video);
    } else {
        title.textContent = 'Track Snapshot';
        const img = document.createElement('img');
        img.id = 'snapshot-viewer';
        img.src = `/vlogs/${filename}`;
        img.style.maxWidth = '100%';
        img.style.maxHeight = '60vh';
        img.style.display = 'block';
        img.style.margin = '0 auto';
        container.appendChild(img);
    }

    // Try to load from server's vlogs directory
    pathDisplay.textContent = `File: ${filename}`;
    modal.style.display = 'flex';
}

function closeMediaModal() {
    const modal = document.getElementById('video-modal');
    if (modal) {
        modal.style.display = 'none';
        const container = modal.querySelector('#modal-media-container');
        container.innerHTML = ''; // Clear to stop video
    }
}

// --- Minimap ---
const minimap = new Minimap({
    canvas: document.getElementById('minimap-canvas'),
    expandButton: document.getElementById('minimap-expand-btn'),
    modal: document.getElementById('minimap-modal'),
    modalCanvas: document.getElementById('minimap-modal-canvas'),
    modalCloseBtn: document.getElementById('minimap-modal-close'),
    readout: document.getElementById('minimap-readout'),
});

socket.on('geolocationUpdate', (data) => minimap.onGeolocationUpdate(data));
socket.on('trackingStatus', (data) => minimap.onTrackingStatus(data));
socket.on('globalTrackUpdate', (data) => minimap.onGlobalTrackUpdate(data));
socket.on('globalTrackRemoved', (data) => minimap.onGlobalTrackRemoved(data));
socket.on('unitTelemetry', (data) => minimap.onUnitTelemetry(data));

// =========================================================================
// Sensor Health — per-tile hover popover (anchored on each Pi's "i" chip).
// Updated by `unitHealth` events (Pi emits at ~1 Hz, server stamps server_ts).
// Devices whose last `unitHealth` arrived > HEALTH_TIMEOUT_MS ago are pruned
// from HEALTH entirely (the popover reverts to "no telemetry").
// Status pill: green (fps≥25, fresh), yellow (degraded), red (stale or low fps).
// =========================================================================
const HEALTH = new Map();  // deviceName -> last unitHealth payload (with server_ts)
const HEALTH_STALE_RED_MS = 5000;
const HEALTH_STALE_YEL_MS = 2000;
const HEALTH_TIMEOUT_MS   = 10000;  // drop device entirely after this many ms with no update

function statusPillFor(h, ageMs) {
    if (ageMs > HEALTH_STALE_RED_MS) return 'red';
    if ((h.fps_30s || 0) < 15) return 'red';
    if (ageMs > HEALTH_STALE_YEL_MS) return 'yellow';
    if ((h.fps_30s || 0) < 25) return 'yellow';
    return 'green';
}

function findHealthSlots(deviceName) {
    // A device can exist in any of the grid slots; return all matching slot DOMs.
    return Array.from(document.querySelectorAll(`.video-slot[data-device="${CSS.escape(deviceName)}"]`));
}

function renderHealthForDevice(deviceName, slotOrNull) {
    if (!deviceName) return;
    const slots = slotOrNull ? [slotOrNull] : findHealthSlots(deviceName);
    if (slots.length === 0) return;
    const h = HEALTH.get(deviceName);
    const now = Date.now();
    for (const slot of slots) {
        const chipPill   = slot.querySelector('.health-info-wrap > .health-pill');
        const popName    = slot.querySelector('.health-popover-name');
        const popPill    = slot.querySelector('.health-popover-title > .health-pill');
        const popBody    = slot.querySelector('.health-popover-body');
        if (!chipPill || !popName || !popPill || !popBody) continue;

        popName.textContent = deviceName;

        if (!h) {
            chipPill.className = 'health-pill';
            chipPill.textContent = '—';
            popPill.className = 'health-pill';
            popPill.textContent = '—';
            popBody.innerHTML = '<div class="health-empty">no telemetry yet</div>';
            continue;
        }

        const ageMs = now - (h.server_ts || 0);
        const pill = statusPillFor(h, ageMs);
        chipPill.className = `health-pill ${pill}`;
        chipPill.textContent = pill === 'green' ? 'OK' : (pill === 'yellow' ? 'WARN' : 'FAIL');
        popPill.className = `health-pill ${pill}`;
        popPill.textContent = pill.toUpperCase();

        const ageS    = Math.max(0, ageMs / 1000);
        const ageTxt  = ageS < 60 ? `${ageS.toFixed(1)}s` : `${(ageS/60).toFixed(1)}m`;
        const uptimeS = h.uptime_s || 0;
        const uptimeTxt = uptimeS < 60 ? `${uptimeS.toFixed(0)}s`
                        : uptimeS < 3600 ? `${(uptimeS/60).toFixed(1)}m`
                        : `${(uptimeS/3600).toFixed(2)}h`;
        const lockTxt = h.active_track_id != null
            ? `t${h.active_track_id}` + (h.active_track_sigma != null ? ` σ${h.active_track_sigma.toFixed(1)}m` : '')
            : '—';
        const tempTxt = (typeof h.cpu_temp_c === 'number' && isFinite(h.cpu_temp_c))
            ? `${h.cpu_temp_c.toFixed(1)} °C` : '—';

        popBody.innerHTML = `
            <div class="health-kv"><span>FPS (30 s)</span><span>${(h.fps_30s || 0).toFixed(1)}</span></div>
            <div class="health-kv"><span>Detect/s</span><span>${(h.detect_per_s_30s || 0).toFixed(2)}</span></div>
            <div class="health-kv"><span>Lock</span><span>${lockTxt}</span></div>
            <div class="health-kv"><span>Passive tracks</span><span>${h.passive_track_count || 0}</span></div>
            <div class="health-kv"><span>CPU temp</span><span>${tempTxt}</span></div>
            <div class="health-kv"><span>Uptime</span><span>${uptimeTxt}</span></div>
            <div class="health-kv"><span>Last seen</span><span>${ageTxt}</span></div>`;
    }
}

socket.on('unitHealth', (data) => {
    if (!data || !data.deviceName) return;
    HEALTH.set(data.deviceName, data);
    renderHealthForDevice(data.deviceName, null);
});

// Tick every second: prune devices we haven't heard from in HEALTH_TIMEOUT_MS,
// and refresh open popovers so the "Last seen" / age column keeps moving.
setInterval(() => {
    const now = Date.now();
    for (const [name, h] of Array.from(HEALTH.entries())) {
        if (now - (h.server_ts || 0) > HEALTH_TIMEOUT_MS) {
            HEALTH.delete(name);
            renderHealthForDevice(name, null);
        }
    }
    // Refresh every active tile's popover (cheap; small DOM updates).
    document.querySelectorAll('.video-slot[data-device]').forEach(slot => {
        const dev = slot.getAttribute('data-device');
        if (dev) renderHealthForDevice(dev, slot);
    });
}, 1000);

