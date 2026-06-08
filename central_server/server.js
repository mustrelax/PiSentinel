const fs = require('fs');
const https = require('https');
const http = require('http');
const express = require('express');
const socketIO = require('socket.io');
const mediasoup = require('mediasoup');
const config = require('./config');
const path = require('path');
const multer = require('multer');
const { spawn } = require('child_process');
const net = require('net');
const { registerGeolocationHandlers, removeUnitFromGlobalTracks, dropTrackFromGlobalTracks } = require('./geolocation');

// Global variables
let worker;
let router;
const PRODUCERS = new Map(); // producerId -> { socketId, producer, deviceName }
let consumers = []; // List of consumers
const ALERTS = []; // Global alerts history
const TRACK_LOGS = []; // Completed track logs (max 100)
const ACTIVE_RECORDINGS = new Map(); // key(device+track) -> { ffmpeg, filepath, trackId, deviceName, plainTransport }
const PENDING_RECORDINGS = new Set(); // key(device+track) (prevent race conditions)
const STOP_TIMEOUTS = new Map(); // key(device+track) -> timeoutId (for delayed stop)

function recordingKey(deviceName, trackId) {
    return `${deviceName}::${trackId}`;
}

// Helper to find free UDP port for FFmpeg
const getFreePort = () => new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.listen(0, '127.0.0.1', () => {
        const port = srv.address().port;
        srv.close(() => resolve(port));
    });
    srv.on('error', reject);
});

const app = express();
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve node_modules for client-side libs (no auth needed)
app.use('/libs', express.static(path.join(__dirname, 'node_modules')));

// --- Auth ---
const SESSIONS = new Map();

// Middleware to check auth
function requireAuth(req, res, next) {
    const p = req.path;

    // Allow login page and its resources
    if (p === '/login' || p === '/login.html' || p === '/style.css' || p === '/favicon.ico' || p === '/pisentinellogo.png') {
        return next();
    }

    // Allow socket.io (it has its own auth if needed)
    if (p.startsWith('/socket.io/')) {
        return next();
    }

    // Check session cookie
    const cookie = req.headers.cookie || '';
    const tokenMatch = cookie.split(';').find(c => c.trim().startsWith('session_token='));

    if (tokenMatch) {
        const val = tokenMatch.split('=')[1];
        if (SESSIONS.has(val)) {
            return next();
        }
    }

    // Not authenticated - redirect to login
    if (req.accepts('html')) {
        return res.redirect('/login.html');
    }
    return res.status(401).json({ error: 'Unauthorized' });
}

// Apply auth middleware BEFORE static files
app.use(requireAuth);

// Now serve static files (protected by auth above)
app.use(express.static('public'));

// Serve video logs from server's vlogs directory
const VLOGS_DIR = path.join(__dirname, 'vlogs');
if (!fs.existsSync(VLOGS_DIR)) {
    fs.mkdirSync(VLOGS_DIR, { recursive: true });
}
app.use('/vlogs', express.static(VLOGS_DIR));

// Video upload storage config
const videoStorage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, VLOGS_DIR),
    filename: (req, file, cb) => cb(null, file.originalname)
});
const videoUpload = multer({ storage: videoStorage, limits: { fileSize: 100 * 1024 * 1024 } });

// Video upload endpoint (from Pi)
app.post('/upload-video', videoUpload.single('video'), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No video file' });
    }
    console.log(`[VIDEO] Received: ${req.file.originalname} (${(req.file.size / 1024 / 1024).toFixed(2)}MB)`);
    res.json({ success: true, filename: req.file.originalname });
});

app.post('/login', (req, res) => {
    const { username, password } = req.body;

    // Read users from JSON file (simple approach for student project)
    let users = [];
    try {
        users = JSON.parse(fs.readFileSync('./users.json', 'utf8'));
    } catch (e) {
        console.error('Could not read users.json:', e);
    }

    // Check if user exists
    const user = users.find(u => u.username === username && u.password === password);

    if (user) {
        const token = Math.random().toString(36).substr(2);
        SESSIONS.set(token, username);
        res.cookie('session_token', token, { maxAge: 43200000, httpOnly: false });
        console.log(`[AUTH] User '${username}' logged in`);
        return res.json({ ok: true, redirect: '/' });
    }

    console.log(`[AUTH] Failed login attempt for '${username}'`);
    return res.status(401).json({ ok: false, error: 'Invalid credentials' });
});

app.get('/stats', (req, res) => {
    res.json({ viewers: VIEWERS.size });
});

// HTTP Server
const webServer = http.createServer(app);

// Socket.IO
const io = socketIO(webServer);

// Track authenticated viewers (not producers)
const VIEWERS = new Set();

// Helper to broadcast stats
function broadcastStats() {
    console.log(`[STATS] Viewers: ${VIEWERS.size}`);
    io.emit('stats', { viewers: VIEWERS.size });
}

// Startup
(async () => {
    try {
        await runMediasoupWorker();
        await createRouter();

        // Start server
        webServer.listen(config.listenPort, config.listenIp, () => {
            console.log(`Server running at http://${config.listenIp}:${config.listenPort}`);
            console.log(`Open http://localhost:${config.listenPort} in your browser.`);
        });
    } catch (err) {
        console.error(err);
    }
})();

// --- Mediasoup Worker ---
async function runMediasoupWorker() {
    worker = await mediasoup.createWorker({
        logLevel: config.mediasoup.worker.logLevel,
        logTags: config.mediasoup.worker.logTags,
        rtcMinPort: config.mediasoup.worker.rtcMinPort,
        rtcMaxPort: config.mediasoup.worker.rtcMaxPort,
    });

    worker.on('died', () => {
        console.error('mediasoup worker died, exiting in 2 seconds... [pid:%d]', worker.pid);
        setTimeout(() => process.exit(1), 2000);
    });

    console.log('Mediasoup Worker created [pid:%d]', worker.pid);

    // Debug Stats & Keyframe Pump
    setInterval(async () => {
        // console.log("[DEBUG] Stats Tick. Producers:", PRODUCERS.size); // Temporary Heartbeat

        // 1. Log Producer Stats (commented out - too spammy)
        // console.log(`[DEBUG] Current Producers count: ${PRODUCERS.size}`);
        for (const producerData of PRODUCERS.values()) {
            try {
                const stats = await producerData.producer.getStats();
                const statsArray = Array.from(stats.values());
                console.log(`[DEBUG] Stats for ${producerData.deviceName}: Found ${statsArray.length} stat entries.`);

                const inbound = statsArray.find(s => s.type === 'inbound-rtp');
                if (inbound) {
                    console.log(`[STATS] ${producerData.deviceName} | SSRC:${inbound.ssrc} | Pkts:${inbound.packetCount} | Bytes:${inbound.byteCount}`);
                } else {
                    console.log(`[STATS] ${producerData.deviceName} - No inbound-rtp stats found.`);

                    // Detailed Debugging for Missing Stats
                    const transport = statsArray.find(s => s.type === 'transport');
                    if (transport) {
                        console.log(`[DEBUG] Transport State: dtlsState=${transport.dtlsState}, iceState=${transport.iceState}`);
                        console.log(`[DEBUG] Transport Bytes: Recv=${transport.bytesReceived}, Sent=${transport.bytesSent}`);
                    }

                    const candidates = statsArray.filter(s => s.type === 'candidate-pair' && s.state === 'succeeded');
                    if (candidates.length > 0) {
                        console.log(`[DEBUG] Active Candidate Pair: ${candidates[0].localCandidateId} <-> ${candidates[0].remoteCandidateId}`);
                    } else {
                        console.log(`[DEBUG] NO SUCCEEDED CANDIDATE PAIR FOUND! (Likely connection failure)`);
                    }
                }
            } catch (e) { console.error("Stats Error:", e); }
        }

        // 2. FORCE Keyframe for all consumers (Brute Force Reliability)
        // If the browser misses the first keyframe (black screen), this ensures it gets another one every 2s.
        for (const c of consumers) {
            try {
                if (!c.consumer.paused) {
                    await c.consumer.requestKeyFrame();
                }
            } catch (e) { }
        }
    }, 2000);
}

// --- Mediasoup Router ---
async function createRouter() {
    const mediaCodecs = config.mediasoup.router.mediaCodecs;
    router = await worker.createRouter({ mediaCodecs });
    console.log('Mediasoup Router created');
}

// --- Video Recording Functions ---

// Actual Stop Logic
function finalizeRecordingByKey(key) {
    const recording = ACTIVE_RECORDINGS.get(key);
    if (!recording) return;

    try {
        // Signal FFmpeg to stop gracefully
        recording.ffmpeg.stdin.end();

        // Remove from map immediately to prevent re-entry
        ACTIVE_RECORDINGS.delete(key);

        // Close mediasoup resources
        if (recording.consumer) recording.consumer.close();
        if (recording.plainTransport) recording.plainTransport.close();

        // Wait for FFmpeg to actually finish writing the file
        recording.ffmpeg.on('close', () => {
            // Calculate duration based on when stop was REQUESTED, not when finalized
            const stopTime = recording.pendingStopTime || Date.now();
            const duration = stopTime - recording.startTime;
            const isShortTrack = duration < 2000; // Less than 2 seconds

            // Find track log to update
            const trackLog = TRACK_LOGS.find(log =>
                log.sentinel_name === recording.deviceName && log.track_id === recording.trackId
            );

            if (isShortTrack) {
                console.log(`[RECORD] Short track detected (${duration}ms). Converting to snapshot.`);

                // Generate snapshot filename (replace .webm with .jpg)
                const snapshotFilename = recording.filename.replace('.webm', '.jpg');
                const snapshotPath = path.join(VLOGS_DIR, snapshotFilename);

                // Extract first frame using FFmpeg
                // -i input -vframes 1 -q:v 2 output.jpg
                const extractArgs = [
                    '-i', recording.filepath,
                    '-vframes', '1',
                    '-q:v', '2',
                    '-y',
                    snapshotPath
                ];

                const extractor = spawn('ffmpeg', extractArgs);

                extractor.on('close', (code) => {
                    if (code === 0) {
                        console.log(`[RECORD] Snapshot created: ${snapshotFilename}`);

                        // Delete original video file
                        try {
                            fs.unlinkSync(recording.filepath);
                            console.log(`[RECORD] Deleted short video: ${recording.filename}`);
                        } catch (e) {
                            console.error(`[RECORD] Failed to delete video: ${e.message}`);
                        }

                        // Update track log
                        if (trackLog) {
                            trackLog.snapshot_path = `/vlogs/${snapshotFilename}`;
                            // We do NOT set video_path

                            // Broadcast update
                            io.emit('trackLogUpdated', trackLog);
                        }
                    } else {
                        console.error(`[RECORD] Snapshot extraction failed with code ${code}`);
                    }
                });

            } else {
                // Standard Video Logic
                if (trackLog) {
                    trackLog.video_path = `/vlogs/${recording.filename}`;
                    // Broadcast update to UI so the button appears
                    io.emit('trackLogUpdated', trackLog);
                }
                console.log(`[RECORD] Finalized Video: ${recording.deviceName} track #${recording.trackId} -> ${recording.filename}`);
            }
        });

        // Trigger stop
        recording.ffmpeg.kill('SIGINT');

    } catch (err) {
        console.error(`[RECORD] Failed to finalize: ${err.message}`);
        ACTIVE_RECORDINGS.delete(key);
    }
}

async function startRecording(deviceName, trackId) {
    const key = recordingKey(deviceName, trackId);

    // 0. Check for pending stop on this same track (Delayed Stop Logic)
    if (STOP_TIMEOUTS.has(key)) {
        const recording = ACTIVE_RECORDINGS.get(key);
        if (recording) {
            console.log(`[RECORD] Resumed ${deviceName} track #${trackId} (cancelled stop)`);
            if (recording.pendingStopTime) delete recording.pendingStopTime;
            clearTimeout(STOP_TIMEOUTS.get(key));
            STOP_TIMEOUTS.delete(key);
            return;
        }
    }

    // Prevent race conditions (multiple LOCKED events for same track)
    if (ACTIVE_RECORDINGS.has(key) || PENDING_RECORDINGS.has(key)) {
        return;
    }

    PENDING_RECORDINGS.add(key);

    try {
        // Find the producer for this device
        let producer = null;
        for (const [id, data] of PRODUCERS) {
            if (data.deviceName === deviceName) {
                producer = data.producer;
                break;
            }
        }

        if (!producer) {
            console.log(`[RECORD] No producer found for ${deviceName}`);
            PENDING_RECORDINGS.delete(key);
            return;
        }

        // 1. Get a free port for FFmpeg to listen on
        const ffmpegPort = await getFreePort();

        // 2. Create PlainTransport (PUSH mode)
        // comedia: false (default) -> We must explicitly connect() to destination
        const plainTransport = await router.createPlainTransport({
            listenIp: { ip: '127.0.0.1' },
            rtcpMux: true, // Use Mux to allow single port connection
            comedia: false
        });

        // 3. Create consumer to receive producer's RTP
        const consumer = await plainTransport.consume({
            producerId: producer.id,
            rtpCapabilities: router.rtpCapabilities,
            paused: true // Start paused
        });

        // 4. Generate SDP so FFmpeg knows format and where to listen
        const rtpParameters = consumer.rtpParameters;
        const payloadType = rtpParameters.codecs[0].payloadType;
        const codecName = rtpParameters.codecs[0].mimeType.split('/')[1]; // VP8

        const sdpContent = `v=0
o=- 0 0 IN IP4 127.0.0.1
s=FFmpeg
c=IN IP4 127.0.0.1
t=0 0
m=video ${ffmpegPort} RTP/AVP ${payloadType}
a=rtpmap:${payloadType} ${codecName}/${rtpParameters.codecs[0].clockRate}
a=rtcp-mux
a=recvonly
`;

        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `${deviceName}_track${trackId}_${timestamp}.webm`;
        const filepath = path.join(VLOGS_DIR, filename);
        const sdpFile = path.join(VLOGS_DIR, `${deviceName}_${trackId}.sdp`);

        fs.writeFileSync(sdpFile, sdpContent);

        // 5. Spawn FFmpeg (Server Mode - Listening on UDP port via SDP)
        const ffmpegArgs = [
            '-protocol_whitelist', 'file,rtp,udp',
            '-i', sdpFile,
            '-c:v', 'copy', // No transcoding, just dump VP8 to WebM
            '-f', 'webm',
            '-y',
            filepath
        ];

        const ffmpeg = spawn('ffmpeg', ffmpegArgs);

        ffmpeg.stderr.on('data', (data) => {
            // Optional debug
            // console.log(`[FFMPEG] ${data}`); 
        });

        ffmpeg.on('close', (code) => {
            if (code !== 0 && code !== 255) { // 255 is SIGINT
                console.log(`[RECORD] FFmpeg closed code ${code} (${deviceName})`);
            }
            try { fs.unlinkSync(sdpFile); } catch (e) { }
        });

        // 6. Connect Mediasoup Transport to FFmpeg's port
        await plainTransport.connect({
            ip: '127.0.0.1',
            port: ffmpegPort
        });

        // 7. Resume Consumer and Force Keyframe
        await consumer.resume();
        await consumer.requestKeyFrame();

        ACTIVE_RECORDINGS.set(key, {
            ffmpeg,
            filepath,
            trackId,
            deviceName,
            plainTransport,
            consumer,
            filename,
            sdpFile,
            startTime: Date.now()
        });

        console.log(`[RECORD] Started: ${deviceName} track #${trackId} -> ${filename}`);

    } catch (err) {
        console.error(`[RECORD] Failed to start: ${err.message}`);
    } finally {
        PENDING_RECORDINGS.delete(key);
    }
}

function stopRecording(deviceName, trackId) {
    const key = recordingKey(deviceName, trackId);
    const recording = ACTIVE_RECORDINGS.get(key);
    if (!recording) return;

    // Check if already waiting to stop
    if (STOP_TIMEOUTS.has(key)) return;

    console.log(`[RECORD] Pending stop for ${deviceName} track #${trackId} (5s)...`);

    // Mark the time we ASKED to stop, so duration calculation is accurate
    recording.pendingStopTime = Date.now();

    const timeout = setTimeout(() => {
        finalizeRecordingByKey(key);
        STOP_TIMEOUTS.delete(key);
    }, 5000); // 5 seconds delayed stop

    STOP_TIMEOUTS.set(key, timeout);
}

function stopAllRecordingsForDevice(deviceName) {
    const keys = Array.from(ACTIVE_RECORDINGS.keys()).filter(k => k.startsWith(`${deviceName}::`));
    for (const key of keys) {
        if (STOP_TIMEOUTS.has(key)) {
            clearTimeout(STOP_TIMEOUTS.get(key));
            STOP_TIMEOUTS.delete(key);
        }
        finalizeRecordingByKey(key);
    }
}

// --- Socket.IO Connection ---
io.on('connection', (socket) => {
    console.log('New client connected:', socket.id);

    // Client must identify itself as 'viewer' or 'producer'
    socket.on('identify', ({ role }) => {
        if (role === 'viewer') {
            VIEWERS.add(socket.id);
            console.log(`[VIEWER] ${socket.id} identified as viewer`);
            broadcastStats();
        }
    });

    socket.on('disconnect', () => {
        console.log('Client disconnected:', socket.id);

        // Remove from viewers if applicable
        if (VIEWERS.has(socket.id)) {
            VIEWERS.delete(socket.id);
        }

        // Cleanup ALL consumers for this socket
        const socketConsumers = consumers.filter(c => c.socketId === socket.id);
        socketConsumers.forEach(c => c.consumer.close());
        consumers = consumers.filter(c => c.socketId !== socket.id);

        // Cleanup if it was a producer (Search in Map)
        for (const [id, producerData] of PRODUCERS) {
            if (producerData.socketId === socket.id) {
                console.log(`Producer disconnected: ${id} (${producerData.deviceName})`);

                // Mark all alerts from this device as LOST
                ALERTS.forEach(alert => {
                    if (alert.deviceName === producerData.deviceName && alert.status !== 'LOST') {
                        alert.status = 'LOST';
                        // Broadcast LOST status to all clients
                        io.emit('trackingStatus', {
                            deviceName: alert.deviceName,
                            trackId: alert.trackId,
                            status: 'LOST'
                        });
                    }
                });

                producerData.producer.close();
                PRODUCERS.delete(id);
                socket.broadcast.emit('producer-closed', { producerId: id });
                stopAllRecordingsForDevice(producerData.deviceName);
                removeUnitFromGlobalTracks(io, producerData.deviceName);
            }
        }

        broadcastStats();
    });

    // --- Signaling Events ---

    // Drone Alert - Producer sends, broadcast to all viewers
    socket.on('droneAlert', (data) => {
        console.log(`[ALERT] Drone detected by ${data.deviceName}, Track: ${data.trackId}`);

        const alertData = {
            deviceName: data.deviceName,
            trackId: data.trackId,
            timestamp: Date.now(),
            status: 'PASSIVE' // Initial status (will be updated to LOCKED if selected as primary)
        };

        // Store in history
        ALERTS.unshift(alertData);
        if (ALERTS.length > 50) ALERTS.pop(); // Keep last 50

        // Broadcast to all connected clients
        io.emit('droneAlert', alertData);
    });

    // Get Alert History
    socket.on('getAlertHistory', (callback) => {
        callback(ALERTS);
    });

    // Tracking Status - ACTIVE/LOST updates from producers
    socket.on('trackingStatus', (data) => {

        // Update history
        const existingAlert = ALERTS.find(a =>
            a.deviceName === data.deviceName && a.trackId === data.trackId
        );
        if (existingAlert) {
            existingAlert.status = data.status;
        }

        // Handle recording start/stop
        if (data.status === 'LOCKED') {
            startRecording(data.deviceName, data.trackId);
        } else if (data.status === 'LOST') {
            stopRecording(data.deviceName, data.trackId);
            // Evict this (device, trackId) from any global track immediately,
            // instead of waiting for pruneStale. On fast re-acquisition under a
            // NEW Pi-side trackId, the orphan-resurrection path then re-uses
            // the same globalId — no ghost-twin globals.
            dropTrackFromGlobalTracks(io, data.deviceName, data.trackId);
        }

        // Broadcast to all viewers
        io.emit('trackingStatus', {
            deviceName: data.deviceName,
            trackId: data.trackId,
            status: data.status  // 'LOCKED', 'ACTIVE', or 'LOST'
        });
    });

    // Radar Telemetry - Phase 1 Verification
    socket.on('radarUpdate', (data) => {
        // Log sparingly (e.g. check timestamp if needed, but for now just log)
        // console.log(`[RADAR] ${data.deviceName} | T:${data.trackId} | Az:${data.azimuth.toFixed(1)}° | Dist:${data.distance.toFixed(0)} | Spd:${data.speed.toFixed(1)}`);

        // Broadcast to clients (for Phase 2)
        io.emit('radarUpdate', data);
    });

    // World-frame geolocation relay (Phase 1: pure fan-out).
    registerGeolocationHandlers(io, socket);

    // Always-on unit telemetry — drives live look-bearing wedge on map even
    // when no drone is being tracked. Pi emits at ~10 Hz; pure fan-out.
    socket.on('unitTelemetry', (data) => {
        if (!data || !data.deviceName) return;
        io.emit('unitTelemetry', data);
    });

    // Periodic Pi health snapshot — drives the Sensor Health dashboard.
    // Pi emits at ~1 Hz. Server stamps server_ts so the UI can compute
    // last-seen latency independently of any Pi clock skew, then fans out.
    socket.on('unitHealth', (data) => {
        if (!data || !data.deviceName) return;
        data.server_ts = Date.now();
        io.emit('unitHealth', data);
    });

    // Track Log Complete - Full log when track is lost
    socket.on('trackLogComplete', (data) => {
        console.log(`[TRACKLOG] ${data.sentinel_name} Track #${data.track_id}: ${data.duration_sec}s, ${data.entry_direction}->${data.exit_direction}`);

        // Save false-positive screenshot to server's vlogs/ if Pi sent image data
        if (data.screenshot_data && data.screenshot_filename) {
            if (!fs.existsSync(VLOGS_DIR)) fs.mkdirSync(VLOGS_DIR, { recursive: true });
            const imgPath = path.join(VLOGS_DIR, data.screenshot_filename);
            fs.writeFileSync(imgPath, Buffer.from(data.screenshot_data, 'base64'));
            data.snapshot_path = `/vlogs/${data.screenshot_filename}`;
            delete data.screenshot_data; // don't keep base64 blob in memory or logs
            console.log(`[FP] Saved false-positive screenshot: ${imgPath}`);
        }

        // Store in history
        TRACK_LOGS.unshift(data);
        if (TRACK_LOGS.length > 100) TRACK_LOGS.pop();

        // Save to date-based JSON file in /logs
        const today = new Date().toISOString().split('T')[0]; // YYYY-MM-DD
        const logFile = path.join(__dirname, 'logs', `tracks_${today}.json`);

        // Ensure logs directory exists
        const logsDir = path.join(__dirname, 'logs');
        if (!fs.existsSync(logsDir)) {
            fs.mkdirSync(logsDir, { recursive: true });
        }

        // Read existing file or create new array
        let dailyLogs = [];
        try {
            if (fs.existsSync(logFile)) {
                dailyLogs = JSON.parse(fs.readFileSync(logFile, 'utf8'));
            }
        } catch (e) { /* ignore parse errors, start fresh */ }

        // Append new log
        dailyLogs.push(data);
        fs.writeFileSync(logFile, JSON.stringify(dailyLogs, null, 2));
        console.log(`[TRACKLOG] Saved to ${logFile}`);

        // Broadcast to all viewers
        io.emit('newTrackLog', data);
    });

    // Get Track Log History
    socket.on('getTrackLogHistory', (callback) => {
        callback(TRACK_LOGS.slice(0, 50));
    });

    // Update track log with video path (from dashboard recording)
    socket.on('updateTrackLogVideo', ({ deviceName, trackId, video_path }) => {
        // Find matching track log and update it
        const trackLog = TRACK_LOGS.find(log =>
            log.sentinel_name === deviceName && log.track_id === trackId
        );
        if (trackLog) {
            trackLog.video_path = video_path;
            console.log(`[VIDEO] Linked ${video_path} to ${deviceName} track #${trackId}`);
            // Broadcast updated log to refresh UI
            io.emit('trackLogUpdated', trackLog);
        }
    });

    // 1. Get Router RTP Capabilities
    socket.on('getRouterRtpCapabilities', (callback) => {
        callback(router.rtpCapabilities);
    });

    // 2. Create Transport
    // isProducer: boolean (true for Pi, false for Viewers)
    socket.on('createTransport', async ({ sender }, callback) => {
        try {
            const transport = await createWebRtcTransport(callback);
            // Store transport on socket or separate map if needed
            socket.transport = transport;
            // Simplified: One transport per socket for this demo

            callback({
                id: transport.id,
                iceParameters: transport.iceParameters,
                iceCandidates: transport.iceCandidates,
                dtlsParameters: transport.dtlsParameters,
            });
        } catch (error) {
            console.error('createTransport error:', error);
            callback({ error: error.message });
        }
    });

    // 3. Connect Transport (DTLS)
    socket.on('connectTransport', async ({ dtlsParameters }, callback) => {
        try {
            await socket.transport.connect({ dtlsParameters });
            callback({});
        } catch (error) {
            console.error('connectTransport error:', error);
            callback({ error: error.message });
        }
    });

    // 4. Produce (Published by Pi)
    socket.on('produce', async ({ kind, rtpParameters, appData }, callback) => {
        try {
            // In this Grid version, we allow multiple producers!
            // But for simplicity of the prompt "2x2 grid", we assume this socket 
            // is just ONE producer. If a new socket comes, it's a new producer.

            const newProducer = await socket.transport.produce({ kind, rtpParameters, appData });

            const producerData = {
                socketId: socket.id,
                producer: newProducer,
                deviceName: (appData && appData.deviceName) ? appData.deviceName : `Camera ${socket.id.substr(0, 4)}`
            };

            // Store globally (or better, in a Map)
            // For this simple example we use a list or map.
            // reusing 'producer' variable is bad if we want multiple.
            // Let's perform a quick refactor to support multiple producers globally.
            PRODUCERS.set(newProducer.id, producerData);

            console.log(`Producer created: ${newProducer.id}, kind: ${newProducer.kind}, name: ${producerData.deviceName}`);

            newProducer.on('transportclose', () => {
                console.log('Producer transport closed');
                PRODUCERS.delete(newProducer.id);
                socket.broadcast.emit('producer-closed', { producerId: newProducer.id });
            });
            newProducer.on('close', () => {
                console.log('Producer closed');
                PRODUCERS.delete(newProducer.id);
                socket.broadcast.emit('producer-closed', { producerId: newProducer.id });
            });

            callback({ id: newProducer.id });

            // Notify all other clients that a new stream is available
            socket.broadcast.emit('newProducer', {
                producerId: newProducer.id,
                deviceName: producerData.deviceName
            });

        } catch (error) {
            console.error('produce error:', error);
            callback({ error: error.message });
        }
    });

    socket.on('getProducers', (callback) => {
        // Return list of active producers
        const list = [];
        for (const [id, prod] of PRODUCERS) {
            list.push({ producerId: id, deviceName: prod.deviceName });
        }
        callback(list);
    });

    // 5. Consume (Requested by Web Client)
    socket.on('consume', async ({ producerId, rtpCapabilities }, callback) => {
        try {
            const producerData = PRODUCERS.get(producerId);
            if (!producerData) {
                return callback({ error: 'No producer active' });
            }

            if (!router.canConsume({
                producerId: producerData.producer.id,
                rtpCapabilities
            })) {
                return callback({ error: 'Cannot consume' });
            }

            const consumer = await socket.transport.consume({
                producerId: producerData.producer.id,
                rtpCapabilities,
                paused: true, // Standard: Start paused, wait for client to be ready
            });

            consumers.push({
                socketId: socket.id,
                consumer: consumer
            });

            console.log(`Consumer created: ${consumer.id}`);

            consumer.on('transportclose', () => {
                console.log('Consumer transport closed');
            });
            consumer.on('producerclose', () => {
                console.log('Producer of consumer closed');
                socket.emit('producer-closed', { producerId: producerData.producer.id });
                consumer.close();
                consumers = consumers.filter(c => c.consumer.id !== consumer.id);
            });

            callback({
                id: consumer.id,
                producerId: producerData.producer.id,
                kind: consumer.kind,
                rtpParameters: consumer.rtpParameters
            });

            // Notify all clients of updated viewer count
            broadcastStats();

        } catch (error) {
            console.error('consume error:', error);
            callback({ error: error.message });
        }
    });

    // 6. Resume Consumer (Client is ready)
    socket.on('resumeConsumer', async ({ consumerId }, callback) => {
        const entry = consumers.find(c => c.consumer.id === consumerId);
        if (entry) {
            await entry.consumer.resume();
            await entry.consumer.requestKeyFrame();
            callback();
        } else {
            callback({ error: 'Consumer not found' });
        }
    });
});

async function createWebRtcTransport(callback) {
    const transport = await router.createWebRtcTransport(config.mediasoup.webRtcTransport);

    transport.on('dtlsstatechange', (dtlsState) => {
        if (dtlsState === 'closed') {
            transport.close();
        }
    });

    return transport;
}
