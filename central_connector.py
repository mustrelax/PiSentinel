import asyncio
import socketio
import logging
import json
import time
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.sdp import candidate_from_sdp, candidate_to_sdp
import config

logger = logging.getLogger("CentralConnector")

class CentralConnector:
    def __init__(self, frame_source):
        self.sio = socketio.AsyncClient(reconnection=False)
        self.frame_source = frame_source
        self.pc = None
        self.producer_id = None
        self.transport_id = None
        self.connected = False
        self.stop_signal = asyncio.Event()
        self.retry_delay = 5
        self.loop = None  # Store event loop reference for thread-safe calls

        # Callback for peer geolocation updates relayed by the server.
        # Signature: (peer_device, peer_track_id, world_dict, uncertainty_m, azimuth_world, elevation, range_m, track_state, timestamp)
        self.on_peer_geolocation = None

        # Register inbound Socket.IO handler up-front so reconnects don't lose it.
        self.sio.on('geolocationUpdate', self._on_geolocation_update)

    async def start(self):
        """Main loop with automatic reconnection."""
        # Store event loop reference for thread-safe calls
        self.loop = asyncio.get_running_loop()
        
        while not self.stop_signal.is_set():
            try:
                logger.info(f"Connecting to Central Server at {config.CENTRAL_SERVER_URL}...")
                await self.sio.connect(config.CENTRAL_SERVER_URL)
                self.connected = True
                logger.info("Socket.IO Connected. Setting up Mediasoup transport...")
                
                await self._setup_transport()
                
                # Keep alive until disconnect
                await self.sio.wait()
                
            except asyncio.CancelledError:
                logger.info("Stopping Central Connector (Cancelled)...")
                break
            except Exception as e:
                logger.warning(f"Connection lost or failed: {e}")
            finally:
                await self._cleanup()
            if not self.stop_signal.is_set():
                logger.info(f"Retrying connection in {self.retry_delay} seconds...")
                await asyncio.sleep(self.retry_delay)

    async def _cleanup(self):
        """Cleanup resources before reconnection attempt."""
        self.connected = False
        if self.pc:
            try:
                await self.pc.close()
            except:
                pass
            self.pc = None
        if self.sio.connected:
            try:
                await self.sio.disconnect()
            except:
                pass

    async def stop(self):
        self.stop_signal.set()
        await self._cleanup()

    def send_alert(self, track_id):
        """Send drone detection alert to server (thread-safe)."""
        if not self.sio.connected or not self.loop:
            return
        
        # Thread-safe: schedule on the stored event loop
        try:
            asyncio.run_coroutine_threadsafe(self._emit_alert(track_id), self.loop)
        except Exception as e:
            logger.warning(f"Could not schedule alert: {e}")
    
    async def _emit_alert(self, track_id):
        try:
            await self.sio.emit('droneAlert', {
                'deviceName': config.DEVICE_NAME,
                'trackId': track_id,
                'timestamp': time.time()
            })
            logger.info(f"[ALERT] Sent drone alert: Track {track_id}")
        except Exception as e:
            logger.warning(f"Failed to send alert: {e}")

    def send_tracking_status(self, track_id, status):
        """Send tracking status update (ACTIVE/LOST) to server."""
        if not self.sio.connected or not self.loop:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self._emit_tracking_status(track_id, status), 
                self.loop
            )
        except Exception as e:
            logger.warning(f"Could not send tracking status: {e}")
    
    async def _emit_tracking_status(self, track_id, status):
        try:
            await self.sio.emit('trackingStatus', {
                'deviceName': config.DEVICE_NAME,
                'trackId': track_id,
                'status': status  # 'LOCKED', 'ACTIVE', or 'LOST'
            })
        except Exception as e:
            logger.warning(f"Failed to send tracking status: {e}")

    def send_radar_telemetry(self, track_id, azimuth, distance, speed):
        """Send radar telemetry (azimuth, dist, speed) to server."""
        if not self.sio.connected or not self.loop:
            return
        
        # Don't flood the queue, maybe rate limit in controller?
        try:
            asyncio.run_coroutine_threadsafe(
                self._emit_telemetry(track_id, azimuth, distance, speed), 
                self.loop
            )
        except Exception as e:
            pass # Silent fail for telemetry to avoid log spam
            
    async def _emit_telemetry(self, track_id, azimuth, distance, speed):
        try:
            await self.sio.emit('radarUpdate', {
                'deviceName': config.DEVICE_NAME,
                'trackId': track_id,
                'azimuth': float(azimuth),
                'distance': float(distance),
                'speed': float(speed)
            })
        except:
            pass

    def send_unit_health(self, health):
        """Emit a `unitHealth` snapshot (called ~1 Hz from the tracking thread).

        health: dict produced by Controller.get_health_snapshot().
        """
        if not self.sio.connected or not self.loop:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self._emit_unit_health(health), self.loop
            )
        except Exception:
            pass

    async def _emit_unit_health(self, health):
        try:
            await self.sio.emit('unitHealth', {
                'deviceName': config.DEVICE_NAME,
                'ts': time.time(),
                'uptime_s': float(health.get('uptime_s', 0.0)),
                'fps_30s': float(health.get('fps_30s', 0.0)),
                'detect_per_s_30s': float(health.get('detect_per_s_30s', 0.0)),
                'active_track_id': health.get('active_track_id'),
                'active_track_sigma': health.get('active_track_sigma'),
                'passive_track_count': int(health.get('passive_track_count', 0)),
                'total_frames': int(health.get('total_frames', 0)),
                'cpu_temp_c': health.get('cpu_temp_c'),
            })
        except Exception:
            pass

    def send_unit_telemetry(self, pan_angle, tilt_angle, has_active_track):
        """Periodic unit telemetry — fires even when no drone is tracked.
        Drives the live look-direction wedge on the command-center map.
        """
        if not self.sio.connected or not self.loop:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self._emit_unit_telemetry(pan_angle, tilt_angle, has_active_track),
                self.loop
            )
        except Exception:
            pass

    async def _emit_unit_telemetry(self, pan_angle, tilt_angle, has_active_track):
        try:
            servo_offset = 90.0 - pan_angle
            look_bearing = (config.CAMERA_FACING_DIRECTION + servo_offset) % 360.0
            await self.sio.emit('unitTelemetry', {
                'deviceName': config.DEVICE_NAME,
                'timestamp': time.time(),
                'position': dict(config.SYSTEM_POSITION_ENU),
                'camera_facing': float(config.CAMERA_FACING_DIRECTION),
                'pan_angle': float(pan_angle),
                'tilt_angle': float(tilt_angle),
                'look_bearing': float(look_bearing),
                'state': 'tracking' if has_active_track else 'scanning',
            })
        except Exception:
            pass

    def send_geolocation(self, track_id, world_estimate):
        """Send world-frame geolocation for the active track (thread-safe).

        world_estimate is a WorldEstimate dataclass from tracking.geolocation
        (or anything exposing an as_dict() method).
        """
        if not self.sio.connected or not self.loop:
            return
        try:
            payload = world_estimate.as_dict() if hasattr(world_estimate, 'as_dict') else dict(world_estimate)
            asyncio.run_coroutine_threadsafe(
                self._emit_geolocation(track_id, payload),
                self.loop
            )
        except Exception:
            pass  # silent fail; this event is fire-and-forget

    async def _emit_geolocation(self, track_id, payload):
        try:
            await self.sio.emit('geolocationUpdate', {
                'deviceName': config.DEVICE_NAME,
                'trackId': track_id,
                'timestamp': time.time(),
                'world': {
                    'x': payload.get('x', 0.0),
                    'y': payload.get('y', 0.0),
                    'z': payload.get('z', 0.0),
                },
                'uncertainty_m': payload.get('uncertainty_m', 0.0),
                'azimuth_world': payload.get('azimuth_world', 0.0),
                'elevation': payload.get('elevation', 0.0),
                'range_m': payload.get('range_m', 0.0),
                'track_state': payload.get('track_state', 'LOCKED'),
                # Unit metadata (Phase 1: ship it inline; Phase 2 moves to server registry).
                'unit': {
                    'position': dict(config.SYSTEM_POSITION_ENU),
                    'camera_facing': float(config.CAMERA_FACING_DIRECTION),
                },
            })
        except Exception:
            pass

    async def _on_geolocation_update(self, data):
        """Socket.IO handler: server relayed a geolocationUpdate from any unit."""
        try:
            peer_device = data.get('deviceName')
            if not peer_device or peer_device == config.DEVICE_NAME:
                return  # ignore our own echo
            if self.on_peer_geolocation is None:
                return
            world = data.get('world') or {}
            self.on_peer_geolocation(
                peer_device,
                data.get('trackId'),
                world,
                data.get('uncertainty_m', 0.0),
                data.get('azimuth_world'),
                data.get('elevation'),
                data.get('range_m'),
                data.get('track_state', 'LOCKED'),
                data.get('timestamp'),
            )
        except Exception as e:
            logger.warning(f"Peer geolocation handler failed: {e}")

    def send_track_log(self, log_summary: dict):
        """Send complete track log when tracking ends (thread-safe)."""
        if not self.sio.connected or not self.loop:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self._emit_track_log(log_summary), 
                self.loop
            )
        except Exception as e:
            logger.warning(f"Could not send track log: {e}")
    
    async def _emit_track_log(self, log_summary: dict):
        try:
            await self.sio.emit('trackLogComplete', log_summary)
            logger.info(f"[TRACKLOG] Sent log for Track #{log_summary.get('track_id')}")
        except Exception as e:
            logger.warning(f"Failed to send track log: {e}")

    async def _setup_transport(self):

        response = await self.sio.call('createTransport', {'sender': True})
        if 'error' in response:
            logger.error(f"createTransport failed: {response['error']}")
            return

        self.transport_id = response['id']
        ice_params = response['iceParameters']
        ice_candidates = response['iceCandidates']
        dtls_params = response['dtlsParameters']

        logger.info(f"Transport created: {self.transport_id}")

        self.pc = RTCPeerConnection()
        
        @self.pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            logger.info(f"ICE Connection State changed to: {self.pc.iceConnectionState}")
            if self.pc.iceConnectionState == "failed":
                logger.error("ICE Connection FAILED. Check network/firewall.")

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection State: {self.pc.connectionState}")
                
        from webrtc_utils import WebRTCVideoTrack
        from aiortc import RTCRtpSender

        track = WebRTCVideoTrack(self.frame_source, target_fps=config.TARGET_FPS)
        
        # Use VP8 (H264 has profile matching issues with Mediasoup)
        transceiver = self.pc.addTransceiver(track, direction='sendonly')
        try:
            capabilities = RTCRtpSender.getCapabilities('video')
            preferences = [c for c in capabilities.codecs if c.name == 'VP8']
            if preferences:
                transceiver.setCodecPreferences(preferences)
                logger.info("Using VP8 codec")
        except Exception as e:
            logger.warning(f"Could not set codec preferences: {e}")

        offer = await self.pc.createOffer()

        remote_sdp = self._generate_remote_sdp(offer.sdp, ice_params, ice_candidates, dtls_params)
        
        await self.pc.setLocalDescription(offer)
        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=remote_sdp, type='answer'))

        local_dtls = self._extract_dtls_from_sdp(offer.sdp)
        
        resp = await self.sio.call('connectTransport', {'dtlsParameters': local_dtls})
        if resp.get('error'):
             logger.error(f"connectTransport failed: {resp['error']}")
             return

        local_rtp = self._extract_rtp_from_sdp(offer.sdp)
        
        produce_resp = await self.sio.call('produce', {
            'kind': 'video', 
            'rtpParameters': local_rtp,
            'appData': {'deviceName': config.DEVICE_NAME}
        })
        
        if produce_resp.get('error'):
            logger.error(f"Produce failed: {produce_resp['error']}")
            return

        self.producer_id = produce_resp['id']
        logger.info(f"Producing successfully! ID: {self.producer_id}")

        # Start periodic sender stats logger
        asyncio.create_task(self._log_sender_stats(transceiver.sender))

    async def _log_sender_stats(self, sender):
        """Periodically log what local RTCPeerConnection thinks it is sending."""
        while True:
            await asyncio.sleep(2)
            try:
                stats = await sender.getStats()
                # We expect an 'outbound-rtp' report
                outbound = next((s for s in stats.values() if s.type == 'outbound-rtp'), None)
                if outbound:
                    logger.info(f"[SENDER STATS] Pkts Sent: {outbound.packetsSent} | Bytes Sent: {outbound.bytesSent}")
                else:
                    logger.warning("[SENDER STATS] No outbound-rtp stats found yet.")
                    
                    # Debug: Print connection state
                    logger.info(f"[SENDER DEBUG] ICE: {self.pc.iceConnectionState} | Signalling: {self.pc.signalingState}")

            except Exception as e:
                logger.error(f"Error reading sender stats: {e}")


    def _generate_remote_sdp(self, local_sdp, ice, candidates, dtls):
        import re
        
        # Extract codec info from local offer (H264 or VP8)
        codec_name = 'VP8'
        codec_pt = 96
        
        # Try H264 first
        h264_match = re.search(r'a=rtpmap:(\d+) H264/90000', local_sdp)
        if h264_match:
            codec_pt = int(h264_match.group(1))
            codec_name = 'H264'
        else:
            # Fallback to VP8
            vp8_match = re.search(r'a=rtpmap:(\d+) VP8/90000', local_sdp)
            if vp8_match:
                codec_pt = int(vp8_match.group(1))
                codec_name = 'VP8'
            
        # Extract MID
        mid = "0"
        mid_match = re.search(r'a=mid:(\S+)', local_sdp)
        if mid_match:
            mid = mid_match.group(1)
            
        sdp_lines = []
        sdp_lines.append("v=0")
        sdp_lines.append("o=- 10000 10000 IN IP4 0.0.0.0")
        sdp_lines.append("s=-")
        sdp_lines.append("t=0 0")
        if ice.get('iceLite'):
            sdp_lines.append("a=ice-lite")
        sdp_lines.append("a=group:BUNDLE " + mid)
        sdp_lines.append("a=msid-semantic: WMS")
        
        sdp_lines.append(f"m=video 9 UDP/TLS/RTP/SAVPF {codec_pt}") 
        sdp_lines.append("c=IN IP4 0.0.0.0")
        sdp_lines.append("a=recvonly")
        
        # ICE
        sdp_lines.append(f"a=ice-ufrag:{ice.get('usernameFragment')}")
        sdp_lines.append(f"a=ice-pwd:{ice['password']}")
        
        for cand in candidates:
            component = cand.get('component', 1) 
            line = f"a=candidate:{cand['foundation']} {component} {cand['protocol']} {cand['priority']} {cand['ip']} {cand['port']} typ {cand['type']}"
            if cand.get('tcpType'):
                line += f" tcptype {cand['tcpType']}"
            sdp_lines.append(line)

        # DTLS fingerprint (use sha-256)
        fp = next((f for f in dtls['fingerprints'] if f['algorithm'] == 'sha-256'), dtls['fingerprints'][0])
        sdp_lines.append(f"a=fingerprint:{fp['algorithm']} {fp['value']}")
        sdp_lines.append("a=setup:passive")
        
        sdp_lines.append(f"a=mid:{mid}")
        sdp_lines.append("a=rtcp-mux")
        sdp_lines.append("a=rtcp-rsize")
        
        # Codec specific
        sdp_lines.append(f"a=rtpmap:{codec_pt} {codec_name}/90000")
        if codec_name == 'H264':
            # H264 specific params
            sdp_lines.append(f"a=fmtp:{codec_pt} level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f")
        sdp_lines.append(f"a=rtcp-fb:{codec_pt} goog-remb")
        sdp_lines.append(f"a=rtcp-fb:{codec_pt} transport-cc")
        sdp_lines.append(f"a=rtcp-fb:{codec_pt} ccm fir")
        sdp_lines.append(f"a=rtcp-fb:{codec_pt} nack")
        sdp_lines.append(f"a=rtcp-fb:{codec_pt} nack pli")
        
        return "\r\n".join(sdp_lines) + "\r\n"

    def _extract_dtls_from_sdp(self, sdp):
        import re
        # Extract ALL fingerprints from offer SDP
        matches = re.findall(r'a=fingerprint:(\S+) (\S+)', sdp)
        if matches:
            fingerprints = [{'algorithm': m[0], 'value': m[1]} for m in matches]
            return {
                'role': 'client',  # We are DTLS client
                'fingerprints': fingerprints
            }
        return {}

    def _extract_rtp_from_sdp(self, sdp):
        """
        Robustly parses SDP to extract RTP parameters for Mediasoup.
        Handles rtpmap, fmtp, rtcp-fb, extmap, and ssrc/cname/mid.
        """
        media_section = []
        in_video = False
        
        # 1. Extract Video Media Section
        for line in sdp.splitlines():
            if line.startswith('m=video'):
                in_video = True
                media_section.append(line)
            elif line.startswith('m='):
                in_video = False
            elif in_video:
                media_section.append(line)
                
        # 2. Parse Attributes
        rtp_map = {}   # pt -> codec_name
        fmtp_map = {}  # pt -> {param: val}
        rtcp_fb = {}   # pt -> [{type:..., parameter:...}]
        ext_map = []   # [{uri:..., id:...}]
        ssrcs = set()
        cname = None
        mid = None
        
        # Default VP8 Payload Type (will be overwritten by rtpmap)
        target_pt = 96
        
        import re
        
        for line in media_section:
            # a=rtpmap:96 VP8/90000
            m_rtp = re.match(r'a=rtpmap:(\d+) (\w+)/(\d+)', line)
            if m_rtp:
                pt = int(m_rtp.group(1))
                name = m_rtp.group(2)
                rtp_map[pt] = name
                # Prefer H264, fallback to VP8
                if name.upper() == 'H264':
                    target_pt = pt
                elif name.upper() == 'VP8' and target_pt == 96:
                    target_pt = pt
                    
            # a=fmtp:96 profile-id=1;sprop-parameter-sets=...
            m_fmtp = re.match(r'a=fmtp:(\d+) (.+)', line)
            if m_fmtp:
                pt = int(m_fmtp.group(1))
                params_str = m_fmtp.group(2)
                params = {}
                for p in params_str.split(';'):
                    if '=' in p:
                        k, v = p.strip().split('=', 1)
                        params[k] = v
                fmtp_map[pt] = params
            
            # a=rtcp-fb:96 nack pli
            m_fb = re.match(r'a=rtcp-fb:(\d+) (.+)', line)
            if m_fb:
                pt = int(m_fb.group(1))
                parts = m_fb.group(2).split()
                fb_type = parts[0]
                fb_param = parts[1] if len(parts) > 1 else ''
                if pt not in rtcp_fb:
                    rtcp_fb[pt] = []
                rtcp_fb[pt].append({'type': fb_type, 'parameter': fb_param})
                
            # a=extmap:3 urn:ietf:params:rtp-hdrext:sdes:mid
            m_ext = re.match(r'a=extmap:(\d+) (\S+)', line)
            if m_ext:
                ext_map.append({'id': int(m_ext.group(1)), 'uri': m_ext.group(2)})
                
            # a=ssrc:1938481 cname:xyz
            m_ssrc = re.match(r'a=ssrc:(\d+)(.*)', line)
            if m_ssrc:
                ssrc_val = int(m_ssrc.group(1))
                ssrcs.add(ssrc_val)
                rest = m_ssrc.group(2)
                if 'cname:' in rest:
                    # simplistic cname extraction
                    cname_match = re.search(r'cname:(\S+)', rest)
                    if cname_match:
                        cname = cname_match.group(1)
            
            # a=mid:0
            m_mid = re.match(r'a=mid:(\S+)', line)
            if m_mid:
                mid = m_mid.group(1)
                
        # 3. Construct Codecs List
        # We only really care about sending the Target Codec (VP8) info
        codecs = []
        if target_pt in rtp_map:
            codec_entry = {
                'mimeType': f"video/{rtp_map[target_pt]}",
                'payloadType': target_pt,
                'clockRate': 90000,
                'parameters': fmtp_map.get(target_pt, {}),
                'rtcpFeedback': rtcp_fb.get(target_pt, [])
            }
            codecs.append(codec_entry)
            
            # If there's an RTX associated (apt=target_pt), add it?
            # Complexity: AIORTC usually doesn't do RTX unless configured.
            # Keeping it simple as per instructions.
            
        else:
            # Fallback if parsing failed somehow
            logger.warning(f"Could not find VP8 in SDP! Falling back to default.")
            codecs.append({
                'mimeType': 'video/VP8',
                'payloadType': 96,
                'clockRate': 90000,
                'parameters': {},
                'rtcpFeedback': [{'type': 'nack'}, {'type': 'nack', 'parameter': 'pli'}]
            })
            
        # 4. Construct UDP Encodings
        # Pick the first SSRC found (usually Video SSRC)
        # Note: In Chrome/Mediasoup, if we don't send ssrc, it might fail to map.
        main_ssrc = list(ssrcs)[0] if ssrcs else 11111111
        
        encodings = [
            {
                'ssrc': main_ssrc,
                'dtx': False
            }
        ]
        
        return {
            'codecs': codecs,
            'encodings': encodings,
            'headerExtensions': ext_map,
            'rtcp': {
                'cname': cname if cname else "cname-default",
                'reducedSize': True
            },
            'mid': mid if mid else "0"
        }

