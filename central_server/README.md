# PiSentinel Central Server Setup (Windows)

This directory contains the central server for the PiSentinel system. It receives the video stream from your Raspberry Pi and broadcasts it to web viewers using Mediasoup.

## Prerequisites

1.  **Node.js**: Download and install Node.js (v16 or higher) from [nodejs.org](https://nodejs.org/).
2.  **Build Tools**: Mediasoup requires compilation of C++ modules.
    -   Open a PowerShell as Administrator.
    -   Run: `npm install --global --production windows-build-tools`
    -   *Alternative:* Install Visual Studio with "Desktop development with C++" workload.
3.  **Python**: Ensure Python is installed and in your PATH (usually handled by the step above or your existing setup).

## Installation

1.  Open Command Prompt (cmd) or PowerShell.
2.  Navigate to this folder:
    ```cmd
    cd central_server
    ```
3.  Install dependencies:
    ```cmd
    npm install
    ```
    *Note: This might take a few minutes as it compiles Mediasoup.*

## Configuration

The default configuration is in `config.js`.
-   **Port**: 3000
-   **IP**: 0.0.0.0 (Listen on all interfaces)
-   **Codecs**: VP8, H264 supported.

## Running the Server

1.  Start the server:
    ```cmd
    npm start
    ```
2.  Allow Node.js through the Windows Firewall if prompted.
3.  You should see: `Server running at http://0.0.0.0:3000`

## Viewing the Stream

1.  Open your browser (Chrome/Edge/Firefox).
2.  Go to `http://localhost:3000` (or your laptop's IP address from other devices).
3.  Login with:
    -   Username: `admin`
    -   Password: `admin`
4.  Click "Connect" to watch.

## Connecting the Raspberry Pi

The Pi code has been updated to connect to this server automatically.
Ensure the Pi and this Laptop are on the same Wi-Fi/Network.
You may need to update the `CENTRAL_SERVER_URL` in the Pi's `config.py` to point to this Laptop's IP address (e.g., `http://192.168.1.50:3000`).
