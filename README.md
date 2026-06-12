# GymPulse Engine

GymPulse is a MindSpore-based dispatch prediction engine with a browser WebUI.
It supports a bundled CSV, custom CSV uploads, and JSON/CSV result downloads.

## Docker

```bash
docker run --rm -p 8080:8080 030220046518/gympulse-engine:latest
```

Open [http://localhost:8080](http://localhost:8080).

To build locally:

```bash
docker compose up --build -d
```

## Windows WebUI

Install Python 3.11 64-bit, then run:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1
```

The distributable application is created in `dist\GymPulseWebUI`. Transfer the
whole folder. Double-click `GymPulseWebUI.exe` to start the local server and open
the default browser automatically.

## Input

The default dataset is `field_dispatch_dataa.csv`. The WebUI also accepts CSV
uploads with the same schema.

## Endpoints

- `/` - Web dashboard
- `/health` - Health check
- `/api/predictions` - GET for bundled data, POST with a `file` upload
