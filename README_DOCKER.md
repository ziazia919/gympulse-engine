# GymPulse Docker WebUI

Start the WebUI:

```powershell
docker compose up --build -d
```

Open:

http://localhost:8080

The dashboard can run the bundled `field_dispatch_dataa.csv`, accept a CSV upload,
and download prediction results as JSON or CSV.

Stop it with:

```powershell
docker compose down
```

When using Docker Desktop's image **Run** action instead of Compose, map host port
`8080` to container port `8080`. Docker Desktop then shows a clickable `8080:8080`
port link for the dashboard.
