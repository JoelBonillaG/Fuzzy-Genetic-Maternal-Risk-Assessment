# Despliegue continuo en VPS

Este proyecto queda preparado para validar y desplegar automaticamente cuando haya cambios en `main`.

## Flujo

- En cada pull request hacia `main`, GitHub Actions instala dependencias, valida el backend y construye el frontend.
- En cada push a `main`, despues de validar, GitHub Actions entra por SSH al VPS, actualiza el repositorio, instala dependencias, construye el frontend y reinicia los servicios configurados.

## Preparar el VPS

Clona el repositorio en una ruta fija. Para este VPS se usara:

```bash
sudo mkdir -p /home/python-project
sudo chown -R deploy:deploy /home/python-project
git clone git@github.com:JoelBonillaG/Fuzzy-Genetic-Maternal-Risk-Assessment.git /home/python-project
```

Crea `backend/.env` en el VPS:

```bash
APP_HOST=127.0.0.1
APP_PORT=8000
APP_RELOAD=false
APP_ENVIRONMENT=production
```

Servicio sugerido para el backend en `/etc/systemd/system/riesgo-materno-backend.service`:

```ini
[Unit]
Description=API riesgo materno
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/python-project/backend
EnvironmentFile=/home/python-project/backend/.env
ExecStart=/home/python-project/backend/.venv/bin/python -m src.app.run
Restart=always
RestartSec=5
User=deploy

[Install]
WantedBy=multi-user.target
```

Recarga systemd y dejalo habilitado. El primer `start` conviene hacerlo despues de ejecutar el primer despliegue, porque ahi se crea `backend/.venv`.

```bash
sudo systemctl daemon-reload
sudo systemctl enable riesgo-materno-backend
```

Configuracion sugerida de Nginx:

```nginx
server {
    listen 80 default_server;
    server_name _;
    return 444;
}

server {
    listen 80;
    server_name med-maternal-risk.stratiumhub.com;

    root /home/python-project/frontend/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Secrets de GitHub

En GitHub entra a `Settings > Secrets and variables > Actions > New repository secret` y configura:

| Secret | Valor |
| --- | --- |
| `VPS_HOST` | IP o dominio del VPS |
| `VPS_USER` | Usuario SSH que despliega, por ejemplo `deploy` |
| `VPS_PORT` | Puerto SSH, normalmente `22` |
| `VPS_SSH_PRIVATE_KEY` | Llave privada SSH con acceso al VPS |
| `VPS_APP_DIR` | `/home/python-project` |
| `VPS_NODE_VERSION` | Opcional, version de Node para `nvm`, por ejemplo `20` |
| `VPS_BACKEND_RESTART_COMMAND` | `sudo systemctl restart riesgo-materno-backend` |
| `VPS_FRONTEND_RESTART_COMMAND` | `sudo systemctl reload nginx` |

Si usas `sudo` en los comandos de reinicio, permite esos comandos sin password para el usuario de despliegue:

```bash
sudo visudo
```

Ejemplo:

```text
deploy ALL=(ALL) NOPASSWD: /bin/systemctl restart riesgo-materno-backend, /bin/systemctl reload nginx
```

## Primera ejecucion

Antes del primer despliegue automatico, entra al VPS y ejecuta una vez:

```bash
cd /home/python-project
APP_DIR=/home/python-project \
BACKEND_RESTART_COMMAND="sudo systemctl restart riesgo-materno-backend" \
FRONTEND_RESTART_COMMAND="sudo systemctl reload nginx" \
bash scripts/deploy-vps.sh
```

Luego, cada cambio que entre a `main` desplegara automaticamente.

Si todavia no habias iniciado el servicio del backend:

```bash
sudo systemctl start riesgo-materno-backend
```
