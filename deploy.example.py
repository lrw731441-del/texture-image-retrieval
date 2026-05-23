"""One-click deployment script — uploads project to cloud server and starts the service.

Usage:
  1. Copy this file to deploy.py:  cp deploy.example.py deploy.py
  2. Edit SERVER_IP and SERVER_USER below to match your server
  3. Run:  set ROOT_PWD=<password> && python deploy.py
"""

import os
import sys
import tarfile
from pathlib import Path

import paramiko
from scp import SCPClient

PROJECT_DIR = Path(__file__).resolve().parent
ARCHIVE_NAME = "texture-search.tar.gz"
ARCHIVE_PATH = PROJECT_DIR / ARCHIVE_NAME
APP_NAME = "texture-search"

# ══════════════════════════════════════════════════════════════════════════════
# EDIT THESE — replace with your own server info
# 修改这里 — 替换成你自己的服务器信息
# ══════════════════════════════════════════════════════════════════════════════
SERVER_IP   = "<YOUR_SERVER_IP>"       # e.g. "1.2.3.4"
SERVER_USER = "<YOUR_USERNAME>"        # e.g. "root" or "ubuntu"
SERVER_PORT = 22
REMOTE_DIR  = "/home/<YOUR_USERNAME>/texture-search"
# ══════════════════════════════════════════════════════════════════════════════

IMAGE_TAG = "texture-search:v1"


def create_archive():
    """Create tar.gz of project files needed for Docker build."""
    include_ext = {".py", ".txt", ".html", ".jpg", ".jpeg", ".png",
                   ".bmp", ".webp", ".tiff", ".tif", ".npy", ".json"}
    include_files = {"Dockerfile", ".dockerignore", "requirements.txt"}
    exclude_dirs = {".git", "__pycache__", ".claude", "venv", ".venv", "env"}

    print("[1/5] Creating project archive ...")
    with tarfile.open(ARCHIVE_PATH, "w:gz") as tar:
        for root, dirs, files in os.walk(PROJECT_DIR):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in include_ext or f in include_files:
                    filepath = os.path.join(root, f)
                    arcname = os.path.relpath(filepath, PROJECT_DIR)
                    tar.add(filepath, arcname=arcname)
    size_mb = os.path.getsize(ARCHIVE_PATH) / (1024 * 1024)
    print(f"  Done. Archive: {size_mb:.1f} MB")
    return True


def connect_ssh():
    """Connect to server, return SSH client."""
    password = os.environ.get("ROOT_PWD", "")
    if not password:
        print("  ERROR: Set ROOT_PWD environment variable first.")
        print("         CMD:       set ROOT_PWD=<password>")
        print("         PowerShell: $env:ROOT_PWD='<password>'")
        sys.exit(1)
    print(f"\n  Connecting to {SERVER_USER}@{SERVER_IP} ...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(SERVER_IP, port=SERVER_PORT,
                       username=SERVER_USER, password=password,
                       look_for_keys=False, allow_agent=False,
                       timeout=15)
        print("  Connected to server.")
        return client
    except paramiko.AuthenticationException:
        print("  ERROR: Wrong password.")
        sys.exit(1)
    except Exception as e:
        print(f"  ERROR: Cannot connect — {e}")
        sys.exit(1)


def upload_archive(ssh_client):
    """SCP the archive to the server."""
    print(f"\n[2/5] Uploading archive to {SERVER_IP} ... (may take 1-2 minutes)")

    _stdin, stdout, _stderr = ssh_client.exec_command(f"mkdir -p {REMOTE_DIR}")
    stdout.read()

    transport = ssh_client.get_transport()
    if transport is None:
        print("  ERROR: No SSH transport")
        sys.exit(1)

    with SCPClient(transport) as scp:
        scp.put(str(ARCHIVE_PATH), f"{REMOTE_DIR}/{ARCHIVE_NAME}")

    _stdin, stdout, _stderr = ssh_client.exec_command(
        f"ls -lh {REMOTE_DIR}/{ARCHIVE_NAME}"
    )
    print(f"  Uploaded: {stdout.read().decode().strip()}")


def build_on_server(ssh_client):
    """Extract archive and build Docker image on the server."""
    print(f"\n[3/5] Building Docker image on server ... (may take 3-5 minutes)")
    commands = f"""
set -e
mkdir -p {REMOTE_DIR}
cd {REMOTE_DIR}
tar -xzf {ARCHIVE_NAME}
echo "  Archive extracted."

sudo docker stop {APP_NAME} 2>/dev/null || true
sudo docker rm {APP_NAME} 2>/dev/null || true

echo "  Building image ..."
sudo docker build -t {IMAGE_TAG} .

echo "  Build complete."
"""
    _stdin, stdout, stderr = ssh_client.exec_command(commands)
    out = stdout.read().decode()
    err = stderr.read().decode()
    if err:
        print(f"  {err[-500:] if len(err) > 500 else err}")
    if "Build complete" in out or "Successfully tagged" in err:
        print("  Build completed successfully.")
    elif out:
        print(out[-300:])


def start_container(ssh_client):
    """Run the container."""
    print(f"\n[4/5] Starting container ...")
    commands = f"""
set -e
sudo docker stop {APP_NAME} 2>/dev/null || true
sudo docker rm {APP_NAME} 2>/dev/null || true

sudo docker run -d \\
    --name {APP_NAME} \\
    --restart unless-stopped \\
    -p 80:8000 \\
    {IMAGE_TAG}

echo "Container started."
sleep 2
sudo docker logs --tail 10 {APP_NAME}
"""
    _stdin, stdout, stderr = ssh_client.exec_command(commands)
    out = stdout.read().decode()
    err = stderr.read().decode()
    print(out)
    if err:
        print(err[-300:])


def verify(ssh_client):
    """Verify the service is running."""
    print(f"\n[5/5] Verifying service ...")
    _stdin, stdout, _stderr = ssh_client.exec_command(
        f"sudo docker ps --filter name={APP_NAME} --format '{{{{.Status}}}}'"
    )
    status = stdout.read().decode().strip()
    if "Up" in status:
        print(f"  Container running: {status}")
    else:
        print(f"  WARNING: Container status: {status}")

    print("  Testing health endpoint ...")
    _stdin, stdout, _stderr = ssh_client.exec_command(
        f"curl -s http://localhost:8000/health 2>/dev/null || echo 'FAILED'"
    )
    health = stdout.read().decode().strip()
    print(f"  Health: {health}")

    print(f"\n{'='*60}")
    print(f"  Deployment complete!")
    print(f"  Visit: http://{SERVER_IP}")
    print(f"{'='*60}")


def main():
    print("=" * 60)
    print("  Texture Image Search — Cloud Deployment")
    print(f"  Server: {SERVER_IP}")
    print("=" * 60)

    create_archive()
    ssh_client = connect_ssh()

    try:
        upload_archive(ssh_client)
        build_on_server(ssh_client)
        start_container(ssh_client)
        verify(ssh_client)
    finally:
        ssh_client.close()

    if ARCHIVE_PATH.exists():
        ARCHIVE_PATH.unlink()
        print("  Local archive cleaned up.")


if __name__ == "__main__":
    main()
