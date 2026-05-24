"""FastAPI web service for texture image search — supports global, local, and rerank modes."""

import base64
import json
import os
import secrets
import tempfile
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import HTTPException, FastAPI, File, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from utils import (
    DEFAULT_PATCH_SCALES,
    DEFAULT_PATCH_SIZE,
    DEFAULT_PATCH_STRIDE,
    extract_clothing_region,
    extract_dense_patches,
    extract_features,
    extract_local_texture,
    load_image,
    preprocess_image,
    template_match_score,
    to_gray,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
IMAGES_DIR = Path(__file__).resolve().parent / "images"

# ── Load indexes ────────────────────────────────────────────────────────────

if not (DATA_DIR / "features.npy").exists():
    raise FileNotFoundError("Index not found. Run 'python build_index.py' first.")

global_features = np.load(str(DATA_DIR / "features.npy"))
with open(DATA_DIR / "index.json", "r", encoding="utf-8") as f:
    global_meta = json.load(f)

_local_features = None
_local_meta = None
if (DATA_DIR / "features_local.npy").exists():
    _local_features = np.load(str(DATA_DIR / "features_local.npy"))
    with open(DATA_DIR / "local_meta.json", "r", encoding="utf-8") as f:
        _local_meta = json.load(f)

# Generate a random access token for API protection
ACCESS_TOKEN = secrets.token_urlsafe(24)
print(f"Access token: {ACCESS_TOKEN}")

# ── Rate limiter (in-memory, per IP) ────────────────────────────────────────

_rate_log: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT = 15        # max requests
RATE_WINDOW = 60.0     # per 60 seconds


def _check_rate(ip: str) -> bool:
    now = time.time()
    window = [t for t in _rate_log[ip] if now - t < RATE_WINDOW]
    _rate_log[ip] = window
    if len(window) >= RATE_LIMIT:
        return False
    _rate_log[ip].append(now)
    return True


# ── In-memory access log ────────────────────────────────────────────────────

_access_log: list[dict] = []   # last 200 entries
MAX_LOG_ENTRIES = 200
_rate_blocked_count = 0
_token_blocked_count = 0


def _log_access(ip: str, filename: str, mode: str, status: str, elapsed_ms: float):
    entry = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ip": ip,
        "file": filename,
        "mode": mode,
        "status": status,
        "elapsed_ms": round(elapsed_ms, 0),
    }
    _access_log.append(entry)
    if len(_access_log) > MAX_LOG_ENTRIES:
        _access_log[:] = _access_log[-MAX_LOG_ENTRIES:]


# ── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    print(f"Global index: {len(global_meta)} images, dim={global_features.shape[1]}")
    if _local_features is not None:
        print(f"Local index: {len(_local_meta)} patches, dim={_local_features.shape[1]}")
    import cv2
    from utils import extract_cnn_features
    dummy = np.zeros((128, 128, 3), dtype=np.uint8)
    try:
        extract_cnn_features(cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB))
        print("CNN model warmed up")
    except Exception as e:
        print(f"CNN warmup skipped: {e}")
    yield


app = FastAPI(
    title="Texture Image Search",
    description="Multi-scale texture-based image retrieval engine",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/images/{path:path}")
async def serve_image(path: str, token: str = Query(default="")):
    if token != ACCESS_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid access token.")
    file_path = IMAGES_DIR / path
    if not file_path.resolve().is_relative_to(IMAGES_DIR.resolve()):
        raise HTTPException(status_code=403, detail="Invalid path.")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(str(file_path))


# ── Helpers ─────────────────────────────────────────────────────────────────

def cosine_similarity(query_vec: np.ndarray, db_features: np.ndarray) -> np.ndarray:
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    db_norm = db_features / (np.linalg.norm(db_features, axis=1, keepdims=True) + 1e-12)
    return db_norm @ q_norm


def _search_global(query_path: str, top_k: int, threshold: float):
    img = load_image(query_path)
    query_feat = extract_features(img)
    sims = cosine_similarity(query_feat, global_features)
    order = np.argsort(-sims)
    results = []
    for i in order:
        score = float(sims[i])
        if score < threshold:
            break
        m = global_meta[i]
        results.append(_make_result(len(results) + 1, score, m, mode_tag="global"))
        if len(results) >= top_k:
            break
    return results


def _search_local(query_path: str, top_k: int, threshold: float):
    if _local_features is None or _local_meta is None:
        return None, "Local index not available. Run: python build_index.py --local_index"

    query_img = load_image(query_path)
    processed = preprocess_image(query_img)
    gray = to_gray(processed)
    qh, qw = gray.shape

    ref_side = 128.0
    if min(qh, qw) < 96:
        scale_factor = min(qh, qw) / ref_side
        ps = max(16, int(round(DEFAULT_PATCH_SIZE * scale_factor)))
        stride = max(8, int(round(DEFAULT_PATCH_STRIDE * scale_factor)))
        scales = DEFAULT_PATCH_SCALES
    else:
        ps = DEFAULT_PATCH_SIZE
        stride = max(DEFAULT_PATCH_STRIDE, int(round(min(qh, qw) / 20)))
        scales = [1.0, 0.7, 0.5, 0.35, 0.25, 0.18]

    patches = extract_dense_patches(gray, scales=scales,
                                     patch_size=ps, stride=stride)
    if not patches:
        return [], None

    q_feats = np.stack(
        [extract_local_texture(p) for p, _, _ in patches], axis=0
    ).astype(np.float32)

    q_norm = q_feats / (np.linalg.norm(q_feats, axis=1, keepdims=True) + 1e-12)
    db_norm = _local_features / (np.linalg.norm(_local_features, axis=1, keepdims=True) + 1e-12)
    sims = db_norm @ q_norm.T

    k = min(3, sims.shape[0])
    top_rows = np.argpartition(-sims, k - 1, axis=0)[:k]

    scores = {}
    for j in range(q_feats.shape[0]):
        for ki in range(k):
            db_idx = top_rows[ki, j]
            s = float(sims[db_idx, j])
            if s > 0:
                img_idx = _local_meta[db_idx]["image_idx"]
                scores[img_idx] = scores.get(img_idx, 0.0) + s

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for img_idx, score in ranked:
        if score < threshold:
            break
        results.append(_make_result(len(results) + 1, score, global_meta[img_idx],
                                     mode_tag="local"))
        if len(results) >= top_k:
            break
    return results, None


def _search_rerank(query_path: str, top_k: int, threshold: float):
    recall_k = min(top_k * 5, len(global_meta))
    coarse = _search_global(query_path, recall_k, threshold)
    scored = []
    for item in coarse:
        cand_path = str(IMAGES_DIR / item["path"])
        try:
            tm_score, _ = template_match_score(query_path, cand_path)
        except Exception:
            continue
        scored.append((tm_score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for rank, (tm_score, item) in enumerate(scored[:top_k], 1):
        results.append({
            "rank": rank,
            "score": round(float(tm_score), 4),
            "path": item["path"],
            "stem": item["stem"],
            "mode": "rerank",
        })
    return results


def _make_result(rank: int, score: float, meta_item: dict, mode_tag: str):
    return {
        "rank": rank,
        "score": round(score, 4),
        "path": meta_item["path"],
        "stem": meta_item["stem"],
        "mode": mode_tag,
    }


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).resolve().parent / "templates" / "index.html"
    raw = html_path.read_text(encoding="utf-8")
    # Inject the token into the page so the frontend can use it
    html = raw.replace("__ACCESS_TOKEN__", ACCESS_TOKEN)
    return HTMLResponse(content=html)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "indexed_images": len(global_meta),
        "local_index": _local_features is not None,
    }


@app.post("/search")
async def search(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = Query(default=10, le=50),
    threshold: float = Query(default=0.0, ge=0.0, le=1.0),
    mode: str = Query(default="global", pattern="^(global|local|rerank)$"),
    auto_crop: bool = Query(default=False),
    token: str = Query(default=""),
):
    global _rate_blocked_count, _token_blocked_count

    ip = request.client.host if request.client else "unknown"

    if not _check_rate(ip):
        _rate_blocked_count += 1
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.")

    if token != ACCESS_TOKEN:
        _token_blocked_count += 1
        raise HTTPException(status_code=403, detail="Invalid access token.")

    suffix = Path(file.filename).suffix.lower() if file.filename else ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}:
        return JSONResponse(status_code=400, content={"error": f"Unsupported format: {suffix}"})

    t_start = time.time()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Auto clothing-region crop (model-photo → fabric matching)
        crop_preview_b64 = ""
        if auto_crop:
            img = load_image(tmp_path)
            h0, w0 = img.shape[:2]
            cropped = extract_clothing_region(img)
            h1, w1 = cropped.shape[:2]
            if h1 > 0 and w1 > 0:
                import cv2
                # Save cropped image for search
                crop_path = tmp_path + "_crop.jpg"
                cv2.imwrite(crop_path, cropped)
                tmp_path = crop_path
                # Encode as base64 for frontend preview
                _, buf = cv2.imencode(".jpg", cropped)
                crop_preview_b64 = base64.b64encode(buf.tobytes()).decode()
                print(f"[ACCESS] Auto-crop: {w0}x{h0} → {w1}x{h1}")
            else:
                auto_crop = False  # fallback: no crop applied

        if mode == "local":
            results, error = _search_local(tmp_path, top_k, threshold)
            if error:
                _log_access(ip, file.filename, mode, "error", (time.time() - t_start) * 1000)
                return JSONResponse(status_code=400, content={"error": error})
        elif mode == "rerank":
            results = _search_rerank(tmp_path, top_k, threshold)
        else:
            results = _search_global(tmp_path, top_k, threshold)

        elapsed = (time.time() - t_start) * 1000
        _log_access(ip, file.filename, mode, "ok", elapsed)
        print(f"[ACCESS] {ip} | {mode} | {file.filename} | {elapsed:.0f}ms | {len(results)} results")

        resp = {
            "query": file.filename,
            "mode": mode,
            "auto_crop": bool(auto_crop),
            "total_indexed": len(global_meta),
            "results": results,
        }
        if crop_preview_b64:
            resp["crop_preview"] = f"data:image/jpeg;base64,{crop_preview_b64}"
        return resp
    except Exception as e:
        _log_access(ip, file.filename, mode, "error", (time.time() - t_start) * 1000)
        return JSONResponse(status_code=400, content={"error": f"Search failed: {e}"})
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.get("/stats", response_class=HTMLResponse)
async def stats(token: str = Query(default="")):
    if token != ACCESS_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid access token.")

    now = time.strftime("%Y-%m-%d %H:%M:%S")

    # ── Aggregate stats ──────────────────────────────────────────────────
    total_ok = sum(1 for e in _access_log if e["status"] == "ok")
    total_err = sum(1 for e in _access_log if e["status"] == "error")
    avg_ms = (
        round(sum(e["elapsed_ms"] for e in _access_log) / len(_access_log))
        if _access_log else 0
    )
    mode_counts = defaultdict(int)
    for e in _access_log:
        if e["status"] == "ok":
            mode_counts[e["mode"]] += 1

    global_pct = round(mode_counts["global"] / max(total_ok, 1) * 100)
    local_pct = round(mode_counts["local"] / max(total_ok, 1) * 100)
    rerank_pct = round(mode_counts["rerank"] / max(total_ok, 1) * 100)

    # ── Request table rows ───────────────────────────────────────────────
    rows = ""
    for entry in reversed(_access_log[-50:]):
        color = "#dc2626" if entry["status"] != "ok" else "#059669"
        rows += (
            f"<tr>"
            f"<td>{entry['time']}</td>"
            f"<td>{entry['ip']}</td>"
            f"<td><span class='badge {entry['mode']}'>{entry['mode']}</span></td>"
            f"<td title='{entry['file']}'>{entry['file'][:30]}{'…' if len(entry['file'])>30 else ''}</td>"
            f"<td class='status-{entry['status']}'>{entry['status']}</td>"
            f"<td>{entry['elapsed_ms']:.0f}ms</td>"
            f"</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>系统监控 — 纹理检索</title>
<style>
  :root {{
    --bg: #f8f9fb; --card: #fff; --text: #1a1a2e; --muted: #6b7280;
    --border: #e5e7eb; --radius: 14px;
    --blue: #4f46e5; --green: #059669; --red: #dc2626; --amber: #d97706;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans SC", sans-serif;
    background: var(--bg); color: var(--text); min-height:100vh;
  }}
  .nav {{
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    padding: 18px 24px; color: #fff;
  }}
  .nav h1 {{ font-size: 1.2rem; font-weight: 700; }}
  .nav .sub {{ font-size: .8rem; opacity: .7; margin-top: 2px; }}
  .nav .actions {{ display:flex; gap:10px; align-items:center; margin-top:10px; }}
  .nav button {{
    padding: 6px 16px; border:1px solid rgba(255,255,255,.3); border-radius:6px;
    background: transparent; color:#fff; cursor:pointer; font-size:.8rem;
  }}
  .nav button:hover {{ background: rgba(255,255,255,.1); }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 20px 24px; }}
  .card {{
    background: var(--card); border-radius: var(--radius);
    padding: 24px; margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,.06); border: 1px solid var(--border);
  }}
  .card-header {{
    display:flex; justify-content:space-between; align-items:center;
    margin-bottom: 20px;
  }}
  .card-header h2 {{ font-size: 1rem; font-weight: 700; }}
  .card-header .hint {{ font-size:.78rem; color:var(--muted); }}

  /* stat cards */
  .metric-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 14px;
  }}
  .metric {{
    padding: 20px; border-radius: 12px;
    background: #f9fafb; border: 1px solid var(--border);
    display: flex; align-items: center; gap: 14px;
  }}
  .metric .icon {{
    width: 44px; height: 44px; border-radius: 10px;
    display:flex; align-items:center; justify-content:center; font-size:1.3rem;
    flex-shrink: 0;
  }}
  .icon-ok {{ background: #ecfdf5; color: var(--green); }}
  .icon-err {{ background: #fef2f2; color: var(--red); }}
  .icon-block {{ background: #fffbeb; color: var(--amber); }}
  .icon-speed {{ background: #eef2ff; color: var(--blue); }}
  .metric .val {{ font-size:1.7rem; font-weight:800; line-height:1.1; }}
  .metric .lbl {{ font-size:.75rem; color:var(--muted); }}

  /* bar chart */
  .bar-row {{ display:flex; align-items:center; gap:12px; margin-bottom:10px; }}
  .bar-row .lbl {{ width:70px; font-size:.8rem; font-weight:600; }}
  .bar-track {{
    flex:1; height:22px; background: #f3f4f6; border-radius:11px;
    overflow:hidden; position:relative;
  }}
  .bar-fill {{
    height:100%; border-radius:11px; transition: width .4s ease;
    display:flex; align-items:center; justify-content:flex-end;
    padding-right: 10px; font-size:.7rem; font-weight:700; color:#fff;
    min-width: 0;
  }}
  .bar-fill.global {{ background: linear-gradient(90deg, #60a5fa, #3b82f6); }}
  .bar-fill.local {{ background: linear-gradient(90deg, #fbbf24, #f59e0b); }}
  .bar-fill.rerank {{ background: linear-gradient(90deg, #34d399, #10b981); }}

  /* table */
  .table-wrap {{ overflow-x:auto; }}
  table {{ width:100%; border-collapse:collapse; font-size:.83rem; }}
  th {{
    text-align:left; padding:10px 12px; border-bottom:2px solid var(--border);
    color:var(--muted); font-weight:600; font-size:.75rem; text-transform:uppercase;
  }}
  td {{ padding:9px 12px; border-bottom:1px solid #f3f4f6; }}
  tr:hover td {{ background:#fafafe; }}

  /* badges */
  .badge {{
    display:inline-block; padding:3px 10px; border-radius:10px;
    font-size:.7rem; font-weight:700; text-transform:uppercase;
  }}
  .badge.global {{ background:#dbeafe; color:#1d4ed8; }}
  .badge.local {{ background:#fef3c7; color:#92400e; }}
  .badge.rerank {{ background:#d1fae5; color:#065f46; }}
  .status-ok {{ color:var(--green); font-weight:600; }}
  .status-error {{ color:var(--red); font-weight:600; }}

  /* footer */
  .footer {{ text-align:center; padding:16px; color:var(--muted); font-size:.75rem; }}

  /* responsive */
  @media (max-width:600px) {{
    .metric-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .container {{ padding:12px; }}
  }}
</style>
</head>
<body>

<div class="nav">
  <h1>纹理检索 — 系统监控面板</h1>
  <div class="sub">实时请求统计与安全概览</div>
  <div class="actions">
    <button onclick="location.reload()">立即刷新</button>
    <span style="font-size:.75rem;opacity:.6">数据保留最近 {MAX_LOG_ENTRIES} 条记录</span>
  </div>
</div>

<div class="container">

  <!-- ── Key metrics ──────────────────────────────────────────────── -->
  <div class="card">
    <div class="card-header">
      <h2>关键指标</h2>
      <span class="hint">更新时间: {now}</span>
    </div>
    <div class="metric-grid">
      <div class="metric">
        <div class="icon icon-ok">&#10003;</div>
        <div>
          <div class="val">{total_ok}</div>
          <div class="lbl">请求成功</div>
        </div>
      </div>
      <div class="metric">
        <div class="icon icon-err">&#9888;</div>
        <div>
          <div class="val">{total_err}</div>
          <div class="lbl">请求失败</div>
        </div>
      </div>
      <div class="metric">
        <div class="icon icon-block">&#128683;</div>
        <div>
          <div class="val">{_rate_blocked_count}</div>
          <div class="lbl">频率限制拦截</div>
        </div>
      </div>
      <div class="metric">
        <div class="icon icon-block">&#128274;</div>
        <div>
          <div class="val">{_token_blocked_count}</div>
          <div class="lbl">Token 拦截</div>
        </div>
      </div>
    </div>
  </div>

  <!-- ── Service status ────────────────────────────────────────────── -->
  <div class="card">
    <div class="card-header">
      <h2>服务运行状态</h2>
    </div>
    <div class="metric-grid">
      <div class="metric">
        <div class="icon icon-speed">&#9889;</div>
        <div>
          <div class="val">{avg_ms}ms</div>
          <div class="lbl">平均响应时间</div>
        </div>
      </div>
      <div class="metric">
        <div class="icon icon-ok">&#128230;</div>
        <div>
          <div class="val">{len(_access_log)}</div>
          <div class="lbl">总请求数（内存中）</div>
        </div>
      </div>
      <div class="metric">
        <div class="icon" style="background:#f3e8ff;color:#7c3aed;">&#128190;</div>
        <div>
          <div class="val">{len(global_meta)}</div>
          <div class="lbl">已索引图片数</div>
        </div>
      </div>
      <div class="metric">
        <div class="icon" style="background:#fce7f3;color:#db2777;">&#128065;</div>
        <div>
          <div class="val" style="font-size:1rem;word-break:break-all">{ACCESS_TOKEN[:10]}...</div>
          <div class="lbl">当前 Token（前缀）</div>
        </div>
      </div>
    </div>
  </div>

  <!-- ── Mode distribution ─────────────────────────────────────────── -->
  <div class="card">
    <div class="card-header">
      <h2>检索模式分布</h2>
      <span class="hint">仅统计成功请求</span>
    </div>
    <div class="bar-row">
      <span class="lbl"><span class="badge global">global</span></span>
      <div class="bar-track">
        <div class="bar-fill global" style="width:{global_pct}%">{global_pct}% ({mode_counts.get('global', 0)})</div>
      </div>
    </div>
    <div class="bar-row">
      <span class="lbl"><span class="badge local">local</span></span>
      <div class="bar-track">
        <div class="bar-fill local" style="width:{local_pct}%">{local_pct}% ({mode_counts.get('local', 0)})</div>
      </div>
    </div>
    <div class="bar-row">
      <span class="lbl"><span class="badge rerank">rerank</span></span>
      <div class="bar-track">
        <div class="bar-fill rerank" style="width:{rerank_pct}%">{rerank_pct}% ({mode_counts.get('rerank', 0)})</div>
      </div>
    </div>
    <p style="margin-top:10px;font-size:.75rem;color:var(--muted);">
      提示：<strong>局部纹理（local）</strong> 适合粗细材质对比，<strong>模板重排（rerank）</strong> 精度最高但略慢
    </p>
  </div>

  <!-- ── Recent requests ───────────────────────────────────────────── -->
  <div class="card">
    <div class="card-header">
      <h2>最近访问记录</h2>
      <span class="hint">最近 50 条</span>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>时间</th><th>IP</th><th>模式</th><th>文件名</th><th>状态</th><th>耗时</th></tr>
        </thead>
        <tbody>
          {rows if rows else '<tr><td colspan="6" style="text-align:center;color:var(--muted);padding:28px">暂无访问记录 — 搜索后记录会出现在这里</td></tr>'}
        </tbody>
      </table>
    </div>
  </div>

  <!-- ── Security summary ──────────────────────────────────────────── -->
  <div class="card">
    <div class="card-header">
      <h2>安全防护状态</h2>
    </div>
    <div class="metric-grid">
      <div class="metric">
        <div class="icon" style="background:#ecfdf5;color:var(--green);">&#9989;</div>
        <div>
          <div class="val" style="font-size:1rem;color:var(--green)">已启用</div>
          <div class="lbl">Token 鉴权</div>
        </div>
      </div>
      <div class="metric">
        <div class="icon" style="background:#ecfdf5;color:var(--green);">&#9989;</div>
        <div>
          <div class="val" style="font-size:1rem;color:var(--green);">{RATE_LIMIT}次/{RATE_WINDOW:.0f}s</div>
          <div class="lbl">IP 频率限制</div>
        </div>
      </div>
      <div class="metric">
        <div class="icon" style="background:#ecfdf5;color:var(--green);">&#9989;</div>
        <div>
          <div class="val" style="font-size:1rem;color:var(--green);">已启用</div>
          <div class="lbl">图片访问保护</div>
        </div>
      </div>
      <div class="metric">
        <div class="icon" style="background:#ecfdf5;color:var(--green);">&#9989;</div>
        <div>
          <div class="val" style="font-size:1rem;color:var(--green);">已启用</div>
          <div class="lbl">路径防穿越</div>
        </div>
      </div>
    </div>
  </div>

</div>

<div class="footer">纹理图像检索系统 &copy; 2026 &middot; 基于 LBP + HOG + pHash + CNN 多尺度特征融合</div>

</body></html>"""
    return HTMLResponse(content=html)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
