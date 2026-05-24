<p align="center">
  <h1 align="center">Texture Image Retrieval Engine<br><sub style="font-size:0.6em">纹理图像检索系统</sub></h1>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/framework-FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/CV-OpenCV-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/DL-PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/deploy-Docker-2496ED?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

<p align="center">
  <b>Multi-scale texture-based reverse image search</b> for fabric, material, and pattern libraries.<br>
  基于多尺度纹理特征的 <b>以图搜图引擎</b>，面向布料 / 材质 / 图案库。
</p>

---

## Concept / 核心思路

```
                    ┌─────────────────────────────────────────┐
  Query Image       │  Feature Extraction Pipeline            │  Ranking
  ┌────────┐        │                                         │
  │  输入  │─────▶  │  LBP (R=1,2,3) × 3 image scales         │──▶ #1  0.998
  │  图片  │        │  + HOG spatial pyramid (1×1 + 2×2)      │──▶ #2  0.944
  │        │        │  + DCT perceptual hash (pHash)          │──▶ #3  0.938
  └────────┘        │  + ResNet-18 CNN (layer3 + layer4)      │──▶ ...
                    │         ↓                               │
                    │  Group-weighted L2-normalised fusion    │
                    │  (LBP×4, HOG×1.5, pHash×1.5, CNN×0.5)   │
                    └─────────────────────────────────────────┘
```

Three retrieval modes, from fast to precise / 三种检索模式，从快到精：

| Mode / 模式 | Method / 方法 | Best for / 适用 |
|:---|:---|:---|
| **Global** 全局 | Full-image feature vector → cosine similarity | Quick preview, flat scans |
| **Local** 局部 | Dense multi-scale LBP patches → per-patch voting | Texture details, cross-scale matching |
| **Rerank** 重排 | Global recall + `TM_CCOEFF_NORMED` spatial verification | Highest precision, model→fabric queries |

---

## Quick Start / 快速开始

### CLI Mode / 命令行

```bash
# 1. Install / 安装依赖
pip install -r requirements.txt

# 2. Build index / 构建索引 (global + local)
python build_index.py --local_index --augment 4

# 3. Search / 搜索
python search.py query.jpg                      # global / 全局检索
python search.py crop.jpg --local_mode           # local patch / 局部纹理
python search.py photo.jpg --rerank_template     # rerank / 模板重排
```

### Web Service / Web 服务 (Recommended / 推荐)

```bash
# Start the server / 启动服务
python app.py
# Open / 打开 → http://localhost:8000
```

### Docker Deploy / Docker 部署

```bash
# Build / 构建
docker build -t texture-search:v1 .

# Run / 运行
docker run -d --name texture-search --restart unless-stopped -p 80:8000 texture-search:v1
```

---

## Project Structure / 项目结构

```
.
├── app.py                        # Web API (FastAPI) / Web 服务入口
├── search.py                     # CLI search tool / 命令行搜索
├── build_index.py                # Index builder / 离线索引构建
├── utils.py                      # Core library / 核心算法库
├── feature_extractor.py          # Standalone LBP+HOG extractor / 独立提取器（旧版）
├── generate_test_images.py       # Synthetic test patterns / 生成测试图
│
├── deploy.py                     # One-click deploys to cloud / 一键部署脚本
├── update.py                     # Incremental code+index update / 增量更新脚本
│
├── Dockerfile                    # Container build / 容器化配置
├── .dockerignore
├── requirements.txt
│
├── templates/
│   └── index.html                # Search frontend / 搜索前端页面
├── images/                       # Image library / 图片库 (gitignored)
├── data/                         # Feature index files / 特征索引文件 (gitignored)
└── .claude/                      # Claude Code harness / AI 辅助配置
```

---

## Feature Details / 功能详情

### Image Preprocessing / 图像预处理

- **Aspect-ratio-preserving resize** → standardised 128×128 input
- **Auto clothing-region crop** (`--auto_crop`) — Haar cascade face detection + torso heuristics for model→fabric cross-domain matching. Zero additional dependencies.
- **Fabric augmentation** (`--augment N`) — perspective warp + elastic deformation + gamma correction. Generates N "worn" variants per flat fabric scan to bridge the 2D-scan → 3D-draped domain gap.

### Feature Engineering / 特征工程

| Descriptor | Details | Dims |
|:---|:---|:---|
| **LBP** | R=1,2,3 rotation-invariant uniform × 3 image pyramid scales | ~700 |
| **HOG** | 1×1 global + 2×2 spatial pyramid × 3 scales, 9-bin orientation | ~23,000 |
| **pHash** | DCT 8×8 low-frequency coefficients (excluding DC) | 63 |
| **CNN** | ResNet-18 layer3 + layer4, each with 1×1+2×2+4×4 grid-average pooling | 16,128 |
| **Local patch** | 54-dim compact LBP (R=1,2,3) — ~29 patches/image @ 48px, 3 scales | 54/patch |

### Search Pipeline / 检索流水线

| Stage | Operation |
|:---|:---|
| 1 | Load pre-built NumPy feature index + JSON metadata |
| 2 | Extract multi-scale fused descriptor from query |
| 3 | Cosine similarity ranking over full index (global) <br> **or** dense LBP-patch voting with source-image aggregation (local) |
| 4 | *(optional)* `TM_CCOEFF_NORMED` spatial re-rank on top-N candidates |
| 5 | Return ranked results with similarity scores |

### Security / 安全防护

| Layer | Mechanism |
|:---|:---|
| **Token Auth** | 32-char random token embedded in frontend page, required for all API calls |
| **Rate Limiting** | Per-IP sliding window: max 15 requests / 60 seconds |
| **Path Sanitisation** | Prevents directory traversal attacks on image serving |
| **Image Protection** | `/images/` endpoint requires valid token |

### Monitoring / 系统监控

访问 `/stats?token=<your-token>` 可查看：
- 请求成功 / 失败 / 拦截计数
- 平均响应时间
- 三种检索模式使用占比（柱状图）
- 最近 50 条访问记录（时间 / IP / 模式 / 文件名 / 耗时）
- 安全防护状态总览

---

## Dependencies / 依赖

```
numpy>=1.24.0       opencv-python>=4.8.0    scikit-image>=0.21.0
scipy>=1.10.0       matplotlib>=3.7.0       torch>=2.0.0
torchvision>=0.15.0  Pillow                  fastapi>=0.100.0
uvicorn[standard]>=0.23.0                    python-multipart>=0.0.6
```

## Hardware Requirements / 硬件要求

| Spec | Minimum / 最低 | Recommended / 推荐 |
|:---|:---|:---|
| CPU | 2 cores / 核 | 4+ cores / 核 |
| RAM | 2 GB | 4 GB |
| Disk / 磁盘 | 5 GB + image storage | 10 GB+ |
| GPU | Not required / 不需要 | Optional for faster CNN / 可选 |

---

## Cloud Deployment / 云部署

1. 修改 `deploy.py` 中的 `SERVER_IP`、`SERVER_USER` 为你的服务器信息
2. 确保服务器已安装 Docker
3. 运行一键部署：

```bash
# Windows CMD
set ROOT_PWD=<服务器密码> && python deploy.py

# 后续更新代码/索引
set ROOT_PWD=<服务器密码> && python update.py
```

> **注意：** `deploy.py` 和 `update.py` 包含服务器连接信息，提交到公开仓库前请确认已脱敏或加入 `.gitignore`。

### Adding New Images / 新增图片

部署后新增图片到图库，**无需重建 Docker 镜像**：

```bash
# 1. Upload new images to server / 上传新图到服务器
scp your-image.jpg ubuntu@<IP>:/home/ubuntu/texture-search/images/

# 2. Rebuild feature index inside container / 在容器内重建索引
sudo docker exec texture-search python build_index.py --local_index --augment 4

# 3. Restart to reload index / 重启容器加载新索引
sudo docker restart texture-search
```

4. 刷新网页，新图片即可被检索。

---

## Performance / 性能指标

Tested on 2-core / 2 GB RAM VM with 53 indexed images / 在 2 核 2GB 云服务器上测试 53 张索引图：

| Mode | Avg Response | Memory |
|:---|:---|:---|
| global | ~85 ms | ~800 MB |
| local | ~200 ms | ~900 MB |
| rerank | ~400 ms | ~800 MB |

Scales linearly with index size — 200 images ≈ same response time, ~3 MB feature index on disk.

检索 200 张图片性能基本不变，索引文件仅 ~3MB。

---

## License / 许可

MIT — use freely for personal, academic, and commercial projects.

可自由用于个人、学术。
