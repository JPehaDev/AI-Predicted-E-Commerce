# AI‑Predicted E‑Commerce 🛒🤖

![Made with](https://img.shields.io/badge/Made%20with-Python%203.8%2B-blue.svg)
![Framework](https://img.shields.io/badge/API-FastAPI-009688.svg)
![Model](https://img.shields.io/badge/Model-DistilBERT-8A2BE2.svg)
![Serving](https://img.shields.io/badge/Serving-Uvicorn-informational.svg)
![Cache](https://img.shields.io/badge/DB-Redis-red.svg)
![Docker](https://img.shields.io/badge/Container-Docker-blue.svg)

> An end‑to‑end **product categorization** service that predicts a predefined category from the product **name + description** using **NLP**. It includes data prep, training, an API, a minimal UI, and Docker Compose orchestration.

---

## 🧭 Table of Contents
- [AI‑Predicted E‑Commerce 🛒🤖](#aipredicted-ecommerce-)
  - [🧭 Table of Contents](#-table-of-contents)
  - [🌐 Overview](#-overview)
  - [🗺️ Workflow Diagram](#️-workflow-diagram)
  - [✨ Features](#-features)
  - [🧱 Architecture](#-architecture)
  - [✅ Project Plan \& Milestones](#-project-plan--milestones)
  - [⚡ Quick Start](#-quick-start)
    - [With Docker (recommended)](#with-docker-recommended)
    - [Local Environment](#local-environment)
  - [📦 Dataset](#-dataset)
  - [🧠 Training](#-training)
  - [🔌 Inference \& API](#-inference--api)
  - [🖥️ Demo UI](#️-demo-ui)
  - [🗄️ Redis Persistence](#️-redis-persistence)
  - [🗂️ Repository Structure](#️-repository-structure)
  - [📈 Metrics](#-metrics)
  - [🧯 Troubleshooting](#-troubleshooting)
  - [🛣️ Roadmap](#️-roadmap)
  - [📚 References](#-references)
  - [🤝 Contributing](#-contributing)
  - [📄 License](#-license)

---

## 🌐 Overview
This project trains and serves a **text classifier** for e‑commerce. Given a product's **name** and **description**, it predicts the most likely **category**. The pipeline covers:

1) **Preprocessing** (cleaning, folding rare categories into `other`, `LabelEncoder`).  
2) **Training** a **DistilBERT** *sequence classification* model.  
3) **Serving** via **FastAPI** + **Uvicorn** (REST endpoints).  
4) **Lightweight storage** in **Redis** (SKU → hash with name/description/category/predict).  
5) A minimal **HTML UI** for demos.  
6) **Jupyter** for EDA and experiments.  
7) **Docker Compose** for a reproducible local stack.

---

## 🗺️ Workflow Diagram

<p align="center">
  <img src="assets/ai_ecommerce_workflow_demo.gif" alt="AI E-Commerce Workflow" width="900">
</p>

---

## ✨ Features
- 🔤 **Transformers NLP**: fine‑tuned **DistilBERT** for multi‑class classification.  
- 🧹 **Reproducible preprocessing**: `text = name + description`, NA filling, `LabelEncoder`.  
- 🚀 **FastAPI** with automatic docs at `/docs`.  
- 💾 **Redis** as a fast key‑value store for items.  
- 🧪 **Notebooks** for EDA/experimentation.  
- 🐳 **Docker Compose**: API, Redis and Jupyter on a shared network.  

---

## 🧱 Architecture
- **`src/preprocessdata.py`**: reads BestBuy JSON, cleans, filters infrequent categories (≥ 100), builds `text`, fits `LabelEncoder`, saves `data.csv`, `label.csv`, `label_encoder.pkl`.
- **`src/train.py`**: tokenization (**DistilBERT**, `max_length=18`), *fine‑tuning* (`epochs=3`, `lr=5e‑5`, `batch_size=16`), evaluation, and persists model/tokenizer in `models/`.
- **`src/predict.py`**: loads artifacts and exposes `main_predict(name, description)` returning the **category id**.
- **`app.py` (FastAPI)**: REST endpoints + server‑side templates.  
- **`templates/`**: simple HTML for the demo form.  
- **`models/`**: artifacts (`classifier/`, `tokenizer/`, `encoder/`).  
- **`docker-compose.yml` + `docker/`**: reproducible local deployment.  

---

## ✅ Project Plan & Milestones
The project follows this **11‑step milestone plan** (adapted to this implementation):

1. **Setup repository & structure** 🧩  
   - Initialize the GitHub repo, create sub‑folders, and scaffold deliverables.  
   - Optionally, prepare an **AWS** EC2 instance/S3 bucket for dataset evaluation and storage.

2. **Download & evaluate the dataset (EDA)** 🔎  
   - Compute **#products**, **#unique categories**, histograms, top categories.  
   - Note: a product can belong to **multiple categories** (taxonomy). We take the **last** (most specific) one.  
   - Keep categories with **≥ 100** samples; map the rest to **`other`**.

3. **Create a training dataset** 🧽  
   - Implement a cleaning function for the text: remove **non‑alphabetical chars**, **punctuation**, and optionally **stop words**; normalize whitespace/casing.  
   - Build `text = name + " " + description`.  
   - (Optional) Persist the cleaned dataset to **S3** (text + final category).  
   - This repo saves `data.csv` / `label.csv` and a `LabelEncoder` locally under `./data/` and `./models/`.

4. **State‑of‑the‑art review** 📚  
   - Compare **tokenization** approaches (WordPiece/BPE), **word embeddings**, and **TF‑IDF**.  
   - Consider alternative **classifiers**: **LightGBM**, **XGBoost**, **CatBoost**, **RandomForest**, **Ensembles/Stacking**, plus an **MLP** baseline.

5. **Classifier & accuracy research** 🧪  
   - Train/evaluate the selected classical models (and an MLP) on the cleaned features.  
   - In this implementation, we prioritize a **Transformer baseline (DistilBERT)** for strong text performance.

6. **Evaluate/test the initial classifier** 📊  
   - Compare **accuracy**, **AUC**, **training time**, and **inference time** across candidates.  
   - Select the **best model** balancing quality and latency. (This repo reports ~0.91 accuracy/F1 with DistilBERT.)

7. **Setup an API for product classification** 🌐  
   - Expose the model through **FastAPI**. Prefer containerization with **Docker** for reproducibility and portability.

8. **Integrate a basic UI & secure the API** 🖥️🔐  
   - Provide a simple web form for demos (this repo includes `templates/`).  
   - Security hardening ideas: **CORS** rules, **rate limiting**, **auth (e.g., JWT)**, secrets in **env vars**, and **HTTPS** behind a reverse proxy.

9. **Fine‑tune / Train additional models** 🎯  
   - After the first evaluation, iterate to improve accuracy/latency (hyperparameters, longer context, better cleaning, class weights, focal loss, etc.).

9.5 **Add API tests (optional)** 🧪  
   - Unit/integration tests for endpoints and a small smoke‑test for inference.

10. **Preview to other teams** 📣  
   - Demo the service and gather feedback for final adjustments (UX, latency, error messages).

11. **Build final presentation** 🧾  
   - Prepare slides/live demo for *Demo Day*. Document reproducibility and limitations.

> The milestones above are **high‑level** and should be refined into smaller tasks as you learn more about the dataset and constraints.

---

## ⚡ Quick Start

### With Docker (recommended)
1. Clone the repo and `cd` to the root (where `docker-compose.yml` lives).  
2. (Optional) Ensure **Git LFS** if you need `*.safetensors`:
   ```bash
   git lfs install
   git lfs pull
   ```
3. Bring the stack up:
   ```bash
   sudo docker-compose up
   ```
4. Open:
   - API: `http://0.0.0.0:8080/` (docs at `http://0.0.0.0:8080/docs`)
   - Jupyter: `http://localhost:8888/`

### Local Environment
> The project is optimized for Docker. If running locally:
- Start **Redis** on `localhost:6379` or change the `host` in `app.py` (defaults to `redis`, the Docker service name).  
- Install deps and launch:
  ```bash
  pip install -r requirements.txt
  uvicorn app:app --reload --port 8080
  ```

---

## 📦 Dataset
- Based on the open **BestBuy** e‑commerce dataset used in the AnyoneAI challenge.  
- Two source files are relevant: `categories.json` and `products.json`.  
- Raw GitHub (example):  
  ```
  https://raw.githubusercontent.com/anyoneai/e-commerce-open-data-set/master/products.json
  ```

**Single product example:**
```json
{
  "sku": 1004695,
  "name": "GoPro - Camera Mount Accessory Kit - Black",
  "type": "HardGood",
  "price": 19.99,
  "upc": "185323000309",
  "category": [
    {"id": "abcat0400000", "name": "Cameras & Camcorders"},
    {"id": "abcat0410022", "name": "Camcorder Accessories"},
    {"id": "pcmcat329700050009", "name": "Action Camcorder Accessories"},
    {"id": "pcmcat240500050057", "name": "Action Camcorder Mounts"},
    {"id": "pcmcat329700050020", "name": "Handlebar/Seatpost Mounts"}
  ],
  "shipping": 5.49,
  "description": "Compatible with most GoPro cameras; includes a variety of camera mounting accessories",
  "manufacturer": "GoPro",
  "model": "AGBAG-001",
  "url": "http://www.bestbuy.com/site/gopro-camera-mount-accessory-kit-black/1004695.p?id=1218249514954&skuId=1004695&cmp=RMXCC",
  "image": "http://img.bbystatic.com/BestBuy_US/images/products/1004/1004695_rc.jpg"
}
```
> Real‑world descriptions can be **short or low‑quality**. The model should be robust to sparse text.

This repo expects `./data/raw/products.json.gz`. During preprocessing we take the **last** category in the list (most specific) and fold categories with **< 100** samples into **"other"**.

---

## 🧠 Training
1. **Preprocess**  
   ```python
   from src.preprocessdata import main_preprocessdata
   main_preprocessdata()
   # Produces data.csv, label.csv and label_encoder.pkl
   ```
2. **Train**  
   ```python
   from src.train import main_train
   main_train()
   # Saves model and tokenizer under ./models/
   ```
3. **Artifacts**  
   - `models/classifier/model.safetensors`  
   - `models/tokenizer/*`  
   - `models/encoder/label_encoder.pkl`

> You can run these steps inside the **Jupyter** container for full reproducibility.

---

## 🔌 Inference & API
**Key FastAPI endpoints**
- `GET /predict_category?name=...&description=...` → returns the category id.  
- `POST /create_item` → stores an item in Redis with `sku`, `name`, `description`, `category`.  
- `GET /get_item?sku=...` → retrieves the hash for a SKU.  
- `PUT /update_predict_category?sku=...` → runs the model and writes `predict_category` to Redis.

Interactive docs: `http://0.0.0.0:8080/docs`

**Security hardening (suggested):**
- CORS policy, API key/JWT, rate limiting, request validation, secrets via env vars, HTTPS behind a reverse proxy (e.g., Nginx/Caddy).

---

## 🖥️ Demo UI
- `GET /` renders `templates/form.html` (simple form).  
- **POST** to `/new_item_and_predict/` creates a random SKU, runs inference and shows `templates/result.html` with the **predicted category**.

> If you build your own frontend, submit `name` and `description` to `/predict_category` or reuse the form flow.

---

## 🗄️ Redis Persistence
- **Key**: `sku`  
- **Hash fields**: `name`, `description`, `category`, `predict_category`  
- The API already includes **create**, **read**, and **update** flows.

---

## 🗂️ Repository Structure
```text
.
├─ app.py
├─ docker-compose.yml
├─ docker/
│  ├─ api
│  └─ jupyter
├─ models/
│  ├─ classifier/
│  ├─ encoder/
│  └─ tokenizer/
├─ notebooks/
├─ src/
│  ├─ preprocessdata.py
│  ├─ train.py
│  ├─ predict.py
│  └─ settings.py
├─ templates/
│  ├─ form.html
│  └─ result.html
└─ assets/
   ├─ ai_ecommerce_workflow_v2.svg
   └─ ai_ecommerce_workflow_v2.png
```

---

## 📈 Metrics
*Reported on validation/test after the reference training:*

| Split       | mean_accuracy | f1 weighted | f1 micro | f1 macro |
|-------------|---------------|-------------|----------|----------|
| Validation  | 0.9109        | 0.9113      | 0.9108   | 0.8697   |
| Test        | 0.9112        | 0.9105      | 0.9111   | 0.8670   |

> Results may vary if you change *max_length*, *epochs*, the category threshold, etc.

---

## 🧯 Troubleshooting
- **Model won't load** → Ensure `git lfs install && git lfs pull` to fetch `model.safetensors`.  
- **`/new_item_and_predict/` fails** → It must be a **POST** with **form-data** (not JSON).  
- **Redis unreachable** → Confirm the `redis` container is running (`docker ps`) or adjust the `host` in `app.py` when running without Docker.  
- **Dataset missing** → Place `products.json.gz` under `./data/raw/`.

---

## 🛣️ Roadmap
- [ ] API tests (pytest).  
- [ ] Improved text cleaning (stopwords, normalization).  
- [ ] Additional metrics (per‑class confusion matrix, **AUC**).  
- [ ] Fine‑tuning with *class weights*/*focal loss* for imbalance.  
- [ ] Batch prediction endpoint.  
- [ ] Optional S3 integration for cleaned datasets and artifacts.

---

## 📚 References
- **Large Scale Product Categorization using Structured and Unstructured Attributes** — Abhinandan Krishnan, Abilash Amarthaluri.  
- **Multi‑Label Product Categorization Using Multi‑Modal Fusion Models** — Pasawee Wirojwatanakul, Artit Wangperawong.

---

## 🤝 Contributing
PRs and suggestions are welcome! Please open an issue to discuss major changes. Style: *black/flake8* recommended.

---

## 📄 License
This project is distributed under the terms listed in `LICENSE`.
