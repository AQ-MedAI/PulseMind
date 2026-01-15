
# PulseMind: A Multi-Modal Medical Model for Real-World Clinical Diagnosis

> Official repository for **"PulseMind: A Multi-Modal Medical Model for Real-World Clinical Diagnosis"**, accepted as an **Oral** paper at **AAAI 2026**.

<p align="center">
  <img src="https://img.shields.io/badge/Conference-AAAI%202026-blue.svg" />
  <img src="https://img.shields.io/badge/Type-Oral-green.svg" />
  <img src="https://img.shields.io/badge/Domain-Multi--Modal%20Medicine-orange.svg" />
</p>

<p align="center">
  <b>Datasets, models, and benchmarks for PulseMind.</b>
</p>

---

## 🌐 Overview

This repository provides the official **codebase and evaluation scripts** for the PulseMind project, together with:

- 🧪 **MediScope**: a large-scale multimodal medical dataset.  
  In this release, we provide a curated subset of **~1,000 cases** (JSON + images). The full dataset is larger and will be gradually released.
- 🧠 **Models**:  
  - `PulseMind-72B`  
- 📊 **Benchmarks**:
  - `MedDiagnose` – 237-sample test set (JSON + images)
  - `CMtMedQA-test` – 1,000-sample test set (JSON)
  - `MedDiagnose-plus` – 937-sample extended test set (JSON + images)

> ⚠️ Due to size and privacy considerations, **all datasets and model checkpoints are hosted externally** and are **not** stored in this GitHub repository.  
> This repo mainly contains **evaluation code**.

---

### 🔗 Dataset Download Links 

[Download link](https://huggingface.co/datasets/AQ-MedAI/PulseMind) 

- - **MediScope (curated ~1k subset)**
- **MedDiagnose (237 samples)**
- **CMtMedQA-test (1,000 samples)**
- **MedDiagnose-plus (937 samples)**

### 🧠 Model Checkpoint Links

- **PulseMind-72B checkpoint**: [Download link](TODO_pulsemind72b_link)  

> After downloading, please follow the recommended directory layout  
> (e.g., place raw data under `data/`, benchmark test sets under `Benchmark/`,  
> and model checkpoints under `model/`), so that the provided evaluation scripts
> can run out of the box.

---

## 📁 Repository Structure (Code Only)

The GitHub repository mainly contains evaluation code and auxiliary configs:

```bash
.
├── data/                        # (empty by default) place downloaded datasets here
│
├── Benchmark/
│   ├── CMtMedQA-test/           # Folder for CMtMedQA-test data (JSON, etc.)
│   ├── MedDiagnose/             # Folder for MedDiagnose data (JSON + images)
│   ├── MedDiagnose-plus/        # Folder for MedDiagnose-plus data (JSON + images)
│   └── Eval/                    # Optional: extra evaluation utilities / configs
│
├── model/                       # Place downloaded model checkpoints here
│
└── README.md

>>>>>>> Initial push from server AAAI_github_PulseMind
