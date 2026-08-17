# 🦵 Knee-o-1: Multi-View MRI Abnormality Detection

> *"He is... Knee-o-1."*  
> High-performance multi-view MRI knee pathology detection leveraging **DINOv2 ViT-Small**, **Multi-Slice Attention Pooling**, and **Test-Time Augmentation (TTA)** for the RSNA Medical Imaging Challenge.

---

## 📌 Overview

**Knee-o-1** is an end-to-end deep learning framework designed to identify 12 distinct anatomical abnormalities across multi-planar MRI scans (`Sagittal`, `Coronal`, and `Axial`). Built to process complex 3D volumetric medical data, the system extracts fine-grained visual representations using self-supervised Vision Transformers and dynamically pools critical slice features via sequence-level attention mechanisms.

---

## 🎯 Target Pathologies (12 Multi-Label Targets)

The pipeline outputs probabilistic predictions for 12 key clinical targets:

* **Ligament Injuries:** `ACL`, `MCL`
* **Cartilage & Meniscus:** `Medial Meniscus`, `Lateral Meniscus`
* **Osteoarthritis (OA):** `Medial OA`, `Lateral OA`, `PF OA`
* **Joint Effusion & Soft Tissue:** `Effusion`, `Synovitis`, `Baker's Cyst`, `Contusion`, `Fracture`

---

## 🚀 Key Architectural Features

* **DINOv2 ViT-Small Backbone:** Utilizes self-supervised Vision Transformer features for precise representation across DICOM slices without requiring manual region annotations.
* **Multi-Slice Attention Pooling:** Dynamically weights key MRI slices per anatomical plane to isolate acute lesions and pathology from normal tissue.
* **Tri-Planar Concatenation:** Seamlessly integrates joint embeddings from Sagittal, Coronal, and Axial series for a unified joint representation.
* **Test-Time Augmentation (TTA):** Blends original and horizontal flip predictions during inference to improve prediction stability and AUC score performance.
* **Multi-GPU Optimization:** Engineered with PyTorch `DataParallel` for efficient distributed inference across **NVIDIA Tesla T4 ×2** setups.

---

## ⚙️ Environment & Setup

* **Framework:** PyTorch (TorchVision)
* **Pretrained Models:** DINOv2 via PyTorch Hub / KaggleHub
* **Hardware Acceleration:** Dual GPU (NVIDIA T4 ×2)
* **Environment:** Kaggle Notebooks (Latest PyTorch Environment)

---

## 📂 Submission Pipeline Workflow

1. **DICOM Ingestion:** Dynamic loading and min-max intensity normalization of MRI DICOM slices.
2. **Feature Extraction:** DINOv2 feature mapping across 5 representative slices per anatomical plane.
3. **Attention Weighting:** Slice-level soft-attention pooling per view.
4. **Ensemble & TTA:** Multi-view output concatenation followed by TTA prediction blending.
5. **Output Generation:** Standardized `submission.csv` formatting with probability thresholds.

---

## 📜 License

This project is released under the [MIT License](LICENSE).
