Knee-o-1: Multi-View MRI Abnormality Detection
"He is... Knee-o-1." — Precision Knee Diagnosis Powered by Vision Transformers

Knee-o-1 is a deep learning pipeline engineered to detect 12 distinct knee pathologies across multi-planar MRI scans (Sagittal, Coronal, Axial). Built specifically for the RSNA Knee Abnormality Detection challenge, the system leverages self-supervised vision models and sequence-level attention mechanisms to process complex volumetric medical data.

Key Features
DINOv2 ViT-Small Backbone: Utilizes self-supervised Vision Transformers for robust, fine-grained feature extraction across DICOM image slices.
Multi-Slice Attention Pooling: Dynamically weights key MRI slices per anatomical plane to isolate critical lesions from normal tissue.
Tri-Planar Integration: Seamlessly concatenates features from Sagittal, Coronal, and Axial series for holistic 3D joint representation.
Test-Time Augmentation (TTA): Employs horizontal flip blending during inference to maximize prediction stability and AUC score performance.
Dual-GPU Accelerated: Optimized for PyTorch DataParallel execution on dual NVIDIA Tesla T4 setups.
Target Pathologies (12 Multi-Label Output)
Ligament Tears: ACL, MCL
Cartilage & Meniscus: Medial Meniscus, Lateral Meniscus
Osteoarthritis (OA): Medial OA, Lateral OA, PF OA
Joint Effusion & Soft Tissue: Effusion, Synovitis, Baker's Cyst, Contusion, Fracture
