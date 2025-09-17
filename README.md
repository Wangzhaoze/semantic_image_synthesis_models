# Semantic Image Synthesis Models

This repository is a lightweight framework for experimenting with **semantic image synthesis** models, where the input is a semantic label map and the output is a realistic image.  
It uses **PyTorch** and **PyTorch Lightning** for training, logging, and modular model design.

---

## 📂 Datasets

Supported datasets (examples):

- **CMP Facade** – lightweight dataset for quick testing.  
- (TBD) **Cityscapes** – street scenes with semantic labels (default benchmark).  

---

## ⚡ Installation

```bash
git clone https://github.com/yourname/semantic-synthesis-playground.git
cd semantic-synthesis-playground

# create environment
conda create -n synth python=3.10 -y
conda activate synth

# install requirements
pip install -r requirements.txt
```

Main dependencies:
- `torch`  
- `pytorch-lightning`  
- `torchvision`  
- `numpy`

---

## 🚀 Training

Example command:

```bash
python train.py --dataset cityscapes --model pix2pix --gpus 1
```

Each model has its own config file in `configs/`.

---

## ✅ TODO: Models to Implement

We plan to progressively add classic **conditional generative models** for semantic-to-image synthesis:

### 🔹 GAN-based
- [ ] **pix2pix** (Isola et al., 2017) – baseline paired image-to-image translation.  
- [ ] **pix2pixHD** (Wang et al., 2018) – high-resolution image synthesis.  
- [ ] **SPADE** (Park et al., 2019) – spatially-adaptive normalization for better semantic alignment.

### 🔹 Diffusion-based
- [ ] **Semantic Diffusion Models (SDM)** – diffusion models conditioned on semantic maps.  
- [ ] **ControlNet (on Stable Diffusion)** – semantic map as structural condition.  

### 🔹 Flow-based
- [ ] **Latent Flow Matching (LFM)** – flow-based generative models in latent space.  

### 🔹 Other Conditional Generators
- [ ] **cGAN with PatchGAN discriminator** (classic baseline).  
- [ ] **VQGAN + semantic conditioning**.  
- [ ] **StyleGAN-based semantic synthesis** (conditioning via masks or embeddings).  

---

## 📊 Roadmap

1. ✅ Setup training loop with PyTorch Lightning.  
2. ✅ Implement datamodules for Cityscapes / CMP Facade.  
3. 🔜 Add **pix2pix** baseline.  
4. 🔜 Extend to **pix2pixHD** and **SPADE**.  
5. 🔜 Integrate **diffusion** and **flow matching** models.  
6. 🔜 Add evaluation metrics (FID, mIoU).  

---

## 📖 References

- Isola et al. *Image-to-Image Translation with Conditional Adversarial Networks*, CVPR 2017.  
- Wang et al. *High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs*, CVPR 2018.  
- Park et al. *Semantic Image Synthesis with Spatially-Adaptive Normalization*, CVPR 2019.  
- Rombach et al. *High-Resolution Image Synthesis with Latent Diffusion Models*, CVPR 2022.  
