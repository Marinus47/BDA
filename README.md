
# [ACM MM 2026] BDA: Learning a Band-Decomposed Adapter for Underwater Instance Segmentation

Junyi Wang, [Guodong Fan](https://ccecfgd.github.io/)<sup>&#42;</sup>, Genji Yuan, [Jinjiang Li](https://scholar.google.com/citations?user=UKD2DtQAAAAJ&hl=zh-CN&oi=sra)<br>
<sup>&#42;</sup> Corresponding author.

## 📚 Introduction
Official implementation of **BDA**, a band-decomposed adapter designed for underwater instance segmentation.

## 📖 Abstract
Fine-tuning vision foundation models (VFMs) has become the dominant paradigm for underwater instance segmentation (UIS), yet existing methods overlook the fact that underwater degradation further aggravates the confusion between target instances and visually cluttered backgrounds, a challenge that general adaptation strategies fail to address effectively. Although frequency-domain style alignment methods have shown some promise, they typically apply uniform operations across the entire spectrum. In contrast, we find that the effects of different degradations are concentrated in different frequency bands of the amplitude spectrum, making band-Decomposed correction a more reasonable strategy. Based on this, we propose Band-Decomposed Adapter (BDA), a parameter-efficient fine-tuning method. Specifically, BDA employs Gaussian functions to partition the amplitude into multiple frequency bands and constructs an independent subspace for each band to perform targeted degradation correction. Furthermore, we design a dynamic routing mechanism that adaptively allocates the contribution of each frequency band according to the global amplitude distribution, thereby enabling robust handling of mixed degradations. Extensive experiments on UIIS and USIS10K show that BDA consistently outperforms existing state-of-the-art methods, validating the effectiveness of band-Decomposed frequency-domain adaptation for UIS.

## 🏗️ Architecture
<p align="center">
  <img src="framework.png" alt="BDA Framework" width="100%">
</p>
