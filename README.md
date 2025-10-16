# [LiVeAction: Lightweight, Versatile, and Asymmetric Codec Design for Real-time Operation](https://ut-sysml.github.io/liveaction)

Modern sensors generate rich, high-fidelity data, yet applications operating on wearable or remote sensing devices remain constrained by bandwidth and power budgets. Standardized codecs such as JPEG and MPEG achieve efficient trade-offs between bitrate and perceptual quality but are designed for human perception, limiting their applicability to machine-perception tasks and non-traditional modalities such as spatial audio arrays, hyperspectral images, and 3D computed tomography. General-purpose compression schemes based on scalar quantization or resolution reduction are broadly applicable but fail to exploit inherent signal redundancies, resulting in suboptimal rate–distortion performance. Recent generative neural codecs, or tokenizers, model complex signal dependencies but are often over-parameterized, data-hungry, and modality-specific, making them impractical for resource-constrained environments. We introduce a \underline{Li}ghtweight, \underline{Ve}rsatile, and \underline{A}symmetric neural codec architecture, called LiVeAction, that addresses these limitations through two key ideas: (1) an FFT-inspired encoder structure that reduces model size and computational complexity, and (2) a simplified rate-based penalty that enhances sample efficiency and generality across modalities. Our design produces codecs that deliver superior rate–distortion performance compared to state-of-the-art generative tokenizers, while remaining practical for deployment on low-power sensors.

- [Paper](https://danjacobellis.net/_static/live_action.pdf)
- [Project webpage](https://ut-sysml.github.io/liveaction)
- [Visual examples (kodak)](https://huggingface.co/danjacobellis/liveaction/tree/main/examples/kodak)
- Additional code accompanying the paper
  - [Training for all modalities](https://github.com/danjacobellis/autocodec/tree/main/train)
  - [Evaluation for all modalities](https://github.com/danjacobellis/autocodec/tree/main/eval)  
- [Pre-trained codecs available on Hugging Face](https://huggingface.co/danjacobellis/autocodec/tree/main)
- [Python package:](https://pypi.org/project/livecodec) `pip install livecodec`


```
@article{jacobellis2025liveaction,
  title={LiVeAction: Lightweight, Versatile, and Asymmetric Codec Design for Real-time Operation},
  author={Jacobellis, Dan and Yadwadkar, Neeraja J.},
  year={2025},
  note={Under review},
  url={https://ut-sysml.github.io/liveaction}
}
```