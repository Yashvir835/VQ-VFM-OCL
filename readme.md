# VQ-VFM-OCL (VVO) : Vector-Quantized Vision Foundation Models for Object-Centric Learning



[![](https://img.shields.io/badge/arXiv-2502.20263-red)](https://arxiv.org/abs/2502.20263)
[![](https://img.shields.io/badge/license-MIT-orange)](LICENSE)
[![](https://img.shields.io/badge/python-3.11-yellow)](https://www.python.org)
[![](https://img.shields.io/badge/pytorch-2.6-green)](https://pytorch.org)
[![](https://img.shields.io/badge/model-checkpoints-blue)](https://github.com/Genera1Z/VQ-VFM-OCL?tab=readme-ov-file#-model-checkpoints--training-logs)
[![](https://img.shields.io/badge/training-logs-purple)](https://github.com/Genera1Z/VQ-VFM-OCL?tab=readme-ov-file#-model-checkpoints--training-logs)



## 🎉 Accepted to ACM MM 2025 as a Poster

Official source code, model checkpoints and training logs for paper "**Vector-Quantized Vision Foundation Models for Object-Centric Learning**".

<img src="res/model_arch_unify.png" style="width:70%">
<img src="res/model_arch_compare.png" style="width:25%">

Supported OCL methods include, categorized by OCL decoding:
- Auto-regressive decoding: [SLATE](https://github.com/singhgautam/slate) vs VVO-Tfd, [STEVE](https://github.com/singhgautam/steve) vs VVO-TfdT, [SPOT](https://github.com/gkakogeorgiou/spot) vs VVO-Tfd9
- Mixture-based decoding: [DINOSAUR](https://github.com/martius-lab/videosaur) vs VVO-Mlp, [VideSAUR](https://github.com/martius-lab/videosaur) vs VVO-SmdT
- Diffusion-based decoding: [SlotDiffusion](https://github.com/Wuziyi616/SlotDiffusion) vs VVO-Dfz

Object discovery performance with DINO2 ViT (s/14) for OCL encoding. VVO is instantiated as VQDINO; Tfd, TfdT, Mlp and Dfz are Transformer, Transformer-temporal, MLP and Diffusion for OCL decoding respectively.

<img src="res/acc_vqdino_all.png" style="width:80%;">

Using higher resolution.

<img src="res/acc_vqdino_r384_coco.png" style="width:40%;">

Qualitative  results.

<img src="res/qualitative.png" style="width:100%;">



## 🌟 Highlights

- ✅ **fp16 fast training** [Automatic mixed precision](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html) training (fp32+fp16) is enabled. Most of the training can be finished less than 4 or 8 hours (for image or video OCL respectively) using one V100 GPU.
- ✅ **less I/O overhead** Datasets are stored in [LMBD](https://lmdb.readthedocs.io) database format to save I/O overhead, beneficial especially on computing cluster.

- ✅ **config-driven experiment** This is totally config-driven framework, largely inspired by [OpenMMLab](https://github.com/open-mmlab), but with much less capsulation.

- ✅ **strong baselines** All models requiring VAE are implemented with StableDiffusion pretrained VAE [TinyVAE](https://huggingface.co/docs/diffusers/v0.30.1/en/api/models/autoencoder_tiny); All models are trained with [strong](https://arxiv.org/abs/2206.07764) data augmentations; All models employ vision foundation model [DINO2](https://huggingface.co/docs/transformers/en/model_doc/dinov2) as their backbone.



## 📂 Repo Stucture

Source code.
```shell
- config-slatesteve/    # configs for SLATE and STEVE
- config-dinosaur/      # configs for DINOSAUR
- config-slotdiffusion/ # configs for SlotDiffusion
- config-vqdino/        # *** configs for our VQDINO ***
- object_centric_bench/
  - datum/              # dataset loading and preprocessing
  - model/              # model building
    - ...
    - vaez.py           # *** for vector-quantization ***
    - vqvfmocl.py       # *** for our VVO model building ***
    - ...
  - learn/              # metrics, optimizers and callbacks
- convert.py
- train.py
- eval.py
- requirements.txt
```

Release.
```shell
- dataset-clevrtex/     # dataset files in LMDB format
- dataset-coco/
- dataset-voc/
- dataset-movi_d/
- slatesteve/           # baseline model checkpoints and training logs
- dinosaur/
- slotdiffusion/
- vqdino_tfd/           # our VQDINO-Tfd models and logs
- vqdino_mlp/           # our VQDINO-Mlp models and logs
- vqdino_dfz/           # our VQDINO-Dfz models and logs
- r384/                 # models and logs trained at resolution 384x384
```



## 🚀 Converted Datasets

Converted datasets, including ClevrTex, COCO, VOC and MOVi-D are available as [releases](https://github.com/Genera1Z/VQ-VFM-OCL/releases).
- [dataset-clevrtex](https://github.com/Genera1Z/VQ-VFM-OCL/releases/tag/dataset-clevrtex): converted dataset [ClevrTex](https://www.robots.ox.ac.uk/~vgg/data/clevrtex).
- [dataset-coco](https://github.com/Genera1Z/VQ-VFM-OCL/releases/tag/dataset-coco): converted dataset [COCO](https://cocodataset.org).
- [dataset-voc](https://github.com/Genera1Z/VQ-VFM-OCL/releases/tag/dataset-voc): converted dataset [VOC](http://host.robots.ox.ac.uk/pascal/VOC).
- [dataset-movi_d](https://github.com/Genera1Z/VQ-VFM-OCL/releases/tag/dataset-movi_d): converted dataset [MOVi-D](https://github.com/google-research/kubric/blob/main/challenges/movi).



## 🧠 Model Checkpoints & Training Logs

***The checkpoints and training logs (@ random seeds 42, 43 and 44) for all models*** are available as [releases](https://github.com/Genera1Z/VQ-VFM-OCL/releases). All backbones are unified as DINO2-S/14.
- [slatesteve](https://github.com/Genera1Z/VQ-VFM-OCL/releases/tag/slatesteve): SLATE on ClevrTex, COCO and VOC; STEVE on MOVi-D.
    - My implementation of paper **Illiterate DALL-E Learns to Compose**, ICLR 2022, achieving much better performance.
    - My implementation of paper **Simple Unsupervised Object-Centric Learning for Complex and Naturalistic Videos**, NeurIPS 2022, achieving much better performance.
- [vqdino_tfd](https://github.com/Genera1Z/VQ-VFM-OCL/releases/tag/vqdino_tfd): VQDINO-Tfd on ClevrTex, COCO and VOC; VQDINO-TfdT on MOVi-D.
    - Our VVO's counterparts to SLATE and STEVE.
- [dinosaur](https://github.com/Genera1Z/VQ-VFM-OCL/releases/tag/dinosaur): DINOSAUR on ClevrTex, COCO and VOC.
    - My implementation of paper **Bridging the Gap to Real-World Object-Centric Learning**, ICLR 2023, replacing its DINO2-B/14 with DINO2-S/14.
- [vqdino_mlp](https://github.com/Genera1Z/VQ-VFM-OCL/releases/tag/vqdino_mlp): VQDINO-Mlp on ClevrTex, COCO and VOC.
    - Our VVO's counterpart to DINOSAUR.
- [slotdiffusion](https://github.com/Genera1Z/VQ-VFM-OCL/releases/tag/slotdiffusion): SlotDiffusion on ClevrTex, COCO and VOC.
    - My implementation of paper **SlotDiffusion: Object-Centric Generative Modeling with Diffusion Models**, NeurIPS 2023, replacing its DINO1-B/8 with DINO2-S/14.
- [vqdino_dfz](https://github.com/Genera1Z/VQ-VFM-OCL/releases/tag/vqdino_dfz): VQDINO-Dfz on ClevrTex, COCO and VOC.
    - Our VVO's counterpart to SlotDiffusion.
- [vqdino-r384](https://github.com/Genera1Z/VQ-VFM-OCL/releases/tag/coco-r384): VQDINO-Tfd/Mlp/Dfz on COCO with resolution 384x384 (336x336).
    - Our VVO's counterparts to the baselines training with higher resolution.



## 🔥 How to Use


### (1) Install requirements

(Using Python version 3.11)
```shell
pip install -r requirements.txt
```
Use package versions no older than the specification.


### (2) Prepare datasets

Download **converted datasets** or convert original datasets into LMDB format: 
```shell
python convert.py
```
But **firstly** download original datasets according to docs of ```XxxDataset.convert_dataset()```.


### (3) Pretrain and train

Run training:
```shell
python train.py
```
But **firstly** change the arguments marked with ```TODO XXX``` to your needs.

Specifically on training:
- For **SLATE/STEVE, SlotDiffusion and VQDINO-Tfd/Mlp/Dfz**, there are two stages for training. For example,
```shell
# 1. pretrain the VAE module
python train.py \
    --seed 42 \
    --cfg_file config-slatesteve/vqvae-coco-c256.py \
    --data_dir path/to/coco

# 2. place the best VAE checkpoint at archive-slatesteve/vqvae-coco-c256/best.pth
mv save archive-slatesteve

# 3. train the OCL model
python train.py \
    --seed 42 \
    --cfg_file config-slatesteve/slate_r_vqvae-coco.py \
    --data_dir path/to/coco \
    --ckpt_file archive-slatesteve/vqvae-coco-c256/best.pth
```

- Please note that:
  - VQDINO-Tfd/Mlp models share the same ``config-vqdino/vqdino-xxx-c256.py`` and corresponding checkpoint as VAE pretraining;
  - VQDINO-Dfz models take ``config-vqdino/vqdino-xxx-c4.py`` and corresponding checkpoint as VAE pretraining.

- For **DINOSAUR**, there is only one training stage. For example,
```shell
python train.py \
    --seed 42 \
    --cfg-file config-dinosaur/dinosaur_r-coco.py \
    --data_dir path/to/coco
```


### (4) Evaluate

Run evaluation:
```shell
python eval.py
```
But **firstly** modify places marked with ``TODO XXX`` according to your needs.



## 💡 Tips

1. Any config file can be converted into typical Python code by changing from
```Python
model = dict(type=ClassName, key1=value1,..)
```
to
```Python
model = ClassName(key1=value1,..)
```

2. All config files follow a similar structure, and you can use file comparator [Meld](https://meldmerge.org) with VSCode plugin [Meld Diff](https://marketplace.visualstudio.com/items?itemName=danielroedl.meld-diff) to check their differences.
<img src="res/meld_diff.png" style="width:75%;">



## 📝 TODO

- ⬜ SPOT & VVO-Tfd9: To be integrated into this framework;
- ⬜ VideoSAUR & VVO-SmdT: To be integrated into this framework.



## 🤗 Contact & Support

I am now working on Object-Centric Learning (OCL). If you have any cool ideas or issues, do not hasitate to contact me!
- WeChat: Genera1Z
- GoogleScholar: [MqlwrKAAAAAJ](https://scholar.google.com/citations?hl=en&user=MqlwrKAAAAAJ&view_op=list_works&sortby=pubdate)
- LinkedIn: [rongzhen-zhao-3b7215247](https://www.linkedin.com/in/rongzhen-zhao-3b7215247)
- eMail: [rongzhen.zhao@aalto.fi](rongzhen.zhao@aalto.fi), [zhaorongzhenagi@gmail.com](zhaorongzhenagi@gmail.com)

If you are **applying OCL (not limited to this repo) to tasks like visual question answering, visual prediction/reasoning, world modeling and reinforcement learning**, let us collaborate!



## ⚗️ Further Research

My further research works on OCL can be found in [my repos](https://github.com/Genera1Z?tab=repositories).



## 📖 Citation

If you find this repo useful, please cite our work.
```
@article{zhao2025vvo,
  title={{Vector-Quantized Vision Foundation Models for Object-Centric Learning}},
  author={Zhao, Rongzhen and Wang, Vivienne and Kannala, Juho and Pajarinen, Joni},
  journal={ACM MM},
  year={2025}
}
```
