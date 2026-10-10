# Datasets and pilot protocols

Download and prepare any subset of the three sources:

```sh
python scripts/run.py data --datasets cifake progan tiny_genimage
```

Archive snapshots are checked against byte sizes and SHA-256 values recorded in
`tools/download_benchmarks.py`. Preparation decodes source images, groups identical
decoded RGB pixels globally per source, removes conflicting-label groups, and
preserves original image bytes. Splits and balancing use preparation seed **42**,
independent of the training seed. Near-duplicate similarity was not audited.

Each prepared variant has `{train,val,test}/{nature,ai}` folders, `manifest.csv`,
`manifest.json`, and a provenance JSONL file. Images and these generated manifests
remain local. Small publication/provider references are in
[`configs/dataset_sources.json`](../configs/dataset_sources.json).

## Available variants

Counts below are total images across both classes; each split is balanced.
These are the verified local snapshot counts after cleaning and balancing.

| Dataset folder | Train | Validation | Test |
|---|---:|---:|---:|
| `cifake` | 87,664 | 9,740 | 19,924 |
| `cifake_poc` | 8,000 | 2,000 | 2,000 |
| `progan` | 5,600 | 1,200 | 1,200 |
| `progan_poc` | 2,800 | 600 | 600 |
| `tiny_genimage` | 27,998 | 3,500 | 3,500 |
| `tiny_genimage_poc` | 2,000 | 500 | 500 |
| `tiny_biggan` | 4,000 | 498 | 500 |
| `tiny_vqdm` | 4,000 | 500 | 486 |
| `tiny_sdv5` | 4,000 | 494 | 500 |
| `tiny_wukong` | 4,000 | 500 | 484 |
| `tiny_adm` | 4,000 | 476 | 500 |
| `tiny_glide` | 4,000 | 500 | 494 |
| `tiny_midjourney` | 4,000 | 496 | 500 |

The POC, full, and per-generator variants deliberately overlap. They are alternative
views of a source, not independent datasets. Never train on a full variant and then
claim its POC test subset is unseen if that subset was used by another training split.

## CIFAKE

[CIFAKE author repository](https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images)
· [paper](https://arxiv.org/abs/2303.14126)
· [download provider](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

The downloaded ZIP is 109,625,224 bytes, approximately 104.5 MiB, containing 120,000
images. Native images are 32x32: real CIFAR-10 and synthetic Stable Diffusion 1.4.
The existing model resizes them to 224x224, so this is a low-resolution experiment.
The provider reports an MIT license; consult the original source terms as well.

Original test groups remain held out. Validation is derived from the original
training groups using approximately 90/10 stratified allocation. Exact duplicates
are grouped before splitting; cross-split duplicates retain held-out priority.
The final full/POC splits are balanced. This is a documented cleaned derivative of
CIFAKE, not an assertion that its counts equal the original published protocol.

## ProGAN / CNNDetection

[CVPR 2020 project](https://peterwang512.github.io/CNNDetection/)
· [author code](https://github.com/peterwang512/CNNDetection)
· [download release](https://huggingface.co/datasets/sywang/CNNDetection)

The selected `progan_val.zip` is 830,792,545 bytes, approximately 792.3 MiB, with
8,000 images at 256x256 in the downloaded release. It combines LSUN real images
and ProGAN images. Release metadata reports CC-BY-NC-SA-4.0.

This project creates a **custom 70/15/15 pilot split from that validation archive**,
stratified by label and LSUN category. It is useful for an initial train/test check,
but it does not reproduce CNNDetection's official training/evaluation experiment.
The paper describes 4,000 validation images; report the actual 8,000-image release
and snapshot checksum when describing this pilot in a paper.

## Tiny GenImage

[Official GenImage repository](https://github.com/GenImage-Dataset/GenImage)
· [NeurIPS 2023 paper](https://arxiv.org/abs/2306.08571)
· [community subset](https://www.kaggle.com/datasets/yangsangtai/tiny-genimage)

The subset ZIP is 8,345,888,545 bytes, approximately 7.77 GiB, with 35,000 images
from seven generators: ADM, BigGAN, GLIDE, Midjourney, SD1.5, VQDM, and Wukong.
It contains no SD1.4 subset. Provider metadata reports CC-BY-NC-SA-4.0; underlying
ImageNet and generated-image source terms also apply.

Supplied training groups stay in train. Supplied validation groups are divided
between validation and test after global exact-duplicate grouping, then balanced.
The seven per-generator exports use the same global split decisions. This is a
community pilot subset and differs from the full official GenImage protocol.

## Transfer existing data instead of downloading again

The datasets are already prepared on the original PC. Copy one desired variant,
including its manifests, to `dataset/prepared/<variant>/` on the destination.
Then run the check and training commands; the data stage is unnecessary.

Alternatively, copy source ZIPs **and their matching `.json` metadata** into
`dataset/downloads/`, then run:

```sh
python scripts/run.py data --datasets progan --prepare-only
```

Preparation verifies archive hashes and regenerates the image index if absent.
Full variants are exported before smaller variants; hard links share image storage
on filesystems that support them, with copying as a fallback. Allow extra disk space
when transferring to another drive or a filesystem without hard links.

## Add a future dataset

Create a new folder with both classes present in each split:

```text
dataset/prepared/my_dataset/
  train/nature/  train/ai/
  val/nature/    val/ai/
  test/nature/   test/ai/
```

PNG, JPG, and JPEG files may be nested under class folders. Keep related source
images and duplicates in one split. Record dataset provenance and split decisions.
Then run `python scripts/run.py train --dataset my_dataset --check-only`.
Use `--data-root D:/Datasets` (Windows) or `/data/datasets` (Linux) for external data.

For conference experiments, cite source publications, describe these custom splits,
and preserve the generated manifests with the experiment. Evaluate against official
protocols when making claims comparable to published benchmark numbers.
