# Download full GenImage

The next dataset is the full eight-generator GenImage release, including SD1.4.
The completed Tiny GenImage runs and original source archives remain intact.
This page covers downloading the official source; these compressed files are
not yet prepared training datasets. The existing seven-generator trainer must
not be pointed at these archives or renamed and described as an 8x8 experiment.

## Start on another PC

Install Git and Python 3.11-3.13. No GPU or Torch installation is needed just to
download the source. For a new checkout on Windows:

```powershell
git clone https://github.com/ChananyuKML/AI-generated-image-detection-research.git
cd AI-generated-image-detection-research
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/download.txt
.\download_genimage.ps1 -Mode list -Source huggingface
# Replace E: with your actual data drive; all files need about 608 GiB plus reserve.
.\download_genimage.ps1 -Mode download -Source huggingface -Output 'E:\GenImage\hf_archives'
```

If the repository already exists, preserve any local source edits, then use
`git pull --ff-only` inside it instead of cloning. Create `.venv` only if absent
and install/update `requirements/download.txt` with that PC's project Python.
Keep each provider's output path unchanged when resuming. The destination PC's
free space and network access must be checked independently.

On Linux:

```sh
git clone https://github.com/ChananyuKML/AI-generated-image-detection-research.git
cd AI-generated-image-detection-research
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements/download.txt
.venv/bin/python tools/download_genimage.py --list --source huggingface
.venv/bin/python tools/download_genimage.py --download --source huggingface --output /data/GenImage/hf_archives
```

For the existing supported training workflows, install model dependencies later
with `python scripts/run.py setup --compute auto` (or `cpu`); the full GenImage
training transition described at the end of this page remains pending.

## Source and storage

The downloader uses the public Google Drive folder linked by the
[official GenImage repository](https://github.com/GenImage-Dataset/GenImage):
[download folder](https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS).
The 28 October 2026 public listing contains 208 file IDs, 207 distinct archive
paths. All eight generators have a final `.zip` and numbered `.z01`, `.z02`, ...
volumes. **Download all volumes, not only the eight `.zip` files.**

Live transfer checks on 28 October found Google Drive quota errors for the GLIDE
and ADM final volumes and an SD1.4 part. Folder metadata remains readable, but
those files could not be downloaded from Drive at that time. Use the explicitly
selected community mirror below when the official source is quota-limited.

| Generator argument | Download files | Compressed GiB |
|---|---:|---:|
| `midjourney` | 71 | 211.85 |
| `sdv4` | 30 | 89.79 |
| `sdv5` | 32 | 94.76 |
| `adm` | 13 | 36.17 |
| `glide` | 11 | 30.05 |
| `wukong` | 31 | 90.56 |
| `vqdm` | 12 | 35.58 |
| `biggan` | 8 | 22.57 |
| **All** | **208** | **611.33** |

GiB means 1024^3 bytes. Sizes are from public Drive metadata, not estimates from
image count. The source has **two IDs for `imagenet_ai_0424_sdv5.z26`**. Both
are downloaded separately and compared by SHA-256. If they differ, the script
stops with both preserved. The canonical filename is never silently overwritten.
The extra comparison file accounts for 3 GiB of the total above.

Allow at least the compressed total plus the script's 10 GiB free-space reserve.
Extraction requires substantial additional space; its size has not been measured.
On the original PC on 28 October, C: had about 68 GiB free and D: 411 GiB. Neither
can hold all source archives simultaneously. Use a larger drive, or fetch selected
generators in stages. The downloader never deletes archives automatically.

## Install the small optional downloader dependency

From the repository root on Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements/download.txt
.\download_genimage.ps1 -Mode list
```

On Linux:

```sh
.venv/bin/python -m pip install -r requirements/download.txt
.venv/bin/python tools/download_genimage.py --list
```

This is independent of Torch installation and does not change the models. Listing
the versioned snapshot requires no network and does not download image bytes.

## Tested alternative when Google Drive has exceeded its quota

The public [jzousz/GenImage community mirror](https://huggingface.co/datasets/jzousz/GenImage)
contains every one of the **207 distinct official archive paths** at matching
byte sizes. `configs/genimage_huggingface.json` pins revision
`71c983e6262684bc2c6b6af99582e8f568c259a5` and the mirror's published SHA-256 for
each file. Downloaded bytes must match those hashes before a file is published as
complete. This verifies the mirrored snapshot; it does not establish byte equality
with official Drive files, whose publisher hashes are unavailable.

Select the mirror explicitly and use a separate output directory from Drive:

```powershell
.\download_genimage.ps1 -Mode download -Source huggingface -Output 'E:\GenImage\hf_archives'
```

The mirror needs **608.33 GiB** for 207 files, plus the 10 GiB reserve and additional
extraction space. It has one version of the duplicated SD1.5 filename, so the
extra 3 GiB Drive duplicate comparison is unnecessary. The repository's unrelated
`genimage_test.zip` and other extras are excluded. The official source remains
the default; no automatic provider switch occurs.

The actual PowerShell launcher downloaded the 51,607,857-byte GLIDE final volume
successfully from the mirror after an intentional interruption at 2,097,152 bytes.
Resumption, exact size, full SHA-256 against the pinned mirror and completed-file
reuse were verified. This checks one archive volume, not full dataset completion
or multipart CRC integrity. To repeat the bounded one-file download:

```powershell
.\download_genimage.ps1 -Mode download -Source huggingface -Output 'D:\GenImage\hf_archives' -Files 'glide/imagenet_glide.zip'
.\download_genimage.ps1 -Mode verify -Source huggingface -Output 'D:\GenImage\hf_archives' -Files 'glide/imagenet_glide.zip'
```

Rerunning without `-Files` downloads the remaining selected generator archives,
subject to disk capacity. A `-Files` run is explicitly labeled
`selected_archive_files_only` and cannot be used with `-TestArchives` because
complete multipart volumes are required for a CRC test. `-Generators` also works
with either provider, for example `-Source huggingface -Generators sdv4`.

## Download all eight generators

Replace `E:\GenImage\archives` with a real directory on a sufficiently large
drive. This is an example destination, not a drive detected on the original PC.

```powershell
.\download_genimage.ps1 -Mode download -Output 'E:\GenImage\archives'
```

Equivalent Python command:

```powershell
.\.venv\Scripts\python.exe tools/download_genimage.py --download --output 'E:\GenImage\archives'
```

```sh
.venv/bin/python tools/download_genimage.py --download --output /data/GenImage/archives
```

The script prints file/byte progress, downloads sequentially, retries transient
transfer failures, continues to the remaining files after individual failures,
and returns a nonzero exit code if anything is incomplete. Disk checks account
for existing files and partial downloads. A directory lock prevents concurrent
writers. Ctrl+C preserves partial data. **Rerun the identical command to resume.**
Completed files are rechecked by size and SHA-256 before being skipped. TLS
verification stays enabled; anonymous public access is used without browser cookies.
Google Drive can throttle files or impose quotas. The script reports those errors
and preserves progress; it cannot override the source's access/quota limits.

## Download selected generators

For example, these fit within D:'s observed free space individually:

```powershell
.\download_genimage.ps1 -Mode download -Output 'D:\GenImage\archives' -Generators sdv4
.\download_genimage.ps1 -Mode download -Output 'D:\GenImage\archives' -Generators biggan
```

Or select several with Python:

```powershell
.\.venv\Scripts\python.exe tools/download_genimage.py --download --output 'D:\GenImage\archives' --generators sdv4 biggan
```

Later commands retain already downloaded generators. A selected-generator success
is labeled `selected_generators_only`; it is not completion of the full dataset.

## Verify, test and extract

The output directory contains the source folder names, `download_catalog.json`
and `download_status.json`. Each successfully downloaded file has its Drive ID,
exact byte count, locally computed SHA-256 and verification timestamp recorded.
These digests establish the local snapshot; the publisher does not supply a
SHA-256 in this catalog, so they are not claimed as publisher-authenticated hashes.

Recheck local files without downloading:

```powershell
.\download_genimage.ps1 -Mode verify -Output 'E:\GenImage\archives'
```

Install [7-Zip](https://www.7-zip.org/) separately to test complete multipart
archives. All numbered volumes and the final `.zip` must be in the same directory:

```powershell
.\download_genimage.ps1 -Mode verify -Output 'E:\GenImage\archives' -TestArchives -SevenZip 'C:\Program Files\7-Zip\7z.exe'
```

This performs an archive CRC test and does not extract. It does not attempt to
validate each `.zNN` as a standalone ZIP. Once testing passes, extract each final
`.zip` with all its adjacent volumes, for example:

```powershell
& 'C:\Program Files\7-Zip\7z.exe' x 'E:\GenImage\archives\stable_diffusion_v_1_4\imagenet_ai_0419_sdv4.zip' '-oE:\GenImage\raw' -aos
```

`-aos` preserves existing extracted files. Keep the downloaded bytes and status
records until extraction and the image inventory have been verified.

## Refresh the public catalog

The checked-in `configs/genimage_downloads.json` makes the download repeatable.
If the publisher changes files or IDs, inspect a fresh listing first:

```powershell
.\.venv\Scripts\python.exe tools/download_genimage.py --refresh-index tmp/genimage_new_catalog.json --list
```

The script discovers the official folder recursively and reads public byte-size
metadata. It rejects missing generators, gaps in numbered volumes and conflicting
duplicate sizes. It never replaces the checked-in snapshot automatically. To use
a reviewed new snapshot, pass `--catalog tmp/genimage_new_catalog.json` and choose
a new output directory if its file identities differ from the existing catalog.

## Training transition

The intended comparison is 8 source-generator detectors per model and 8 target
tests per detector: **40 training runs and 320 test cells for five models, seed 42**.
This is a new full-GenImage batch; the old Tiny 7x7 results remain a separate
experiment. All supplied test images must remain held out; checkpoint-selection
validation must be reserved from supplied training images. Exact duplicate
auditing, prepared manifests, full eight-generator queue/evaluation support and
the actual full-data training run are still pending. Downloading the source does
not claim that those steps have run or that the original paper's training recipe
has been reproduced.
