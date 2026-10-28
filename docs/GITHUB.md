# Upload the prepared code to ChananyuKML

Repository: **ChananyuKML/AI-generated-image-detection-research**. The project title
is "AI-generated image detection research"; the GitHub repository name uses hyphens
in place of spaces. The repository already exists. The upload helper previews its
actions unless passed `--push`; historical first-publication instructions remain
below for reference. Push when explicitly requested by the owner.

## Current instructions: pull and download full GenImage

The full eight-generator source downloader, provider catalogs, tests and usage
instructions are in this repository. On another PC, follow
[FULL_GENIMAGE: Start on another PC](FULL_GENIMAGE.md#start-on-another-pc)
to clone or pull, create a local virtual environment and download using the
tested community mirror. The official Google Drive source hit quota limits in
the 28 October check. The mirror is pinned and its published hashes are verified.

For an existing Windows checkout:

```powershell
git status --short
git pull --ff-only
# Create the venv only if this PC does not have one yet:
# python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/download.txt
.\download_genimage.ps1 -Mode list -Source huggingface
# Example: start one generator on your chosen data drive.
.\download_genimage.ps1 -Mode download -Source huggingface -Output 'D:\GenImage\hf_archives' -Generators sdv4
```

All eight generators require 608.33 GiB of compressed mirror archives, plus the
script's 10 GiB reserve and additional extraction space. Use a sufficiently large
drive for the complete release. Full-data preparation and eight-generator
training integration are still pending; current Tiny checkpoints are not full
GenImage results. Models, datasets, local test downloads and experiment outputs
remain outside Git and need a separate transfer if required.

## Create the empty GitHub repository

Sign into the intended `ChananyuKML` account and open
[GitHub's new repository page](https://github.com/new). Set the repository name to
`AI-generated-image-detection-research`. Choose private or public as appropriate for the
research. Leave GitHub's README, .gitignore, and license initialization unchecked;
the prepared local project supplies the first commit.

For a private repository, the other PC must also authenticate with an account
that can access it. Git for Windows can use its normal browser authentication.
Do not put access tokens in repository URLs, scripts, or documentation.

## Review the local source snapshot

Open a terminal in `AI-generated-image-detection-research`:

```sh
git status --short
git ls-files
git log -1 --oneline
python scripts/publish_github.py
```

Only source, tests, small configuration, and documentation should be tracked.
The helper checks for tracked image datasets, environments, checkpoints, local
archives, credentials files, and large files, then previews the destination/commands.

If using a source-only ZIP export, initialize and commit it first:

```sh
git init -b main
git add .
git diff --cached --stat
git commit -m "Prepare portable binary detection workflow"
```

If Git needs an identity, configure your own name/email before committing.

## Push after the empty repository exists

```sh
python scripts/publish_github.py --push
```

This adds `origin` when absent and runs `git push -u origin main`. It never replaces
an origin pointing at another destination and never force-pushes. If the remote
already contains commits, fetch and review them before integrating histories.

Without the helper, the equivalent commands are:

```sh
git remote add origin https://github.com/ChananyuKML/AI-generated-image-detection-research.git
git push -u origin main
```

## Clone, update, and train on another PC

For later source updates on the current PC, first read [UPDATE_LOG.md](UPDATE_LOG.md)
and review `git status --short` and `git diff`. Stage only the intended source,
tests, small configuration, and documentation, then create a local commit.
Run `python scripts/publish_github.py` to preview the destination and source-only
checks, then use `git push origin main` or the helper's explicit `--push` option
when a push is requested. Update-log results summarize local measurements;
the ignored datasets and complete run folders require a separate transfer.

```sh
git clone https://github.com/ChananyuKML/AI-generated-image-detection-research.git
cd AI-generated-image-detection-research
python scripts/run.py pipeline --dataset progan_poc --compute cu128 --epochs 20
```

For later updates, commit any local source edits before pulling, then run:

```sh
git pull --ff-only
python scripts/run.py setup --compute cu128
python scripts/run.py train --dataset progan_poc --device cuda --check-only
```

Existing data remains on that PC. Set up CPU instead if the destination has no
supported NVIDIA GPU. Use `python3` on Linux when needed. See [SETUP.md](SETUP.md)
and [DATASETS.md](DATASETS.md) for machine setup and transferring datasets.

## What stays outside Git

`.venv/`, downloaded/prepared datasets, derived image indices/manifests,
`runs/`, `checkpoints/`, `output/`, `tmp/`, and `local_archive/` are ignored.
Store model/results backups separately. No code license was inferred from the
research files; add the license you choose before a public release.
