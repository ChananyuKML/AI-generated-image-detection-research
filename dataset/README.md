# Local data workspace

Image datasets are not uploaded to GitHub. The data command creates download,
prepared-image, index, and manifest folders here:

```sh
python scripts/run.py data --datasets progan
```

See [docs/DATASETS.md](../docs/DATASETS.md) for all variants, download sizes,
provenance, custom split protocols, and transferring existing data to another PC.

Add future datasets as `prepared/<name>/{train,val,test}/{nature,ai}/` or select an
external root with `--data-root`. Both labels must be present in each used split.
