# Fixed benchmark split: baseline124-v1

These four artifacts are copied byte-for-byte from [HiimRaccoon/cs406-geoguessr](https://github.com/HiimRaccoon/cs406-geoguessr/tree/33cb29d72974f1677efab0f56ed92360e5a9f712/artifacts/baseline124), commit `33cb29d72974f1677efab0f56ed92360e5a9f712`:

- `manifest.csv`: exact membership, original label IDs, image SHA-256 and raw dimensions.
- `summary.json`: original preparation metadata and fingerprints.
- `labels.json`: original ordered 124-label mapping.
- `class_support.csv`: original class support counts.

Do not regenerate the split from seed 406, reorder the label mapping, rewrite CSVs, change newline conventions, or edit the source metadata. The original `data_root_at_preparation` is historical metadata, not a required runtime path. Resolve `relative_path` against the configured data root.

`provenance.json` pins the source commit, source Git blob IDs, SHA-256 and byte length of all four files, including `summary.json`. `.gitattributes` disables newline conversion for these exact artifacts.

Expected counts: train 34,997; validation (`val`) 7,497; test 7,502; total 49,996. Class coverage: 124 / 106 / 111. Exact deduplication has already removed one raw duplicate. Geographic independence is not verified.

Before any pilot or training, compare actual file SHA-256 with `provenance.json` and the frozen preprocessing config, compare the three source fingerprints with `summary.json`, validate label IDs, and check split counts and class support. Fail on disagreement. Local copied bytes have been verified against the pinned source Git blobs.
