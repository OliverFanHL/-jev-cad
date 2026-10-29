# Running Laya and CLM on Kaggle

Each model has a notebook builder, an input packager, and a result importer. Packaging
runs locally; upload the resulting ZIP to a private Kaggle Dataset and attach
it to the notebook. Download the completed output ZIP and import it locally.

Prepare the cohort, descriptors, questions, and surface features first, following
the [workflow guide](../docs/REPRODUCTION.md). The packagers check the inputs
against their recorded hashes before creating a ZIP.

## Laya adaptation

Prepare the training examples and build the four-epoch input bundle:

```bash
python run.py --config configs/study_full_ft.yaml study-laya-ft-prepare
python kaggle/laya/build_upload.py
python kaggle/laya/create_notebook.py
```

Open `laya/laya_finetune_kaggle.ipynb` on Kaggle, attach the ZIP from
`laya/upload/`, select a GPU, and follow the notebook setup cells.

For the ten-epoch run, build the base bundle above first, then:

```bash
python kaggle/laya/build_upload_10ep.py
python kaggle/laya/create_notebook_10ep.py
```

Use `laya/laya_finetune_10epoch_kaggle.ipynb` with that ZIP. It uses the same
prepared examples and a separate output directory.

Import the downloaded result from the repository root:

```bash
python kaggle/laya/import_results.py /path/to/jev_tmcad_laya_ft_results.zip
```

Add `--ten-epoch` for `jev_tmcad_laya_ft_10epoch_results.zip`. The importer checks
the input bundle, checkpoint hash, and sample coverage. It prints the evaluation
command for the imported run.

## CLM inference

With the shared cohort, descriptors, surface features, and duplicate audit available:

```bash
python kaggle/clm/build_upload.py
python kaggle/clm/create_notebook.py
```

Use `clm/clm_tmcad_kaggle.ipynb` with the resulting ZIP from `clm/upload/`.
The notebook uses vLLM; `clm/study_clm_vllm.yaml` contains its runtime settings.
The local alternative uses `configs/study_clm.yaml` and the Transformers backend.

Import the downloaded CLM result:

```bash
python kaggle/clm/import_results.py /path/to/jev_tmcad_clm_results.zip
```

The default destination is `outputs/study/clm/kaggle_import/`.

## Updating the notebooks

Generate the notebooks from their Python builders:

```bash
python kaggle/laya/create_notebook.py
python kaggle/laya/create_notebook_10ep.py
python kaggle/clm/create_notebook.py
```

Run the Laya builders in that order. Its base notebook includes the existing
snapshot-recovery and continuation helpers. Rebuild the input ZIPs after source
changes so the notebook and bundled Python package use the same imports.
