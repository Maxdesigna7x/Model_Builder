# User guide

## Train a model

1. Create a model project and select an architecture and compatible task.
2. Import a dataset, inspect its samples, and configure preprocessing and splits.
3. Build the model graph or select a preset/pretrained backbone.
4. Train, compare runs, and select a checkpoint.
5. Open inference to inspect predictions.

Model projects and dataset-curation projects are separate workspaces.

## Architectures and tasks

| Family | Supported work |
| --- | --- |
| MLP | Tabular classification and regression |
| CNN 1D | Sequence classification, regression, forecasting |
| CNN 2D | Image classification, regression, single-box localization, detection, image-to-image |
| RNN / GRU / LSTM | Sequence classification, regression, forecasting |
| Transformer encoder | Sequence tasks and text classification |
| Causal Transformer | Token-based language modeling |
| ViT | Image classification and regression |
| U-Net | Binary/multiclass segmentation and image-to-image |
| Autoencoder | Tabular and image reconstruction |

The backend [catalog](../backend/modelbuilder/catalog.py) defines valid task/architecture combinations. Available templates do not imply support for arbitrary external model checkpoints.

## Data preparation

Use local data, the curated Hugging Face catalog, or Kaggle for image classification. The app displays the expected local format for each task: tables, class folders, sequences, labeled images, masks, or text.

Inspect samples and original image statistics, add preprocessing steps such as resize/channel conversion, then configure train/validation/test splits. Official or temporal splits may be locked to preserve their contract. Some adapters materialize images at a fixed size before the editable pipeline; resizing them later cannot restore lost detail.

For Kaggle, enter an `owner/dataset` identifier or dataset URL. Supported layouts include class folders, split/class folders, class/split folders, and images with an unambiguous CSV mapping filenames to class labels. Select the intended collection or label CSV when necessary. Unlabeled test images require labels rather than guessed classes. Kaggle image-classification imports combine labeled source splits before you choose new splits; imports of supported local partitioned exports can retain their splits.

Private or restricted datasets need authorized credentials and any required license acceptance. Configure credentials using the service's supported mechanisms on the backend machine; do not put tokens in project files. Downloads are cached locally.

## Build and validate

Add layers or presets, connect their ports, and inspect shapes in the graph. Use Add or Concatenate to join branches, and use groups to organize sequences. The inspector edits layer properties; the minimap navigates large graphs.

Validation checks connections, shapes, and supported execution. Where available, **View solutions** proposes repairs verified against a copied graph with prepared data. Applying a proposed repair is explicit and can be undone; it does not optimize model accuracy.

Curated pretrained checkpoints use pinned revisions and do not allow remote model code. Supported vision backbones retain their processor; supported text models retain their tokenizer. Freezing, full fine-tuning, gradual unfreezing, and LoRA depend on the chosen model/task. A backbone is not automatically exposed as fully editable internal layers.

## Train and predict

Configure epochs, optimizer, loss, batch size, and supported advanced options. Training runs in a separate PyTorch process. CPU or CUDA is used according to availability and configuration. Curves distinguish training and validation; accuracy uses a 0–1 scale.

You can pause, resume, or stop a run. Early stopping is off by default; when enabled, it monitors validation loss independently of the best-checkpoint selection criterion. Best/last checkpoints, metrics, and run configuration are kept locally. A new training run does not overwrite a completed run.

Inference uses the selected checkpoint and task. Image classification can show the input, predicted class/probability, and known label. Other tasks provide their corresponding outputs, such as values, boxes, reconstructions, or masks.

## Curate an image dataset

Create a dataset project from the home screen. Import images or extract frames from video, organize classes/groups, review samples, and edit bounding boxes. Background jobs expose progress and support cancellation/recovery where implemented.

The tools include YOLO proposals/crops and EmbeddingGemma-based similarity or clustering. These require their own model weights and compatible runtimes. Proposed labels and boxes should be reviewed before acceptance. YOLO dataset jobs currently use CPU; the desktop NVIDIA selection is not a promise that every auxiliary tool uses CUDA.

Export reviewed samples as image classification or YOLO detection data. Preview omissions and split settings before exporting. Supported exported layouts can be reimported for training. See [known limitations](LIMITATIONS.md) for remaining QA and model coverage.
