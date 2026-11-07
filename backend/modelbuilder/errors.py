from __future__ import annotations

from typing import Any


class BackendError(Exception):
    """Exception that carries a stable error code and optional detail for the frontend."""

    def __init__(self, code: str, detail: str | None = None) -> None:
        self.code = code
        self.detail = detail
        message = code if detail is None else f"{code}|{detail}"
        super().__init__(message)


ERROR_MESSAGES: dict[str, str] = {
    # Project / storage
    "PROJECT_INVALID_ID": "Invalid project identifier.",
    "PROJECT_FOLDER_MISMATCH": "The selected folder does not match the project ID.",
    "PROJECT_ALREADY_EXISTS": "A project with that identifier already exists.",
    "PROJECT_MISSING_MANIFEST": "Select a project folder that contains a project.json file.",
    "PROJECT_INCOMPATIBLE_SCHEMA": "The project uses an incompatible schema version.",

    # Data splits
    "SPLITS_INVALID_PERCENTAGES": "Splits must be positive and add up to 100%.",
    "SPLITS_TEMPORAL_ONLY_SEQUENCES": "Temporal splitting is only supported for sequence tasks.",

    # Pipeline validation
    "PIPELINE_TRANSFORM_BEFORE_SPLIT": "{detail} must run after the Split node.",
    "PIPELINE_AUGMENTATION_SCOPE_TRAIN": "Augmentations must be limited to the train split.",
    "PIPELINE_RANDOM_FLIP_REGRESSION_TARGET": "Random flip is unsafe for visual regression unless the target is also transformed.",
    "PIPELINE_NODE_OUT_OF_ORDER": "{detail} is out of order.",
    "PIPELINE_CLIP_INVALID_QUANTILES": "Quantiles must satisfy 0 ≤ lower < upper ≤ 100.",
    "PIPELINE_RESIZE_IMAGES_ONLY": "Resize is only supported for images.",
    "PIPELINE_CROP_IMAGES_ONLY": "Center crop is only supported for images.",
    "PIPELINE_CROP_TOO_LARGE": "Crop size {detail} is larger than the image size.",
    "PIPELINE_CHANNELS_IMAGES_ONLY": "Channel conversion is only supported for images.",
    "PIPELINE_TOO_FEW_SAMPLES": "The pipeline left fewer than three samples.",

    # Pipeline diagnostics
    "DATA_SAMPLES_NON_FINITE": "{detail} samples contain non-finite values.",
    "DATA_SAMPLES_VALID": "{detail} samples verified; shapes and values are valid.",
    "DATA_NONFINITE_VALIDATION_FAILED": "Validation found {detail} samples with NaN or infinity; add an Exclude non-finite node.",
    "DATA_SAMPLES_EXCLUDED_NON_FINITE": "{detail} non-finite samples excluded.",
    "DATA_SAMPLES_EXCLUDED_DUPLICATES": "{detail} exact duplicates removed.",

    # Dataset import / generation
    "DATASET_NUMERIC_REQUIRED": "All variables must be numeric in this version.",
    "IMPORT_PATH_NOT_FOUND": "The selected path does not exist.",
    "IMPORT_TABULAR_REQUIRES_CSV": "This task requires a CSV file.",
    "IMPORT_CSV_TOO_FEW_ROWS": "The CSV needs at least 10 rows.",
    "IMPORT_TEXT_INVALID_EXTENSION": "Text tasks require a .txt, .csv or .tsv file.",
    "IMPORT_TEXT_CLASSIFICATION_TOO_FEW_LINES": "Text classification requires at least 10 label<TAB>text lines.",
    "IMPORT_TEXT_CORPUS_TOO_FEW_LINES": "The corpus needs at least 10 non-empty lines.",
    "IMPORT_SEQUENCE_REQUIRES_CSV": "This task requires a long CSV file.",
    "IMPORT_SEQUENCE_MISSING_COLUMNS": "The columns sequence_id and timestep are missing.",
    "IMPORT_SEQUENCE_INCONSISTENT_LENGTH": "Sequences must have the same length and at least two steps.",
    "IMPORT_SEQUENCE_NO_FEATURES": "No input variables were found.",
    "IMPORT_SEQUENCE_NUMERIC_REQUIRED": "Sequence variables must be numeric.",
    "IMPORT_SEQUENCE_CLASSIFICATION_TARGET_MISSING": "Classification requires a target column.",
    "IMPORT_SEQUENCE_FORECAST_FUTURE_MISSING": "Forecast requires future_1, future_2, … columns in each sequence.",
    "IMPORT_SEQUENCE_REGRESSION_TARGET_MISSING": "Regression requires a target column.",
    "IMPORT_IMAGE_CLASSIFICATION_MIN_CLASSES": "At least two class folders with images are required.",
    "IMPORT_IMAGE_REGRESSION_LABELS_MISSING": "Visual regression requires a labels.csv file with filename,target columns.",
    "IMPORT_IMAGE_REGRESSION_LABELS_INVALID": "labels.csv requires filename and target columns.",
    "IMPORT_IMAGE_REGRESSION_IMAGE_INVALID": "labels.csv references a missing or incompatible image.",
    "IMPORT_IMAGE_RECONSTRUCTION_MIN_IMAGES": "Visual reconstruction requires at least 10 images.",
    "IMPORT_SEGMENTATION_FOLDERS_MISSING": "Segmentation requires images/ and masks/ folders.",
    "IMPORT_SEGMENTATION_MASK_MISSING": "Image {detail} needs exactly one mask.",
    "IMPORT_SEGMENTATION_MIN_PAIRS": "At least 10 image/mask pairs are required.",
    "IMPORT_SEGMENTATION_MULTICLASS_MIN_CLASSES": "Multiclass masks contain fewer than two classes.",
    "IMPORT_TASK_UNSUPPORTED": "No importer exists for this task.",
    "IMPORT_IMAGE_READ_FAILED": "Could not read image {detail}.",

    # Curated Hugging Face datasets
    "HF_DEPENDENCIES_MISSING": "Install the Hugging Face dataset dependencies before downloading public datasets.",
    "HF_DATASET_UNKNOWN": "Unknown Hugging Face dataset: {detail}.",
    "HF_DATASET_INCOMPATIBLE": "The selected dataset is not compatible with this task: {detail}.",
    "HF_DOWNLOAD_FAILED": "The Hugging Face dataset could not be downloaded: {detail}.",
    "HF_DATASET_INVALID": "The downloaded dataset failed validation: {detail}.",
    "HF_MODEL_DEPENDENCIES_MISSING": "Transformers and pretrained-model dependencies are missing.",
    "HF_MODEL_LOAD_FAILED": "The pretrained model could not be loaded: {detail}.",
    "HF_GATED_MODEL_ACCEPTANCE_REQUIRED": "This gated Hugging Face model requires authentication and license acceptance: {detail}.",
    "HF_TOKENIZER_FAILED": "The pinned tokenizer could not be loaded or executed: {detail}.",
    "HF_RAW_TEXT_REQUIRED": "The model requires raw text: {detail}.",
    "HF_DOUBLE_NORMALIZATION": "Remove dataset normalization; the pinned image processor already normalizes inputs.",

    # Graph / model
    "GRAPH_EMPTY": "The graph is empty.",
    "GRAPH_EDGE_TO_MISSING_NODE": "A connection points to a block that does not exist.",
    "GRAPH_CYCLE": "The graph contains a cycle.",
    "GRAPH_SINGLE_INPUT_OUTPUT_REQUIRED": "Exactly one input and one output block are required.",
    "GRAPH_NO_PATH_INPUT_OUTPUT": "There is no path from input to output.",
    "GRAPH_DISCONNECTED_BLOCKS": "Blocks disconnected from the input: {detail}.",
    "GRAPH_BLOCK_MISSING_INPUT": "Block {detail} has no input.",
    "GRAPH_BLOCK_EXECUTION_FAILED": "Block {detail} failed.",
    "GRAPH_MISSING_INPUT_OUTPUT": "The graph needs an input and an output block.",
    "GRAPH_DRY_RUN_NON_FINITE": "The dry run produced non-finite values.",
    "GRAPH_DRY_RUN_GRADIENT_INVALID": "The dry run produced missing or non-finite gradients.",
    "GRAPH_OUTPUT_SHAPE_MISMATCH": "Graph output shape {detail} does not match the task.",

    # CNN vision contracts and training
    "GRAPH_ADD_INPUT_SHAPE_MISMATCH": "Add inputs must have identical shapes: {{detail}}.",
    "GRAPH_CONCAT_AXIS_INVALID": "Concat axis is outside the tensor rank: {{detail}}.",
    "GRAPH_CONCAT_INPUT_SHAPE_MISMATCH": "Concat inputs must match outside the concatenation axis: {{detail}}.",
    "GRAPH_DETECTION_OUTPUT_NAMES_REQUIRED": "Detection requires outputs named boxes and classes. {{detail}}",
    "GRAPH_DETECTION_OUTPUT_SHAPE_INVALID": "Detection heads must be boxes [4,H,W] and classes [K,H,W] on the same grid: {{detail}}.",
    "GRAPH_LOCALIZATION_OUTPUT_SHAPE_INVALID": "Localization must return four normalized box coordinates: {{detail}}.",
    "GRAPH_MERGE_REQUIRES_TWO_INPUTS": "Add and Concat need at least two connected inputs: {{detail}}.",
    "GRAPH_MULTI_OUTPUT_RECIPE_REQUIRED": "These outputs need an explicit training recipe: {{detail}}.",
    "GRAPH_OUTPUT_NAME_DUPLICATE": "Output names must be unique: {{detail}}.",
    "GRAPH_PORT_OCCUPIED": "This input port already has a connection: {{detail}}.",
    "GRAPH_SINGLE_INPUT_REQUIRED": "The graph must have one input and one or more outputs.",
    "GRAPH_SOURCE_PORT_NOT_FOUND": "The source port does not exist: {{detail}}.",
    "GRAPH_SOURCE_PORT_REQUIRED": "Choose a named output port on this block: {{detail}}.",
    "HF_SPATIAL_FEATURES_UNAVAILABLE": "The selected backbone does not expose enough spatial feature maps: {{detail}}.",
    "IMPORT_CSV_HEADER_INVALID": "Could not read the header in {{detail}}.",
    "IMPORT_DETECTION_ANNOTATION_INVALID": "The detection annotation is invalid: {{detail}}.",
    "IMPORT_DETECTION_BOX_OUT_OF_RANGE": "A detection box is outside the image or has zero area: {{detail}}.",
    "IMPORT_DETECTION_CLASSES_MISSING": "Provide classes.txt or class names in the annotation file.",
    "IMPORT_DETECTION_CLASS_UNKNOWN": "A detection label refers to an unknown class: {{detail}}.",
    "IMPORT_DETECTION_IMAGE_UNKNOWN": "An annotation references an image that was not imported: {{detail}}.",
    "IMPORT_DETECTION_JSON_INVALID": "The detection JSON format is invalid: {{detail}}.",
    "IMPORT_DETECTION_LABELS_FOLDER_MISSING": "The YOLO dataset needs images/ and labels/ folders.",
    "IMPORT_DETECTION_TOO_FEW_IMAGES": "At least three detection images are required.",
    "IMPORT_DETECTION_YOLO_LINE_INVALID": "The YOLO annotation line is invalid: {{detail}}.",
    "IMPORT_FEATURE_COLUMNS_EMPTY": "Select at least one input feature.",
    "IMPORT_FEATURE_COLUMNS_INVALID": "The selected features include missing columns or the target column.",
    "IMPORT_IMAGE_PAIRS_DUPLICATE_IDS": "Paired image folders contain duplicate IDs: {{detail}}.",
    "IMPORT_IMAGE_PAIRS_FOLDERS_MISSING": "Image-to-image data needs inputs/ and targets/ folders.",
    "IMPORT_IMAGE_PAIRS_SIZE_MISMATCH": "A paired input and target have different original sizes: {{detail}}.",
    "IMPORT_IMAGE_PAIRS_TOO_FEW": "At least three image pairs are required.",
    "IMPORT_IMAGE_PAIRS_UNMATCHED": "Input and target filenames must match exactly: {{detail}}.",
    "IMPORT_LOCALIZATION_BOX_INVALID": "The localization box is invalid: {{detail}}.",
    "IMPORT_LOCALIZATION_LABELS_INVALID": "labels.csv needs filename,xmin,ymin,xmax,ymax columns.",
    "IMPORT_LOCALIZATION_LABELS_MISSING": "Localization data needs labels.csv.",
    "IMPORT_LOCALIZATION_TOO_FEW": "At least three labeled localization images are required.",
    "IMPORT_TARGET_COLUMN_NOT_FOUND": "The requested target column was not found: {{detail}}.",
    "IMPORT_UPLOAD_EMPTY": "No files were selected.",
    "IMPORT_UPLOAD_INVALID_FILE": "Could not read file {{detail}}.",
    "IMPORT_UPLOAD_INVALID_PATH": "The selected folder contains an invalid file path.",
    "IMPORT_UPLOAD_TOO_LARGE": "The selection exceeds the 64 MB upload limit.",
    "MODEL_INVALID_ID": "The model has an invalid identifier.",
    "MODEL_REVISION_EMPTY": "A revision needs at least one block and connection.",
    "PIPELINE_CROP_REMOVED_LOCALIZATION_BOX": "The crop removed the labeled box for image {{detail}}.",
    "TRAINING_DETECTION_BINDINGS_REQUIRED": "Detection needs boxes and classes outputs connected to its target recipe.",
    "TRAINING_DETECTION_CELL_COLLISION": "Two objects fall into the same output cell; this one-box-per-cell recipe cannot assign both: {{detail}}.",
    "TRAINING_DETECTION_TARGET_INVALID": "The detection target is invalid: {{detail}}.",
    "TRAINING_INITIAL_ARCHITECTURE_MISMATCH": "Initial weights belong to a different architecture: {{detail}}.",
    "TRAINING_INITIAL_CHECKPOINT_INVALID": "The initial checkpoint is invalid: {{detail}}.",
    "TRAINING_INITIAL_CHECKPOINT_MISSING": "The initial checkpoint was not found: {{detail}}.",
    "TRAINING_INITIAL_INPUT_CHANNELS_MISMATCH": "The checkpoint input channel count is incompatible: {{detail}}.",
    "TRAINING_INITIAL_LOAD_REPORT_INVALID": "The checkpoint did not provide a valid compatible-weight report: {{detail}}.",
    "TRAINING_INITIAL_NO_COMPATIBLE_WEIGHTS": "No checkpoint weights match this model by name and shape: {{detail}}.",
    "TRAINING_OUTPUTS_EMPTY": "The model returned no outputs for this training task.",
    "TRAINING_OUTPUT_TARGET_SHAPE_MISMATCH": "An output does not match its target shape: {{detail}}.",
    "TRAINING_RESUME_AND_INITIAL_CHECKPOINT_CONFLICT": "Choose either resume training or initialize a new run from weights.",
    "TRAINING_RESUME_ARCHITECTURE_MISMATCH": "The resume checkpoint architecture has changed: {{detail}}.",
    "TRAINING_RESUME_DATASET_MISMATCH": "The resume checkpoint dataset has changed: {{detail}}.",
    "TRAINING_RESUME_GRAPH_MISMATCH": "The resume checkpoint graph has changed: {{detail}}.",
    "TRAINING_RESUME_SPLITS_MISMATCH": "The resume checkpoint data splits have changed: {{detail}}.",
    "TRAINING_RESUME_TASK_MISMATCH": "The resume checkpoint task has changed: {{detail}}.",

    # Model blocks
    "MODEL_SEQUENCE_TOO_LONG": "The sequence exceeds the maximum positional length.",
    "MODEL_D_MODEL_NOT_DIVISIBLE_BY_HEADS": "d_model must be divisible by the number of heads.",
    "MODEL_BLOCK_NOT_SUPPORTED": "Block not yet supported: {detail}.",

    # Inference
    "INFERENCE_NO_CHECKPOINT": "No trained checkpoint exists.",
    "INFERENCE_CHECKPOINT_MISSING": "The active run checkpoint no longer exists.",
    "INFERENCE_MANUAL_IMAGE_FAILED": "Failed to process the manual image: {detail}.",
    "INFERENCE_MANUAL_TABULAR_WRONG_FEATURES": "Expected {detail} feature values.",
    "INFERENCE_DATASET_ARTIFACT_MISSING": "The source dataset artifact is required for test-split inference: {detail}.",

    # Training
    "TRAINING_LOSS_NON_FINITE": "The training loss stopped being finite.",
    "TRAINING_NO_TRAINABLE_PARAMETERS": "The model has no trainable parameters.",
    "TRAINING_CONTROL_INVALID": "The requested training control state is invalid: {detail}.",
    "MODEL_BUNDLE_INVALID": "The selected model bundle is incomplete or invalid: {detail}.",
    "PEFT_DEPENDENCIES_MISSING": "PEFT is required for the selected LoRA recipe.",

    # Engine
    "ENGINE_UNKNOWN_ACTION": "Unknown action: {detail}.",

    # Catalog
    "CATALOG_TASK_UNSUPPORTED": "Unsupported task: {detail}.",
    "CATALOG_ARCHITECTURE_INCOMPATIBLE": "Incompatible architecture {detail}.",
}


def get_message(code: str, detail: str | None = None) -> str:
    """Return the English message for a code, optionally formatting it with detail."""
    template = ERROR_MESSAGES.get(code, code)
    return template.format(detail=detail) if detail else template
