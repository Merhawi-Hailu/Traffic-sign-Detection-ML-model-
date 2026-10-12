# Traffic-Sign Classification with PySpark and YOLOv8

An academic computer-vision pipeline that converts traffic-sign annotations into a classification dataset, then trains and evaluates a 76-class YOLOv8 model.

**Recorded test result:** 86.56% top-1 and 99.60% top-5 accuracy on 4,718 cropped signs. These scores measure classification of supplied sign crops; full-image detection was not evaluated.

## Implementation

1. Download Mapillary/DFG data from Kaggle with `kagglehub`.
2. Parse XML annotations with PySpark RDDs and filter invalid bounding boxes.
3. Aggregate class counts, attach image paths and split by source-image key using a deterministic hash.
4. Crop annotated objects to 224 × 224 RGB images using partition-level processing.
5. Fine-tune `yolov8n-cls.pt` for five epochs on CPU.
6. Evaluate validation/test splits, generate predictions and export run metadata.

## Saved-run results

| Split | Images | Top-1 accuracy | Top-5 accuracy |
|---|---:|---:|---:|
| Training | 21,250 | — | — |
| Validation | 4,589 | 86.49% | 99.69% |
| Test | 4,718 | 86.56% | 99.60% |

The saved run uses 76 classes and 30,557 prepared crops. Results come from the notebook outputs and have not been independently rerun for this documentation update.

## Run the notebook

Open [MIT_805_Merhawi_Hailu_Road_sign_detection.ipynb](MIT_805_Merhawi_Hailu_Road_sign_detection.ipynb) in Jupyter or Google Colab and run its cells in order.

```bash
pip install kagglehub pyspark ultralytics pillow pandas scikit-learn
```

A working Java runtime is required by Spark. The source download is approximately 11.6 GB; allow additional space for extracted data and crops. The notebook trains on CPU by default.

Observed environment: Python 3.12.11, PySpark 3.5.1, PyTorch 2.8.0 and Ultralytics 8.3.207.

## Files and generated outputs

- `MIT_805_Merhawi_Hailu_Road_sign_detection.ipynb`: primary implementation and saved results.
- `Machine_Translation_model_for_African_language.ipynb`: separate NLP notebook retained in this repository; see the [dedicated translation project](https://github.com/u17095736-coder/Machine-Translation-Model).
- `data_cls/`: generated train/validation/test crop directories.
- `runs/classify/`: generated training and evaluation outputs.
- `artifacts/run_info.json`: generated run-location metadata.

## Limitations and next steps

The five-epoch run is an academic experiment. Check class imbalance, per-class errors, image-path collisions and the consistency of plotting data when reproducing the analysis. Future work includes longer training, GPU experiments and evaluation on external data.

## Data and attribution

- [Mapillary and DFG traffic-sign dataset on Kaggle](https://www.kaggle.com/datasets/nomihsa965/traffic-signs-dataset-mapillary-and-dfg)
- [Ultralytics documentation](https://docs.ultralytics.com/)
- [PySpark documentation](https://spark.apache.org/docs/latest/api/python/)

Developed by **Merhawi Hailu** for MIT 805. Dataset and pretrained-model usage remain subject to their respective licences.
