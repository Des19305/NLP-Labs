# LAB 04 — Text Classification

## Folder contents

- `classification.ipynb`: practical workflow for sections 11–23 of W4.pdf.
- `preprocessing.py`: reusable dataset-loading and light text-cleaning functions.
- `results.csv`: output table header; notebook cells append experiment results.
- `confusion_matrix.png`: created when the confusion-matrix cell is run.
- `calculations.pdf`, `prediction.pdf`, `error_analysis.md`, `reflection.md`: intentionally not created; add these yourself later.

## Dataset

Use the SMS Spam Collection dataset. Download it separately and place the dataset file in this folder, for example:

- `SMSSpamCollection` (the original tab-separated UCI file), or
- `spam.csv` (a CSV variant containing `v1`/`v2` or `label`/`text` columns).

If your file has another name or location, change `DATA_PATH` in the first setup cell of the notebook.

## Run

1. Install dependencies in your environment:
   `pip install pandas numpy scikit-learn matplotlib seaborn gensim`
2. Open `classification.ipynb`.
3. Run cells from top to bottom.
4. The notebook does not download the dataset automatically

## Notes

- The notebook splits the data before fitting TF-IDF to avoid data leakage.
- Section 17 uses two standalone example sentences and does not require the SMS dataset.
- Average Word2Vec uses a small Word2Vec model trained on the training messages only. This is a practical baseline, not a pretrained embedding.
- `confusion_matrix.png` is exported by the notebook when the relevant cell is run.
- Interpret your own experimental results and complete the prediction, calculation, error-analysis, and reflection deliverables yourself, following the course AI policy.
