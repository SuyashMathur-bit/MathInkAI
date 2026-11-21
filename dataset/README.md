# CROHME Handwritten Mathematical Expressions Dataset

This folder contains the preprocessed CROHME dataset used for training and evaluating **MathInk.ai**.

---

### Dataset Details
- **File**: `processed_crohme.pkl`
- **Total Samples**: 21,265 mathematical expressions
- **Format**: Python pickle containing `(processed_data, skipped_files)`
- **Data Structure**:
  - `label`: Raw LaTeX equation string (e.g. `x^2 + 5x + 6 = 0`)
  - `strokes`: Sequential stroke points `[[x1, y1], [x2, y2], ...]`, converted into `[x, y, pen_state]` sequences for CNN-BiLSTM training.

---

### Kaggle Dataset Download Link
Due to GitHub's file size limits, the full 196 MB preprocessed dataset is hosted on Kaggle:
👉 **[suyashma/math-expression1 on Kaggle](https://www.kaggle.com/datasets/suyashma/math-expression1)**

Download `processed_crohme.pkl` from the Kaggle link and place it in this `dataset/` directory before running training.
