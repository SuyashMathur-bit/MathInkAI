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

### Kaggle Dataset Source
You can also access or download this dataset directly on Kaggle:
👉 **[suyashma/math-expression1 on Kaggle](https://www.kaggle.com/datasets/suyashma/math-expression1)**
