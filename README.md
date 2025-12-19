# Tweet Sentiment Phrase Extraction using RoBERTa
# 1. Project Overview
This project addresses the **Tweet Sentiment Phrase Extraction** task by fine-tuning
a pre-trained **RoBERTa-base** model to extract the exact text span that justifies
a given sentiment (positive, negative, or neutral) within a tweet.

The task is formulated as an **Extractive Question Answering** problem, where the
sentiment label acts as the query and the tweet text serves as the context. The
model predicts the start and end token positions of the sentiment-bearing phrase.
# 2. Dataset Description
The dataset consists of:
- **Training set**: Approximately 27,500 tweets, each annotated with:
  - "text": original tweet content
  - "sentiment": sentiment polarity (positive, negative, neutral)
  - "selected_text": ground-truth sentiment phrase
  - "textID": ID of each text
- **Test set**: Tweets with sentiment labels only (no ground-truth spans)
# 3. Methodology
- **Model**: RoBERTa-base with two linear span prediction heads
- **Input Format**:  
  `[CLS] sentiment [SEP] tweet text [SEP]`
- **Training Strategy**:
  - Stratified 3-Fold Cross-Validation
  - AdamW optimizer (learning rate = 3e-5)
  - Partial freezing of lower Transformer layers
- **Evaluation Metric**: Character-level F1 score
# 4. Repository Structure
```
├── tokenizer/
├── .gitattributes 
├── 62FIT4ATI_Group-30_Topic-4-TWEET-SENTIMENT-PHRASE-EXTRACTION.ipynb
├── 62FIT4ATI_Group-30_Topic-4-TWEET-SENTIMENT-PHRASE-EXTRACTION.docx 
├── README.md
├── result.csv
└── roberta_span_final.pt
```
# 5. Environment Setup
- Install required dependencies using pip:
   ```pip install transformers torch datasets scikit-learn tqdm matplotlib seaborn```
- Requirements:
  - Python ≥ 3.8
  - PyTorch ≥ 1.10
  - GPU recommended (CUDA), but CPU is supported
# 6. How to Run the Project
# 6.1 Training and Validation
    jupyter notebook 62FIT4ATI_Group-30_Topic-4-TWEET-SENTIMENT-PHRASE-EXTRACTION.ipynb
The notebook includes:
  - Data preprocessing
  - Model training with 3-fold cross-validation
  - Performance evaluation and visualization
# 6.2 Inference and Submission
After training, the final model is used to generate predictions on the test set.
The resulting file is saved as: 
```result.csv```
This file contains:
  - textID
  - selected_text
# 7. Results Summary
  - Mean Character-level F1 (3-Fold CV): 0.6247
  - Stable loss convergence across folds
  - Consistent performance on positive and negative sentiment samples
Note: Neutral samples naturally yield higher F1 scores due to task definition.
# 8. Notes
  - The confusion matrix included in the analysis reflects sentiment distribution
rather than model prediction errors, as the model performs span extraction and
not sentiment classification.
  - Post-processing heuristics are applied during inference to ensure valid span
predictions.
  - Results may vary slightly due to random initialization and GPU nondeterminism.















