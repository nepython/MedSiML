# MedSiML: A Multilingual Approach for Simplifying Medical Texts
Health literacy is crucial yet often hampered by complex medical terminology. Existing simplification approaches are limited by small, sentence-level, and monolingual datasets. To address this, we introduce **MedSiML**, a large-scale dataset designed to simplify and translate medical texts into the ten most spoken languages, improving global health literacy. This repository contains our 64k paragraph level medical text simplification dataset created from multiple sources using automated annotation process involving Gemini model and manual scrutiny. We also share our code for fine-tuning and evaluating the models. The model checkpoints being very large were not committed. They would be later made available on DropBox post publication and the link would be shared here.

![MedSiML Logo](path_to_logo_image)

## Table of Contents
- [Introduction](#introduction)
- [Dataset Overview](#dataset-overview)
- [Methodology](#methodology)
- [Results](#results)
- [Usage](#usage)
- [Contributors](#contributors)
- [Citation](#citation)
- [License](#license)

## Dataset Overview

**MedSiML** includes over 64,000 paragraphs from PubMed, Wikipedia, and Cochrane reviews, simplified into English, Mandarin, Spanish, Arabic, Hindi, Bengali, Portuguese, Russian, Japanese, and Punjabi. An additional super-simplified English version is available for those with learning disabilities.

- **Sources**: PubMed, Wikipedia, Cochrane Reviews
- **Languages**: English, Mandarin, Spanish, Arabic, Hindi, Bengali, Portuguese, Russian, Japanese, Punjabi
- **Simplification Model**: Flash-1.5 Gemini model
- **Fine-tuning Model**: Text-To-Text Transfer Transformer (T5) base model

## Methodology

### Data Collection

We compiled data from three main sources:
1. **PubMed**: 50,000 abstracts from biomedical articles.
2. **Wikipedia**: Biomedical articles sourced from Hugging Face 2022 Wikipedia corpus.
3. **Cochrane Reviews**: Derived from existing Cochrane datasets.

### Data Cleaning

The collected data underwent rigorous cleaning to remove noise, ensure quality, and eliminate duplicates.

### Annotation

Using the Flash-1.5 Gemini model, we simplified and translated the texts into ten languages and created a super-simplified English version.

### Model Training

We fine-tuned the T5 base model on this paragraph-level, multilingual data, achieving significant improvements in readability and semantic similarity.

## Results

Our fine-tuned model showed improvements over existing models in various metrics:
- **ROUGE1**: +10.61%
- **SARI**: +11.01%
- **Semantic Similarity**: +49.1%
- **Readability Scores**: FK +0.38, ARI +1.06

## Usage

To use the MedSiML dataset and models, follow these steps:

1. **Clone the Repository**
    ```bash
    git clone https://github.com/nepython/MedSiML.git
    cd MedSiML
    ```

2. **Installation**
    ```bash
    pip install -r requirements.txt
    ```

3. **Accessing the Dataset**

   Using `datasets` library
    ```python
    from datasets import load_dataset
    
    raw_datasets = load_dataset("csv", data_files='data/data.tsv', delimiter='\t')
    ```

   Using `pandas` library
    ```python
    import pandas as pd
    
    df = pd.read_csv('data/data.tsv', sep='\t')
    ```

5. **Model Inference**

   * Download the `checkpoints` from DropBox.
   * Move the `checkpoints` into `notebooks` directory.
   * Unzip the `checkpoints`.
   * Open the notebook `T5_base.ipynb` and run cells in all sections skipping `Preprocessing`, `Finetuning`.

## Contributors

- Hardik A. Jain
- Chirayu Patel
- Riyasatali Umatiya
- Sajib Mistry
- Aneesh Krishna

## Citation

If you use this dataset or model in your research, please cite our paper:
```
@inproceedings{Jain2024MedSiML,
  title={MedSiML: A Multilingual Approach for Simplifying Medical Texts},
  author={Hardik A. Jain, Chirayu Patel, Riyasatali Umatiya, Sajib Mistry, Aneesh Krishna},
  booktitle={Proceedings of the Conference},
  year={2024}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
