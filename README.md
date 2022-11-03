# Impact Pharmacie

We identify high-quality publications demonstrating the impact of pharmacists on health outcomes. Impact Pharmacy is transparent, reproducible, and evidence-based.

---

## Context

The amount of scientific literature describing the impact of pharmacists on health outcomes is increasing. Pharmacists have limited time to keep up with current evidence. We offer a curation service for these publications by identifying high-quality publications through [a rigorous, transparent and reproducible methodology](https://impactpharmacie.net/methodology/). We broadcast these publications weekly in [our mailing list](https://impactpharmacy.net). Through this methodology, we are building a dataset of publications related to pharmacy practice. The features of this dataset are the title and abstracts of all publications retrieved by our search strategy. The labels indicate whether or not this publication met our inclusion criteria (as determined indepentently by two reviewers and through consensus when there is disagreement), the study design, the pharmacy practice field for which this publication is applicable as well as the practice setting (see details in methodology). This dataset grows every week. The complete dataset is available as csv files updated weekly within the present repository under the `data/second_gen` directory. Alternatively, the complete extraction logs and machine learning predictions and ground truths, as well as ratings for the current week, can be found in a Google Spreadsheet [here](https://docs.google.com/spreadsheets/d/1Zm_Wx19BhAf-d3MM18hbxyc8us_Irrsloy_YP_5g-Ao/edit?usp=sharing). We develop machine learning models to help determine if a given abstract meets our inclusion criteria, and to predict its design, field and setting labels. We plan to improve upon these models as our new dataset grows. This repository includes 

- The raw data extracted from the first generation platform
- The dataset from our current generation platform
- The code used to transform the original data into a machine learning compatible dataset
- The code to train and evaluate our machine learning models
- The code to build our current dataset as well as to update our website and to generate our mailing list.

## Files

### Files contained in the credentials directory

These files are examples (containing no actual credentials) of how to create json files containing the authentication credentials for each API acessed by our scripts.

### Files contained in the data/first_gen directory

These files are the raw HTML scraped from the first generation Impact Pharmacie website and are used by the `create_impact_dataset.ipynb` notebook to generate the machine learning dataset.

### Files contained in the data/second_gen directory

These CSV files contain:

- A log of the data extractions made from PubMed (`extraction_log.csv`).
- The titles, abstract texts and ratings of all papers evaluated with our methodology (`ratings.csv`).
- The machine learning tag predictions and ground truths for papers which met our inclusion criteria (`predictions.csv`).
- The exclusion thresholds for the model that predicts if a paper can be automatically excluded or needs to be reviewed manually, as well as associated metrics computed when determining the threshold (`thresholds.csv`).

### create_impact_dataset.ipynb

This notebook creates a dataset containing as features the titles and abstracts from publications included in the original Impact Pharmacie website, as well as titles and abstracts from publications that were not included.

### inclusion_basic_models.ipynb

Coming soon.

### inclusion_transformers.ipynb

This notebook uses Hugging Face Transformers to train and evaluate transformer models on the second generation dataset to determine if the paper should be included in Impact Pharmacie. The notebook also includes the training of production models.

### labels_basic_models.ipynb

This notebook uses scikit-learn to train and evaluate a large number of "classic" machine learning models on the original dataset for label predictions (design, field and setting of included papers).

### labels_transformers.ipynb

This notebook uses Hugging Face Transformers to train and evaluate transformer models on the original dataset for label predictions (design, field and setting of included papers). These models performed better than the basic models, therefore the notebook also includes the training, evaluation and explainability testing of models on the test set as well as the training of production models.

### update_data.py

This script is used to build our new dataset by performing an automated PubMed search.

### update_site.py

This script is used to update our website from our dataset and to generate our newsletter.

### update_inclusion_model.py

This script is used to update the model that predicts if abstracts can be excluded automatically or need to be reviewed for inclusion.

# Prerequisites

Developed using Python 3.9.7

Requires:

- BeautifulSoup
- Gspread
- Hugging Face Datasets
- Hugging Face Transformers
- Numpy
- Pandas
- PyTorch
- Ray Tune
- Scikit-learn
- Scipy
- Shap
- Tqdm


# Contributors

Maxime Thibault
Cynthia Tanguay

# References

- [First generation Impact Pharmacie platform](http://impactpharmacie.org)
- [New Impact Pharmacie website](https://impactpharmacie.net)

# License

GNU GPL v3

Copyright (C) 2022 Maxime Thibault, Cynthia Tanguay.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
