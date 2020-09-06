# DeepSurvivalAnalysis

DeepSigSurvNet  is a not-fully connected deep learning model used for survival analysis and pathway analysis.

## 1. Run the code

To run the code, you need first download TCGA dataset from below following link and process it into coresspoding format:

| **Cancer Type** | **URLs**                                                     |
| --------------- | ------------------------------------------------------------ |
| BRCA  (n=1057)  | [https://xenabrowser.net/datapages/?cohort=TCGA%20Breast%20Cancer%20(BRCA)&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443](https://xenabrowser.net/datapages/?cohort=TCGA Breast Cancer (BRCA)&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443) |
| LUAD  (n=500)   | [https://xenabrowser.net/datapages/?cohort=TCGA%20Lung%20Adenocarcinoma%20(LUAD)&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443](https://xenabrowser.net/datapages/?cohort=TCGA Lung Adenocarcinoma (LUAD)&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443) |
| GBM  (n=484)    | [https://xenabrowser.net/datapages/?cohort=TCGA%20Glioblastoma%20(GBM)&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443](https://xenabrowser.net/datapages/?cohort=TCGA Glioblastoma (GBM)&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443) |
| SKCM  (n=358)   | [https://xenabrowser.net/datapages/?cohort=TCGA%20Melanoma%20(SKCM)&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443](https://xenabrowser.net/datapages/?cohort=TCGA Melanoma (SKCM)&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443) |

The model is implemented by `Keras=2.2.4`. To analysis pathways, you need also install [Innvestigate package](https://github.com/albermax/innvestigate) 

To train the model, open `train.py` file and set apporiate parameters for data load path, save path, model parameters, etc. Then run the file.

To investigate pathway, open `pathway_analysis.py` file and set apporiate parameters for model and data load path(save path when you train the model), model parameters, etc. Then run the file.

## 2. Cite the paper

if you find it is helpful in your research, please cite:

https://www.biorxiv.org/content/10.1101/2020.04.13.039487v1.abstract **Investigate the relevance of major signaling pathways in cancer survival using a biologically meaningful deep learning model**

