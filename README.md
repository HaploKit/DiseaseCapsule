# DiseaseCapsule
Predicting the prevalence of complex genetic diseases from individual genotype profiles using capsule networks

## Installation and dependencies
- Linux OS; GPU hardware support
- Python >= v3.7
- PyTorch v1.5.0 (GPU)
- TensorFlow v2.1.0 (GPU)
- sklearn v0.22.2

No need to install the source code. Dependencies can be installed with a few minutes.

## Running code
Our proposed DiseaseCapsule:
- capsule.GPU.py 

Other approaches for comparison:
- MLP.GPU.py
- CNN.GPU.py  
- basicML.py  (used for Gene-PCA)       
- basicML_allpca.py  (used for All-PCA)
- classifier_PRS.py

Potentially diease-related core genes and non-additive genes selection:
- select_core_genes.ipynb
- select_nonadditive_genes.py

## Experimental data
Since the real data used in this study is only available to bona-fide researchers upon request, just to facilitate understanding, synthetic data can be seen here: https://drive.google.com/open?id=1Mya0YdT4Hf9wUfbcX6y5mubEFWky6Jg- 