# DiseaseCapsule
Predicting the prevalence of complex genetic diseases from individual genotype profiles using capsule networks

## Installation and dependencies
- Linux OS; GPU hardware support
- Python >= v3.7
- PyTorch v1.5.0 (GPU)
- TensorFlow v2.1.0 (GPU)

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
