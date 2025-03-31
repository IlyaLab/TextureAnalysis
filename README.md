# TextureAnalysis
## Predicting microsatellite instability from whole slide images using texture features
Nilus Swanson$`^1`$ , Mauro A. A. Castro$`^2`$ , A. Gordon Robertson$`^3`$ , Ilya Shmulevich$`^1`$ ,+, Bahar Tercan$`^1`$ ,*   

1 Institute for Systems Biology, Seattle, WA, US  
2 Bioinformatics and Systems Biology Laboratory, Federal University of Paraná, Curitiba, PR 81520-260, Brazil  
3 Dxige Research Inc., Courtenay, British Columbia, Canada, V9N 1C2  
+Deceased, in memoriam  
*Correspondence: btercan@systemsbiology.org  


## Pipeline Overview
![Image of the texture analysis pipeline for predciting MSI.](/images/MSI-vs-MSS-fig.png)

This project utilizes texture features extracted from hematoxylin and eosin (H&E) stained whole slide images (WSIs) in order to predict microsatellite instability. 

## Install and usage
Using the already segmented and MSI labelled tiles from the Kather dataset that can be found at DOI: 10.5281/zenodo.2530834 or at https://zenodo.org/records/2530835, the following example can be implemented.  

In order to use XGBoost, an NVIDIA A100 Tensor Core GPU machine with CUDA toolkit enabled was used through Google Colab.

1. Clone repository.
```
git clone https://github.com/IlyaLab/TextureAnalysis.git 
cd TextureAnalysis
```

2. Install dependencies.  
   NOTE: xgboost changes greatly between versions, so using `xgboost==2.1.4` is the only guarenteed version that works for this example.
```
conda evn create -f environment.yaml
```

3. In a Jupyter notebook, set the paths to tile directories, seperated by labels (already done in Kather dataset).
```
MSI_dir = r"/path/to/TCGA/CRC/MSI/tiles"
MSS_dir = r"/path/to/TCGA/CRC/MSS/tiles"
```

4. Run `main` with the following parameters in order to predict MSI in TCGA-CRC (colorectal cancer) cohort. This will produce a per-tile ROC curve, a bar graph of the feature importance of the model, and a per-sample ROC curve.
```
main(MSI_dir, MSS_dir, sample_size=5000, n_jobs=-1, cohort_name='TCGA-CRC', validation=True, subsets=True, model='CRC', study='TCGA', ML='XGBoost')
```


## License
Apache-2.0 license

## Author Contributions
I.S. conceptualized the study, NS implemented the analyses and visualizations and wrote the initial manuscript draft. B.T., I.S., M.C., G.R. provided guidance, I.S. and B.T. administered the project, B.T., N.S., M.C., G.R.  did manuscript reviewing and editing.

## Acknowledgements
We would like to dedicate this paper to the memory of Ilya Shmulevich, who initiated this study, and co-mentored Nilus Swanson with Bahar Tercan. Nilus Swanson and Bahar Tercan are grateful to the Institute for Systems Biology.

## Citation










