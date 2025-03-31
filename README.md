## Predicting microsatellite instability from whole slide images using texture features
This project utilizes texture features extracted from hematoxylin and eosin (H&E) stained whole slide images (WSIs) in order to predict microsatellite instability. 
Contact: Bahar Tercan btercan@systemsbiology.org  

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








