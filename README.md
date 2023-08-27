# Deep Learning Prediction for Percutaneous Recanalization of Chronic Total Occlusion Using Coronary CT Angiography

# Introduction
**Background**: CT imaging is helpful in diagnosis and guiding revascularization of chronic total occlusion (CTO), but time-consuming conventional prediction scores need to be improved. Deep learning (DL) has superior performance and may be used to predict efficiently the percutaneous coronary intervention (PCI) of CTO.

**Purpose**: An end-to-end automated prediction framework is proposed to predict the 30-min guidewire (guidewire) crossing and percutaneous coronary intervention (PCI) success for chronic total occlusion (CTO). The framework first segments the coronary artery and then detects candidate CTO lesions based on the delineated coronary artery. Then, it extracts the pathological features of CTO lesions and predicts the PCI success rate for CTO. This framework consists of the Patch-UCTNet for coronary delineation, a strategy for CTO lesion detection, the Swin Transformer network for CTO feature extraction, and a classification module for CTO PCI prediction.

# Flowchart of the model
<div align="center">    
<img src="(https://github-production-user-asset-6210df.s3.amazonaws.com/143240318/263509656-c28efb4a-58ab-43b7-8699-91014f4fb93b.png)" height="90%" width="90%" />
</div>

# Comparison with other scores 


# Requirements
```python=3.7.1 torch=1.7.0```
