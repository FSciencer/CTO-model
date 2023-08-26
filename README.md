# Deep Learning Prediction for Percutaneous Recanalization of Chronic Total Occlusion Using Coronary CT Angiography

# Absract
To automatically predict the 30-min guidewire (guidewire) crossing and percutaneous coronary intervention (PCI) success for chronic total occlusion (CTO), an end-to-end automated prediction framework is proposed, which first segments the coronary artery and then detects candidate CTO lesions based on the delineated coronary artery. After that, this framework extracts the pathological features of CTO lesions and predicts the PCI success rate for CTO. This framework consists of the Patch-UCTNet for coronary delineation, a strategy for CTO lesion detection, the Swin Transformer network for CTO feature extraction, and a classification module for CTO PCI prediction.

# Flowchart of the model
<div align="center">    
<img src="https://user-images.githubusercontent.com/64700979/194312703-2a28018f-a050-4413-8917-98e181258954.png" height="66%" width="66%" />
</div>
![flow_chart](https://github.com/FSciencer/CTO-model/assets/143240318/a77c5ffa-1802-4dff-bbcf-38ac3f70601d)

# Requirements
```python=3.7.1 torch=1.7.0```
