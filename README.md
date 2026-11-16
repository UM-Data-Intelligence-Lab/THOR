Official implementation for **THOR: Inductive Link Prediction over Hyper-Relational Knowledge Graphs**. THOR is a fully inductive link prediction technique for hyper-relational knowledge graphs that leverages two parallel NBFNet-based encoders and a transformer decoder to process dual foundation graphs, which supports efficient masked training while obviating the reliance on costly negative sampling via random corruption. To set up the execution environment with Python 3.9+, please run the code below:
To set up the execution environment with python 3.9+, please run the code below:
```bash
pip install numpy==1.26.0
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-geometric==2.4.0 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install ninja easydict pyyaml six
```
To reproduce our results, please run the code below with the default settings:
```bash
python3 -u run.py
```
