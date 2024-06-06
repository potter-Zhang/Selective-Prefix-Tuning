# Selective-Prefix-Tuning
Implementation of paper Selective Prefix Tuning

We use ![P-tuning-v2](https://github.com/THUDM/P-tuning-v2) as our codebase. 
Please note that the tau in this project refers to alpha, and the alpha in this project refers to lambda in the paper.

### Setup
We recommend using a conda environment for this project. To create a conda environment:
'''
    conda create --name spt python=3.8.5
    conda activate spt
'''
install pytorch via:

'''
conda install -n spt pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
'''

install dependencies:
'''
pip install -r requirements.txt
'''

We noticed that the results could be sensitive to the environment. 
On our server, when CUDA_VISIBLE_DEVICES=1, it is a RTX 3090, when CUDA_VISIBLE_DEVICES=0, it is a RTX 3090 Ti.

### Running
'''
bash bash_script/run_boolq_bert_alpha.sh
'''

### Data
For NER tasks, we use the data that is exactly the same as P-tuning-v2 where the dataset could be downloaded.
For SuperGLUE tasks, the data can be obtained through hugging face.

### Results
In results/, we include some of our results.





