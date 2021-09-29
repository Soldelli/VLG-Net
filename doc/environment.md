# Below commands will allow you to create an environment to run our code

Tested on Linux machine.

```bash
conda create --name vlg python=3.7 -y

conda activate vlg

conda install pytorch==1.5.1 torchvision==0.6.1 -c pytorch -y
conda install -c anaconda pyyaml==5.3.1 h5py==2.10.0 scipy==1.1.0 joblib=0.16.0 -y
conda install -c conda-forge spacy==2.0.12 terminaltables==3.1.0 -y
conda install -c cogsci pyspellchecker==0.5.2 -y

pip install yacs==0.1.7
pip install stanfordcorenlp==3.9.1.1 
pip install torchtext==0.6.0

# Might be necessary 
conda deactivate
```

