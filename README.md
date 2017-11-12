# Lyrics
## Introduction
- Lyrics generation using machine learning techniques.
- FIT 4-508, Tsinghua University.

## Setup
- You should add this 'Lyrics' folder to python path.(the parent folder of this project)
- Add the following line to your ~/.bashrc or ~/.profile

```bash
export PYTHONPATH=$PYTHONPATH:<path_to_Lyrics>
```

- Then setup the dependencies.
- Anaconda is highly recommended.

```bash
conda create -n lyrics python=3.6
source activate lyrics
pip install -r requirements.txt
```

- Meanwhile, you can pull the data and model which will be used.
- Tip: Dropbox is blocked by the GFW. Make sure your terminal can do the work.
- If not, peek into saved/pull.sh and manually download them in your browser and place them under saved/.

```bash
cd saved
bash pull.sh
```

## Subdirectories
### docs
- Documentation can be found at [Lyrics Documentation](https://thucqb.github.io/Lyrics/)

### saved
- Saved data including training set and trained models are kept here.

### utils
- Utility classes handling data and lyrics.

### topic
- Topic model, keyword extraction, clustering.

### seq2seq
- seq2seq implemented in PyTorch.

### rhythm
- Use bigram and word vector distance to predict word, constrained by rhythm.

### crawler
- Lyrics crawler.

### preprocess
- Data cleaning, preprocessing

### app
- App served on [Deep Lyrics](deeplyrics.eastasia.cloudapp.azure.com).
- Frontend in React and backend in Flask.


