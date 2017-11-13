# Lyrics
## Introduction
- Lyrics generation using machine learning techniques.
- FIT 4-508, Tsinghua University.

## Setup
- You should add this 'Lyrics' folder to python path.
- Add the following line to your ~/.bashrc or ~/.profile

```bash
export PYTHONPATH=$PYTHONPATH:<path_to_Lyrics>
```

- Then setup the dependencies.
- Anaconda is highly recommended.

```bash
conda env create -f environment.yml
source activate lyrics
```

- Meanwhile, you can pull the data and model which will be used.
- Tip: Dropbox is blocked by the GFW. Make sure your terminal can do the work.
- If not, peek into saved/pull.sh and manually download them in your browser and place them under saved/.

```bash
cd saved
bash pull.sh
```

## Subdirectories
### experiments
- Jupyter notebooks containing experimental records and study notes are kept here.

### docs
- Documentation can be found at [Lyrics Documentation](https://thucqb.github.io/Lyrics/)

### utils
- Utility classes handling data and lyrics.

### attribute
- Topic model, keyword extraction, clustering.

### saved
- Saved data including training set and trained models are kept here.

### app
- App served on [Deep Lyrics](http://deeplyrics.eastasia.cloudapp.azure.com).
- Frontend in React and backend in Flask.

### archived
- Code not in use but may come into use in the future.

#### seq2seq
- seq2seq implemented in PyTorch.

#### rhythm
- Use bigram and word vector distance to predict word, constrained by rhythm.

#### preprocess
- Data cleaning, preprocessing

#### crawler
- Lyrics crawler.



