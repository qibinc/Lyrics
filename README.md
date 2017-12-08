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

## References
-   Survey & Guide
    -   [Stanford 2017, Neural Text Generation: A Practical Guide](http://arxiv.org/abs/1711.09534v1)
    -   [Survey of the State of the Art in Natural Language Generation: Core tasks, applications and evaluation](http://arxiv.org/abs/1703.09902v1)
    -   [Deep Learning for NLG](http://mi.eng.cam.ac.uk/~thw28/talks/DL4NLG_20160906.pdf)

-   Style Transfer
    -   [AAAI 2018, Style Transfer in Text: Exploration and Evaluation](http://arxiv.org/abs/1711.06861v2)
    -   [ACL 2017, Hafez: an Interactive Poetry Generation System](http://www.aclweb.org/anthology/P17-4008)
    -   [NIPS 2017 Workshop, Improved Neural Text Attribute Transfer with Non-parallel Data](http://arxiv.org/abs/1711.09395v2)
    -   [EMNLP 2016, Stylistic Transfer in Natural Language Generation Systems Using Recurrent Neural Networks](http://www.aclweb.org/old_anthology/W/W16/W16-60.pdf#page=55)

-   Thematic & Coherent
    -   [COLING 2016, Chinese Poetry Generation with Planning based Neural Network](http://arxiv.org/abs/1610.09889v2)
    -   [ICML 2017, Toward Controlled Generation of Text](http://arxiv.org/abs/1703.00955v2)
    -   [EMNLP 2016, Globally Coherent Text Generation with Neural Checklist Models](http://www.aclweb.org/anthology/D16-1032)


-   Refining & Editing
    -   [NIPS 2017, Deliberation networks: Sequence generation beyond one-pass decoding](http://papers.nips.cc/paper/6775-deliberation-networks-sequence-generation-beyond-one-pass-decoding.pdf)
    -   [Generating Sentences by Editing Prototypes](http://arxiv.org/abs/1709.08878v1)

-   VAE
    -   [Tutorial on Variational Autoencoders](http://arxiv.org/abs/1606.05908v2)
    -   [Google, Generating Sentences from a Continuous Space](http://arxiv.org/abs/1511.06349v4)
    -   [ACL 2017, A Conditional Variational Framework for Dialog Generation](http://arxiv.org/abs/1705.00316v4)
    -   [Improved Variational Autoencoders for Text Modeling using Dilated Convolutions](http://arxiv.org/abs/1702.08139v2)
    -   [A Hybrid Convolutional Variational Autoencoder for Text Generation](http://arxiv.org/abs/1702.02390v1)
    -   [A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues](http://arxiv.org/abs/1605.06069v3)
    -   [Generating Thematic Chinese Poetry with Conditional Variational Autoencoder](http://arxiv.org/abs/1711.07632v1)



-   Reinforcement Learning
    -   [BBQ-Networks: Efficient Exploration in Deep Reinforcement Learning for Task-Oriented Dialogue Systems](http://arxiv.org/abs/1711.05715v2)




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



