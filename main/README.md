# Chinese song generation

Main code is from [here](https://github.com/suriyadeepan/easy_seq2seq) and [here](https://github.com/billycyy/chinese_song_generation)

## Setup

tensorflow version: 0.12.1

## Intro

1. data 文件夹下是 train 时所需的数据，需 train.enc, train.dec，两个文件行数相同，一一对应，其中，'，'为单个输入中句子与句子的分隔符，'\n'为多次输入间的分隔符。 test.enc, test.dec 类似。编码为 utf-8。
2. preprocess 文件夹下包含原始数据（形式为 title 和 lyrics 的列表），以及将这些数据转化成模型输入（即 train/test.enc/dec）的代码。**用 data.py 来生成训练数据并拷到外面 data 文件夹下**
3. working_dir 为工作目录，运行程序前需 mkdir working_dir 确保该文件夹存在

## Training

```bash
# edit seq2seq.ini file to set 
#		mode = train
# put lyrics into train.enc, train.dec, (test.enc, test.dec)
python execute.py
# or use custom ini file
#		python execute.py my_custom_conf.ini
```

## Testing

```bash
# edit seq2seq.ini file to set 
#		mode = test
python execute.py
```

## Serve

```bash
# configuration : seq2seq_serve.ini
# move work_dir to ui folder. modify checkpoint to be the checkpoint you want
python ui/app.py
# wait until this message shows up
#		"Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)"
# open up the address in browser, chat with the bot
```

