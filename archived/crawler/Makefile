all: convert crawl 

convert:
	jupyter nbconvert lyrics_crawler/spiders/lyrics_spider.ipynb --to python

crawl:
	scrapy crawl lyrics -o output.json

