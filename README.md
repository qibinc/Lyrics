# Lyrics

## crawler
- 用 scrapy 框架的爬虫代码，之前是从 lrcgc.com 上爬的 lrc 文件，该网站提供的 karaoke 歌词格式为 trc，需登录+金币。

## preprocess
- 最先的预处理，筛掉广告、非中文、繁体转简体等。对歌词的简单分析，统计频率等。
- 上面 crawler 和 preprocess 的最终输出是两个文件，titles.pkl 和 lyrics.pkl，分别是歌名和歌词的两个 list。

## main
主要部分