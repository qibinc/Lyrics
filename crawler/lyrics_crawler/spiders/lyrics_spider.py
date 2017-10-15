
# coding: utf-8

# In[15]:


import scrapy, logging


# In[10]:


class LyricsSpider(scrapy.Spider):
    name = "lyrics"
    
    start_urls = [
        'http://www.lrcgc.com/lyric-106-276252.html',
    ]

    allowed_domains = [
        'lrcgc.com'
    ]
    
    def parse(self, response):
        for lrc in response.css('#J_downlrc::attr(href)').extract():
            yield response.follow(lrc, callback=self.parse_lrc)

        for href in response.css('a::attr(href)'):
            yield response.follow(href, callback=self.parse)
    
    def parse_lrc(self, response):
        yield {
            'title': response.headers['Content-Disposition'].decode('utf-8')[22:-1],
            'lyric': response.body.decode('utf-8')
        }

