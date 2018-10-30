#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 23:19:29 2018

@author: ljk
"""

import requests

url='https://note.youdao.com/yws/api/personal/file'

YNOTE_CSTK='a9LBBfUj'#固定值
headers={
'Content-Type':'application/x-www-form-urlencoded;charset=UTF-8',
'Cookie':'YNOTE_CSTK='+YNOTE_CSTK+';YNOTE_SESS=v2|04Z_pCF2uRlGhf6zO4640PBRLeyP4gF0qy0MpF0HJF0QZhf6y6MgF0q40H6L0HqS0e4nLlMkLOM0gBOLlEhHP40pK6M640HUG0; YNOTE_LOGIN=5||1540432705735',
'Host':'note.youdao.com',
'Origin':'https://note.youdao.com',
'Referer':'https://note.youdao.com/web/index.html',
'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36'
        }

para={
'method':'listRecent',
'offset':'0',
'limit':'30',
'keyfrom':'web',
'cstk':YNOTE_CSTK
        }
req=requests.post(url=url,headers=headers,params=para)
rj=req.json()