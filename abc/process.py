# coding:utf-8
# !/usr/bin/python
# -*- coding:UTF-8 -*-
import re
import codecs
import sys
import xlrd
reload(sys)
sys.setdefaultencoding('utf8')
import unicodedata
import jieba


old_file =  'old_init.txt'
new_file  = 'new_init.txt'
old_f = codecs.open(old_file, 'w', 'utf-8')
new_f = codecs.open(new_file, 'w', 'utf-8')
f = open('abc_new.txt','r')


def first_step():
    dict = {}
    count =0
    for line in f.readlines():
        count +=1
        dict[count] = line
    print 'yes'
    print count
    sorted_dict = sorted(dict.items(), lambda x, y: cmp(x[0], y[0]))
    print 'ok'
    cnt =0
    for item in sorted_dict:
        arr = item[1].split("\\")
        filtrate = re.compile(u'[^\u4E00-\u9FA50-9]')  # 非中文,非数字
        old_str = filtrate.sub(r' ', arr[0].strip().replace(" ",'').decode('utf-8'))
        new_str = filtrate.sub(r' ',arr[1].strip().replace(" ",'').decode('utf-8'))

        old_str = "".join(old_str.strip().encode('utf-8').split())
        new_str = "".join(new_str.strip().encode('utf-8').split())
        if new_str == '' or old_str=='' or len(new_str)==0 or len(old_str)==0:
            continue


        old_f.write(old_str+"\n")
        new_f.write(new_str+"\n")
        cnt +=1
    old_f.flush()
    new_f.flush()
    old_f.close()
    new_f.close()
    print count,cnt
    print 'done'




def filter_process(files):
    cnt = 0
    pro_f = codecs.open('all_new.txt', 'w', 'utf-8')
    for item in files:
        print item
        print cnt
        if item.endswith('txt'):
            if item =="sjfj30000.txt":
                for line in open(item, 'r').readlines():
                    arr = line.split("\\")
                    filtrate = re.compile(u'[a-zA-Z]')  # 去掉英文字符
                    old_str = filtrate.sub(r' ', arr[0].strip().decode('utf-8'))
                    new_str = filtrate.sub(r' ', arr[1].strip().decode('utf-8'))

                    old_str = old_str.encode('utf-8')
                    new_str = new_str.encode('utf-8')

                    if new_str == '' or old_str == '' or len(new_str) == 0 or len(old_str) == 0:
                        continue
                    seg_list = jieba.cut(new_str, cut_all=False)
                    old_str = " ".join(list(old_str.encode("utf-8").decode("utf-8")))

                    res = old_str + " \ " + " ".join(seg_list) + "\n"
                    print res
                    cnt = cnt + 1
                    pro_f.write(res)
            else:
                for line in open(item, 'r').readlines():
                    arr = line.split("\\")
                    filtrate = re.compile(u'[a-zA-Z]')  # 去掉英文字符
                    old_str = filtrate.sub(r' ', arr[0].strip().decode('utf-8'))
                    new_str = filtrate.sub(r' ', arr[1].strip().decode('utf-8'))

                    old_str = old_str.encode('utf-8')
                    new_str = new_str.encode('utf-8')
                    res = old_str + " \ " + new_str + "\n"
                    if new_str == '' or old_str == '' or len(new_str) == 0 or len(old_str) == 0:
                        continue
                    cnt = cnt + 1
                    pro_f.write(res)



        else:
            data = xlrd.open_workbook(item)
            table = data.sheets()[1]
            row_num = table.nrows
            col_num = table.ncols
            values = []
            for i in range(1, row_num):
                current_val = table.row_values(i)
                old = current_val[0]
                new_ = current_val[2]
                filtrate = re.compile(u'[a-zA-Z]')  # 去掉英文字符
                old_str = filtrate.sub(r' ', old.strip().decode("utf-8").encode('utf-8'))
                new_str = filtrate.sub(r' ', new_.strip().decode('utf-8').encode('utf-8'))




                old_str = old_str.decode("utf-8").encode('utf-8')
                new_str = new_str.decode("utf-8").encode('utf-8')

                if new_str == '' or old_str == '' or len(new_str) == 0 or len(old_str) == 0:
                    continue

                seg_list = jieba.cut(new_str, cut_all=False)

                old_str = " ".join(list(old_str.encode("utf-8").decode("utf-8")))

                res = old_str + " \ " + " ".join(seg_list) + "\n"
                cnt = cnt + 1
                pro_f.write(res)



    print cnt

    pro_f.flush()
    pro_f.close()


import requests
from bs4 import BeautifulSoup
if __name__ =='__main__':
    ret = requests.get('https://hanyu.baidu.com/s?wd=亟&from=zici')
    soup = BeautifulSoup(ret.text)
    div = soup.find(name='div', id='pinyin')
    li_list = div.find_all(name='b')