# coding:utf-8
# !/usr/bin/python
# -*- coding:UTF-8 -*-
import re
import codecs
import sys
reload(sys)
sys.setdefaultencoding('utf8')




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




def filter_process():
    pro_f = codecs.open('short_new.txt', 'w', 'utf-8')
    for line in f.readlines():
        arr = line.split("\\")
        filtrate = re.compile(u'[^\u4E00-\u9FA50-9]')  # 非中文,非数字
        old_str = filtrate.sub(r' ', arr[0].strip().decode('utf-8'))
        new_str = filtrate.sub(r' ', arr[1].strip().decode('utf-8'))

        old_str =old_str.encode('utf-8')
        new_str = new_str.encode('utf-8')
        res = old_str +" \ " +new_str+"\n"
        if new_str == '' or old_str == '' or len(new_str) == 0 or len(old_str) == 0:
            continue
        pro_f.write(res)
    pro_f.flush()
    pro_f.close()



if __name__ =='__main__':
    first_step()

