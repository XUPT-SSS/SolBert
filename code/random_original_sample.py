import linecache
import random
import  config as cf

def get_sample(total,num):
    originFile = cf.target_data_path
    # 原文件，从这里边随机获取一行
    f_new = open('/data/xqke/paperCode1/code/random_10k_sample.txt', 'w')
    # 新文件，随机获取的都写到这个里
    random_num = []
    curtotal = 0
    while(curtotal<num):
        # 随机获取一行数据
        lineNumber = random.randint(1, total)
        # 随机数作为行数
        line = linecache.getline(originFile, lineNumber)
        # 随机读取一行
        if(lineNumber  in random_num or len(line)==0):
            continue
        if len(line) == 0:
            continue
	    # 写入新的一个文件
        print(line)
        f_new.write(line)
        curtotal = curtotal+1
        random_num.append(lineNumber)
	# 不再读取时，需要清除缓存
    linecache.clearcache()
    # 关闭文件
    f_new.close()

if __name__ =='__main__':
    file = open(cf.target_data_path)
    total =  len(file.readlines())
    print(total)
    get_sample(total, 100)

