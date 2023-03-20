from tqdm import tqdm
if __name__ == '__main__':
    total = 0
    lowFour =0
    lowFive = 0
    lowSeven = 0
    others = 0
    with open("","r") as file:
        for line in file:
            total = total+1
            leng = len(line.strip().split(' '))
            if leng <=400:
                lowFour = lowFour+1
            if leng <=512:
                lowFive = lowFive+1
            if leng <=768:
                lowSeven = lowSeven+1
            if leng >768:
                others = others+1
    file.close()
    print("total:{}".format(total))
    print("lowerFour:{}".format(lowFour))
    print("lowerFive:{}".format(lowFive))
    print("loweSeven:{}".format(lowSeven))
    print("others:{}".format(others))
    print("*"*10)
    print("lowerFour:{}".format(lowFour/total))
    print("lowerFive:{}".format(lowFive/total))
    print("loweSeven:{}".format(lowSeven/total))
    print("others:{}".format(others/total))
    print("Statistics completed")
