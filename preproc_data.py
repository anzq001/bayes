import os
import random
import re

filepath = "./raw_data"
data_train = "./mid_data/data.train"
data_test = "./mid_data/data.test"
pattern = re.compile(r"[\w\u4e00-\u9fff]+")
threshold = 0.8

def setLabel(filename):
    if filename.find("business") != -1:
        return "business"
    elif filename.find("yule") != -1:
        return "yule"
    elif filename.find("it") != -1:
        return "it"
    elif filename.find("sports") != -1:
        return "sports"
    elif filename.find("auto") != -1:
        return "auto"
    else:
        return None

def filetolist(filename):
    wordlist = []
    with open(filename, "r", encoding="utf-8") as file:
        for line in file.readlines():
            for word in line.split():
                if pattern.fullmatch(word):
                    wordlist.append(word)
    return wordlist

def preprocData(filepath):
    try:
        trainfile = open(data_train, "w", encoding="utf-8")
        testfile = open(data_test, "w", encoding="utf-8")
        taglist = []
        filelist = []
        for filename in os.listdir(filepath):
            tag = setLabel(filename)
            filelist.append(filename)
            taglist.append(tag)
            wordlist = filetolist(os.path.join(filepath, filename))

            if random.random() < threshold: #训练数据
                trainfile.write(tag + "#")
                trainfile.write(" ".join(wordlist))
                trainfile.write("\n")
            else:
                testfile.write(tag + "#")
                testfile.write(" ".join(wordlist))
                testfile.write("\n")
        trainfile.close()
        testfile.close()

    except IOError as e:
            print(e)
            exit(0)



if __name__ == "__main__":
    preprocData(filepath)