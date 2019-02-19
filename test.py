import os
import train
import math
import numpy as np
test_file = "./mid_data/data.test"
model_file = "./model/bayes.model"
tag_list = ["business", "yule", "it", "sports", "auto"]

def load_model():
    class_prob = dict()
    word_prob = dict()
    default_prob = dict()
    if ~os.path.isfile(model_file):
        train.train_model()
    try:
        with open(model_file, "r", encoding="utf-8") as outfile:
            class_prob, word_prob, default_prob = outfile.readlines()
    except IOError as e:
        print(e)
        exit(0)
    return eval(class_prob), eval(word_prob), eval(default_prob)

if __name__ == "__main__":
    class_prob, word_prob, default_prob = load_model()
    predict_list = [0, 0, 0, 0, 0]
    real_tags = []
    predict_tags = []
    correct = 0
    with open(test_file, "r", encoding="utf-8") as file:
        for line in file.readlines():
            real_tag, sentence = line.split("#")
            word_list = sentence.split(" ")
            for tag in word_prob.keys():
                predict_list[tag_list.index(tag)] = sum([math.log(word_prob[tag].get(word, default_prob[tag])) for word in word_list])
            predict_tag = tag_list[np.argmax(np.array(predict_list))]
            real_tags.append(real_tag)
            predict_tags.append(predict_tag)
    predict_count = len(real_tags)
    for i in range(len(real_tags)):
        if real_tags[i] == predict_tags[i]:
            correct += float(1)/predict_count
    print(predict_count, correct)
