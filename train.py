import os
import preproc_data

train_file = "./mid_data/data.train"
model_file = "./model/bayes.model"
default_count = 1


def get_corpus(filename):
    """
    get corpus info from filename
    :param filename: string
    :return: wordset, word_dict:{tag:{word:count}}, wc_dict:{tag:word_count}
    """
    wordset = set()
    word_dict = dict()
    wc_dict = dict()
    if ~os.path.isfile(filename):
        preproc_data.preprocData(preproc_data.filepath)
    try:
        with open(filename, "r", encoding="utf-8") as file:
            line_num = 0
            for line in file.readlines():
                tag = line.split("#")[0]
                words = line.split("#")[1]
                if word_dict.get(tag) is None:
                    word_dict[tag] = dict()
                wordlist = words.split()
                for word in wordlist:
                    word_dict[tag][word] = word_dict[tag].get(word, 0) + 1
                wordset.update(set(wordlist))
                wc_dict[tag] = wc_dict.get(tag, 0) + len(wordlist)
                line_num += 1
    except IOError as e:
        print(e)
        exit(0)
    return wordset, word_dict, wc_dict


def compute_model(wordset, word_dict, wc_dict):
    """
    :param wordset: word set
    :param word_dict: word count per word in class dict
    :param wc_dict: word count per class dict
    :return: model
    """
    class_prob = dict()
    word_prob = dict()
    default_prob = dict()
    wc = sum(wc_dict.values())
    for tag in wc_dict.keys():
        class_prob[tag] = float(wc_dict[tag])/wc
    for tag in word_dict.keys():
        word_prob[tag] = dict()
        adjust_sum = wc_dict[tag] + default_count * len(wordset)
        for word, count in word_dict[tag].items():
            word_prob[tag][word] = float(count + default_count)/adjust_sum
            default_prob[tag] = float(default_count)/adjust_sum
    return class_prob, word_prob, default_prob


def train_model():
    """
    two steps, first step is get word info from corpus, last step is compute model by corpus
    :return: class_prob: {class: probability from corpus}, word_prob: {class: {word: probability in the class}}
    """
    wordset, word_dict, wc_dict = get_corpus(train_file)
    class_prob, word_prob, default_prob = compute_model(wordset, word_dict, wc_dict)
    return class_prob, word_prob, default_prob


def save_model(*model):
    with open(model_file, "w", encoding="utf-8") as file:
        for ele in model:
            file.write(str(ele) + "\n")


if __name__ == "__main__":
    class_prob, word_prob, default_prob = train_model()
    save_model(class_prob, word_prob, default_prob)