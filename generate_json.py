import json
import random


def extract(file_src, file_tgt, src, tgt):

    lines_src = open(file_src, 'r').readlines()
    lines_tgt = open(file_tgt, 'r').readlines()

    assert len(lines_src) == len(lines_tgt)

    for i in range(len(lines_src)):
        tmp = {}
        tmp["input"] = lines_src[i].strip()
        tmp["input_lang"] = src
        tmp["output"] = lines_tgt[i].strip()
        tmp["output_lang"] = tgt
        res.append(tmp)
    return


res = []

# en <-> cs
extract("dataset/experiment_alma/cs-en/wmt.17-20.en",
        "dataset/experiment_alma/cs-en/wmt.17-20.cs",
        "English", "Czech")
extract("dataset/experiment_alma/cs-en/flores200.dev.en",
        "dataset/experiment_alma/cs-en/flores200.dev.cs",
        "English", "Czech")
extract("dataset/experiment_alma/cs-en/wmt.17-20.cs",
        "dataset/experiment_alma/cs-en/wmt.17-20.en",
        "Czech", "English")
extract("dataset/experiment_alma/cs-en/flores200.dev.cs",
        "dataset/experiment_alma/cs-en/flores200.dev.en",
        "Czech", "English")

# en <-> de
extract("dataset/experiment_alma/de-en/wmt.17-20.en",
        "dataset/experiment_alma/de-en/wmt.17-20.de",
        "English", "German")
extract("dataset/experiment_alma/de-en/flores200.dev.en",
        "dataset/experiment_alma/de-en/flores200.dev.de",
        "English", "German")
extract("dataset/experiment_alma/de-en/wmt.17-20.de",
        "dataset/experiment_alma/de-en/wmt.17-20.en",
        "German", "English")
extract("dataset/experiment_alma/de-en/flores200.dev.de",
        "dataset/experiment_alma/de-en/flores200.dev.en",
        "German", "English")

# en <-> ru
extract("dataset/experiment_alma/ru-en/wmt.17-20.en",
        "dataset/experiment_alma/ru-en/wmt.17-20.ru",
        "English", "Russian")
extract("dataset/experiment_alma/ru-en/flores200.dev.en",
        "dataset/experiment_alma/ru-en/flores200.dev.ru",
        "English", "Russian")
extract("dataset/experiment_alma/ru-en/wmt.17-20.ru",
        "dataset/experiment_alma/ru-en/wmt.17-20.en",
        "Russian", "English")
extract("dataset/experiment_alma/ru-en/flores200.dev.ru",
        "dataset/experiment_alma/ru-en/flores200.dev.en",
        "Russian", "English")

# en <-> zh
extract("dataset/experiment_alma/zh-en/wmt.17-20.en",
        "dataset/experiment_alma/zh-en/wmt.17-20.zh",
        "English", "Chinese")
extract("dataset/experiment_alma/zh-en/flores200.dev.en",
        "dataset/experiment_alma/zh-en/flores200.dev.zh",
        "English", "Chinese")
extract("dataset/experiment_alma/zh-en/wmt.17-20.zh",
        "dataset/experiment_alma/zh-en/wmt.17-20.en",
        "Chinese", "English")
extract("dataset/experiment_alma/zh-en/flores200.dev.zh",
        "dataset/experiment_alma/zh-en/flores200.dev.en",
        "Chinese", "English")


random.shuffle(res)

outfile = open("wmt_test-flores200_dev.cs_de_ru_zh.json", "w")
json.dump(res, outfile, ensure_ascii=False, indent=2)
