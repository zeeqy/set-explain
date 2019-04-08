import json, sys, os

"""
inherit gold set from set expansion into json with id and title

"""

if __name__ == '__main__':
    with open("../data/gold_set.json", "r") as f:
        gold_set_raw = json.loads(f.read())
    f.close()
    wiki_eCaSe = os.listdir("../data/filter_sets/wiki2/")

    gold_set_wiki_eCaSe = []
    for s in wiki_eCaSe:
        if s.replace('.set','') in gold_set_raw.keys():
            gold_set_wiki_eCaSe.append(s.replace('.set',''))

    gold_set_list = []
    for st in gold_set_wiki_eCaSe:
        title = gold_set_raw[st]["title"]
        print(title)
        item = {"id":st, "title":title}
        st += ".set"
        with open("../data/filter_sets/wiki2/{}".format(st), "r") as f:
            set_entity = f.read().split("\n")[0:-1]
        f.close()
        item.update({"entities":set_entity})
        gold_set_list.append(item)

    with open("../data/gold_set_inherit.data", "w+") as f:
        for item in gold_set_list:
            f.write(json.dumps(item) + "\n")
    f.close()