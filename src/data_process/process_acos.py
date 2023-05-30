import pandas as pd
import csv

num2sent = {"0": "negative", "1": "neutral", "2": "positive"}


def process(input_file, output_file):
    """
    convert the original data to unified format for MvP
    """
    wf = open(output_file, 'w')
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        lines = []
        for line in reader:
            cur_sent = line[0]
            quads = line[1:]
            new_quads = []
            for quad in quads:
                words = cur_sent.split()
                a, c, s, o = quad.split()
                a_start, a_end = a.split(',')
                a_start = int(a_start)
                a_end = int(a_end)
                if a_start == -1 or a_end == -1:
                    a = "NULL"
                else:
                    a = " ".join(words[a_start:a_end])
                c = c.replace('#', ' ').lower()
                s = num2sent[s]
                o_start, o_end = o.split(',')
                o_start = int(o_start)
                o_end = int(o_end)
                if o_start == -1 or o_end == -1:
                    o = "NULL"
                else:
                    o = " ".join(words[o_start:o_end])
                new_quads.append([a, c, s, o])
            wf.writelines(cur_sent + '####')
            wf.writelines(str(new_quads))
            wf.write('\n')
    wf.close()


if __name__ == '__process__':
    postfix = ['train', 'test', 'dev']
    for p in postfix:
        process(f'./acos/rest/rest16_quad_{p}.tsv', f'./acos/rest/{p}.txt')

    for p in postfix:
        process(f'./acos/laptop/laptop_quad_{p}.tsv', f'./acos/laptop/{p}.txt')

