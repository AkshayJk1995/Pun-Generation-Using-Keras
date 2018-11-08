# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 16:04:22 2018

@author: AkshayJk
"""

import csv
with open('Data\shortjokes.csv') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter = ',')
    jokes_count = 0
    text_set = set()
    with open('Data\whatjokes.txt', 'w+') as txtfile:
        for row in csv_reader:
            row[1] = row[1].lower();
            if "what do you call" in row[1]:
                jokes_count += 1
                text_set.add(row[1])
                txtfile.write(row[1] + '\n');
    print("Total number of question jokes = " + str(jokes_count))
    print("Number of unique question jokes = " + str(len(text_set)))
    completetext = '\n'.join(text_set)
    chars = sorted(list(set(completetext)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    #print(indices_char)
    lettercount = 0
    lettermax = 0
    lettermin = len(next(iter(text_set)))
    for line in text_set:
        lettercount += len(line)
        if len(line) > lettermax:
            lettermax = len(line)
        if len(line) < lettermin:
            lettermin = len(line)
    lettercount = lettercount/len(text_set)
    print("Maximum sentence length = ", lettermax)
    print("Minimum sentence length = ", lettermin)
    print("Average sentence size = ", lettercount)