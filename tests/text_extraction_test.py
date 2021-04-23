# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 12:36:01 2020
@author: Latifa H
"""

import unittest
import sys
import os
import pandas
import numpy

# set current working directory to where this file is saved
thisdir = os.path.dirname(os.path.abspath(__file__)) + "\\"
os.chdir(thisdir)
# Add higher directory to python module's path
sys.path.append("..")

# Local Application Modules
from main import DocSim_Function

class TestCase(unittest.TestCase):

    def test_get_src_txt(self):
        # get_src_txt(srcdir=thisdir, saveas="raw_corpus.csv"):
        # """Extract each line of each file in a directory [srcdir] 
        # of word documents and possibly skill inking files. 
        # Combine into a single CSV of labeled lines from documents. 
        # Save output as raw_corpus.csv or other given filedir."""

        output = main.DocSim_Function.get_src_txt("testDir")

        self.assertEqual(5, output.shape[0])  # rows output
        self.assertEqual("this", output.rawtext[0].values[0].split(" ")[0])  # first word
        self.assertEqual("a", output.rawtext[0].values[0].split(" ")[2])  # middle word
        self.assertEqual("file", output.rawtext[0].values[0].split(" ")[4])  # last word
        self.assertEqual(2, len(output.filename[0].values))  # file names included

    def test_removedonetext(self):

        a = ["a"]
        b = "hello this is just a text"

        dd = {"a": a, "b": b}

        new_df = main.DocSim_Function.removedonetext(dd, "a", "b", "hello")
        self.assertEqual("this is just a text", new_df["b"])

        a = "a"
        dd = {"a": a}

        new_df = main.DocSim_Function.removedonetext(dd, "a", "a", "b")
        self.assertEqual("a", new_df["a"])

        a = numpy.nan
        dd = {"a": a}

        new_df = main.DocSim_Function.removedonetext(dd, "a", "a", "b")
        self.assertTrue(numpy.isnan(new_df["a"])) 

    def test_add_col_from_extract(self):
        data = {'Name': ['Andrew', 'Princess', 'Alex', 'Joe'],
                'Age': [27, 24, 22, 32],
                'Country': ['USA', 'UK', 'China', 'Japan'],
                'Qualification': ['MS', 'MA', 'BS', 'Phd']}
        df = pandas.DataFrame(data)
        new_df = main.DocSim_Function.add_col_from_extract(df, "Qualification", "Qualification_Short", "[A-Za-z]", False, "")

        self.assertEqual("", new_df['Qualification'][0])
        self.assertEqual("", new_df['Qualification'][1])
        self.assertEqual("", new_df['Qualification'][2])
        self.assertEqual("", new_df['Qualification'][3])

        self.assertEqual("M", new_df['Qualification_Short'][0])
        self.assertEqual("M", new_df['Qualification_Short'][1])
        self.assertEqual("B", new_df['Qualification_Short'][2])
        self.assertEqual("P", new_df['Qualification_Short'][3])

        data = {'Name': ['Andrew', 'Princess', 'Alex', 'Joe'],
                'Age': [27, 24, 22, 32],
                'Country': ['USA', 'UK', 'China', 'Japan'],
                'Qualification': ['MS', 'MA', 'BS', 'Phd']}
        df = pandas.DataFrame(data)
        new_df = main.DocSim_Function.add_col_from_extract(df, "Qualification", "Qualification_Short", "[A-Za-z]", True, "")

        self.assertEqual("", new_df['Qualification'][0])
        self.assertEqual("", new_df['Qualification'][1])
        self.assertEqual("", new_df['Qualification'][2])
        self.assertEqual("", new_df['Qualification'][3])

        self.assertEqual(['M', 'S'], new_df['Qualification_Short'][0])
        self.assertEqual(['M', 'A'], new_df['Qualification_Short'][1])
        self.assertEqual(['B', 'S'], new_df['Qualification_Short'][2])
        self.assertEqual(['P', 'h', 'd'], new_df['Qualification_Short'][3])

    def test_group_by_speaker(self):
        data = {
            'script_impl': ['a', 'b', 'b', 'a'],
            'src': ['a', 'b', 'b', 'a'],
            'filename': ['a', 'b', 'b', 'a'],
            'rawtext': ['x', 'v', 'd', 'w'],
            'speaker': ['a', 'b', 'b', 'a'],
            'text': ['r', 't', 'y', 'u']
        }
        df = pandas.DataFrame(data)
        new_df = main.DocSim_Function.group_by_speaker(df)

        self.assertEqual(2, len(new_df))
        self.assertEqual("a", new_df['speaker'][0])
        self.assertEqual("x w", new_df['rawtext'][0])

        self.assertEqual("b", new_df['speaker'][1])
        self.assertEqual("v d", new_df['rawtext'][1])

    def test_clean_text(self):
        data = {
            'rawtext': ['Speaker1 0:02  Hello, good morning.Transcribed by https://otter.ai 15422514 youth.mp4'],
            'filedir': ['period\\document'],
            'file_index': ['first column'],
            'script_impl': ['first column'],
            'src': ['first column'],
            'filename': ['first column'],
            'txtlen': [10]

        }
        df = pandas.DataFrame(data)
        new_df = main.DocSim_Function.clean_text(df)

        self.assertEqual("Speaker1 0:02 Hello, good morning.", new_df['text'][0])  # text replacements, no timestamps
        self.assertTrue(numpy.isnan(new_df['summary_keywords'][0]))
        self.assertTrue(numpy.isnan(new_df['speakers'][0]))

    def test_linkages_connection(self):
        corpus = {
            'rawtext': ['Speaker1 0:02  Hello, good morning.Transcribed by https://otter.ai 15422514 youth.mp4'],
            'filedir': ['period\\document'],
            'file_index': ['first column'],
            'script_impl': ['first column'],
            'src': ['20'],
            'filename': ['1-1A.txt'],
            'txtlen': [10],
            'speaker': ["s"],
            'text': ["Hello, good morning.Transcribed by https://otter.ai"]

        }
        df_corpus = pandas.DataFrame(corpus)
        linkage = {
            'rawtext': ['Speaker1 0:02  Hello, good morning.Transcribed by https://otter.ai 15422514 youth.mp4'],
            'filedir': ['period\\document'],
            'file_index': ['first column'],
            'script_impl': ['first column'],
            'src': ['20'],
            'filename': ['first column'],
            'txtlen': [10],
            'model': [10],
            'id': ['1'],
            'skill': ["a"],
            'speaker': ["s"],
            'period': ['20']

        }
        df_linkage = pandas.DataFrame(linkage)
        new_df = main.DocSim_Function.linkages_connection(df_corpus, df_linkage)

        self.assertEqual("1-1A.txt", new_df['filename'][0])
        self.assertEqual("a", new_df['skill'][0])
        self.assertEqual(10, new_df['model'][0])
        self.assertEqual("Hello, good morning.Transcribed by https://otter.ai", new_df['text'][0])

if __name__ == '__main__':   
    
    # Start unit test
    print("=========== Unit Testing is initiated =========\n")
    unittest.main(exit=False)