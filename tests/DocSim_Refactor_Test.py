# Setting Directories
import pandas
import sys
import unittest
import numpy
import os
import collections

sys.path.append('C:\\Users\\DaDa\\Documents\\GitHub\\Capstone-NLP-Edu-Interventions\\Duplicate_DocSim\\DocSim\\main')


import DocSim_class


data = pandas.read_csv("all_documents.csv")
data.head()

def compare(s, t):
    if collections.Counter(s) == collections.Counter(t):
        return True
    else: 
        return False


class TestCase(unittest.TestCase): # inherit from unittest.TestCase

    # Test if all functions can run without error
    def test_if_all_functions_can_run(self):
        
        DocSim1 = DocSim_class.DocSim(data = data, 
                         skill = 'skill', 
                         study = 'study',
                         doc_type = 'doctype',
                         doc_id = 'filename',
                         text = 'text') 
        
        try:
            DocSim1.preprocessing()
            DocSim1.get_preprocessed_text()
            DocSim1.get_tfidf_feature_names()
            DocSim1.get_lsa_feature_names()
            # DocSim1.check_preprocessing_input()
            DocSim1.get_skill()
            DocSim1.get_doc_type()
            DocSim1.get_study(skill_id = [2])
            DocSim1.get_study()
            DocSim1.normal_comparison()
            DocSim1.pairwise_comparison()
            DocSim1.within_study_normal_average()
            DocSim1.across_study_within_skill_normal_average()
            DocSim1.across_study_across_skill_normal_average()
            x = True
        except:
            x = False

        self.assertEqual(x, True)
        
        
    # # Test simple Get functions 
    def test_get_functions(self):
        
        docsim1 = DocSim_class.DocSim(data = data, 
                         skill = 'skill', 
                         study = 'study',
                         doc_type = 'doctype',
                         doc_id = 'filename',
                         text = 'text')
        
        # Test
        self.assertEqual(compare(docsim1.get_skill(), 
                                 numpy.array([1,2,3,4])), 
                         True)

        self.assertEqual(compare(docsim1.get_doc_type(), 
                                 numpy.array(['transcript', 'script'])), 
                         True)


    # Test different combinations of hyperparameters
    def test_normal_diff_combinations(self):
        
        docsim2 = DocSim_class.DocSim(data = data, 
                                      skill = 'skill', 
                                      study = 'study',
                                      doc_type = 'doctype',
                                      doc_id = 'filename',
                                      text = 'text')
        try:
            for a in [True, False]:
                for b in [True, False]:
                    for c in [True, False]:
                        for d in [True, False]:
                            for e in [50, 80]:
                                for f in ['full', 'skill', 'study', 'document']:
                                    for g in [False]:
                                        for h in [1,2,3]:
                                            docsim2.normal_comparison(
                                                remove_stopwords = a,
                                                stem = b,
                                                lemm = g,
                                                tfidf = c,
                                                tfidf_level = f,
                                                lsa = d,
                                                lsa_n_components = e,
                                                ngram = h
                                            )

            for a in [True, False]:
                for b in [False]:
                    for c in [True, False]:
                        for d in [True, False]:
                            for e in [50, 80]:
                                for f in ['full', 'skill', 'study', 'document']:
                                    for g in [True, False]:
                                        for h in [1,2,3]:
                                            docsim2.normal_comparison(
                                                remove_stopwords = a,
                                                stem = b,
                                                lemm = g,
                                                tfidf = c,
                                                tfidf_level = f,
                                                lsa = d,
                                                lsa_n_components = e,
                                                ngram = h
                                            )
                                    # print(a,b,c,d,e,f)

            x = True
        except:
            x = False

        self.assertEqual(x, True)


    def test_pairwise_diff_combinations(self):
        
        docsim3 = DocSim_class.DocSim(data = data, 
                                      skill = 'skill', 
                                      study = 'study',
                                      doc_type = 'doctype',
                                      doc_id = 'filename',
                                      text = 'text')
        try:
            for a in [True, False]:
                for b in [True, False]:
                    for c in [True, False]:
                        for d in [True, False]:
                            for e in [50, 80]:
                                for f in ['full', 'skill', 'study', 'document']:
                                    for g in [False]:
                                        for h in [1,2,3]:
                                            docsim3.pairwise_comparison(
                                                remove_stopwords = a,
                                                stem = b,
                                                lemm = g,
                                                tfidf = c,
                                                tfidf_level = f,
                                                lsa = d,
                                                lsa_n_components = e,
                                                ngram = h
                                            )
            for a in [True, False]:
                for b in [False]:
                    for c in [True, False]:
                        for d in [True, False]:
                            for e in [50, 80]:
                                for f in ['full', 'skill', 'study', 'document']:
                                    for g in [True, False]:
                                        for h in [1,2,3]:
                                            docsim3.pairwise_comparison(
                                                remove_stopwords = a,
                                                stem = b,
                                                lemm = g,
                                                tfidf = c,
                                                tfidf_level = f,
                                                lsa = d,
                                                lsa_n_components = e,
                                                ngram = h
                                            )
                                
            x = True
        except:
            x = False

        self.assertEqual(x, True)


    def test_within_diff_combinations(self):
        
        docsim4 = DocSim_class.DocSim(data = data, 
                                      skill = 'skill', 
                                      study = 'study',
                                      doc_type = 'doctype',
                                      doc_id = 'filename',
                                      text = 'text')
        try:
            for a in [True, False]:
                for b in [True, False]:
                    for c in [True, False]:
                        for d in [True, False]:
                            for e in [50, 80]:
                                for f in ['full', 'skill', 'study', 'document']:
                                    for g in [False]:
                                        for h in [1,2,3]:
                                            docsim4.within_study_normal_average(
                                                remove_stopwords = a,
                                                stem = b,
                                                lemm = g,
                                                tfidf = c,
                                                tfidf_level = f,
                                                lsa = d,
                                                lsa_n_components = e,
                                                ngram = h
                                            )
            for a in [True, False]:
                for b in [False]:
                    for c in [True, False]:
                        for d in [True, False]:
                            for e in [50, 80]:
                                for f in ['full', 'skill', 'study', 'document']:
                                    for g in [True, False]:
                                        for h in [1,2,3]:
                                            docsim4.within_study_normal_average(
                                                remove_stopwords = a,
                                                stem = b,
                                                lemm = g,
                                                tfidf = c,
                                                tfidf_level = f,
                                                lsa = d,
                                                lsa_n_components = e,
                                                ngram = h
                                            )
            x = True
        except:
            x = False

        self.assertEqual(x, True)


    def test_across_study_within_skill_diff_combinations(self):
        
        docsim5 = DocSim_class.DocSim(data = data, 
                         skill = 'skill', 
                         study = 'study',
                         doc_type = 'doctype',
                         doc_id = 'filename',
                         text = 'text')
        try:
            for a in [True, False]:
                for b in [True, False]:
                    for c in [True, False]:
                        for d in [True, False]:
                            for e in [50, 80]:
                                for f in ['full', 'skill', 'study', 'document']:
                                    for g in [False]:
                                        for h in [1,2,3]:
                                            docsim5.across_study_within_skill_normal_average(
                                                remove_stopwords = a,
                                                stem = b,
                                                lemm = g,
                                                tfidf = c,
                                                tfidf_level = f,
                                                lsa = d,
                                                lsa_n_components = e,
                                                ngram = h
                                            )
            for a in [True, False]:
                for b in [False]:
                    for c in [True, False]:
                        for d in [True, False]:
                            for e in [50, 80]:
                                for f in ['full', 'skill', 'study', 'document']:
                                    for g in [True, False]:
                                        for h in [1,2,3]:
                                            docsim5.across_study_within_skill_normal_average(
                                                remove_stopwords = a,
                                                stem = b,
                                                lemm = g,
                                                tfidf = c,
                                                tfidf_level = f,
                                                lsa = d,
                                                lsa_n_components = e,
                                                ngram = h
                                            )
            x = True
        except:
            x = False

        self.assertEqual(x, True)
    
    
    def test_across_diff_combinations(self):
        
        docsim6 = DocSim_class.DocSim(data = data, 
                                      skill = 'skill', 
                                      study = 'study',
                                      doc_type = 'doctype',
                                      doc_id = 'filename',
                                      text = 'text')
        try:
            for a in [True, False]:
                for b in [True, False]:
                    for c in [True, False]:
                        for d in [True, False]:
                            for e in [50, 80]:
                                for f in ['full', 'skill', 'study', 'document']:
                                    for g in [False]:
                                        for h in [1,2,3]:
                                            docsim6.across_study_across_skill_normal_average(
                                                remove_stopwords = a,
                                                stem = b,
                                                lemm = g,
                                                tfidf = c,
                                                tfidf_level = f,
                                                lsa = d,
                                                lsa_n_components = e,
                                                ngram = h
                                            )
                                            # print(a,b,c,d,e,f,g,h)
            for a in [True, False]:
                for b in [False]:
                    for c in [True, False]:
                        for d in [True, False]:
                            for e in [50, 80]:
                                for f in ['full', 'skill', 'study', 'document']:
                                    for g in [True, False]:
                                        for h in [1,2,3]:
                                            docsim6.across_study_across_skill_normal_average(
                                                remove_stopwords = a,
                                                stem = b,
                                                lemm = g,
                                                tfidf = c,
                                                tfidf_level = f,
                                                lsa = d,
                                                lsa_n_components = e,
                                                ngram = h
                                            )
                                            # print(a,b,c,d,e,f,g,h)
            x = True
        except:
            x = False

        self.assertEqual(x, True)

if __name__ == '__main__':   
    
    # Start unit test
    print("=========== Unit Testing is initiated =========\n")
    unittest.main(exit = False)
