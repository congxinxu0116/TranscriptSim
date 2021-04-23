#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 16:19:44 2021

@author: Ashley
"""
# =============================================================================
#  DocSim class preprocessing function unit tests
# =============================================================================
#%%
import unittest
import pandas as pd
from DocSim_Class_complete import *

#%%
# Testing the __int__ function
class DocSimIntTestCase(unittest.TestCase): # inherit from unittest.TestCase
    
    def test_input_nonexisting_column_name(self):
        # What if “text_col” is not in the column names?
        # Set up
        documents = pd.DataFrame({'id':[1,2,3,4,5],\
                                  'text':['this is the first','this is the second','this is the third','this is the fourth','this is the fifth'],\
                                  'id2':['1a','2a','3a','4a','5a']}) # create instance
        
        # Test
        self.assertRaises(SystemExit,DocSim, documents, text_col='tex')


        
    def test_large_LSA_n_components(self):
        # What if “text_col” is not in the column names?
        
        # Set up
        documents = pd.DataFrame({'id':[1,2,3,4,5],\
                                  'text':['one','two','three','four','five']}) # create instance
        
        doc = DocSim(documents, 'text')
        
        # Test
        self.assertRaises(SystemExit,doc.preprocessing, LSA=True, LSA_n_components=5)




if __name__ == '__main__':
    unittest.main() 

#%%

class DocSimPreprocessingAllOutputTestCase(unittest.TestCase): # inherit from unittest.TestCase

# =============================================================================
# No Preprocessing Test Case 1/16
# =============================================================================
    def test_All_False(self):
        # Case 1: All parameters = False
        #Check that the number of columns = the number of unique words (8 -> 8)
                        # [this, is, the, first, second, third, fourth, fifth]
        #Check that the vectors contain integers

        # Set up
        documents = pd.DataFrame({'id':[1,2,3,4,5],\
                                 'text':['this is the first','this is the second','this is the third','this is the fourth','this is the fifth'],\
                                 'id2':['1a','2a','3a','4a','5a']}) # create instance
        doc1 = DocSim(documents, 'text')
        documents_processed = doc1.preprocessing(remove_stopwords = False,
                  filler_words = [], stem = False, tfidf = False,
                  LSA = False, LSA_n_components = 2)
        
        # Test
        #is the number of columns correct?
        print(len(documents_processed["cleaned_vectorized_document"][0]))
        self.assertEqual(len(documents_processed["cleaned_vectorized_document"][0]), 8) # should be 8
        #does the lists only contain integers?
        self.assertEqual(type(documents_processed["cleaned_vectorized_document"][0][0]), int) # should be int

# =============================================================================
# One Preprocessing Step Test 5/16
# =============================================================================
        
    def test_remove_stopwords(self):
        # Case 2: remove_stopwords=True, filler_words = ['word']
        #Check that the number of columns = the number of non-stopwords (reduced from 19 -> 4)
               # [so, of, yourself, himself, only, me, third, untl, it, fifth, by, fourth, second, had, i, hers, yours, first, what]
               # [second, third, fourth, fifth]
        #Check that the vectors contain integers
        
        # Set up
        documents = pd.DataFrame({'id':[1,2,3,4,5],\
                                 'text':['I me SO only BY first','himself yours second','of until had third','what yourself HERS fourth','it fifth'],\
                                 'id2':['1a','2a','3a','4a','5a']}) # create instance
        doc2 = DocSim(documents, 'text')
        documents_processed = doc2.preprocessing(remove_stopwords = True, 
                  filler_words = ['first'], stem = False, tfidf = False, 
                  LSA = False, LSA_n_components = 2)
        
        # Test
        #is the number of columns correct?
        self.assertEqual(len(documents_processed["cleaned_vectorized_document"][0]), 4) # should be 4
        #does the lists only contain integers?
        self.assertEqual(type(documents_processed["cleaned_vectorized_document"][0][0]), int) # should be int
         
# =============================================================================
    def test_stem(self):
        # Case 3: stem=True
        #Check that the number of columns = the number of stemmed words (reduced from 6 -> 3)
            # [running, run, swimming, swim, jumping, jump] -> [run, swim, jump]
        #Check that the vectors contain integers
        
        # Set up
        documents = pd.DataFrame({'id':[1,2,3,4,5],\
                                 'text':['running swim','run swimming','swim jump','jumping run','run jump'],\
                                 'id2':['1a','2a','3a','4a','5a']}) # create instance
        doc3 = DocSim(documents, 'text')
        documents_processed = doc3.preprocessing(remove_stopwords = False,
                  filler_words = [], stem = True, tfidf = False,
                  LSA = False, LSA_n_components = 2)
        
        # Test
        #is the number of columns correct?
        self.assertEqual(len(documents_processed["cleaned_vectorized_document"][0]), 3) # should be 3
        #does the lists only contain integers?
        self.assertEqual(type(documents_processed["cleaned_vectorized_document"][0][0]), int) # should be int
        
# =============================================================================
    def test_tfidf(self):
        # Case 4: tfidf=True
        #Check that the number of columns = the number of unique words (8 -> 8)
            # [this, is, the, first, second, third, fourth, fifth]
        #Check that the vectors contain floats
        
        # Set up
        documents = pd.DataFrame({'id':[1,2,3,4,5],\
                                 'text':['this is the first','this is the second','this is the third','this is the fourth','this is the fifth'],\
                                 'id2':['1a','2a','3a','4a','5a']}) # create instance
        doc4 = DocSim(documents, 'text')
        documents_processed = doc4.preprocessing(remove_stopwords = False,
                  filler_words = [], stem = False, tfidf = True,
                  LSA = False, LSA_n_components = 2)
        
        # Test
        #is the number of columns correct?
        self.assertEqual(len(documents_processed["cleaned_vectorized_document"][0]), 8) # should be 8
        #does the lists only contain integers?
        self.assertEqual(type(documents_processed["cleaned_vectorized_document"][0][0]), float) # should be float
 
# =============================================================================
    def test_LSA(self):
      # Case 5: LSA=True    
      # Check if the number of columns = the number of n_components (3)
      #Check that the vectors contain floats
        
        # Set up
        documents = pd.DataFrame({'id':[1,2,3,4,5],\
                                 'text':['this is the first','this is the second','this is the third','this is the fourth','this is the fifth'],\
                                 'id2':['1a','2a','3a','4a','5a']}) # create instance
        doc5 = DocSim(documents, 'text')
        documents_processed = doc5.preprocessing(remove_stopwords = False,
                  filler_words = [], stem = False, tfidf = False,
                  LSA = True, LSA_n_components = 3)
        
        # Test
        #is the number of columns correct?
        self.assertEqual(len(documents_processed["cleaned_vectorized_document"][0]), 3) # should be 3
        #does the lists only contain integers?
        self.assertEqual(type(documents_processed["cleaned_vectorized_document"][0][0]), float) # should be float
# =============================================================================
#  Two Preprocessing Steps Test 11/16      
# =============================================================================
    def test_remove_stopwords_AND_stem(self):
        # Case 6: remove_stopwords=True, filler_words = ['first'], stem=True
        #Check that the number of columns = the number of non-stop and stemmed words (20 -> 5)
                # [so, of, himself, me, third, until, it, fifth, by, fourth, second, had, i, hers, yours, first, what, jumping, jumply, jump]
                # [second, third, fourth, fifth, jump]
        #Check that the vectors contain integers
        
        # Set up
        documents = pd.DataFrame({'id':[1,2,3,4,5],\
                                  'text':['I me SO BY first','himself yours jump second','of until jump had third','what HERS fourth','it jumping fifth'],\
                                  'id2':['1a','2a','3a','4a','5a']}) # create instance
        doc6 = DocSim(documents, 'text')
        documents_processed = doc6.preprocessing(remove_stopwords = True,
                  filler_words = ['first'], stem = True, tfidf = False,
                  LSA = False, LSA_n_components = 2)
        
        # Test
        #is the number of columns correct?
        self.assertEqual(len(documents_processed["cleaned_vectorized_document"][0]), 5) # should be 5
        #does the lists only contain integers?
        self.assertEqual(type(documents_processed["cleaned_vectorized_document"][0][0]), int) # should be integer
       
# =============================================================================
    def test_remove_stopwords_AND_tfidf(self):
        # Case 7: remove_stopwords=True, filler_words = ['first'], tfidf=True
        #Check that the number of columns = the number of non-stopwords (reduced from 19 -> 4)
               # [so, of, yourself, himself, only, me, third, untl, it, fifth, by, fourth, second, had, i, hers, yours, first, what]
               # [second, third, fourth, fifth]
        #Check that the vectors contain float
        
        # Set up
        documents = pd.DataFrame({'id':[1,2,3,4,5],\
                                 'text':['I me SO only BY first','himself yours second','of until had third','what yourself HERS fourth','it fifth'],\
                                 'id2':['1a','2a','3a','4a','5a']}) # create instance
        doc7 = DocSim(documents, 'text')
        documents_processed = doc7.preprocessing(remove_stopwords = True, 
                  filler_words = ['first'], stem = False, tfidf = True, 
                  LSA = False, LSA_n_components = 2)
        
        # Test
        #is the number of columns correct?
        self.assertEqual(len(documents_processed["cleaned_vectorized_document"][0]), 4) # should be 4
        #does the lists only contain integers?
        self.assertEqual(type(documents_processed["cleaned_vectorized_document"][0][0]), float) # should be float
        
# =============================================================================
    def test_remove_stopwords_AND_LSA(self):
      # Case 8: remove_stopwords=True, filler_words = ['first'], LSA=True 
      # Check is the number of columns = LSA_n_components (3)
      # Check that the vectors contain floats
      
        # Set up
        documents = pd.DataFrame({'id':[1,2,3,4,5],\
                                 'text':['I me SO only BY first','himself yours second','of until had third','what yourself HERS fourth','it fifth'],\
                                 'id2':['1a','2a','3a','4a','5a']}) # create instance
        doc8 = DocSim(documents, 'text')
        documents_processed = doc8.preprocessing(remove_stopwords = True, 
                  filler_words = ['first'], stem = False, tfidf = False, 
                  LSA = True, LSA_n_components = 3)
        
        # Test
        #is the number of columns correct?
        self.assertEqual(len(documents_processed["cleaned_vectorized_document"][0]), 3) # should be 3
        #does the lists only contain integers?
        self.assertEqual(type(documents_processed["cleaned_vectorized_document"][0][0]), float) # should be float
       
# =============================================================================
    def test_stem_AND_tfidf(self):
        # Case 9: stem=True, tfidf=True
        #Check that the number of columns = the number of stemmed words (reduced from 6 -> 3)
            # [running, run, swimming, swim, jumping, jump] -> [run, swim, jump]
        #Check that the vectors contain floats
        
        # Set up
        documents = pd.DataFrame({'id':[1,2,3,4,5],\
                                 'text':['running swim','run swimming','swim jump','jumping run','run jump'],\
                                 'id2':['1a','2a','3a','4a','5a']}) # create instance
        doc9 = DocSim(documents, 'text')
        documents_processed = doc9.preprocessing(remove_stopwords = False,
                  filler_words = [], stem = True, tfidf = True,
                  LSA = False, LSA_n_components = 2)
        
        # Test
        #is the number of columns correct?
        self.assertEqual(len(documents_processed["cleaned_vectorized_document"][0]), 3) # should be 3
        #does the lists only contain integers?
        self.assertEqual(type(documents_processed["cleaned_vectorized_document"][0][0]), float) # should be float
       
# =============================================================================
    def test_stem_AND_LSA(self):
        # Case 10: stem=True, LSA=True
        #Check that the number of columns = LSA_n_components (5)
        #Check that the vectors contain floats
        
        # Set up
        documents = pd.DataFrame({'id':[1,2,3,4,5],\
                                 'text':['running climb swim','run dive swimming','swim climbing jump','jumping diving then run','run to jump'],\
                                 'id2':['1a','2a','3a','4a','5a']}) # create instance
        doc10 = DocSim(documents, 'text')
        documents_processed = doc10.preprocessing(remove_stopwords = False,
                  filler_words = [], stem = True, tfidf = False,
                  LSA = True, LSA_n_components = 5)
        
        # Test
        #is the number of columns correct?
        self.assertEqual(len(documents_processed["cleaned_vectorized_document"][0]), 5) # should be 5
        #does the lists only contain integers?
        self.assertEqual(type(documents_processed["cleaned_vectorized_document"][0][0]), float) # should be float
       
# =============================================================================
    def test_tfidf_AND_LSA(self):
        # Case 11: tfidf=True,, LSA=True
        #Check that the number of columns = LSA_n_components (3)
        #Check that the vectors contain floats
        
        # Set up
        documents = pd.DataFrame({'id':[1,2,3,4,5],\
                                 'text':['running climb swim','run dive swimming','swim climbing jump','jumping diving then run','run to jump'],\
                                 'id2':['1a','2a','3a','4a','5a']}) # create instance
        doc11 = DocSim(documents, 'text')
        documents_processed = doc11.preprocessing(remove_stopwords = False,
                  filler_words = [], stem = True, tfidf = False,
                  LSA = True, LSA_n_components = 3)
        
        # Test
        #is the number of columns correct?
        self.assertEqual(len(documents_processed["cleaned_vectorized_document"][0]), 3) # should be 3
        #does the lists only contain integers?
        self.assertEqual(type(documents_processed["cleaned_vectorized_document"][0][0]), float) # should be float
# =============================================================================
#  Three Preprocessing Steps Test 15/16
# =============================================================================
    def test_remove_stopwords_AND_stem_AND_tfidf(self):
        # Case 12: remove_stopwords=True, stem=True, tfidf=True
        #Check that the number of columns = number of non-stop & stemmed words (reduced from 12 -> 7)
            # [i, running, run, swimming, swim, jumping, jump, first, second, third, fourth, fifth] 
            # -> [run, swim, jump, second, third, fourth, fifth]
        #Check that the vectors contain floats
        
        # Set up
        documents = pd.DataFrame({'id':[1,2,3,4,5],\
                                  'text':['running swim first','run swimming second','swim jump third','i jumping run fourth','run jump fifth'],\
                                  'id2':['1a','2a','3a','4a','5a']}) # create instance
        doc12 = DocSim(documents, 'text')
        documents_processed = doc12.preprocessing(remove_stopwords = True,
                  filler_words = ['first'], stem = True, tfidf = True,
                  LSA = False, LSA_n_components = 2)
        
        # Test
        #is the number of columns correct?
        self.assertEqual(len(documents_processed["cleaned_vectorized_document"][0]), 7) # should be 7
        #does the lists only contain integers?
        self.assertEqual(type(documents_processed["cleaned_vectorized_document"][0][0]), float) # should be float


# =============================================================================
    def test_remove_stopwords_AND_stem_AND_LSA(self):
        # Case 13: remove_stopwords=True, filler_words = ['first'], stem=True, LSA=True
        #Check if the number of columns = LSA_n_components (5)
        #Check that the vectors contain floats
        
        # Set up
        documents = pd.DataFrame({'id':[1,2,3,4,5],\
                                 'text':['I me SO only BY first sixth','himself yours second seventh','of until had third nine','what yourself HERS fourth ten','it fifth twelve'],\
                                 'id2':['1a','2a','3a','4a','5a']}) # create instance
        doc13 = DocSim(documents, 'text')
        documents_processed = doc13.preprocessing(remove_stopwords = True, 
                  filler_words = ['first'], stem = True, tfidf = False, 
                  LSA = True, LSA_n_components = 5)
        
        # Test
        #is the number of columns correct?
        self.assertEqual(len(documents_processed["cleaned_vectorized_document"][0]), 5) # should be 5
        #does the lists only contain integers?
        self.assertEqual(type(documents_processed["cleaned_vectorized_document"][0][0]), float) # should be float

# =============================================================================
    def test_stem_AND_tfidf_AND_LSA(self):
        # Case 14: stem=True, tfidf=True, LSA=True
        #Check if the number of columns = LSA_n_components (6)
        #Check that the vectors contain floats
        
        # Set up
        documents = pd.DataFrame({'id':[1,2,3,4,5],\
                                  'text':['I me SO only BY first sixth','himself yours second seventh','of until had third nine','what yourself HERS fourth ten','it fifth twelve'],\
                                  'id2':['1a','2a','3a','4a','5a']}) # create instance
        doc14 = DocSim(documents, 'text')
        documents_processed = doc14.preprocessing(remove_stopwords = False, 
                  filler_words = [], stem = True, tfidf = True, 
                  LSA = True, LSA_n_components = 4)
        
        # Test
        #is the number of columns correct?
        self.assertEqual(len(documents_processed["cleaned_vectorized_document"][0]), 4) # should be 3
        #does the lists only contain integers?
        self.assertEqual(type(documents_processed["cleaned_vectorized_document"][0][0]), float) # should be float

# =============================================================================
    def test_remove_stopwords_AND_tfidf_AND_LSA(self):
       # Case 15: remove_stopwords=True, filler_words = ['first'], tfidf=True, LSA=True
       #Check if the number of columns = LSA_n_components (6)
       #Check that the vectors contain floats
        
        # Set up
        documents = pd.DataFrame({'id':[1,2,3,4,5,6],\
                                  'text':["He is a good dog.","The dog and cow is too lazy.","That is a brown cat.","That is a brown cow.","The cat is very active.","I have brown cat and dog."],
                                  'id2':['1a','2a','3a','4a','5a','6a']}) # create instance
        doc15 = DocSim(documents, 'text')
        documents_processed = doc15.preprocessing(remove_stopwords = True, 
                  filler_words = [], stem = False, tfidf = True, 
                  LSA = True, LSA_n_components = 6)
        
        # Test
        #is the number of columns correct?
        self.assertEqual(len(documents_processed["cleaned_vectorized_document"][0]), 6) # should be 6
        #does the lists only contain integers?
        self.assertEqual(type(documents_processed["cleaned_vectorized_document"][0][0]), float) # should be float
# =============================================================================
# All Preprocessing Steps Test Case 16/16
# =============================================================================
    def test_all_true(self):
       # Case 16: remove_stopwords=True, filler_words = ['first'], stem=True, tfidf=True, LSA=True'
       #Check if the number of columns = LSA_n_components (6)
       #Check that the vectors contain floats
        
        # Set up
        documents = pd.DataFrame({'id':[1,2,3,4,5,6],\
                                  'text':["He is a good dog.","The dog and cow is too lazy.","That is a brown cat.","That is a brown cow.","The cat is very active.","I have brown cat and dog."],
                                  'id2':['1a','2a','3a','4a','5a','6a']}) # create instance
        doc15 = DocSim(documents, 'text')
        documents_processed = doc15.preprocessing(remove_stopwords = True, 
                  filler_words = ['lazy'], stem = True, tfidf = True, 
                  LSA = True, LSA_n_components = 5)
        
        # Test
        #is the number of columns correct?
        self.assertEqual(len(documents_processed["cleaned_vectorized_document"][0]), 5) # should be 6
        #does the lists only contain integers?
        self.assertEqual(type(documents_processed["cleaned_vectorized_document"][0][0]), float) # should be float

            
if __name__ == '__main__':
    unittest.main()  