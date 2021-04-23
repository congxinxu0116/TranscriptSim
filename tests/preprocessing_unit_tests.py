#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 12:59:01 2020

@author: Ashley
"""
import sys
import unittest
import os

#set current working directory to where this file is saved
thisdir = os.path.dirname(os.path.abspath(__file__)) + "\\" 
os.chdir(thisdir)
# Add higher directory to python module's path
sys.path.append("..") 

#Local Application Modules
from main import DocSim_Function

# =============================================================================
class PreprocessingTestCase(unittest.TestCase):
    
    def test_input_nonexisting_column_name(self):
        # What if “text_col_name” is not in the column names?

        # Set up
        documents = pd.DataFrame({'id':[1,2,3,4,5],\
                                  'text':['this is the first','this is the second','this is the third','this is the fourth','this is the fifth'],\
                                  'id2':['1a','2a','3a','4a','5a']}) # create instance
        
        # Test
        self.assertRaises(SystemExit,preprocessing,documents,text_col_name='tex')
        
# =============================================================================
#     def test_text_column_wrong_datatype(self): DOESN'T WORK
#         # What if the “text_col_name” column is not object type or category type?
#         
#         #Set up
#         documents = pd.DataFrame({'id':[1,2,3,4,5],\
#                                   'text':['this is the first','this is the second','this is the third','this is the fourth','this is the fifth'],\
#                                   'id2':['1a','2a','3a','4a','5a']}) # create instance
#         
#         # Test
#         self.assertRaises(SystemExit, preprocessing, documents, text_col_name='id')
# =============================================================================
                
            
if __name__ == '__main__':
    unittest.main()  


class PreprocessingAllOutputTestCase(unittest.TestCase): 
    
    def test_All_False(self):
        # Case 1: All parameters = False
        #Check that the number of columns = the number of unique words (8 -> 8)
                        # [this, is, the, first, second, third, fourth, fifth]
        #Check that the vectors contain integers

        # Set up
        documents = pd.DataFrame({'id':[1,2,3,4,5],\
                                 'text':['this is the first','this is the second','this is the third','this is the fourth','this is the fifth'],\
                                 'id2':['1a','2a','3a','4a','5a']}) # create instance
        documents_processed = preprocessing(documents, text_col_name='text',remove_stopwords = False,
                  filler_words = [], stem = False, tfidf = False,
                  LSA = False, LSA_n_components = 2)
        
        # Test
        #is the number of columns correct?
        print(len(documents_processed["cleaned_vectorized_document"][0]))
        self.assertEqual(len(documents_processed["cleaned_vectorized_document"][0]), 8) # should be 8
        #does the lists only contain integers?
        self.assertEqual(type(documents_processed["cleaned_vectorized_document"][0][0]), int) # should be int

# =============================================================================
#     
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
        documents_processed = preprocessing(documents, text_col_name='text',remove_stopwords = True, 
                  filler_words = ['first'], stem = False, tfidf = False, 
                  LSA = False, LSA_n_components = 2)
        
        # Test
        #is the number of columns correct?
        self.assertEqual(len(documents_processed["cleaned_vectorized_document"][0]), 4) # should be 4
        #does the lists only contain integers?
        self.assertEqual(type(documents_processed["cleaned_vectorized_document"][0][0]), int) # should be int
# =============================================================================
#         
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
        
        documents_processed = preprocessing(documents, text_col_name='text', remove_stopwords = False,
                  filler_words = [], stem = True, tfidf = False,
                  LSA = False, LSA_n_components = 2)
        
        # Test
        #is the number of columns correct?
        self.assertEqual(len(documents_processed["cleaned_vectorized_document"][0]), 3) # should be 3
        #does the lists only contain integers?
        self.assertEqual(type(documents_processed["cleaned_vectorized_document"][0][0]), int) # should be int
# =============================================================================
#         
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
        
        documents_processed = preprocessing(documents, text_col_name='text', remove_stopwords = False,
                  filler_words = [], stem = False, tfidf = True,
                  LSA = False, LSA_n_components = 2)
        
        # Test
        #is the number of columns correct?
        self.assertEqual(len(documents_processed["cleaned_vectorized_document"][0]), 8) # should be 8
        #does the lists only contain integers?
        self.assertEqual(type(documents_processed["cleaned_vectorized_document"][0][0]), float) # should be float
# =============================================================================
# 
# =============================================================================
# =============================================================================
#     def test_LSA(self):
#       # Case 5: LSA=True    
#       # Check if the number of columns = the number of n_components (5)?
#       #Check that the vectors contain floats
#         
#         # Set up
#         documents = pd.DataFrame({'id':[1,2,3,4,5],\
#                                  'text':['this is the first','this is the second','this is the third','this is the fourth','this is the fifth'],\
#                                  'id2':['1a','2a','3a','4a','5a']}) # create instance
#         
#         documents_processed = preprocessing(documents, text_col_name='text', remove_stopwords = False,
#                   filler_words = [], stem = False, tfidf = False,
#                   LSA = True, LSA_n_components = 3)
#         
#         # Test
#         #is the number of columns correct?
#         self.assertEqual(len(documents_processed["cleaned_vectorized_document"][0]), 3) # should be 8
#         #does the lists only contain integers?
#         self.assertEqual(type(documents_processed["cleaned_vectorized_document"][0][0]), float) # should be float
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
        
        documents_processed = preprocessing(documents, text_col_name='text', remove_stopwords = True,
                  filler_words = ['first'], stem = True, tfidf = False,
                  LSA = False, LSA_n_components = 2)
        
        # Test
        #is the number of columns correct?
        self.assertEqual(len(documents_processed["cleaned_vectorized_document"][0]), 5) # should be 5
        #does the lists only contain integers?
        self.assertEqual(type(documents_processed["cleaned_vectorized_document"][0][0]), int) # should be integer

# =============================================================================
#         
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
        documents_processed = preprocessing(documents, text_col_name='text',remove_stopwords = True, 
                  filler_words = ['first'], stem = False, tfidf = True, 
                  LSA = False, LSA_n_components = 2)
        
        # Test
        #is the number of columns correct?
        self.assertEqual(len(documents_processed["cleaned_vectorized_document"][0]), 4) # should be 4
        #does the lists only contain integers?
        self.assertEqual(type(documents_processed["cleaned_vectorized_document"][0][0]), float) # should be int
        
# =============================================================================
#     def test_remove_stopwords_AND_LSA(self):
#       # Case 8: remove_stopwords=True, filler_words = ['first'], LSA=True 
#         
#         # Set up
#         
#         # Test
#         self.assertEqual(student1.numCourses, 3)
# =============================================================================
# =============================================================================
#         
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
        
        documents_processed = preprocessing(documents, text_col_name='text', remove_stopwords = False,
                  filler_words = [], stem = True, tfidf = True,
                  LSA = False, LSA_n_components = 2)
        
        # Test
        #is the number of columns correct?
        self.assertEqual(len(documents_processed["cleaned_vectorized_document"][0]), 3) # should be 3
        #does the lists only contain integers?
        self.assertEqual(type(documents_processed["cleaned_vectorized_document"][0][0]), float) # should be float
        
# =============================================================================
#     def test_stem_AND_LSA(self):
#         # Case 10: stem=True, LSA=True
#         # Set up
#         
#         # Test
#         self.assertEqual(student1.numCourses, 3)
# =============================================================================
        
# =============================================================================
#     def test_tfidf_AND_LSA(self):
#         # Case 11: tfidf=True,, LSA=True
#         # Set up
#         
#         # Test
#         self.assertEqual(student1.numCourses, 3)
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
        
        documents_processed = preprocessing(documents, text_col_name='text', remove_stopwords = True,
                  filler_words = ['first'], stem = True, tfidf = True,
                  LSA = False, LSA_n_components = 2)
        
        # Test
        #is the number of columns correct?
        self.assertEqual(len(documents_processed["cleaned_vectorized_document"][0]), 7) # should be 7
        #does the lists only contain integers?
        self.assertEqual(type(documents_processed["cleaned_vectorized_document"][0][0]), float) # should be float

# =============================================================================
#     def test_remove_stopwords_AND_stem_AND_LSA(self):
#        # Case 13: remove_stopwords=True, filler_words = ['first'], stem=True, LSA=True
# =============================================================================

# =============================================================================
#     def test_stem_AND_tfidf_AND_LSA(self):
#        # Case 14: stem=True, tfidf=True, LSA=True
# =============================================================================

# =============================================================================
#     def test_remove_stopwords_AND_tfidf_AND_LSA(self):
#        # Case 15: remove_stopwords=True, filler_words = ['first'], tfidf=True, LSA=True
# =============================================================================

# =============================================================================
#     def test_all_true(self):
#        # Case 16: remove_stopwords=True, filler_words = ['first'], stem=True, tfidf=True, LSA=True
# =============================================================================

# =============================================================================
            
if __name__ == '__main__':
    unittest.main() 
