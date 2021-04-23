# %%
#Standard Library Modules
import pandas
import sys
import unittest
import os

# set current working directory to where this file is saved
# thisdir = os.path.dirname(os.path.abspath(__file__)) + "\\" 
# os.chdir(thisdir)

# Add higher directory to python module's path
# sys.path.append("..") 

#Local Application Modules
import main.DocSim_Function
# %%
class TestCase(unittest.TestCase): # inherit from unittest.TestCase
    
    # Test mode = normal
    def test_normal_model(self):
        # Set up
        d1 = "plot: two teen couples go to a church party, drink and then drive."
        d2 = "films adapted from comic books have had plenty of success , whether they're about superheroes ( batman , superman , spawn ) , or geared toward kids ( casper ) or the arthouse crowd ( ghost world ) , but there's never really been a comic book like from hell before . "
        d3 = "every now and then a movie comes along from a suspect studio , with every indication that it will be a stinker , and to everybody's surprise ( perhaps even the studio ) the film becomes a critical darling . "
        d4 = "damn that y2k bug . "

        data = {
            'DocumentID': [1,2,3,4],
            'RawText':[d1,d2,d2,d2]
        }

        documents = pandas.DataFrame.from_dict(data)
        
        output = main.DocSim_Function.doc_sim(document_matrix = documents, 
                    text_col_name = 'RawText', mode = 'normal', 
                    method = 'cosine', remove_stopwords = False,
                    filler_words = [], stem = False, 
                    tfidf = False, LSA = False)

        output2 = main.DocSim_Function.doc_sim(document_matrix = documents, 
                    text_col_name = 'RawText', mode = 'normal', 
                    method = 'cosine', remove_stopwords = True,
                    filler_words = [], stem = True, 
                    tfidf = True, LSA = True)

        # Test
        self.assertEqual(documents.shape[1] + 2, output.shape[1])
        self.assertEqual(output.iloc[0, 3].round(0), 1)
        self.assertEqual(output2.iloc[0, 3].round(0), 1)

    # Test mode = pairwise
    def test_pairwise_model(self):
       
        d1 = "plot: two teen couples go to a church party, drink and then drive."
        d2 = "films adapted from comic books have had plenty of success , whether they're about superheroes ( batman , superman , spawn ) , or geared toward kids ( casper ) or the arthouse crowd ( ghost world ) , but there's never really been a comic book like from hell before . "
        d3 = "every now and then a movie comes along from a suspect studio , with every indication that it will be a stinker , and to everybody's surprise ( perhaps even the studio ) the film becomes a critical darling . "
        d4 = "damn that y2k bug . "

        data = {
            'DocumentID': [1,2,3,4],
            'RawText':[d1,d2,d3,d4]
        }

        documents = pandas.DataFrame.from_dict(data)
        
        output = main.DocSim_Function.doc_sim(document_matrix = documents, 
                    text_col_name = 'RawText', mode = 'pairwise', 
                    method = 'cosine', remove_stopwords = False,
                    filler_words = [], stem = False, 
                    tfidf = False, LSA = False)
        
        output2 = main.DocSim_Function.doc_sim(document_matrix = documents, 
                    text_col_name = 'RawText', mode = 'pairwise', 
                    method = 'cosine', remove_stopwords = True,
                    filler_words = [], stem = True, 
                    tfidf = True, LSA = True)
        # Test
        self.assertEqual(documents.shape[1] + 2, output.shape[1])
        self.assertNotEqual(output.iloc[0, 3].round(0), 1)
        self.assertNotEqual(output2.iloc[0, 3].round(0), 1)

    # Test 1 row data frame
    def test_1_row(self):
       
        d1 = "plot: two teen couples go to a church party, drink and then drive."
        d2 = "films adapted from comic books have had plenty of success , whether they're about superheroes ( batman , superman , spawn ) , or geared toward kids ( casper ) or the arthouse crowd ( ghost world ) , but there's never really been a comic book like from hell before . "
        d3 = "every now and then a movie comes along from a suspect studio , with every indication that it will be a stinker , and to everybody's surprise ( perhaps even the studio ) the film becomes a critical darling . "
        d4 = "damn that y2k bug . "

        data = {
            'DocumentID': [1],
            'RawText':[d1]
        }

        documents = pandas.DataFrame.from_dict(data)
        
        output = main.DocSim_Function.doc_sim(document_matrix = documents, 
                    text_col_name = 'RawText', mode = 'normal', 
                    method = 'cosine', remove_stopwords = False,
                    filler_words = [], stem = False, 
                    tfidf = False, LSA = False)
        
        # Cannot operate pairwise mode with 1 row document_matrix 
        # output2 = DocSim_Function.doc_sim(document_matrix = documents, 
        #             text_col_name = 'RawText', mode = 'pairwise', 
        #             method = 'cosine', remove_stopwords = True,
        #             filler_words = [], stem = True, 
        #             tfidf = True, LSA = True)

        # Test
        self.assertEqual(documents.shape[1] + 2, output.shape[1])
        self.assertEqual(output.iloc[0, 3].round(0), 1)
        # self.assertEqual(output2.iloc[0, 3].round(0), 1)

if __name__ == '__main__':   
    
    # Start unit test
    print("=========== Unit Testing is initiated =========\n")
    unittest.main(exit=False)


# # %%

# #Standard Library Modules
# import pandas
# import sys
# import unittest
# import os

# #set current working directory to where this file is saved
# thisdir = os.path.dirname(os.path.abspath(__file__)) + "\\" 
# os.chdir(thisdir)

# # Add higher directory to python module's path
# sys.path.append("..") 

# #Local Application Modules
# from main import DocSim_Function

# d1 = "plot: two teen couples go to a church party, drink and then drive."
# d2 = "films adapted from comic books have had plenty of success , whether they're about superheroes ( batman , superman , spawn ) , or geared toward kids ( casper ) or the arthouse crowd ( ghost world ) , but there's never really been a comic book like from hell before . "
# d3 = "every now and then a movie comes along from a suspect studio , with every indication that it will be a stinker , and to everybody's surprise ( perhaps even the studio ) the film becomes a critical darling . "
# d4 = "damn that y2k bug . "

# data = {
#     'DocumentID': [1,2,3,4],
#     'RawText':[d3,d3,d3,d2]
# }

# documents = pandas.DataFrame.from_dict(data)

# output = DocSim_Function.doc_sim(document_matrix = documents, 
#             text_col_name = 'RawText', mode = 'pairwise', 
#             method = 'cosine', remove_stopwords = False,
#             filler_words = [], stem = False, 
#             tfidf = False, LSA = False)
# output.head()
# # %%
# DocSim_Function.average_similarity(output, mode = 'pairwise')
# # %%
# output.similarity_score.mean()
# # %%
