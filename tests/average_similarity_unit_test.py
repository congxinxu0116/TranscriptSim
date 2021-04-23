# %%
#Standard Library Modules
import pandas
import sys
import unittest
import os
import numpy

#set current working directory to where this file is saved
thisdir = os.path.dirname(os.path.abspath(__file__)) + "\\" 
os.chdir(thisdir)

# Add higher directory to python module's path
sys.path.append("..") 

#Local Application Modules
from main import DocSim_Function
# %% Unit Test for average_similarity function

class TestCase(unittest.TestCase): # inherit from unittest.TestCase

    # Test On Random Data Frame for calculation
    def test_calculation(self):
       
        df1 = pandas.DataFrame(numpy.array([1.0, 2.0, 3.0]),
                               columns=['a'])
        df3 = pandas.DataFrame(numpy.array([[1.0, 1.0, 1.0], 
                                            [4.0, 5.0, 2.0], 
                                            [7.0, 8.0, 3.0]]),
                               columns=['a', 'b', 'c'])

        # Test
        self.assertEqual(DocSim_Function.\
                         average_similarity(df1, mode = 'normal'), 2.5)
        self.assertEqual(DocSim_Function.\
                         average_similarity(df1, mode = 'pairwise'), 2)
        self.assertEqual(DocSim_Function.\
                         average_similarity(df3, mode = 'normal'), 2.5)
        self.assertEqual(DocSim_Function.\
                         average_similarity(df3, mode = 'pairwise'), 2)


    # Test mode = normal after doc_sim()
    def test_normal_model(self):
        # Set up
        d1 = "plot: two teen couples go to a church party, drink and then drive."
        d2 = "films adapted from comic books have had plenty of success , whether they're about superheroes ( batman , superman , spawn ) , or geared toward kids ( casper ) or the arthouse crowd ( ghost world ) , but there's never really been a comic book like from hell before . "
        d3 = "every now and then a movie comes along from a suspect studio , with every indication that it will be a stinker , and to everybody's surprise ( perhaps even the studio ) the film becomes a critical darling . "
        d4 = "damn that y2k bug . "

        data = {
            'DocumentID': [1,2,3,4],
            'RawText':[d1,d1,d1,d1]
        }

        documents = pandas.DataFrame.from_dict(data)
        
        output = DocSim_Function.doc_sim(document_matrix = documents, 
                    text_col_name = 'RawText', mode = 'normal', 
                    method = 'cosine', remove_stopwords = False,
                    filler_words = [], stem = False, 
                    tfidf = False, LSA = False)

        output2 = DocSim_Function.doc_sim(document_matrix = documents, 
                    text_col_name = 'RawText', mode = 'normal', 
                    method = 'cosine', remove_stopwords = True,
                    filler_words = [], stem = True, 
                    tfidf = True, LSA = True)

        # Test
        self.assertEqual(round(DocSim_Function.average_similarity(output), 0),
                         1)
        self.assertEqual(round(DocSim_Function.average_similarity(output2), 0),
                         1)

    # Test mode = pairwise after doc_sim()
    def test_pairwise_model(self):
       
        d1 = "plot: two teen couples go to a church party, drink and then drive."
        d2 = "films adapted from comic books have had plenty of success , whether they're about superheroes ( batman , superman , spawn ) , or geared toward kids ( casper ) or the arthouse crowd ( ghost world ) , but there's never really been a comic book like from hell before . "
        d3 = "every now and then a movie comes along from a suspect studio , with every indication that it will be a stinker , and to everybody's surprise ( perhaps even the studio ) the film becomes a critical darling . "
        d4 = "damn that y2k bug . "

        data = {
            'DocumentID': [1,2,3,4],
            'RawText':[d2,d2,d2,d2]
        }

        documents = pandas.DataFrame.from_dict(data)
        
        output = DocSim_Function.doc_sim(document_matrix = documents, 
                    text_col_name = 'RawText', mode = 'pairwise', 
                    method = 'cosine', remove_stopwords = False,
                    filler_words = [], stem = False, 
                    tfidf = False, LSA = False)
        
        output2 = DocSim_Function.doc_sim(document_matrix = documents, 
                    text_col_name = 'RawText', mode = 'pairwise', 
                    method = 'cosine', remove_stopwords = True,
                    filler_words = [], stem = True, 
                    tfidf = True, LSA = True)
        # Test
        self.assertEqual(round(DocSim_Function.\
                               average_similarity(output, mode = 'pairwise'), 
                               4), 
                         round(output.iloc[1, 3], 4))
        self.assertEqual(round(DocSim_Function.\
                               average_similarity(output2, mode = 'pairwise'), 
                               4), 
                         round(output2.iloc[1, 3], 4))

    


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
# import numpy

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
#     'RawText':[d1,d2,d2,d2]
# }

# documents = pandas.DataFrame.from_dict(data)

# output = DocSim_Function.doc_sim(document_matrix = documents, 
#             text_col_name = 'RawText', mode = 'pairwise', 
#             method = 'cosine', remove_stopwords = False,
#             filler_words = [], stem = False, 
#             tfidf = False, LSA = False)
# output.head()
# # %%
# DocSim_Function.average_similarity(output)

# # %%
