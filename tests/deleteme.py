# %% 
import pandas
import sys
import unittest
import numpy
import os
import collections
import re
import string
import datetime

#Related 3rd Party Modules
import nltk
nltk.download('wordnet')
import scipy
import string
#sklearn does not automatically import its subpackages
import sklearn
import sklearn.metrics
import sklearn.decomposition
import sklearn.feature_extraction

sys.path.append('C:\\Users\\DaDa\\Documents\\GitHub\\Capstone-NLP-Edu-Interventions\\Duplicate_DocSim\\DocSim\\main')
import DocSim_class


data = pandas.read_csv("all_documents.csv")
data.head()
# %%
# wnl = nltk.stem.WordNetLemmatizer()
# %%
DocSim1 = DocSim_class.DocSim(data = data, 
                              skill = 'skill', 
                              study = 'study',
                              doc_type = 'doctype',
                              doc_id = 'filename',
                              text = 'text') 

# %% 
output = DocSim1.across_study_across_skill_normal_average(method = 'cosine', 
				   remove_stopwords = False,
				   filler_words = [], 
				   stem = False, 
                   lemm = False,
				   tfidf = False, 
				   tfidf_level = 'skill',
				   lsa = False, 
				   lsa_n_components = 5,
                   ngram = h)
output.head()

# %% 
output = DocSim1.pairwise_comparison(method = 'cosine', 
				   remove_stopwords = False,
				   filler_words = [], 
				   stem = False, 
                   lemm = False,
				   tfidf = False, 
				   tfidf_level = 'skill',
				   lsa = False, 
				   lsa_n_components = 5)
output.head()

# %% 
output = DocSim1.within_study_normal_average(method = 'cosine', 
				   remove_stopwords = False,
				   filler_words = [], 
				   stem = False, 
                   lemm = False,
				   tfidf = False, 
				   tfidf_level = 'skill',
				   lsa = False, 
				   lsa_n_components = 5)
output.head() 

# %% 
output = DocSim1.across_study_across_skill_normal_average(method = 'cosine', 
				   remove_stopwords = False,
				   filler_words = [], 
				   stem = False, 
                   lemm = False,
				   tfidf = False, 
				   tfidf_level = 'skill',
				   lsa = False, 
				   lsa_n_components = 5)
output.head() 
# %% 
output = DocSim1.across_study_across_skill_normal_average(method = 'cosine', 
				   remove_stopwords = False,
				   filler_words = [], 
				   stem = False, 
                   lemm = False,
				   tfidf = False, 
				   tfidf_level = 'skill',
				   lsa = False, 
				   lsa_n_components = 5)
output.head() 
# %%
text = 'text'
doc_id = 'filename' 
remove_stopwords = False
filler_words = []
stem = False
tfidf = True
tfidf_level = 'study'
lsa = False
lsa_n_components = 2

# Sort by doc_Id and make a copy of the data
data = data.sort_values(doc_id)
df = data.copy()
    
# Isolate text column as lowercase
text = data[text].str.lower()
# %% 

# Get all unique words for mapping
vectorizer = sklearn.feature_extraction.text.\
    CountVectorizer(lowercase = True, stop_words = filler_words)
vectors = vectorizer.fit_transform(text.tolist())

# Save feature names
unique_words = vectorizer.get_feature_names()
df_all = pandas.DataFrame(index = unique_words)
df_all.head()
# %% 
# Write the text so far back to data for loop       
# DocSim1.data = self.data.sort_values(self.skill)
# df = df.sort_values(self.skill) # For consistency
# self.data[self.text] = text
skill = 'skill'


data = data.sort_values(skill)
df = df.sort_values(skill) # For consistency
data[text] = text


# Create empty list to store results
vectors = list()
# self.factors = list()
factors = list()
def get_skill(data, skill):
    return data[skill].unique()

get_skill(data, skill)
skills = 1
# %% 

for skills in get_skill(data, skill):
    # Extract the script for this skill
    # tmp_text = self.data.loc[self.data[
    #     self.skill] == skills][self.text]
    tmp_text = data.loc[data[skill] == skills, [doc_id,'text']]

    # Vectorize
    vectorizer = sklearn.feature_extraction.text.\
        TfidfVectorizer(lowercase = True, 
                        stop_words = filler_words)
    tmp_vectors = vectorizer.fit_transform(tmp_text['text'].tolist()) # self.text

    # Get the TF-IDF weights and feature names
    tmp_weights = tmp_vectors.todense().tolist()
    tmp_factors = vectorizer.get_feature_names()

    # Store weights and feature names
    df_skill = pandas.DataFrame(numpy.transpose(tmp_weights), index = tmp_factors)
    vectors += df_all \
               .join(df_skill, how = "left") \
               .fillna(0) \
               .to_numpy() \
               .T \
               .tolist()
    print(len(df_all \
              .join(df_skill, how = "left") \
              .fillna(0) \
              .to_numpy() \
              .T \
              .tolist()))
    # self.factors += [(skills, tmp_factors)]
    factors += [(skills, tmp_factors)]

vectors = scipy.sparse.csr_matrix(vectors)

# %%
import pandas
import sys
import unittest
import numpy
import os
import collections
import re
import string
import datetime

#Related 3rd Party Modules
import nltk
import scipy
import string
#sklearn does not automatically import its subpackages
import sklearn
import sklearn.metrics
import sklearn.decomposition
import sklearn.feature_extraction
sys.path.append('C:\\Users\\DaDa\\Documents\\GitHub\\Capstone-NLP-Edu-Interventions\\Duplicate_DocSim\\DocSim\\main')
import DocSim_class

# data = pandas.read_csv("all_documents.csv")

# Raw Text
d1 = """films adapted from comic books have had plenty of success , whether 
        they're about superheroes ( batman , superman , spawn ) , or geared 
        toward kids ( casper ) or the arthouse crowd ( ghost world ) , 
        but there's never really been a comic book like from hell before . """
d2 = """films adapted from comic books have had plenty of success , whether 
        they're about superheroes ( batman , superman , spawn )"""

# Set up a example data frame      
data = {'document_id': ['123.txt','456.txt'],
        'study_id': ['Behavioral Study 1', 'Behavioral Study 1'], 
        'skill_id': [1, 1], 
        'type_id': ['script', 'transcript'],
        'raw_text': [d1, d2]}

data = pandas.DataFrame(data = data)

DocSim1 = DocSim_class.DocSim(data = data, 
                              skill = 'skill_id', 
                              study = 'study_id',
                              doc_type = 'type_id',
                              doc_id = 'document_id',
                              text = 'raw_text') 


# %%
# Make a copy of the data
output = DocSim1.normal_comparison(method = 'cosine', 
				   remove_stopwords = False,
				   filler_words = [], 
				   stem = False, 
				   tfidf = False, 
				   tfidf_level = 'skill',
				   lsa = False, 
				   lsa_n_components = 5)
output.head()
# Isolate text column as lowercase
# %%

level = 'filename' 
filler_words = []
text = DocSim1.data[DocSim1.text].str.lower()

# df = df.sort_values(level) # For consistency
# Get all unique words for mapping
vectorizer = sklearn.feature_extraction.text.\
        CountVectorizer(lowercase = True, stop_words = filler_words)
vectors = vectorizer.fit_transform(text.tolist())

# Save feature names as a dictionary of Data Frame
unique_words = vectorizer.get_feature_names()
df_all = pandas.DataFrame(index = unique_words)

# Write the text back to self.data and sort by skill
#   so that the ordering is correct
DocSim1.data = DocSim1.data.sort_values(level)
DocSim1.data[DocSim1.text] = text

# Create empty list to store results
vectors = list()
tfidf_factors = list()

for index in DocSim1.data[level].unique():                  

    # Extract the raw text for this study group
    tmp_text = DocSim1.data.loc[
        DocSim1.data[level] == index, [DocSim1.doc_id, DocSim1.text]]

    # Train and Fit TF-IDF 
    vectorizer = sklearn.feature_extraction.text.\
        TfidfVectorizer(lowercase = True, 
                        stop_words = filler_words)
    tmp_vectors = vectorizer.fit_transform(
        tmp_text[DocSim1.text].tolist())

    # Get the TF-IDF weights and feature names
    tmp_weights = tmp_vectors.todense().tolist()
    tmp_factors = vectorizer.get_feature_names()

    # Store weights as a data frame
    tmp_df = pandas.DataFrame(numpy.transpose(tmp_weights), 
                              index = tmp_factors)
    
    # Match with all unique words for identical structure
    vectors += df_all \
                .join(tmp_df, how = "left") \
                .fillna(0) \
                .to_numpy() \
                .T \
                .tolist()

    # Store features
    tfidf_factors += [(index, tmp_factors)]

# Convert vectors back to sparse matrix
# return scipy.sparse.csr_matrix(vectors)