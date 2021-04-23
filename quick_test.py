import TranscriptSim.DocSim_class
import pandas

d1 = """films adapted from comic books have had plenty of success , whether 
        they're about superheroes ( batman , superman , spawn ) , or geared 
        toward kids ( casper ) or the arthouse crowd ( ghost world ) , 
        but there's never really been a comic book like from hell before . """
d2 = """films adapted from comic books have had plenty of success , whether 
        they're about superheroes ( batman , superman , spawn )"""

# Set up a example data frame      
data = {'document_id': ['123.txt','456.txt'],
        'study_id': ['Behavioral Study', 'Behavioral Study 1'], 
        'skill_id': [1, 1], 
        'type_id': ['script', 'transcript'],
        'raw_text': [d1, d2]}
data = pandas.DataFrame(data = data)

# Create the DocSim class object
DocSim1 = TranscriptSim.DocSim_class.DocSim(data = data, 
										    skill = 'skill_id', 
										    study = 'study_id',
										    doc_type = 'type_id',
										    doc_id = 'document_id',
										    text = 'raw_text')

# Running the normal_comparison function
output = DocSim1.normal_comparison(method = 'cosine', 
				   remove_stopwords = False,
				   filler_words = [], 
				   stem = False, 
				   tfidf = False, 
				   tfidf_level = 'skill',
				   lsa = False, 
				   lsa_n_components = 5)

# Preview
output.head()

# Successful
print('Installation is successful!')