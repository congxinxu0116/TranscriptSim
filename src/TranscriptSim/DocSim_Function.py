# Standard Library Modules
import string
import os
import re

#Related 3rd Party Modules
import pandas
import numpy
import nltk
import sklearn
import sklearn.decomposition
import scipy

#Local Application Modules
#   NA

#set current working directory as the location of this file
#newly created files will be saved here unless otherwise specified
thisdir = os.path.dirname(os.path.abspath(__file__)) + "\\" 
os.chdir(thisdir)

def get_src_txt(srcdir=thisdir, saveas="raw_corpus.csv"):
    """Extract each line of each file in a directory [srcdir] 
    of word documents and possibly skill inking files. 
    Combine into a single CSV of labeled lines from documents. 
    Save output as raw_corpus.csv or other given filedir."""

    # collect all files in given directory
    f = [os.path.join(root, name) \
            for root, _, files in os.walk(srcdir) \
            for name in files]

    # start the dataframe with two empty rows
    collect_df = pandas.DataFrame([["", "",0,"","",""],
            ["", "",0,"","",""]], 
        columns=["rawtext", "src", 'i', \
            'parentdir1', 'parentdir2', 'filename'])

    # for each file identified, extract transcript files and skill linking files
    for i, file in enumerate(f):
        if "skills.csv" in file:
            # combine skill linking files as is
            df = pandas.read_csv(file, encoding="utf-8")
        elif ".txt" in file:
            with open(file, encoding="cp1252") as f:
                lines = [line.encode("ascii", 
                        "ignore").decode().rstrip('\n') for line in f]
                #text = ' '.join(lines)
            df = pandas.DataFrame(lines,columns=["rawtext"])
            df['rawtext'] = df['rawtext'].str.strip()
            df['rtlen'] = df.rawtext.str.len()
            df = df[df.rtlen > 0]
            df = df.dropna()
        else:
            #if not a skills doc or doc file, do not include it
            continue
        df['src'] = file
        df['i'] = i
        fs = file.split("\\")[-3:]
        df = df.assign(**{'parentdir1':fs[0], 
                'parentdir2':fs[1], 
                'filename':fs[2]})
                
        collect_df = collect_df.append(df)
        
    collect_df = collect_df[2:]
    print(f"Text extracted from {srcdir}")
    return collect_df

def removedonetext(row, extracted, xfrom, regex):
    """Apply vectorization function on a dataframe row [row]. 
    Once data is extracted, remove extract string [extracted] 
    from the indicated source column[xfrom]."""
    extr_ls = row[extracted]
    if pandas.isna(row[xfrom]) or \
        (type(extr_ls) != list and pandas.isna(extr_ls)):
        return row
    if type(extr_ls) != list:        
        extr_ls = [extr_ls]
    extr_ls = [x for x in extr_ls if x]
    for x in extr_ls:
        row[xfrom] = re.sub(regex, "", row[xfrom]).strip()
    return row

def extr_col(x, pattern, mult):
    """Function for Pandas Apply vectorizing. 
    Extract from src text [x] to add to a separate column 
    if any match of the given regex [pattern]. 
    If [mult]=True then extract multiple regex pattern group matches.
    """
    out = re.findall(pattern, x)
    if not out: #exit if no matches at all
        return numpy.nan
    if type(out[0]) == tuple:
        out = [x.strip() for x in list(out[0]) if x]
    elif len(out) > 1:
        out = [x.strip() for x in out if x]
    if not out: #exit if matches are all empty
        return numpy.nan
    if type(out) == list and not mult:
        out = out[0]
    return out

def add_col_from_extract(df1, colfrom, newcolname, regex, \
        mult=False, shift_equals=""):
    """Return the original given dataframe [df1] with a 
    new column [newcolname] created from matches returned from 
    the given regex pattern [regex] applied to a src column [colfrom]. 
    If [mult]=True, returns list of all matches, not just first.
    If shift_equals, returns [regex] match from row where prior row
    equal to [shift_equals] string."""

    # create empty column
    df1[newcolname] = numpy.nan

    # if shifting, use shift function inside where equals [shift_equals]
    if shift_equals:
        df1[newcolname] = numpy.where( \
            df1[colfrom].shift(1, axis = 0) == shift_equals,
            df1[colfrom], numpy.nan)
        # remove extracted text from src column
        df1[colfrom] = numpy.where(~df1[newcolname].isnull(), "",df1[colfrom])
    else:
    # otherwise, add regex match to new column
        df1[newcolname] = df1[colfrom].apply( \
            lambda x: extr_col(x, regex, mult))
        # remove extracted text from src column
        df1[colfrom] = df1[colfrom].apply(lambda x: \
            re.sub(regex, "", x).strip())

    # clean up beginning/end of reduced src column
    df1[colfrom] = df1[colfrom].str.lstrip(": ").str.strip()
    df1[colfrom] = df1[colfrom].str.lstrip("-")
    # return output 
    print(f"{newcolname} extracted into a new column")
    return df1

# convert timestamp to numeric second counter
def colon_delim_timestamp_to_second(x):
    """Apply vectorizer function, accepts raw text like timestamp,
    returns number of hours, minutes, and seconds converted to 
    a single numeric seconds value."""
    if pandas.isna(x): #if nothing here, return null
        return numpy.nan

    # get numeric timestamp matches
    nums = re.findall(r"(\d\d)?\:?(\d\d)\:(\d\d)",x)
    if not nums: #if no matches, return null
        return numpy.nan
    
    # convert regex match outcome into list of integers
    secs = 0
    incr = 0
    for num in range(len(list(nums[0])), -1, -1):
        #aggregate time values moving backward from 0, 
        #increase multiple by order of 60 as time in seconds
        if num:
            secs += int(num) * (incr * 60) 
            incr += 1
    return secs

def linkage_periods(df, saveas=""):
    """Extract linkage data from line strings 
    as extracted from source files."""
    
    links = df[(df['parentdir1'] == 'linking') | (df['parentdir2'] == 'linking')].copy()
    links['period'] = links['filename'].str.replace("_skills.csv", "")
    links = links[['filename', 'id', 'skill', 'coach', 'period']]
    if not saveas:
        return links
    else:
        links.to_csv(saveas, index=False)
    print("Linkage data has been extracted")

def group_by_speaker(clean):
    """Group raw and processed speech from [clean], 
    space concatenated, by speaker. If provided,
    save output as [saveas]. Returns dataframe."""
    
    #limit to columns of interest
    cleanlim = clean[["script_impl", "src", \
        "filename", "rawtext", "speaker", "text"]]
    cleanlim = cleanlim.query( \
        "src not in ['reference', 'scripts']").copy()

    # concatenate raw text by speaker
    joinraw = cleanlim.groupby(["script_impl", "src", \
        "filename", "speaker"])["rawtext"].apply(' '.join).reset_index()
    # concatenate cleaned text by speaker
    joinprocess = cleanlim.groupby(["filename", "speaker"])[ \
        "text"].apply(' '.join).reset_index()

    # join together to have raw text in same file as cleaned text
    joinedconcat = pandas.merge(joinraw, joinprocess,
        on=["filename", "speaker"],
        sort=False)

    # remove file name lines, not necessary
    joinedconcat = joinedconcat.query("speaker != 'File name'")

    print("Speaker names extracted")
    return joinedconcat

def clean_text(src_corpus_df):
    """Apply functions above to clean up raw_corpus content."""
    df = src_corpus_df.copy()
    
    # remove unhelpful \r character
    df = df.replace("\r", "")
    #  separate multi-line text to individual rows
    df['rawtext'] = df['rawtext'].str.split("\n")
    df = df.explode('rawtext')
    # clean up any resulting empty rows 
    df = df.dropna(subset=['rawtext'])
    
    df.columns = ["rawtext", "filedir", "file_index",
        "script_impl", "src", "filename", "txtlen"] #, 
        #"linkrow", "linkid", "linkskill", "linkcoach"]

    # extract period and document text, remove filedir
    df[['period', 'document']] = df.apply(lambda x: \
        x['filedir'].split("\\")[-2:], \
        axis='columns', result_type='expand')
    df = df.drop(["filedir"], axis=1)

    # incrementally process raw text in duplicate column
    df['text'] = df['rawtext'].values

    # extract otter service notes
    # do this first to ensure - 1 - is captured 
    # before leading dash removed
    df = add_col_from_extract(df, 'text', \
        'otter_notes', r"(Transcribed by https\:\/\/otter\.ai|\- \d* \-)")
    
    # replace typos and unhelpful strings
    replacements = {r"(00\:\,00:)": "00:00:",
                    r"(\:\:)": ":",
                    r"(Colleen \& Respondent)": "Colleen",
                    r"(\s{2,})": " ",
                    #r"(Speaker )\d": "Speaker",
                    r"(\–)": "-",
                    "(•)": "-",
                    r"( \- )": " ",
                    r"(’)": "'",
                    r"(see that as\:)": "see that as",
                    r"([Aa]ll? ?righty?)": "alright",
                    r"([Mm]+?\-?hm+)": "mhmm"}
    df['text'] = df['text'].replace(replacements, regex=True)

    # extract datetimes, convert to python datetime
    df = add_col_from_extract(df, 'text', \
        'date_time', \
        r"([A-Z][a-z]{,5}, \d{,2}\/\d{,2}(?:\s*\d{,2}:\d\d" \
            +r"\s?(?:A|P)M)?\s?(?:- \d{,2}:\d\d)?)")

    # add timestamp col and remove from text
    df = add_col_from_extract(df, 'text', \
        'speaker', r"^(?:([^\d:\[\]\n]*) \d\d\:\d\d" \
            + r"|[\[\]\d\: ]* ([\w ]{,25}?)\: |([\w ]*)\: )")

    # in some files, utterances by single speaker broke into multiple lines
    # where an utterance is not yet labeled by speaker,
    # fill down the last identified speaker
    df["speaker"] = df["speaker"].fillna(method='ffill')

    # extract character markers, unknown purpose/interpretation
    df = add_col_from_extract(df, 'text', \
        'char_marker', r"^([\d\_a-zC]{8,}(?:\_Transcript)?(?: - )?(?:\d*)?)")

    # extract timestamp values
    df = add_col_from_extract(df, 'text', \
        'timestamp', r"(\[?[\d\d\:]{5,}]?)")

    # extract time in seconds
    df['seconds'] = df['timestamp'].apply(lambda x: \
        colon_delim_timestamp_to_second(x))

    # extract audio notes in brackets
    df = add_col_from_extract(df, 'text', \
        'audio_note', r"(\[.*?\])", mult=True)

    # extract audio filename in format *.mp4
    df = add_col_from_extract(df, 'text', \
        'audio_file', r"([^\s]*\.mp4)$")
    df = add_col_from_extract(df, 'text', \
        'summary_keywords', "", shift_equals="SUMMARY KEYWORDS")
    df = add_col_from_extract(df, 'text', \
        'speakers', "", shift_equals="SPEAKERS")

    removeus = ["SUMMARY","KEYWORDS","SPEAKERS", r"\[unintelligible"]
    for ru in removeus:
        df['text'] = df['text'].replace(ru, "", regex=True)
        df['text'] = df['text'].apply(lambda x: 
            str(x).strip())
    print("Raw corpus text has been cleaned.")
    return df

def filename_to_link(x):
    """Extract linking information from each file name.
    Returns dictionary of (if existing) subject ID, 
    internal identifier, and group letter"""

    regs = [r"(?P<id>\d*)-(?P<internal>\d)(?P<grp>[A-Z])(?:\.doc|\.txt)",
        r".*?(?P<id>\d*)_(?P<internal>\d{,2})(?P<grp>[A-Za-z])"+ \
            r"?_(?:TC )?(?:Transcript)?(?:\.doc|\.txt)"]
    dic = {"id": "", "grp": "", "internal": ""}
    match = None
    for reg in regs:
        match = re.match(reg, x)
        if match:
            dic.update(match.groupdict())
    return dic


def linkages_connection(corpus_df, linkage_df):
    """Combine corpus and linkages to associate
    transcripts with scripts. 
    Needs more work without clear association guidance. """

    stc = corpus_df.copy()
    lp = linkage_df.copy()
    
    # # include coach speech only
    # stc = stc[stc['speaker'] == 'coach'] 

    # add time period column from given folder
    stc['period'] = stc.src.apply(lambda x: \
        x if "20" in x else numpy.nan)

    # create linkages dictionary from regex groups analyzing file name
    stc = stc.reset_index()
    stc["link_cols"] = stc['filename'].\
        apply(filename_to_link)

    # add linkages dictionaries as columns in stc dataframe
    fn_to_cols = pandas.concat([stc.drop('link_cols', axis=1), \
        pandas.DataFrame(stc['link_cols'].tolist())], axis=1)
    fn_to_cols['grp'] = fn_to_cols.grp.str.lower()

    # merge together linkages and src data
    link_cols = pandas.merge(fn_to_cols, lp, 
        on=["period", "id", "speaker"], how="inner")

    print("Corpus linked with content")
    return link_cols

#Preprocessing function
def preprocessing(data, text_col_name, remove_stopwords = False, 
                  filler_words = [], stem = False, tfidf = False, 
                  LSA = False, LSA_n_components = 2):
    """
    Parameters
    ----------
    data : Dataframe
    text_col_name : String
        Specify the name of the column that contains the document text.
    remove_stopwords : Boolean, optional
        Remove stopwords and punctuation from the Natural Language Toolkit's (NLTK) 
        pre-specified list. The default is False.
    filler_words : List of strings, optional
        Specify any additional stop words to be removed. The default is [].
    stem : Boolean, optional
        Replace all word derivatives with a single stem. The default is False.
    tfidf : Boolean, optional
        Weight each word frequency using term frequency–inverse document frequency 
        (tf-idf) weighting. The default is False.
    LSA : Boolean, optional
        Apply the dimentionality reduction technique Latent Semantic Analysis (LSA). 
        The default is False.
    LSA_n_components : Int, optional
        The number of topics in the output data. The default is 2.

    Returns
    -------
    df : Dataframe
        A copy data with an additional column containing the preprocessed and vectorized documents.
    
    
    Package Dependencies
    ------
    nltk
    string
    sklearn

    """

    # Check if column name specified is correct
    if text_col_name not in data.columns:
        raise SystemExit("Incorrect 'text_col_name' used. \
            Cannot find this column in the document_matrix")

    #make a copy of the data
    df = data.copy()
      
    #isolate text column as lowercase
    text = data[text_col_name].str.lower()
    
    # print("tokenize") #tracking size
    # print(len(text[0].values.tolist()[0]))
    if text.empty:
        return text

    #Define a set of stopwords
    if remove_stopwords:
        filler_words = set(nltk.corpus.stopwords.words('english')).\
            union(set(string.punctuation), set(filler_words))
    #Define stopwords as None 
    else:
        filler_words = None
        
        
    #Stem and remove stopwords that are in filler_words
    if stem:
        if filler_words == None:
            filler_words = [filler_words]
        text = text.apply(lambda x: nltk.tokenize.casual.casual_tokenize(x)) 
        # print("tokenize") #tracking size
        # print(len(text[0].values.tolist()[0]))
        # remove stopwords and stem
        text = text.apply(lambda x: [nltk.stem.SnowballStemmer('english').\
                                     stem(item) for item in x \
                                     if item not in filler_words])
        # print("stopwords and stem") #tracking size
        # print(len(text[0].values.tolist()[0]))
        # combine
        text = text.apply(lambda x: ' '.join([item for item in x]))
        
        #prevent removing stop words twice
        filler_words = None
        
    #Vectorize the text: using tf-idf weights 
    if tfidf:
        #vectorize
        vectors = sklearn.feature_extraction.text.\
            TfidfVectorizer(lowercase = True, stop_words=filler_words).\
                fit_transform(text.tolist())
        # print("Vectorize the text: using tf-idf weights") #tracking size
        # print(vectors.shape)
    
    #Vectorize the text: using word counts
    else:
        #count vectorize
        vectors = sklearn.feature_extraction.text.\
            CountVectorizer(lowercase = True, stop_words=filler_words).\
                fit_transform(text.tolist())
    
    #Apply LSA using the vectorized text
    if LSA:

        #rename the vectors
        vectorized_text = vectors
        #Define the LSA function
        LSA_function = sklearn.decomposition.TruncatedSVD( \
            n_components = LSA_n_components, random_state = 100)
        #Convert text to vectors
        vectors = LSA_function.fit_transform(vectorized_text)
        vectors = sklearn.preprocessing.Normalizer(copy=False).\
                  fit_transform(vectorized_text)
    # print("LSA") #tracking size
    # print(vectors.shape)
    
    #Append preprocessed text to the dataframe and return  
    dense = vectors.todense()
    denselist = dense.tolist()
    # print("denselist") #tracking size
    # print(len(denselist[0]))
    df["cleaned_vectorized_document"] = denselist
    return df


# doc_sim Function
def doc_sim(document_matrix, text_col_name, mode = 'normal', method = 'cosine', 
            remove_stopwords = False, filler_words = [], stem = False, 
            tfidf = False, LSA = False, LSA_n_components = 2):
    """Input:
    - document_matrix, pandas Data Frame with specific structure
    - The last column of this Data Frame must be the cleaned vectorized 
        word list. This requirement may change due to preprocessing
    - If mode = 'normal', the first row of the DF must be either benchmark
        script or session script from the first study.

    - mode:
    - Default: 'normal'
    --------------------------------------------------------------------
    Use Case 1: Within Study - Benchmark/Ideal Script vs. Transcript(s)
        Assume the first document/row is the Benchmark/Ideal script 
        Assume the rest of the documents are the Transcript(s)
    --------------------------------------------------------------------
    Use Case 2: Across Study - Transcript A1 vs. All Transcripts in Study B 
        Assume the first document is the Transcript from Study A
        Assume the rest of the documents are the Transcript from study B
    --------------------------------------------------------------------
    - pairwise
        - Calculate the pairwise similiary for each Transcript within the
            same study
    - method:
        - Default: 'cosine'
        - more can be added
    - remove_stop_words, Boolean
        - Default: True, use stop words
        - False, does not remove stop words
    - stem, Boolean
        - Default: True, stem all words
        - False, disable stem
    - tfidf: Boolean, Term frequency-inverse document-frequency 
        - Default: True, enable tfidf
        - False, disable tfidf 
    - LSA: Boolean, Latent Semantic Analysis
        - Default: True, enable lsa
        - False, disable lsa 
    - LSA_n_components: Int, Latent Semantic Analysis Components
        - Default: 2
        - Must be greater than or equal to 2

    Output: 
    - The function will create one additional column at the end of the input
        `document_matrix`, called 'similarity_score' with dtype = float

    Dependency:
    - Package: 
        - pandas
        - sklearn.feature_extraction.text
        - sklearn.metrics.pairwise
        - preprocessing()"""

    # Check if each input has the correct type
    if mode not in ('normal', 'pairwise'):
        raise SystemExit("Incorrect 'mode' used. \
            Choose 'normal' or 'pairwise'")

    # Check if the method is coded correctly
    if method not in ('cosine'):
        raise SystemExit("Incorrect 'method' used. Use 'cosine'")
    
    # Check preprocessing settings: 
    if remove_stopwords not in (True, False):
        raise SystemExit("Incorrect 'remove_stopwords' used. \
            Use True or False")
    if stem not in (True, False):
        raise SystemExit("Incorrect 'stem' used. Use True or False")
    if tfidf not in (True, False):
        raise SystemExit("Incorrect 'tfidf' used. Use True or False")
    if LSA not in (True, False):
        raise SystemExit("Incorrect 'lsa' used. Use True or False")
    if type(LSA_n_components) is not int:
        raise SystemExit("Incorrect 'LSA_n_components' used. \
            LSA_n_components must be an integer")
    elif LSA_n_components < 2:
        raise SystemExit("Incorrect 'LSA_n_components' used. \
            Set LSA_n_components as an int that is greater than or equal to 2")

    # NLP Preprocessing: 
    document_matrix = preprocessing(data = document_matrix, 
                                    remove_stopwords = remove_stopwords, 
                                    filler_words = filler_words,
                                    text_col_name = text_col_name,
                                    stem = stem, 
                                    tfidf = tfidf,
                                    LSA = LSA,
                                    LSA_n_components = LSA_n_components)
    #make a copy to reset the index 
    # and avoid column addition translations
    firstcol = list(document_matrix.columns)[0]
    document_matrix = document_matrix.reset_index().copy().loc[:,firstcol:]
    
    if document_matrix.empty:
        raise SystemExit("Insufficient data to process.")

    # Convert the vectorized column to a sparse matrix
    #   A sparse matrix is a matrix that majority of the elements are zero.
    #   To save memory space and to increase computational efficiency, we 
    #   convert the vectorized column to a sparse matrix to improve the 
    #   performance.
    #   
    #   `cleaned_vectorized_document` is the column generated by the 
    #   preprocessing() function above. It is the reserved column name that
    #   specifies the output of preprocessing.
    count_matrix = scipy.sparse.\
        csr_matrix([i for i in document_matrix['cleaned_vectorized_document']])

    # Main Section:
    if mode == 'normal':   

        # Calculate the Similarity Score
        if method == 'cosine':           
            # Calculate the cosine similarity score
            similarity_score = sklearn.metrics.pairwise.\
                               cosine_similarity(count_matrix[0],
                                                 count_matrix,
                                                 dense_output = True)
            
            # Write the similarity score back to the orignial DF
            document_matrix['similarity_score'] = \
                similarity_score.reshape(-1, 1) #.round(6)

            # Return the output data frame
            return(document_matrix)
    
    if mode == 'pairwise':

        # Create an empty pandas Series to store the 
        #   average pairwise similarity score
        scores = pandas.Series(dtype = float)
            
        # Calculate the Similarity Score
        if method == 'cosine':
            # Calculate the pairwise cosine similarty for each document
            # Assume the all document is the session scripts
            # Assume all documents are in the same study
            #for index, _ in document_matrix.iterrows():
            for index in range(document_matrix.shape[0]):
                # Calculate the cosine similarity for each document      
                similarity_score = sklearn.metrics.pairwise.\
                                   cosine_similarity(count_matrix[index],
                                                     count_matrix,
                                                     dense_output = True)
                # Add each average pairwise score to the storage Series
                scores.at[index] = average_similarity(
                    pandas.DataFrame(similarity_score.reshape(-1, 1))
                )

            # Write the similarity score back to the orignial DF
            document_matrix['similarity_score'] = scores #.round(6)
            
            # Return the output data frame
            return(document_matrix)

#  Main Function 2: average_similarity
def average_similarity(document_matrix, mode = 'normal'):
    """Input:
    - document_matrix
    - This should be the output of the doc_sim() function.

    - mode:
    - Default: 'normal'
        - Calculate the average similarity scores against the benchmark script
        - Calculate the average across study similarity scores.
    - pairwise
        - Calculate the average pairwise within study similarity scores"""

    # Check if each input has the correct type
    if mode not in ('normal', 'pairwise'):
        raise SystemExit("Incorrect 'mode' used. \
            Choose 'normal' or 'pairwise'")
    
    # Get the last column of the document_matrix
    scores = document_matrix[document_matrix.columns[-1]] 

    # Check to see if the type of scores is 
    if scores.dtype != float:
        raise SystemExit("Incorrect 'document_matrix' used. \
            Make sure the last column of the `document_matrix` \
            is the similarity_score.")

    # Compute the average similiarty
    # We subtract 1 from the total score because the similarity score 
    #   of the first script withitself is always 1
    if mode == 'normal':
        return((sum(scores) - 1) / (len(scores) - 1))
    # For 'pairwise' mode, all similarity score are already properly calculated
    #   and there is no benchmark script or sessions from other study exist in
    #   the document matrix.
    if mode == 'pairwise':
        return scores.mean()
