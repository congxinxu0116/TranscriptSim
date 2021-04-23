#Standard Library Modules
import os

#Related 3rd Party Modules
import textract

def doc_to_txt(source_dir):
    """Accept directory address [source_dir], 
    find all .doc or .docx files in that directory
    and save each file as [same name].txt in 
    [source_dir]. If there are subfolders, any doc
    files in the subfolders will be saved as .txt 
    in [source_dir]."""

    #collect all files in given directory
    f = [os.path.join(root, name) \
            for root, _, files in os.walk(source_dir) \
            for name in files]

    count_convert = 0
    for file in f:
        filecln = file.split("\\")[-1]
        svhere = file.replace(filecln, "")
        filecln = filecln.split(".")[0]
        if ".doc" in file:
            #extract content of doc files from sessions by line
            text = textract.process(file).decode("utf-8") #.split('\n\n')
            text = text.encode("ascii", "ignore").decode()
            #print(len(text))
            with open(svhere + filecln + ".txt", "w") as text_file:
                text_file.write(text)
            count_convert += 1

    print(f"{count_convert} .doc files were saved as .txt in {source_dir}")

if __name__ == "__main__":
    srcdir = r"D:\Git\Capstone-NLP-Edu-Interventions" + \
        r"\Duplicate_DocSim\library\scripts"
    doc_to_txt(srcdir)