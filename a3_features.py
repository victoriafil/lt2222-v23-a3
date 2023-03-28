import os
import sys
import argparse
import numpy as np
import pandas as pd
# Whatever other imports you need
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

def strip_mail(text):
    """function that removes email headers"""
    for index, line in enumerate(text):
        if line.startswith("X-FileName:"):
            text = text[index+1:]
            return text

def remove_footer(text):
    """function that removes email signatures"""
    #I am aware that this is probably not the most efficient way of stripping the signatures
    things_to_strip = ["Susan Bailey", "Dean","Dutch","dq", "Eric", "Holden", "Jim Schwieger","Jim","Swig","dutch","Susan S. Bailey","Dan","-----Original Message-----", "Stephanie","Stephanie Panus", " -----Original Message-----","Larry May", "Larry","Craig", "Tom Donohoe", "Tom Donohoe.", "Kam", "KK", ":cc", "Ken","Ken Lay"]
    for index, line in enumerate(text):
        for phrase in things_to_strip:
            if phrase in line:
                text = text[:index]
                if '\n' in text:
                    text.remove('\n')
    return text     
         
def read_files(inputdir):
    """function that returns a list of mails and a list of author of same length given an input directory. Each mail has a corresponding author"""
    authors = []
    texts = [] 
    corpus = [] #one list with every mail as a single string
    for author in os.listdir(inputdir):
        author_dir = os.path.join(inputdir, author)
        #print(author_dir)
        mails = []
        if os.path.isdir(author_dir):
            for file in os.listdir(author_dir):
                file_path = os.path.join(author_dir, file)
                if os.path.isfile(file_path):
                    with open(file_path, encoding="UTF8") as myfile:
                        text = myfile.readlines()
                        text = strip_mail(text)
                        text = remove_footer(text)
                        mails.append(text)
                        authors.append(str(author))
        texts.append(mails)
    for mail_list in texts:
        for mails in mail_list:
            mail = ' '.join(mails)
            corpus.append(mail)
    return authors, corpus

def extract_features(inputdir, dim):
    """function that returns a dimensionality-reduced (to a fixed dimension, given as a parameter) word-based representation of all the texts labelled by author name"""
    authors, texts = read_files(inputdir)
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(texts)
    svd = TruncatedSVD(n_components=dim)
    reduced_features = svd.fit_transform(x)
    return reduced_features, authors

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()

    print("Reading {}...".format(args.inputdir))
    # Do what you need to read the documents here.
    authors, texts = read_files(args.inputdir)
    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
    # Build the table here.
    reduced_features, authors = extract_features(args.inputdir, args.dims)
    df = pd.DataFrame(reduced_features)
    df1 = pd.concat([df,pd.Series(authors,name='label')],axis=1)
    print("Writing to {}...".format(args.outputfile))
    # Write the table out here.
    train_features, test_features, train_labels, test_labels = train_test_split(reduced_features, authors, test_size=args.testsize/100, random_state=42)
    training_set = pd.concat([pd.DataFrame(train_features), pd.DataFrame(train_labels, columns=['label'])],axis=1)
    test_set = pd.concat([pd.DataFrame(test_features), pd.DataFrame(test_labels, columns=['label'])],axis=1)
    training_set['set'] = 'training set'
    test_set['set'] = 'testing set'
    training_set['author'] = training_set['label']
    test_set['author'] = test_set['label']
    args.outputfile = pd.concat([training_set, test_set])
    args.outputfile = args.outputfile[['author', 'set'] + list(range(args.dims))]
    args.outputfile.to_csv('all_data.csv', index=False)


    print("Done!")
    
