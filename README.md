# LT2222 V23 Assignment 3

Put any documentation here including any answers to the questions in the 
assignment on Canvas.

# Part 4 - Enron data ethics

When using the Enron corpus for machine learning several ethical concerns are raised. One of the main reasons is the way this corpus became public and therefore, free to use by anyone. The individuals whose e-mails came out in public sight because of this corpus, didn't give their explicit consent. Although this data was legally owned by the company (since the corporate mails were being used), the use of the e-mails without the individuals' consent is not morally justifiable. 

More specifically, there are ethical concerns about privacy and confidentiality. The mails were most probably not intended for the public's eye and using them as evidence could have had serious consequences on the people involved. As a result, using this sensitive information for machine learning experimentation can contribute to the perpetual violation of privacy confidentiality. Researchers must address and handle this corpus with careful consideration of its potentianl impact on those involved. 

In the end, most of the ethical concerns raised in the previous paragraphs rely on the way the Enron corpus is used for machine learning experimentaions. What is the nature of research and are there any potential risks for the parties involved? These are some questions we should have in mind before we embark on navigating through the complicated world of data use in machine learning research. One possible suggestion would be to always be evident about the data used and to obtain consent from everyone involved. This might help mitigate parts of the issues raised here and ensure the moral and responsible use of data.

# Part documentation 

Command-line Options:
-----------------------------
To run the code of the ***a3_features*** file, users should provide the following and predefined command-line options:

an input directory: a path to or a directory itself that contains a set of folders of mails named after their author
the name of the output file: just a name of the file that will contain the data

number of dimensions: the number of the wanted output dimensions of the word-based representations of all the texts

percentage of test instances (optional): an integer that would represent the percentage of instances to be included in the test set.

Example of how the code should be called like:
```python .\a3_features.py /scratch/lt2222-v23/enron_sample dataset 20 (--test 20)```

To run the code of the ***a3_model*** script, users should provide the following command-line options:

feature file: this is a required predi=efined argument. It represents the name of the file containing all the extracted and processed data from the mails of the previous script. Those scripts are interconnected, so it is assumed that the csv file created by a3_features.py is the input feature file.

-- epochs: optional argument made by me. The number of epochs it takes to train the model. If this argument is not provided the default number of epochs is 4.

-- linearity: optional argument created by me. The name of the non-linear activation function to use. The options are the relu or the sigmoid activation functions

If the optional arguments are missing their default values are 4 and ```None``` respectively.

Example of how the code should be called like:
```python .\a3_models.py /.dataset --epochs 4 --linearity relu/sigmoid```

Design
-----------------------------
***a3_features:***

Takes an input directory of files named after the author of the mails they contain. 
There are 3 functions that preprocess the data before they are converted into vectors, using the CountVectorizer. It was thought that the order of the words in each mail wasn't necessary to be preserved, so the bag of words method was implemented.
The first function removes the headers and the second removes the signatures, while both are used in the third function that stores every mail as a string inside a corpus list and every author inside an authors list. The two lists are of the same length and they correlate with each other by index positions. The remove_footer() function could have been implemented better without using the list of possible signatures.
The vecotrizing is done inside a vectorize() function which has an output of the vectorized mails with reduced dimensionality to the given number of dimensions by using the truncated value decomposition. This allows for a more compact and efficient representation of the text data, without too many rows of vectors, capturing important patterns and relationships between words.
Finally, given an output file name from the command line, the code stores all the data in a csv folder, where the author's name is stated in the first column as well as whether the instance was included in the training or the testing set. These particular columns will help extract the authors' names and the training and testing instances to train and test the model created in a3_model.py script.

***a3_model:***

The model is a simple Perceptron model that has an optional hidden layer (the hidden size can be modified to be 0 when the model is establieshed in the code). The size of the hidden layer is defined to be 50, because the confusion matrix had better values overall, with more correct predictions on the diagonall. The model has an input layer that takes the number of input features, and the output layer generates the probability distribution over the possible output labels. The hidden layer, when included, applies a non-linear transformation (defined by the user) to the input features before being passed to the output layer. The non-linear transformation applied to the hidden layer depends on the activation function specified by the user when calling the script. The options include the relu function, the sigmoid function, and the identity function, when none linear functions are defined. The model is defined in a class and it's trained in a function using the negative log-likelihood loss function and the Adam optimizer. The LabelEncoder is used to encode the target labels.
After the valuation of the trained model, a confusion matrix is printed.
