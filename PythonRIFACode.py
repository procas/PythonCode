import subprocess

subprocess.run(["pip", "install", "sentence_transformers"])
subprocess.run(["pip", "install", "pandas"])
subprocess.run(["pip", "install", "sklearn"])

from sentence_transformers import SentenceTransformer, util
import pandas as pd
import re
import sys
from sklearn.cluster import SpectralClustering

print('Finished importing packages')

model = SentenceTransformer('all-mpnet-base-v2')

print('Finished loading model')
def remove_special_chars(text):
    # define the pattern of special characters to remove
    pattern = r'[^a-zA-Z0-9\s]'
    # replace the special characters with an empty string
    text = re.sub(pattern, '', text)
    # remove slashes and brackets
    text = re.sub(r'[^\w\s]', " ", text)
    return text

# prsdf = pd.read_csv(sys.argv[1])
# prsdf['Feedback'] = prsdf['Feedback'][3:]
# for i in range(0, len(prsdf)):
#     prsdf['Feedback'][i] = remove_special_chars(str(prsdf['Feedback'][i]))
# prsdf.dropna()
# prsfeedbacks = prsdf['Feedback'].tolist()
# prsfeedbacks = [x for x in prsfeedbacks if str(x) != 'nan']
#
# print('Finished loading CSV')

prsfeedbacks = ["Customer service is messy", "Improve service please", "Not able to contact sales", "Product is not functional", "Fix bugs in product", "I love the product!!", "Not able to access tax folder"]

#encode the sentences
embeddings = model.encode(prsfeedbacks, convert_to_tensor=True)
embeddings

# perform Spectral Clustering without specifying the number of clusters
clustering = SpectralClustering(assign_labels='discretize', random_state=0).fit(embeddings)

# print the cluster labels
labels = clustering.labels_
labels

_dict = {}
for i in range(0, max(labels)+1):
    _dict[i] = []
for i in range(0, len(embeddings)):
    _dict[labels[i]].append(prsfeedbacks[i])

# our output is _dict
print(_dict)