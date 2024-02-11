from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

openrice = pd.read_csv(r"/Users/joycechung/Downloads/Unit 1/01Foundations/11-APIs/data/open-rice.csv", encoding = 'utf8')
openrice['target_text'] = openrice.apply(lambda x: f"Name: {x['name']} Food type: {x['food_type']} Bookmarks: {x['bookmarks']}", axis=1)
target_docs = openrice['target_text'].tolist()

model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder="./model_folder/")

docs_encoding = model.encode(target_docs, show_progress_bar=True)
#converted openrice data into numbers
# progress bar added to check time of completion

while True:
    my_test_sentence = input("Search restaurants ('q' to quit): ")
    if my_test_sentence == 'q':
        break

    test_encoding = model.encode(my_test_sentence)
    #turning the words in my sentence input into numbers

    weights = util.dot_score(test_encoding, docs_encoding).numpy()[0]
#sentence transformer will compare the similarity of the test sentence and the openrice database
#util = it conducts cosine similarity search

    plt.hist(weights, bins=20)
    plt.show()

    idxs = np.argwhere(weights)[:, 0]
    relev_score = weights[idxs]

    # print(idxs, relev_score)

    relev_idxs = idxs[np.argsort(relev_score)[::-1]]
    #
    # print(relev_idxs)

    print(len(relev_idxs))

    for idx in relev_idxs[:10]:
        print(target_docs[idx])


