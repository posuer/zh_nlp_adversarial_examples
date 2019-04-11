# zh_nlp_adversarial_examples
Generating Natural Language Adversarial Examples for Chinese NLI model

0) Prepare the counter_fitted_vectors

Collecting Chinese synonyms and antonyms word pairs

Train Chinese counter-fitted vectors by using https://github.com/nmrksic/counter-fitting

1) Build the vocabulary and embeddings matrix.
```
python build_embeddings.py
```

That will take like a minute, and it will tokenize the dataset and save it to a pickle file. It will also compute some auxiliary files like the matrix of the vector embeddings for words in our dictionary. All files will be saved under `aux_files` directory created by this script.

2) Train the NLI model.
```
python train_model.py
```

3) Pre-compute the distances between embeddings of different words (required to do the attack) and save the distance matrix.

```
python compute_dist_mat.py 

```
4) Now, we are ready to try some attacks ! You can do so by running the `AttackDemo.ipynb` jupyter notebook !


