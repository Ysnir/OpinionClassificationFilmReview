{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Sentiment Analysis in a movie review corpus\n",
    "## 1- First model comparison with raw text\n",
    "### a) Importing dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as p\n",
    "\n",
    "#Importing dataset into a Dataframe\n",
    "corpus_path = 'data/dataset.csv'\n",
    "label_path = 'data/labels.csv'\n",
    "label = p.read_table(label_path, header=None, names=['label'])\n",
    "text = p.read_table(corpus_path, header=None, names=['review'])\n",
    "\n",
    "##to align label and text\n",
    "label.reset_index(drop=True, inplace=True)\n",
    "text.reset_index(drop=True, inplace=True)\n",
    "\n",
    "#concatenate label and text together aligned\n",
    "\n",
    "review = p.concat([text.reset_index(drop=True), label.reset_index(drop=True)], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "pandas.core.frame.DataFrame"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "type(label)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10000, 2)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "review.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>review</th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>I ended up watching this whole (very long) mov...</td>\n",
       "      <td>-1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Where do I start? Per the title of this film I...</td>\n",
       "      <td>-1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>This is absolutely the dumbest movie I've ever...</td>\n",
       "      <td>-1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Bad Movie - saw it at the TIFF and the movie g...</td>\n",
       "      <td>-1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>What I found so curious about this film--I saw...</td>\n",
       "      <td>-1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>The Cat in the Hat is just a slap in the face ...</td>\n",
       "      <td>-1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>This is going to be the most useless comment I...</td>\n",
       "      <td>-1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>I loved Adrianne Curry before this show. I tho...</td>\n",
       "      <td>-1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>I don't really post comments, but wanted to ma...</td>\n",
       "      <td>-1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>Would have better strengthened considerably by...</td>\n",
       "      <td>-1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                              review  label\n",
       "0  I ended up watching this whole (very long) mov...     -1\n",
       "1  Where do I start? Per the title of this film I...     -1\n",
       "2  This is absolutely the dumbest movie I've ever...     -1\n",
       "3  Bad Movie - saw it at the TIFF and the movie g...     -1\n",
       "4  What I found so curious about this film--I saw...     -1\n",
       "5  The Cat in the Hat is just a slap in the face ...     -1\n",
       "6  This is going to be the most useless comment I...     -1\n",
       "7  I loved Adrianne Curry before this show. I tho...     -1\n",
       "8  I don't really post comments, but wanted to ma...     -1\n",
       "9  Would have better strengthened considerably by...     -1"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "review.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>review</th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>9990</th>\n",
       "      <td>What can be said, really... \"The Tenant\" is a ...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9991</th>\n",
       "      <td>Wow this was a movie was completely captivatin...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9992</th>\n",
       "      <td>A quite good film version of the novel, though...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9993</th>\n",
       "      <td>I saw this movie with my friend and we couldnt...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9994</th>\n",
       "      <td>While the story is sweet, and the dancing and ...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9995</th>\n",
       "      <td>An utterly beautiful film, one of a handful of...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9996</th>\n",
       "      <td>Being that this movie has a lot of fine entert...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9997</th>\n",
       "      <td>This was my favourite film as a child, and I h...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9998</th>\n",
       "      <td>Paris, je t'aime (2006) is a film made up of 1...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9999</th>\n",
       "      <td>John Water's (\"Pink Flamingos\"...) \"Pecker\" is...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                 review  label\n",
       "9990  What can be said, really... \"The Tenant\" is a ...      1\n",
       "9991  Wow this was a movie was completely captivatin...      1\n",
       "9992  A quite good film version of the novel, though...      1\n",
       "9993  I saw this movie with my friend and we couldnt...      1\n",
       "9994  While the story is sweet, and the dancing and ...      1\n",
       "9995  An utterly beautiful film, one of a handful of...      1\n",
       "9996  Being that this movie has a lot of fine entert...      1\n",
       "9997  This was my favourite film as a child, and I h...      1\n",
       "9998  Paris, je t'aime (2006) is a film made up of 1...      1\n",
       "9999  John Water's (\"Pink Flamingos\"...) \"Pecker\" is...      1"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "review.tail(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "-1    5000\n",
       " 1    5000\n",
       "Name: label, dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "review.label.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "FEATURES = review.review\n",
    "CLASS = review.label "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(7500,)\n",
      "(2500,)\n",
      "(7500,)\n",
      "(2500,)\n"
     ]
    }
   ],
   "source": [
    "#creating testing and training set\n",
    "from sklearn.model_selection import train_test_split\n",
    "train_feature, test_feature, train_class, test_class = train_test_split(FEATURES, CLASS, random_state=1)\n",
    "print(train_feature.shape)\n",
    "print(test_feature.shape)\n",
    "print(train_class.shape)\n",
    "print(test_class.shape)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Vectorizing the text data\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "vect = TfidfVectorizer()\n",
    "#create a matrix with terms in each documents\n",
    "train_feature_m = vect.fit_transform(train_feature)\n",
    "test_feature_m = vect.transform(test_feature)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### b) Building Naive Bayes model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Wall time: 9.53 ms\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.naive_bayes import MultinomialNB\n",
    "nb = MultinomialNB()\n",
    "\n",
    "#Mesure time for model training\n",
    "%time nb.fit(train_feature_m, train_class)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn import metrics\n",
    "\n",
    "#Prediction of the class for the test set\n",
    "pred_class = nb.predict(test_feature_m)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8964"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "metrics.accuracy_score(test_class, pred_class)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1177,   94],\n",
       "       [ 165, 1064]], dtype=int64)"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "metrics.confusion_matrix(test_class, pred_class)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "false_negative = test_feature[(pred_class < test_class)]\n",
    "false_positive = test_feature[(pred_class > test_class)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "6111    This was a great movie with a good cast, all o...\n",
       "7781    There is a certain genius behind this movie. I...\n",
       "7775    If you are a fan, then you will probably enjoy...\n",
       "8074    Besides the fact that my list of favorite movi...\n",
       "5131    Dream Quest was a surprisingly good movie. The...\n",
       "9112    I thought this movie would be dumb, but I real...\n",
       "7782    I remember seeing this movie when I was about ...\n",
       "7214    For a \"no budget\" movie this thing rocks. I do...\n",
       "5482    I was up late flipping cable channels one nigh...\n",
       "9629    Will all of you please lay the hell off Todd S...\n",
       "Name: review, dtype: object"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "false_negative.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'There is a certain genius behind this movie. I was laughing throughout. The scene in the phone sex office, discussing how love heals the doppelganger was a nice attempt at this genius/humor. Execution is poor, but you can see the writer\\'s message and they do have some talent. The doppelganger split at the end was like... \"ok, wasn\\'t quite expecting that but let\\'s see what the movie has to say\". Certainly ridiculous, but a sweet idea and actually very coherent to the story in a strange way.Is the point of a movie to be logical or is it to be entertaining or communicate on an emotional level? i\\'m easily bored by many movies, but this one kept my interest throughout.I think the story may have some auto-biographical roots, but that\\'s just a guess. Horribly bad, but good. I\\'m looking for other movies this person may have done (with more experience).'"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_feature[7781]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'This was a great movie with a good cast, all of them hitting on all cylinders. And when Dianne Keaton is at her best, well, it just doesn\\'t get any better than that. But Tom Everett Scott, always underrated, was even better. He should be a star.My only complaint is with one aspect of the screenplay. None of the characters ever acknowledged that the dead daughter wasn\\'t always a good person. And neither was her mother, played by Keaton. At one point she breaks a promise she made to one character not to reveal that he had been sleeping around.One of the other commentators said the movie had a \"political agenda\". That is a baffling thing to say. There was no politics at all in this movie.'"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_feature[6111]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### c) Building K-Nearest Neighbors model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Wall time: 7.52 ms\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',\n",
       "           metric_params=None, n_jobs=1, n_neighbors=5, p=2,\n",
       "           weights='uniform')"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "\n",
    "#K-nearest neighbors with default parameters take k = 5\n",
    "knn = KNeighborsClassifier()\n",
    "\n",
    "%time knn.fit(train_feature_m, train_class)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "pred_class = knn.predict(test_feature_m)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7912"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "metrics.accuracy_score(test_class, pred_class)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Poor accuracy score for KNN with k=5 so we are trying to test with different values for k (commented long execution time)\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "\n",
    "knn_accuracy = {}\n",
    "\n",
    "for k in range(1, 50) :\n",
    "    knn = KNeighborsClassifier(n_neighbors=k)\n",
    "    knn.fit(train_feature_m, train_class)\n",
    "    pred_class = knn.predict(test_feature_m)\n",
    "    knn_accuracy[k] = metrics.accuracy_score(test_class, pred_class)\n",
    "    \n",
    "#loading the dictionary into a pandas series\n",
    "s = p.Series(data=knn_accuracy, index=knn_accuracy.keys())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x217dd8ef400>"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x217dd8e25c0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "%matplotlib inline\n",
    "\n",
    "import matplotlib\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "s.plot(kind='line')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "41"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#maximum accuracy for k=\n",
    "s.idxmax()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8256"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "s.max()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### d) Building Decision Tree model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Wall time: 4.59 s\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,\n",
       "            max_features=None, max_leaf_nodes=None,\n",
       "            min_impurity_decrease=0.0, min_impurity_split=None,\n",
       "            min_samples_leaf=1, min_samples_split=2,\n",
       "            min_weight_fraction_leaf=0.0, presort=False, random_state=None,\n",
       "            splitter='best')"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn import tree\n",
    "dtc = tree.DecisionTreeClassifier()\n",
    "\n",
    "%time dtc.fit(train_feature_m, train_class)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "pred_class = dtc.predict(test_feature_m)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7484"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "metrics.accuracy_score(test_class, pred_class)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2- Model Evaluation on Preprocessed corpora\n",
    "See : text-processing.ipynb"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### a) Importing corpora with different preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as p\n",
    "#Importing dataset into a Dataframe\n",
    "label = p.read_table('data/labels.csv', header=None, names=['label'])\n",
    "stopword = p.read_table('data/stopwords.csv', header=None, names=['review'])\n",
    "lemmatized = p.read_table('data/lemmatized.csv', header=None, names=['review'])\n",
    "postagged = p.read_table('data/postagged.csv', header=None, names=['review'])\n",
    "\n",
    "\n",
    "\n",
    "##to align label and text\n",
    "label.reset_index(drop=True, inplace=True)\n",
    "stopword.reset_index(drop=True, inplace=True)\n",
    "lemmatized.reset_index(drop=True, inplace=True)\n",
    "postagged.reset_index(drop=True, inplace=True)\n",
    "\n",
    "#concatenate label and text together aligned in a Dataframe & extract features en class separatly\n",
    "stop_corpus = p.concat([stopword.reset_index(drop=True), label.reset_index(drop=True)], axis=1)\n",
    "STOP_FEATURES = stop_corpus.review\n",
    "STOP_CLASS = stop_corpus.label \n",
    "lemma_corpus = p.concat([lemmatized.reset_index(drop=True), label.reset_index(drop=True)], axis=1)\n",
    "LEMMA_FEATURES = lemma_corpus.review\n",
    "LEMMA_CLASS = lemma_corpus.label \n",
    "pos_corpus = p.concat([postagged.reset_index(drop=True), label.reset_index(drop=True)], axis=1)\n",
    "POS_FEATURES = pos_corpus.review\n",
    "POS_CLASS = pos_corpus.label \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### b) Creating different pipelines for model evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import cross_val_score\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.linear_model import SGDClassifier\n",
    "\n",
    "#Using pipeline to fit the vectorization on each \"train fold\" only and not on the fold used for testing\n",
    "naive_bayes = Pipeline([('vect', TfidfVectorizer()), ('nb', MultinomialNB())])\n",
    "knn = Pipeline([('vect', TfidfVectorizer()), ('knn', KNeighborsClassifier(n_neighbors=41))])\n",
    "decision_tree = Pipeline([('vect', TfidfVectorizer()), ('dtc', DecisionTreeClassifier())])\n",
    "svm = Pipeline([('vect', TfidfVectorizer()), ('svc', SVC(kernel='linear', C=1))])\n",
    "sgd = Pipeline([('vect', TfidfVectorizer()), ('sgd', SGDClassifier(max_iter=100))])\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### c) Running K-fold crossvalidation on the pipelines"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### - Naive Bayes evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.898"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Running 10-fold crossvalidation with the brut text corpus\n",
    "cross_val_score(naive_bayes, STOP_FEATURES, STOP_CLASS, cv=10).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8975000000000002"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#On the lemmatized corpus\n",
    "cross_val_score(naive_bayes, LEMMA_FEATURES, LEMMA_CLASS, cv=10).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8916000000000001"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#On the lemmatized and postagged corpus\n",
    "cross_val_score(naive_bayes, POS_FEATURES, POS_CLASS, cv=10).mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### - K-nearest neighbors evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8465"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#On the brut text corpus\n",
    "cross_val_score(knn, STOP_FEATURES, STOP_CLASS, cv=10).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8360999999999998"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#On the lemmatized corpus\n",
    "cross_val_score(knn, LEMMA_FEATURES, LEMMA_CLASS, cv=10).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8347999999999999"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#On the lemmatized and postagged corpus\n",
    "cross_val_score(knn, POS_FEATURES, POS_CLASS, cv=10).mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### - Decision tree classifier evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7619"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#On the brut text corpus\n",
    "cross_val_score(decision_tree, STOP_FEATURES, STOP_CLASS, cv=10).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7636000000000001"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#On the lemmatized corpus\n",
    "cross_val_score(decision_tree, LEMMA_FEATURES, LEMMA_CLASS, cv=10).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7583"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#On the lemmatized and postagged corpus\n",
    "cross_val_score(decision_tree, POS_FEATURES, POS_CLASS, cv=10).mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### - Support vector machine evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9205"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#On the brut text corpus\n",
    "cross_val_score(svm, STOP_FEATURES, STOP_CLASS, cv=10,n_jobs=-1).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9192"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#On the lemmatized corpus\n",
    "cross_val_score(svm, LEMMA_FEATURES, LEMMA_CLASS, cv=10).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9139999999999999"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#On the lemmatized and postagged corpus\n",
    "cross_val_score(svm, POS_FEATURES, POS_CLASS, cv=10).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Wall time: 6min\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "0.9259000000000001"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ngram_svm = Pipeline([('vect', TfidfVectorizer(ngram_range=(1,2))), ('svc', SVC(kernel='linear', C=1))])\n",
    "\n",
    "%time cross_val_score(ngram_svm, STOP_FEATURES, STOP_CLASS, cv=10,n_jobs=-1).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Wall time: 8min 35s\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "0.9277000000000001"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "%time cross_val_score(ngram_svm, FEATURES, CLASS, cv=10,n_jobs=-1).mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### - Stochastic gradiant descent evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9198999999999999"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#On the brut text corpus\n",
    "cross_val_score(sgd, STOP_FEATURES, STOP_CLASS, cv=10,n_jobs=-1).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9192000000000002"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#On the lemmatized corpus\n",
    "cross_val_score(sgd, LEMMA_FEATURES, LEMMA_CLASS, cv=10,n_jobs=-1).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9137000000000001"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#On the lemmatized and postagged corpus\n",
    "cross_val_score(sgd, POS_FEATURES, POS_CLASS, cv=10,n_jobs=-1).mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### d) Parameter tuning for the best models"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### - Support Vector Machine"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "GridSearchCV(cv=10, error_score='raise',\n",
       "       estimator=Pipeline(memory=None,\n",
       "     steps=[('vect', TfidfVectorizer(analyzer='word', binary=False, decode_error='strict',\n",
       "        dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',\n",
       "        lowercase=True, max_df=1.0, max_features=None, min_df=1,\n",
       "        ngram_range=(1, 1), norm='l2', preprocessor=None, smooth_idf=True,\n",
       "  ...,\n",
       "  max_iter=-1, probability=False, random_state=None, shrinking=True,\n",
       "  tol=0.001, verbose=False))]),\n",
       "       fit_params=None, iid=True, n_jobs=-1,\n",
       "       param_grid={'vect__ngram_range': ((1, 1), (1, 2))},\n",
       "       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',\n",
       "       scoring='accuracy', verbose=0)"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.model_selection import GridSearchCV\n",
    "\n",
    "#,'svc__kernel': ['linear', 'rbf'], 'svc__C':[1, 10]}\n",
    "\n",
    "parameters = {'vect__ngram_range':((1,1),(1, 2))}\n",
    "\n",
    "svm = Pipeline([('vect', TfidfVectorizer()), ('svc', SVC())])\n",
    "\n",
    "grid = GridSearchCV(svm, parameters, cv=10, scoring='accuracy',n_jobs=-1)\n",
    "grid.fit(STOP_FEATURES, STOP_CLASS)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "E:\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\deprecation.py:122: FutureWarning: You are accessing a training score ('mean_train_score'), which will not be available by default any more in 0.21. If you need training scores, please set return_train_score=True\n",
      "  warnings.warn(*warn_args, **warn_kwargs)\n",
      "E:\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\deprecation.py:122: FutureWarning: You are accessing a training score ('split0_train_score'), which will not be available by default any more in 0.21. If you need training scores, please set return_train_score=True\n",
      "  warnings.warn(*warn_args, **warn_kwargs)\n",
      "E:\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\deprecation.py:122: FutureWarning: You are accessing a training score ('split1_train_score'), which will not be available by default any more in 0.21. If you need training scores, please set return_train_score=True\n",
      "  warnings.warn(*warn_args, **warn_kwargs)\n",
      "E:\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\deprecation.py:122: FutureWarning: You are accessing a training score ('split2_train_score'), which will not be available by default any more in 0.21. If you need training scores, please set return_train_score=True\n",
      "  warnings.warn(*warn_args, **warn_kwargs)\n",
      "E:\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\deprecation.py:122: FutureWarning: You are accessing a training score ('split3_train_score'), which will not be available by default any more in 0.21. If you need training scores, please set return_train_score=True\n",
      "  warnings.warn(*warn_args, **warn_kwargs)\n",
      "E:\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\deprecation.py:122: FutureWarning: You are accessing a training score ('split4_train_score'), which will not be available by default any more in 0.21. If you need training scores, please set return_train_score=True\n",
      "  warnings.warn(*warn_args, **warn_kwargs)\n",
      "E:\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\deprecation.py:122: FutureWarning: You are accessing a training score ('split5_train_score'), which will not be available by default any more in 0.21. If you need training scores, please set return_train_score=True\n",
      "  warnings.warn(*warn_args, **warn_kwargs)\n",
      "E:\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\deprecation.py:122: FutureWarning: You are accessing a training score ('split6_train_score'), which will not be available by default any more in 0.21. If you need training scores, please set return_train_score=True\n",
      "  warnings.warn(*warn_args, **warn_kwargs)\n",
      "E:\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\deprecation.py:122: FutureWarning: You are accessing a training score ('split7_train_score'), which will not be available by default any more in 0.21. If you need training scores, please set return_train_score=True\n",
      "  warnings.warn(*warn_args, **warn_kwargs)\n",
      "E:\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\deprecation.py:122: FutureWarning: You are accessing a training score ('split8_train_score'), which will not be available by default any more in 0.21. If you need training scores, please set return_train_score=True\n",
      "  warnings.warn(*warn_args, **warn_kwargs)\n",
      "E:\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\deprecation.py:122: FutureWarning: You are accessing a training score ('split9_train_score'), which will not be available by default any more in 0.21. If you need training scores, please set return_train_score=True\n",
      "  warnings.warn(*warn_args, **warn_kwargs)\n",
      "E:\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\deprecation.py:122: FutureWarning: You are accessing a training score ('std_train_score'), which will not be available by default any more in 0.21. If you need training scores, please set return_train_score=True\n",
      "  warnings.warn(*warn_args, **warn_kwargs)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "{'mean_fit_time': array([117.45812931, 171.12647622]),\n",
       " 'mean_score_time': array([11.50697193, 14.2025346 ]),\n",
       " 'mean_test_score': array([0.6764, 0.7386]),\n",
       " 'mean_train_score': array([0.68071111, 0.69544444]),\n",
       " 'param_vect__ngram_range': masked_array(data=[(1, 1), (1, 2)],\n",
       "              mask=[False, False],\n",
       "        fill_value='?',\n",
       "             dtype=object),\n",
       " 'params': [{'vect__ngram_range': (1, 1)}, {'vect__ngram_range': (1, 2)}],\n",
       " 'rank_test_score': array([2, 1]),\n",
       " 'split0_test_score': array([0.686, 0.796]),\n",
       " 'split0_train_score': array([0.67777778, 0.785     ]),\n",
       " 'split1_test_score': array([0.692, 0.754]),\n",
       " 'split1_train_score': array([0.68188889, 0.70266667]),\n",
       " 'split2_test_score': array([0.676, 0.687]),\n",
       " 'split2_train_score': array([0.68822222, 0.64211111]),\n",
       " 'split3_test_score': array([0.704, 0.739]),\n",
       " 'split3_train_score': array([0.68411111, 0.64966667]),\n",
       " 'split4_test_score': array([0.642, 0.677]),\n",
       " 'split4_train_score': array([0.67933333, 0.65555556]),\n",
       " 'split5_test_score': array([0.686, 0.721]),\n",
       " 'split5_train_score': array([0.68066667, 0.65255556]),\n",
       " 'split6_test_score': array([0.678, 0.78 ]),\n",
       " 'split6_train_score': array([0.68411111, 0.73877778]),\n",
       " 'split7_test_score': array([0.675, 0.71 ]),\n",
       " 'split7_train_score': array([0.68133333, 0.66944444]),\n",
       " 'split8_test_score': array([0.676, 0.78 ]),\n",
       " 'split8_train_score': array([0.67977778, 0.74255556]),\n",
       " 'split9_test_score': array([0.649, 0.742]),\n",
       " 'split9_train_score': array([0.66988889, 0.71611111]),\n",
       " 'std_fit_time': array([11.72721111, 31.97188745]),\n",
       " 'std_score_time': array([0.52040566, 2.62376638]),\n",
       " 'std_test_score': array([0.01768728, 0.03815285]),\n",
       " 'std_train_score': array([0.00458047, 0.0465224 ])}"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "grid.cv_results_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "GridSearchCV(cv=10, error_score='raise',\n",
       "       estimator=Pipeline(memory=None,\n",
       "     steps=[('vect', TfidfVectorizer(analyzer='word', binary=False, decode_error='strict',\n",
       "        dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',\n",
       "        lowercase=True, max_df=1.0, max_features=None, min_df=1,\n",
       "        ngram_range=(1, 1), norm='l2', preprocessor=None, smooth_idf=True,\n",
       "  ...,\n",
       "  max_iter=-1, probability=False, random_state=None, shrinking=True,\n",
       "  tol=0.001, verbose=False))]),\n",
       "       fit_params=None, iid=True, n_jobs=-1,\n",
       "       param_grid={'svc__kernel': ['linear', 'rbf'], 'svc__C': [1, 10]},\n",
       "       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',\n",
       "       scoring='accuracy', verbose=0)"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.model_selection import GridSearchCV\n",
    "\n",
    "#,'svc__kernel': ['linear', 'rbf'], 'svc__C':[1, 10]}\n",
    "\n",
    "parameters = {'svc__kernel': ['linear', 'rbf'], 'svc__C':[1, 10]}\n",
    "\n",
    "svm = Pipeline([('vect', TfidfVectorizer()), ('svc', SVC())])\n",
    "\n",
    "grid = GridSearchCV(svm, parameters, cv=10, scoring='accuracy',n_jobs=-1)\n",
    "grid.fit(STOP_FEATURES, STOP_CLASS)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "E:\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\deprecation.py:122: FutureWarning: You are accessing a training score ('mean_train_score'), which will not be available by default any more in 0.21. If you need training scores, please set return_train_score=True\n",
      "  warnings.warn(*warn_args, **warn_kwargs)\n",
      "E:\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\deprecation.py:122: FutureWarning: You are accessing a training score ('split0_train_score'), which will not be available by default any more in 0.21. If you need training scores, please set return_train_score=True\n",
      "  warnings.warn(*warn_args, **warn_kwargs)\n",
      "E:\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\deprecation.py:122: FutureWarning: You are accessing a training score ('split1_train_score'), which will not be available by default any more in 0.21. If you need training scores, please set return_train_score=True\n",
      "  warnings.warn(*warn_args, **warn_kwargs)\n",
      "E:\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\deprecation.py:122: FutureWarning: You are accessing a training score ('split2_train_score'), which will not be available by default any more in 0.21. If you need training scores, please set return_train_score=True\n",
      "  warnings.warn(*warn_args, **warn_kwargs)\n",
      "E:\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\deprecation.py:122: FutureWarning: You are accessing a training score ('split3_train_score'), which will not be available by default any more in 0.21. If you need training scores, please set return_train_score=True\n",
      "  warnings.warn(*warn_args, **warn_kwargs)\n",
      "E:\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\deprecation.py:122: FutureWarning: You are accessing a training score ('split4_train_score'), which will not be available by default any more in 0.21. If you need training scores, please set return_train_score=True\n",
      "  warnings.warn(*warn_args, **warn_kwargs)\n",
      "E:\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\deprecation.py:122: FutureWarning: You are accessing a training score ('split5_train_score'), which will not be available by default any more in 0.21. If you need training scores, please set return_train_score=True\n",
      "  warnings.warn(*warn_args, **warn_kwargs)\n",
      "E:\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\deprecation.py:122: FutureWarning: You are accessing a training score ('split6_train_score'), which will not be available by default any more in 0.21. If you need training scores, please set return_train_score=True\n",
      "  warnings.warn(*warn_args, **warn_kwargs)\n",
      "E:\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\deprecation.py:122: FutureWarning: You are accessing a training score ('split7_train_score'), which will not be available by default any more in 0.21. If you need training scores, please set return_train_score=True\n",
      "  warnings.warn(*warn_args, **warn_kwargs)\n",
      "E:\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\deprecation.py:122: FutureWarning: You are accessing a training score ('split8_train_score'), which will not be available by default any more in 0.21. If you need training scores, please set return_train_score=True\n",
      "  warnings.warn(*warn_args, **warn_kwargs)\n",
      "E:\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\deprecation.py:122: FutureWarning: You are accessing a training score ('split9_train_score'), which will not be available by default any more in 0.21. If you need training scores, please set return_train_score=True\n",
      "  warnings.warn(*warn_args, **warn_kwargs)\n",
      "E:\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\deprecation.py:122: FutureWarning: You are accessing a training score ('std_train_score'), which will not be available by default any more in 0.21. If you need training scores, please set return_train_score=True\n",
      "  warnings.warn(*warn_args, **warn_kwargs)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "{'mean_fit_time': array([ 56.14385962,  96.56347032,  45.88044491, 105.83062532]),\n",
       " 'mean_score_time': array([ 5.9133364 ,  9.77280705,  4.78523433, 11.95111091]),\n",
       " 'mean_test_score': array([0.9205, 0.6764, 0.9123, 0.6764]),\n",
       " 'mean_train_score': array([0.98712222, 0.68071111, 1.        , 0.68071111]),\n",
       " 'param_svc__C': masked_array(data=[1, 1, 10, 10],\n",
       "              mask=[False, False, False, False],\n",
       "        fill_value='?',\n",
       "             dtype=object),\n",
       " 'param_svc__kernel': masked_array(data=['linear', 'rbf', 'linear', 'rbf'],\n",
       "              mask=[False, False, False, False],\n",
       "        fill_value='?',\n",
       "             dtype=object),\n",
       " 'params': [{'svc__C': 1, 'svc__kernel': 'linear'},\n",
       "  {'svc__C': 1, 'svc__kernel': 'rbf'},\n",
       "  {'svc__C': 10, 'svc__kernel': 'linear'},\n",
       "  {'svc__C': 10, 'svc__kernel': 'rbf'}],\n",
       " 'rank_test_score': array([1, 3, 2, 3]),\n",
       " 'split0_test_score': array([0.915, 0.686, 0.915, 0.686]),\n",
       " 'split0_train_score': array([0.98722222, 0.67777778, 1.        , 0.67777778]),\n",
       " 'split1_test_score': array([0.923, 0.692, 0.92 , 0.692]),\n",
       " 'split1_train_score': array([0.98711111, 0.68188889, 1.        , 0.68188889]),\n",
       " 'split2_test_score': array([0.929, 0.676, 0.913, 0.676]),\n",
       " 'split2_train_score': array([0.98666667, 0.68822222, 1.        , 0.68822222]),\n",
       " 'split3_test_score': array([0.93 , 0.704, 0.911, 0.704]),\n",
       " 'split3_train_score': array([0.98722222, 0.68411111, 1.        , 0.68411111]),\n",
       " 'split4_test_score': array([0.92 , 0.642, 0.922, 0.642]),\n",
       " 'split4_train_score': array([0.98644444, 0.67933333, 1.        , 0.67933333]),\n",
       " 'split5_test_score': array([0.911, 0.686, 0.912, 0.686]),\n",
       " 'split5_train_score': array([0.98755556, 0.68066667, 1.        , 0.68066667]),\n",
       " 'split6_test_score': array([0.923, 0.678, 0.91 , 0.678]),\n",
       " 'split6_train_score': array([0.98722222, 0.68411111, 1.        , 0.68411111]),\n",
       " 'split7_test_score': array([0.912, 0.675, 0.903, 0.675]),\n",
       " 'split7_train_score': array([0.98766667, 0.68133333, 1.        , 0.68133333]),\n",
       " 'split8_test_score': array([0.922, 0.676, 0.915, 0.676]),\n",
       " 'split8_train_score': array([0.98688889, 0.67977778, 1.        , 0.67977778]),\n",
       " 'split9_test_score': array([0.92 , 0.649, 0.902, 0.649]),\n",
       " 'split9_train_score': array([0.98722222, 0.66988889, 1.        , 0.66988889]),\n",
       " 'std_fit_time': array([ 6.6243823 ,  5.78541154,  2.62869591, 13.45732353]),\n",
       " 'std_score_time': array([0.58119774, 0.44697956, 0.54454301, 3.42821364]),\n",
       " 'std_test_score': array([0.00608687, 0.01768728, 0.00606712, 0.01768728]),\n",
       " 'std_train_score': array([0.00035294, 0.00458047, 0.        , 0.00458047])}"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "grid.cv_results_"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
