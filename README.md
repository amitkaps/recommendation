# Recommendation Systems

This is a workshop on using Machine Learning and Deep Learning Techniques to build Recommendation Systesm

- **Theory**: ML & DL Formulation, Prediction vs. Ranking, Similiarity, Biased vs. Unbiased
- **Paradigms**: Content-based, Collaborative filtering, Knowledge-based, Hybrid and Ensembles
- **Data**: Tabular, Images, Text (Sequences)
- **Models**: (Deep) Matrix Factorisation, Auto-Encoders, Wide & Deep, Rank-Learning, Sequence Modelling
- **Methods**: Explicit vs. implicit feedback, User-Item matrix, Embeddings, Convolution, Recurrent, Domain Signals: location, time, context, social,
- **Process**: Setup, Encode & Embed, Design, Train & Select, Serve & Scale, Measure, Test & Improve
- **Tools**: python-data-stack: numpy, pandas, scikit-learn, keras, spacy, implicit, lightfm

## Notes & Slides

- Basics: [Deep Learning](Notes/Deep-Learning-Basics.pdf)
- AI Conference 2019: [WhiteBoard Notes](Notes/AIConf-CA-2019-Notes.pdf) | [In-Class Notebooks](https://notes.pipal.in/2019/AIConf-CA/) 


## Notebooks

- [Movies - Movielens](MovieLens)
    - [01-Acquire](MovieLens/01-Acquire.ipynb)
    - [02-Augment](MovieLens/02-Augment.ipynb)
    - [03-Refine](MovieLens/03-Refine.ipynb)
    - [04-Transform](MovieLens/04-Evaluation.ipynb)
    - [05-Evaluation](MovieLens/05-Evaluation.ipynb)
    - [06-Model-Baseline](Movielens/06-Model-Baseline.ipynb)
    - [07-Feature-extractor](Movielens/07-Feature-Extractor.ipynb)
    - [08-Model-Matrix-Factorization](Movielens/08-Model-MF-Linear.ipynb)
    - [09-Model-Matrix-Factorization-with-Bias](Movielens/09-MF-Linear-Bias.ipynb)
    - [10-Model-MF-NNMF](Movielens/10-Model-MF-NNMF.ipynb)
    - [11-Model-Deep-Matrix-Factorization](Movielens/11-Model-Deep-Factorisation.ipynb)
    - [12-Model-Neural-Collaborative-Filtering](Movielens/12-Neural-CF.ipynb)
    - [13-Model-Implicit-Matrix-Factorization](Movielens/13-Implicit-CF.ipynb)
    - [14-Features-Image](Movielens/14-Image-Features.ipynb)
    - [15-Features-NLP](Movielens/15-Doc-Embedding.ipynb)

- [Ecommerce - YooChoose](YooChoose)
    - [01-Data-Preparation](YooChoose/01-Data-Preparation.ipynb)     
    - [02-Models](YooChoose/02-Models.ipynb)
    
- [News - Hackernews](HackerNews)
- [Product - Groceries](Groceries)
    

## Python Libraries

Deep Recommender Libraries
- [Tensorrec](https://github.com/jfkirk/tensorrec) - Built on Tensorflow
- [Spotlight](https://github.com/maciejkula/spotlight) - Built on PyTorch
- [TFranking](https://github.com/tensorflow/ranking) - Built on TensorFlow (Learning to Rank)

Matrix Factorisation Based Libraries
- [Implicit](https://github.com/benfred/implicit) - Implicit Matrix Factorisation
- [QMF](https://github.com/quora/qmf) - Implicit Matrix Factorisation
- [Lightfm](https://github.com/lyst/lightfm) - For Hybrid Recommedations
- [Surprise](http://surpriselib.com/) - Scikit-learn type api for traditional alogrithms

Similarity Search Libraries
- [Annoy](https://github.com/spotify/annoy) - Approximate Nearest Neighbour
- [NMSLib](https://github.com/nmslib/nmslib) - kNN methods
- [FAISS](https://github.com/facebookresearch/faiss) - Similarity search and clustering

Content-based Libraries
- [Cornac](https://github.com/PreferredAI/cornac) - Leverage Auxiliary Data (Images, Text, Social Networks)

## Learning Resources

### Reference Slides
- [Deep Learning in RecSys by Balázs Hidasi](http://pro.unibz.it/projects/schoolrecsys17/DeepLearning.pdf)
- [Lessons from Industry RecSys by Xavier Amatriain](http://pro.unibz.it/projects/schoolrecsys17/RecsysSummerSchool-XavierAmatriain.pdf)
- [Architecting Recommendation Systems by James Kirk](https://www.slideshare.net/JamesKirk58/boston-ml-architecting-recommender-systems)
- [Recommendation Systems Overview by Raimon and Basilico](http://nn4ir.com/ecir2018/slides/08_RecommenderSystems.pdf)

### Benchmarks
- [MovieLens Benchmarks for Traditional Setup](https://github.com/microsoft/recommenders/blob/master/benchmarks/movielens.ipynb)
- [Microsoft Tutorial on Recommendation System at KDD 2019](https://github.com/microsoft/recommenders)


### Algorithms & Approaches
- [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf)
- [Bayesian Personalised Ranking for Implicit Data](https://arxiv.org/pdf/1205.2618)
- [Logistic Matrix Factorisation](https://web.stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf)
- [Neural Network Matrix Factorisation](https://arxiv.org/abs/1511.06443)
- [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)
- [Variational Autoencoders for Collaborative Filtering](https://arxiv.org/abs/1802.05814)

### Evaluations
- [Evaluating Recommendation Systems](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/EvaluationMetrics.TR_.pdf)

