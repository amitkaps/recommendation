# Recommendation Systems



## Outline for the workshop

### Session #1: Introduction
- Why build recommendation systems? 
    - Scope and evolution of recsys
    - Prediction and Ranking
    - Relevance, novelty, serendipity & diversity
- Paradigms in recommendations: Content-based, Collaborative filtering, Knowledge-based, Hybrid and Ensembles
- Key concepts in recsys: 
  - Explicit vs. implicit feedback
  - User-Item matrix
  - Domain signals: location, time, context, social
- Why use deep learning for recsys?
    - Primer on deep learning
    - Traditional vs deep learning approaches
    - Examples and use-cases

### Session #2: Content-Based 
- Introduction to the case #1: product recommendation
- Environment setup for hands-on session
- Feature extraction using deep learning: Embeddings for Hetrogenous data
- *Exercise: Recommending items using similarity measures*

### Session #3: Colloborative-Filtering
- Overview of traditional Colloborative-Filtering for recsys
- Primer on deep learning approaches
    - Deep matrix factorisation
    - Auto-Encoders
- *Exercise: Recommending items using Colloborative-Filtering*

### Session #4: Learning-to-Rank
- Why learning-to-rank? Prediction vs Ranking
- Rank-learning approaches: pointwise, pairwise and listwise
- Deep learning approach to combine prediction and ranking
- *Exercise: Recommending items using Learning-to-Rank*


### Session #5: Hybrid Recommender
- Introduction to the case #2: text recommendation
- Combining content-based and collaborative filtering
- Primer on Wide & Deep Learning for Recommender Systems
- *Exercise: Recommending items using Hybrid recommender*

### Session #6: Time and Context
- Adding temporal component: window and decay-based
- Adding context context through group recommendations
- Dynamic and Sequential modelling using Recurrent Neural Networks
- *Exercise: Recommending items using RNN recommender*

### Session #7: Deployment & Monitoring
- Deploying the recommendation system models
- Measuring improvements from recommendation system
- Improving the models based on the feedback from production
- Architecture design for recsys: Offline, Nearline and Online 

### Session #8:  Evaluation, Challenges & Way Forward
- A/B testing for recommendation systems
- Challenges in recsys: 
  - Building explanations
  - Model debugging
  - Scaling-out & up
  - Fairness, accountability and trust
- Bias in recsys: training data, UI → Algorithm → UI, private
- When not to use deep learning for recsys
- Recap and next steps, Learning Resources

## Pre-requisites

The workshop is approximately 50% theory and 50% hands-on.

- Programming knowledge and basics of Python is necessary to follow the hands-on part.
- No machine learning knowledge is assumed.
- Basics of Linear Algebra will be good-to-have.

## Software requirements

We will be doing this on cloud. Laptop with a browser is all you need for the workshop.

If you want to setup a local enviroment for it, the follow the next set of instructions.

- We will be using Python data stack for the workshop. Please install Ananconda for Python 3 (https://www.continuum.io/downloads) BEFORE coming to the workshop. 
- After installing Anaconda, please install the following libraries for recommendation systems.

```
pip install lightfm
pip install scikit-surprise
pip install mlxtend  
```