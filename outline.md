# Workshop Outline

### Introduction
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

### Colloborative-Filtering
- Introduction to the case #1
- Environment setup for hands-on session
- Overview of traditional Colloborative-Filtering for recsys
- Primer on deep learning approaches
- Deep matrix factorisation
- *Exercise: Recommending items using Colloborative-Filtering*

### Content-Based 
- Feature extraction using deep learning: Embeddings for Hetrogenous data
- *Exercise: Recommending items using similarity measures*

### Learning-to-Rank
- Why learning-to-rank? Prediction vs Ranking
- Rank-learning approaches: pointwise, pairwise and listwise
- Deep learning approach to combine prediction and ranking
- *Exercise: Recommending items using Learning-to-Rank*

### Hybrid Recommender
- Combining content-based and collaborative filtering
- Primer on Wide & Deep Learning for Recommender Systems
- *Exercise: Recommending items using Hybrid recommender*

### Time and Context
- Adding temporal component: window and decay-based
- Dynamic and Sequential modelling using RNNs/1D CNNs
- *Exercise: Recommending items using RNN recommender*

### Deployment & Monitoring
- Deploying the recommendation system models
- Measuring improvements from recommendation system
- Improving the models based on the feedback from production
- Architecture design for recsys: Offline, Nearline and Online 

### Evaluation, Challenges & Way Forward
- A/B testing for recommendation systems
- Challenges in recsys: 
  - Building explanations
  - Model debugging
  - Scaling-out & up
  - Fairness, accountability and trust
- Bias in recsys: training data, UI → Algorithm → UI, private
- When not to use deep learning for recsys
- Recap and next steps, Learning Resources
