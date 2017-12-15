# Recommendation Systems

> Sure. I do marathons…. on Netflix

In the digital world, recommendation systems play a significant role - both for the users and for the company/platform/sellers.

For the users, a new world of options are thrown up - that were hitherto tough to find. For companies, it helps drive up user engagement and satisfaction, directly impacting their bottom line.

If you’ve shopped on e-commerce platforms like Amazon or Flipkart, you would’ve seen options like:

“People who viewed this product also viewed…”
“Products similar to this one…”

These are the results from recommendation systems. Netflix threw up a major data science challenge last decade: a million dollars to anyone who can improve their recommendation system by 10%. A recent estimate pegs the value of Netflix’s recommendation system to be worth $ 1 Billion.

In this full-day workshop, we will walk you through the various types of recommendation system. By the end of the workshop, you will have enough knowledge to build one for your problem.

## Outline for the workshop

### Session 1

- What are recommendation systems?
  - Definition
  - Some examples
  - Discussions around how to define the data for recommendation systems
- Common types of recommendation systems
  - Explicit Feedback
  - Implicit Feedback
  - item-based recommendation system
  - user-based recommendation system
  - Content-based recommendation system

### Session 2
- Collaborative Filtering (hands-on)
- Matrix Factorization (hands-on)

### Session 3

- Content-based Recommendation System (hands-on)
- Deep Learning in Recommendation System (theory only)

### Session 4
- Recommendation System in production
- Deploying the recommendation system models
- Measuring improvements from recommendation system
- Improving the models based on the feedback from production
- Closing thoughts and next steps

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