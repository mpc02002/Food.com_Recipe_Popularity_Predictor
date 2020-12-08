# Food.com_Recipe_Popularity_Predictor

<h3> Overview </h3>
<p>
The goal of this project is to predict which online recipes (from Food.com) will become popular over time.  Recipe popularity prediction seems to be a complex and difficult problem in machine-learning, which continues to inspire research for state-of-the-art methods.  I was particularly influenced in the design of this project by the article <a href="https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-018-0149-5#Sec10">On the predictability of the popularity of online recipes</a> by Trattner, Moesslang, and Elsweiler appearing in EPJ Data Science (2018). 
  
Concretely, the mathematical model designed in this project takes three initial parameters, which may be set by the user:

T = a popularity threshold, measured in number of user reviews of the recipe<br>
t1 = an initial date, representing the present moment in time<br>
t2 = a later date, at which time we will make popularity predictions<br>

Our model reads recipe data which is available prior to t1 only, and then <b>predicts which recipes will receive at least T many user reviews by time t2</b>.  The model also gives probabilistic estimates of this target outcome, allowing sorting of recipes from most likely to become popular, down to least likely.

Successfully predicting the future popularity of recipes is especially useful for promotion and marketing, because placing popular content (or content with strong potential for popularity) on entry pages can drive user engagement and increase site traffic.  The idea here is that our model can tell Food.com proprietors which recipes they should devote resources to promoting.  In addition, post-hoc feature analysis may help recipe authors and publishers improve the structure of their content to increase visibility and user engagement.

All scripts are written in Python 3.  This repository was last updated 12/8/20.
  
<h3> The Datasets </h3>
<p>
We use datasets containing over 180K recipes from Food.com (2001-2018), together with over 700K user ratings, obtained from <a href="https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions">Kaggle.com</a>.   The datasets presently live in IBM Cloud Storage, which the Jupyter Notebooks in this repository can access.
  
<h3> Inherent Challenges in Recipe Popularity Prediction </h3>
<p>
<h4> 1.  Data Imbalance</h4>
<p>
The problem is a "needle in a haystack" search, because for any threshold sufficiently high to be interesting (say T >= 25) the percentage of recipes on Food.com which will ever achieve T many user interactions is very small.  For example, the proportion of Food.com recipes which cross the threshold of T = 50 user ratings during the entirety of 2017 is about 1 in 2000.  It is difficult to train a machine learning algorithm to be sensitive to such a small target, because standard algorithms tend to favor majority data (in this case, the 99.95% of recipes that do not achieve many interactions).

To tackle the imbalance problem, we make heavy use of a Random Undersampling (RUS) technique; see below for more details.

<h4> 2. Data Leakage </h4>
<p>
To train and validate a supervised machine learning algorithm, we need to be able to "look into the future" to determine which recipes from the past (say, 2016) were able to achieve popularity later (say, 2018).  Thus when working with the available data, there is a high risk that we may accidentally build future information into the predictive algorithm, training an overly optimistic model that won't generalize to real-life forecasting.

To prevent data leakage, we engineer two distinct feature sets (features_train.csv and features_test.csv) as follows.  We compute the time interval D = t2 - t1 from the given parameters, and we assign previous times m1 and m2 so that m1 < m2 < t1 < t2 and m2 - m1 = D.  We also choose m1 so it's on the same day of the year as t1 (to capture a little seasonality).  Then the training set features_train.csv is built only from data available at time m1, with target prediction time m2.  We do a test-train split on the training set to validate our model-building.  Afterward, the model is evaluated purely on the test set, which is built from data available at time t1, and has target prediction time t2.
  
<h3> The Model </h3>
<p>
We build an ensemble model using a family of weak Naive Bayes estimators, together with a logistic regression classifier.  The modeling is performed in two essentially discrete steps as follows.
  
<h4> I. Random Undersampling (RUS) Boosted Classifier with Naive Bayes Estimators</h4>

For each recipe, the text of its reviews is collected and aggregated to train a gradient-boosted model composed of 50 weak estimators (Naive Bayes models).  I've selected Naive Bayes for the weak learners because it is computationally very fast and performs well on "bag of words" text data.  At each step of the boosting, the new weak learner is trained by undersampling the majority class (about 4 unpopular recipes for every popular one).  This makes the model fast to train, and much more sensitive to the minority class.  The model outputs a natural estimate of the probability that a given recipe will cross the popularity threshold.

In validation experiments, this RUS-NB classifier performed with extremely high recall (generally > 0.98) even at very low probability decision thresholds, so it succesfully finds all popular recipes.  But precision is far too low for our purposes (generally < 0.005)-- the model makes thousands of false positive guesses.  So it needs tailoring to be pragmatically useful.

<h4> II. Logistic Regression</h4>

The last classifier in the ensemble model is a simple logistic regression trained on 13 features.  One of the 13 features is the probability estimate returned by the RUS-NB classifier, so the LR model is working together directly with the preceding model.  Two more of the features are the "Innovation Jaccard" and "Average Innovation Jaccard," which are features computed by network analysis on the recipe/ingredient network.

These latter features were made computationally feasible by significantly trimming the size of the full network, using the RUS-NB classifier's predictions.  It's still a bit slow.  In practice, engineering these two features for the training set has taken about ~1 hour of runtime using 8 cores in IBM Watson Studio.

<i>Remark:  This project was completed as a capstone for IBM's Advanced Data Science specialization on Coursera.org, and in partial fulfillment of the requirements, we also trained a feed-forward neural network as a possible second component in the model instead of the logistic regression.  The neural network was outperformed by the logistic regression in all metrics, though, so I dropped it in favor of the simpler model.  I think it's a sign of good feature engineering!</i>

<h3> The Features </h3>
<p>
<h4>1-7.  Current age of recipe; current total number of ratings; current mean rating; number of steps; number of ingredients; cook time; number of calories.</h4>
Self-explanatory.

<h4>8.  Rating pace.</h4>
(Current total number of ratings) / (current age of recipe)

<h4>9.  RUS-NB classifier probability prediction.</h4>
The probability that the recipe will become popular, according to the RUS-NB classifier.  It's generally over-optimistic, but informative.<br>

<i>Remark: Features 10--13 below (or close variants) were identified as strongly predictive of recipe popularity in the <a href="https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-018-0149-5#Sec10">paper</a> of  Trattner, Moesslang, and Elsweiler, which encouraged me to build them into the model.</i>

<h4>10.  Recipe instructions readability modified LIX score</h4>
Intended to measure the overall readability of the recipe instructions on a scale from 0-100.  It's computed by the formula:
  
100 * (number of words with > 6 characters) / (number of words) + (number of words) / (number of steps in the recipe)

<h4>11.  Recipe Innovation IDF score</h4>
Each ingredient is assigned a rarity score (number of recipes containing the ingredient) / (total number of recipes).  Then each recipe is assigned the average rarity of its ingredients.  Recipes with a score close to 0 are novel in the sense of having uncommon ingredients, while recipes close to 1 contain only common ingredients.
  
<h4>12 and 13.  Recipe Innovation Jaccard and Average Innovation Jaccard</h4>
We form the network consisting of all recipes and all ingredients, with an edge connecting an ingredient to a recipe if and only if the former is contained in the latter.  This is a pretty sizable bipartite graph (~250 thousand nodes, ~2 million edges) and doing computations on it can be taxing.  In order to trim the size of the computation, we examine only two relatively small subsets of the nodes:

popular_recipes = {all recipes which have received at least 50 ratings}<br>
candidate_recipes = {all recipes being considered by the model, for which the RUS NB classifier predicts a target probability of > 0.49}<br>

For interesting thresholds T, the set popular_recipes is always small (~2000 recipes). The set candidate_recipes varies in size depending on choices of initial parameters in the model, but in exploratory data analysis it often came out to around ~50000 recipes.

Given a recipe r contained in candidate_recipes, its Jaccard index with a recipe p in popular_recipes is given by

J(r, p) = (# of ingredients contained in both r and p) / (# of ingredients contained in either r or p)

The recipes r and p will have a high Jaccard index if they have mostly the same ingredients; and a low Jaccard index if they have mostly different ingredients.  Then features 12 and 13 are computed as follows:

jaccard_innovation(r) = 1 - max{ J(r, p) : p in popular_recipes}

jaccard_innovation_avg(r) = 1 - mean{ J(r, p) : p in popular_recipes}

A recipe with high Jaccard innovation scores (close to 1) is innovative in the sense that its ingredients differ significantly from any pre-existing popular recipe.

For recipes in the feature sets which are NOT considered candidate_recipes, we assign them scores of 0 for computational simplicity. For all intents and purposes, the RUS NB classifier has already eliminated them from our overall model.
  
<h3> Evaluation Metrics </h3>
<p>
We are trying to predict a rare event, so of course we care about precision and recall.  Our primary metric is Average Precision, because we want a model that makes good guesses especially at high thresholds, but is conservative and doesn't overguess too much.  Because Food.com's marketing team can't do much with a recommendation list that is thousands of recipes long.

Since the logistic regression gives a natural probability estimate, we compute its optimal decision threshold as the one which maximizes the harmonic mean of its precision and recall on the training set.  Then, we take the precision and recall scores at this threshold on the test set.

Area under the ROC curve is sometimes taken as a proxy for a balance between precision and recall, and so we include this metric in our testing.  However, a low False Positive Rate is not necessarily meaningful for this highly imbalanced dataset, so we consider ROC_AUC as a secondary metric only.
  
<h3> Results </h3>
<p>
In our first experiment's run, we set the initial parameters at 

T = 50<br>
t1 = December 31, 2016<br>
t2 = March 30, 2018<br>

The test set contains 228553 recipes.  The number of recipes which will achieve the popularity threshold by the target time is 123.  So our model attempts to predict an outcome achieved by 0.054% of recipes.

When evaluated on the test set, the final model has average precision score 0.355, which I think is remarkably good given the difficulty of the prediction problem.  The model also achieved an ROC_AUC of 0.999.

Of the 228553 recipes, with the model makes 67 positive predictions.  Of these, 34 guesses are correct, for a recall score of 0.276 and a precision score of 0.507.

The model also returns probability estimates of crossing the popularity threshold for each recipe, so recipes can be ranked in descending order of likelihood.  Thus, Food.com may take the top 30 recipes returned by the popularity predictor and increase promotion for them, thus driving greater traffic and user engagement.  If this forecasting were conducted contemporaneously at the end of 2016, then more than half of these 30 would have crossed the threshold even without additional intervention, justifying the company's efforts.

<h3> Post-Hoc Feature Analysis </h3>
The logistic regressions step assigns feature weights to the (normalized) training data, which may be interpreted as a measure of feature importance.  Here is the full list of feature weights, with parameters set as above.

total_ratings_curr	9.045011<br>
age_in_days	-1.189304<br>
rating_pace	0.284021<br>
RUSNB_proba_pred	0.577928<br>
mean_rating_curr	-0.989466<br>
minutes	-0.000419<br>
n_steps	0.603184<br>
n_ingredients	0.343740<br>
calories	0.002737<br>
LIX_score	-0.351900<br>
innovation_IDF	0.924896<br>
innovation_jaccard	0.194669<br>
innovation_jaccard_avg	0.917794<br>

Unsurprisingly, total_ratings_curr and age_in_days are the most predictive features.  It is interesting to note that innovation_IDF and innovation_jaccard_avg also have relatively high weights, providing evidence in favor of these as important metrics for recipe popularity potential.
  
<h3> The Notebooks </h3>
<p>
<h4> a. data_exp.01 and data_exp.02</h4>  Just scratch work and exploration.

<h4> b. etl.01</h4>  Just cleaning and uploading data to cloud storage; not much to see here.

<h4> c. feature_eng.01</h4>  The notebook where features_train.csv and features_test.csv are engineered.

<h4> d. model_def_and_train.01</h4>  Definition and validation of logistic regression, decision tree, gradient-boosted, and feed-forward neural network classifiers.  Logistic regression performs best and is exported at the end.

<h4> e. model_eval.01</h4>  The model is tested on features_test.csv and metrics are reported.
