# UncertainPooling
A package to analyse and solve pooling problems when subjected to uncertain parameter values. There are many ways in which uncertainty can affect the pooling problem and the quality of the solution obtained using various methods for solving the problem. The focus of the four problems/models already implemented in this project is on uncertainty in the quality of the feed streams into the network, and how it affects the satisfaction of the quality constraints on the product streams. More details will hopefully soon be available in a published research article.

<img src="https://user-images.githubusercontent.com/46780228/219687176-1451b462-2a8f-46b1-aa7a-7e62e79e744a.jpg" width="500" >

Hopefully, the examples are sufficient for others to create models for their specific pooling system. Important things to note is that not only does the network structure need to be defined, but also a consequence of products violating the quality constraints also needs to be encoded. The default in all the models is that the product cannot be sold if it violates the quality constraints, but no cost is incurred in disposing of it. Also, a model should also be constructed in the UncertainModel class to evaluate the performance of a solution obtained using a proxy model when the true nature of the uncertainty is respected.

Adjustable parameters:
* All models:
  * Standard deviation - as an input to any uncertain model (including proxymodels. For the scenario proxy this will have a direct affect on the standard scenario generation methods and for the stochastic programming approach it will directly be incorporated into the model to be solved. For the robust proxy it won't have a direct effect on the solution provided by the proxy, but will, in general, affect the quality of the solution when evaluated with the ss_evaluator() method.
* Scenario approach:
  * Number of scenarios - As the number of scenarios increases, the discrete probability distribution more accurately captures what would originally be a continuous distribution. It also icnreases the computaitonal requirements of solving the method. The default number is three scenarios per uncertain parameter
  * Scenario generation method - passed as an optional input to the class. The included options are 'Lee' which follows the method outlined in the 2010 paper by Li, Armagan, Tomasgard and Barton. The other is a "basic" approach that takes a value for the for the "outer" scenarios and calculates their deviation from the mean such that the standard deviation of the discrete distribution is equal to that of the continuous distribution. The basic approach is only valid for the case when the number of scenarios is 3.
* Robust approach:
  * Uncertainty set radius - The "radius" used for the uncertainty set. The model is currently only set up for an uncertainty set based on the infinity norm of the vector of uncertain feed qualities. This radius thus translates into the allowed distance each feed quality can stray away from its mean (and since we are dealing with the infinity norm, they can all stray independently of each other).

## Future work
* It would be nice to more dynamically be able to specify problem models, and for instance not have to define it once in a (each of the) proxy model(s) and also in the ss_evaluator(), but rather define one problem once, and automatically adjust the optimisation formulations to the different model types
* The robust uncertainty model is a bit restrictive and it would be nice to look at uncertainty sets that are, for instance, induced by a 2 norm or a 1 norm. Also uncertainty budgets could be interesting.
* The stochastic programming approach would greatly benefit from some refined convex relaxation schemes and also bounds tightening, especially for the standard deviations of compositions as other variables get restricted in the spatial branch and bound procedure

## Dependencies
* NumPy - For various computational structures (arrays) and linear algebra operations.
* SciPy - Primarily for computing the error function and some norms.
* GurobiPy - For solving mixed-integer bilinear programming problems (particularly for the scenario proxy and robust proxy). Also requires an active license.
* Pandas - First and foremost for loading and saving the result tables as .csv files
* Time - For timing the methods. This forms part of the results table
