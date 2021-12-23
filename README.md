# ML-project

The goal is to predict whether there is some precipitation (rain, snow etc.) on the next day in Pully, getting measurements from different weather stations in Switzerland.

Our repository is composed of:

- src folder: it contains the code which is organized in 3 Pluto notebooks. One for the exploration of the data, one for the linear methods and one for the non-linear ones.
- data folder: it contains the training data that we used to find relevant models, the test data that we used to do our Kaggle submission after selecting the best predicted models on the training data. It contains also an example of a Kaggle submission and our two best submissions : one with Lasso regularized regression (submission_regression.csv) and one with full-connected three-layer network (submission_nn_128.csv).
- the report: a pdf file that summarize our results
- the README file

Our results can be reproduced using Julia version 1.7.0 (2021-11-30).
