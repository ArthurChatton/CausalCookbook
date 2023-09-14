Causal cookbook recipes
================
Arthur Chatton and Julia Rohrer
2023-09-13

-   <a href="#goals" id="toc-goals">Goals</a>
-   <a href="#implementing-the-recipes"
    id="toc-implementing-the-recipes">Implementing the Recipes</a>
    -   <a href="#generating-synthetic-data"
        id="toc-generating-synthetic-data">Generating synthetic data</a>
    -   <a href="#positivity" id="toc-positivity">Positivity</a>
    -   <a href="#inverse-probability-weighting"
        id="toc-inverse-probability-weighting">Inverse-probability-weighting</a>
    -   <a href="#g-computation" id="toc-g-computation">G-computation</a>
    -   <a href="#doubly-robust-standardisation"
        id="toc-doubly-robust-standardisation">Doubly-robust standardisation</a>
    -   <a href="#bootstrapping-to-compute-the-variance"
        id="toc-bootstrapping-to-compute-the-variance">Bootstrapping to compute
        the variance</a>
-   <a
    href="#illustration-of-the-impact-of-the-weighting-schemes-on-the-pseudo-sample-characteristics"
    id="toc-illustration-of-the-impact-of-the-weighting-schemes-on-the-pseudo-sample-characteristics">Illustration
    of the impact of the weighting schemes on the pseudo-sample
    characteristics</a>
-   <a href="#references" id="toc-references">References</a>

# Goals

This document contains additional materials for the Causal Cookbook
(Chatton & Rohrer, 2023). In the first sections, we illustrate how to
implement the recipes of the cookbook in R, namely:

1.  Inverse-probability weighting (also known as propensity-score
    weighting)

2.  g-computation

3.  Doubly-robust standardisation

Below, we also provide an illustration of how the different weighting
schemes mentioned in the cookbook affect the resulting pseudo-sample.

We assume only basic knowledge of R.

# Implementing the Recipes

## Generating synthetic data

First, we need some data to work on. We generate synthetic data here
because this allows us to know the true causal effect, which means that
we can evaluate how the different recipes perform.

``` r
datasim <- function(n) { 
  # This small function simulates a dataset with n rows
  # containing covariats, and action, an outcome and
  # the underlying potential outcomes
  
  # Two binary covariates
  x1 <- rbinom(n, size = 1, prob = 0.5) 
  x2 <- rbinom(n, size = 1, prob = 0.65)
  
  # Two continuous, normally distributed covariates
  x3 <- rnorm(n, 0, 1)
  x4 <- rnorm(n, 0, 1)
  
  # The action (independent variable, treatment, exposure...)
  # is a function of x2, x3, x4 and 
  # the product of (ie, an interaction of) x2 and x4
  A <- rbinom(n, size = 1, prob = plogis(-1.6 + 2.5*x2 + 2.2*x3 + 0.6*x4 + 0.4*x2*x4)) 
  
  # Simulate the two potential outcomes
  # as functions of x1, X2, X4 and the product of x2 and x4
  
  # Potential outcome if the action is 1
  # note that here, we add 1
  Y.1 <- rbinom(n, size = 1, prob = plogis(-0.7 + 1 - 0.15*x1 + 0.45*x2 + 0.20*x4 + 0.4*x2*x4)) 
  # Potential outcome if the action is 0
  # note that here, we do not add 1 so that there
  # is a different the two potential outcomes (ie, an effect of A)
  Y.0 <- rbinom(n, size = 1, prob = plogis(-0.7 + 0 - 0.15*x1 + 0.45*x2 + 0.20*x4 + 0.4*x2*x4)) 
  
  # Observed outcome 
  # is the potential outcomes (Y.1 or Y.0)
  # corresponding to action the individual experienced (A)
  Y <- Y.1*A + Y.0*(1 - A) 

  # Return a data.frame as the output of this function
  data.frame(x1, x2, x3, x4, A, Y, Y.1, Y.0) 
} 
```

We have 4 covariates (`x1` to `x4`), but only `x2` and `x4` are
confounders. Thus, the minimal adjustment set C for achieving
(conditional) exchangeability is (`x2`, `x4`). These four predictors are
used for simulating the action `A` (say the Alcoholic Anonymous
attendance: 1 if attendee and 0 otherwise) and the outcome `Y` (say
abstinence at the 1-year follow-up: 1 if abstinent and 0 otherwise).

We also see the variables `Y.1` and `Y.0,` which are the two *potential
outcomes* observed in a hypothetical world in which all individuals are
attenders (A=1) or non-attenders (A=0), respectively. In practice, the
potential outcomes are unknown, but using synthetic data we know them
which allows us to determine the true causal effect.

Let’s actually generate the data.

``` r
set.seed(120110) # for reproducibility
ObsData <- datasim(n = 500000) # really large sample
TRUE_EY.1 <- mean(ObsData$Y.1); TRUE_EY.1  # mean outcome under A = 1
```

    ## [1] 0.619346

``` r
TRUE_EY.0 <- mean(ObsData$Y.0); TRUE_EY.0  # mean outcome under A = 0
```

    ## [1] 0.388012

``` r
TRUE_ATE <- TRUE_EY.1 - TRUE_EY.0; TRUE_ATE # true average treatment effect is the difference
```

    ## [1] 0.231334

``` r
TRUE_MOR <- (TRUE_EY.1*(1 - TRUE_EY.0))/((1 - TRUE_EY.1)*TRUE_EY.0); TRUE_MOR # true marginal OR
```

    ## [1] 2.56626

The true ATE (average treatment effect on the entire population, i.e.,
risk difference over all individuals) is around 0.231, while the true
marginal odds ratio is around 2.57. Both are valid causal effects, but
their scale differs.

In other words, in our simulated data, attending AA increases the
chances of abstinence by 23.1 percentage points (from 38.8% to 61.9%)

## Positivity

Before modelling, we must check the positivity assumption (*i.e.*, all
individuals must have a non-extreme probability to experience the each
level of `A`) regardless of the method used afterwards. For this, we use
the PoRT algorithm (Danelian *et al.*, 2023).

``` r
library(devtools)
source_url('https://raw.githubusercontent.com/ArthurChatton/PoRT/619c2a65ae81f7092c3edf5d3bc8c44d470e9b66/port.r') #upload the port() function.
ObsData$x4_round <- round(ObsData$x4, 0) # to reduce computational time (huge sample size), in practice use meaningful cutoff.
port(group="A", cov.quanti="x4_round", cov.quali="x2", data=ObsData)
```

    ## [1] "No problematic subgroup was identified."

So, positivity seems respected here. We can go away and start
~~cooking~~ modelling. Else, we face a choice:

-   Changing the target population by excluding the problematic
    subgroup(s) identified by PoRT

-   Targeting an estimand for which the identified violation(s) are not
    meaningful (*e.g.*, ATT only requires the attendees have a
    non-extreme probability to be non-attendee)

-   Using an approach able to extrapolate over the problematic
    subgroup(s) such as the g-computation (but a correct extrapolation
    is not guaranteed).

## Inverse-probability-weighting

Next, we illustrate the recipe provided in Box 2.

In the first step, we need to fit the nuisance model g(C), *i.e.*, the
propensity score. (Here, we do know how the data were generated, so this
step is easy—if we were using actual data, we would first need to decide
which covariates to include and how to precisely model the action)

``` r
# Fit a logistic regression model predicting A from the relevant confounders
# And immediately predict A for all observations (fitted.values)
g <- glm(A ~ x2 + x4 + x2*x4, data = ObsData, family=binomial)$fitted.values
```

The coefficients of this model don’t matter, so we don’t even look them
(see Weistreich & Greenland, 2013, for an explanation).

In the second step, we compute the weights to recover the causal effect
of interest. Here, we use the unstabilised ATE weights.

``` r
# Assign the weights depending on the action group
# See also Table 2 in the Causal Cookbook
omega <- ifelse(ObsData$A==1, 1/g, 1/(1-g))
summary(omega)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##   1.037   1.385   1.625   1.999   2.313  17.559

We need to check if this model balance the two groups in the resulting
pseudo-sample (*i.e.*, weighted sample). We can use the `tableone`
package for this (Yoshida & Bartel, 2022).

``` r
## Load the packages
library(tableone); library(survey)

## Weighted data (pseudo-sample)
pseudo <- survey::svydesign(ids = ~ 1, data = ObsData, weights = ~ omega)

## Construct the table (This is quite slow)
tabWeighted <- tableone::svyCreateTableOne(vars = c("x2", "x4"), strata = "A", data = pseudo, test = FALSE, addOverall = TRUE)

## Show table with SMD
print(tabWeighted, smd = TRUE)
```

    ##                 Stratified by A
    ##                  Overall          0                1                SMD   
    ##   n              999696.74        499708.33        499988.41              
    ##   x2 (mean (SD))      0.65 (0.48)      0.65 (0.48)      0.65 (0.48)  0.001
    ##   x4 (mean (SD))      0.00 (1.00)      0.00 (1.00)      0.00 (1.00)  0.003

A correct balance of the confounders is achieved when the standardised
mean difference (`SMD` in the table) between the two action groups in
the pseudo sample is lower than 10% (Ali *et al.*, 2015). Here, the
weights seems to have balanced the groups. However, if we have
unmeasured or omitted confounders some imbalance can remains unnoticed.
Note that the pseudo-sample size is twice the actual sample size. The
stabilisation of the weights correct this phenomenon.

In the third step, we fit the marginal structural model.

``` r
# Logistic regression model returns the marginal OR
msm_OR <- glm(Y~A, weights = omega, data=ObsData, family=binomial)
# Linear model returns the ATE (risk difference)
msm_RD <- lm(Y~A, weights = omega, data=ObsData)
```

The choice of the marginal structural model depends on the causal effect
of interest. A logistic model gives us an estimate of the marginal OR,
while a linear model give is an estimate of the risk difference.

Fourth step, the resulting estimates:

``` r
MOR_IPW <- exp(msm_OR$coef[2])
RD_IPW <- msm_RD$coef[2]
c(MOR_IPW,RD_IPW) |> round(digits=3)
```

    ##     A     A 
    ## 2.568 0.231

We obtain a marginal OR of 2.57 and an ATE of 0.231—these are indeed the
true values that we recovered above by contrasting the potential
outcomes.

For the variance of these estimates, we can use either a robust-SE
matrix as below or bootstrapping (see end of this document).

``` r
library(sandwich)
library(lmtest)
coeftest(msm_RD, vcov = sandwich)
```

    ## 
    ## t test of coefficients:
    ## 
    ##              Estimate Std. Error t value  Pr(>|t|)    
    ## (Intercept) 0.3888322  0.0011077  351.04 < 2.2e-16 ***
    ## A           0.2314945  0.0015505  149.30 < 2.2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
coeftest(msm_OR, vcov = sandwich)
```

    ## 
    ## z test of coefficients:
    ## 
    ##               Estimate Std. Error z value  Pr(>|z|)    
    ## (Intercept) -0.4522236  0.0046611 -97.021 < 2.2e-16 ***
    ## A            0.9431586  0.0065535 143.918 < 2.2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

## G-computation

Next, we illustrate the recipe from Box 3.

First step, we need to fit the nuisance model Q(A,C):

``` r
# Fit a logistic regression model predicting Y from the relevant confounders
Q <- glm(Y ~ A + x2 + x4 + x2*x4, data = ObsData, family=binomial)
```

Again, the coefficients of this model don’t matter. In contrast to
propensity-score-based methods, g-computation doesn’t need a “balance”
assumption.

Does the Q-model look familiar? Indeed, it’s a classical model used to
control for confounding. But when we look at the estimate of the
marginal OR

``` r
exp(Q$coef[2])
```

    ##        A 
    ## 2.736306

This estimate is not the causal effect of interest which we already
learned is 2.57.

This is because the marginal OR is non-collapsible; we need a specific
causal method to target it properly. (What we get here is not the
marginal OR, but an unbiased estimate of the *conditional OR*. This is
the OR when all covariates are set to 0; its value with thus depend on
how we coded the covariates (e.g., whether we centered them, which
reference category was used for dummy variables) and which terms we
included (linear, quadratic, cubic, interactions…). It can be viewed as
a kind of \`\`average” of causal effects in all possible subgroups
defined by the adjustment set. In contrast, the marginal OR depends on
the actual distribution of covariates in the data.)

Second step, let’s create hypothetical worlds!

``` r
# Copy the "actual" (simulated) data twice
A1Data <- A0Data <- ObsData
# In one world A equals 1 for everyone, in the other one it equals 0 for everyone
# The rest of the data stays as is (for now)
A1Data$A <- 1; A0Data$A <- 0
head(ObsData); head(A1Data); head(A0Data)
```

    ##   x1 x2         x3         x4 A Y Y.1 Y.0
    ## 1  0  1 -0.2909924  0.8374205 1 1   1   0
    ## 2  1  1 -0.1598913 -1.6984448 0 0   1   0
    ## 3  1  0 -0.5454640  1.6010621 0 0   1   0
    ## 4  0  1 -0.4733918 -0.8286665 0 0   1   0
    ## 5  1  1 -0.8476024  1.1711469 1 1   1   1
    ## 6  1  0 -1.2864879  1.5281848 0 0   1   0

    ##   x1 x2         x3         x4 A Y Y.1 Y.0
    ## 1  0  1 -0.2909924  0.8374205 1 1   1   0
    ## 2  1  1 -0.1598913 -1.6984448 1 0   1   0
    ## 3  1  0 -0.5454640  1.6010621 1 0   1   0
    ## 4  0  1 -0.4733918 -0.8286665 1 0   1   0
    ## 5  1  1 -0.8476024  1.1711469 1 1   1   1
    ## 6  1  0 -1.2864879  1.5281848 1 0   1   0

    ##   x1 x2         x3         x4 A Y Y.1 Y.0
    ## 1  0  1 -0.2909924  0.8374205 0 1   1   0
    ## 2  1  1 -0.1598913 -1.6984448 0 0   1   0
    ## 3  1  0 -0.5454640  1.6010621 0 0   1   0
    ## 4  0  1 -0.4733918 -0.8286665 0 0   1   0
    ## 5  1  1 -0.8476024  1.1711469 0 1   1   1
    ## 6  1  0 -1.2864879  1.5281848 0 0   1   0

Our hypothetical worlds are identical except for the action status.
`A1Data`represents a hypothetical world in which all individuals are
attendees, while `A0Data` represents the opposite worlds in which all
individuals are non-attendees.

In the third step, we make counterfactual predictions. For this, we use
the model `Q` as a prediction model and estimate the outcome’s
probability in each hypothetical world:

``` r
# Predict Y if everybody attends
Y_A1 <- predict(Q, A1Data, type="response") 
# Predict Y if nobody attends
Y_A0 <- predict(Q, A0Data, type="response") 
# Taking a look at the predictions
data.frame(Y_A1=head(Y_A1), Y_A0=head(Y_A0), TRUE_Y=head(ObsData$Y)) |> round(digits = 2)
```

    ##   Y_A1 Y_A0 TRUE_Y
    ## 1 0.77 0.55      1
    ## 2 0.42 0.21      0
    ## 3 0.63 0.39      0
    ## 4 0.55 0.31      0
    ## 5 0.80 0.60      1
    ## 6 0.63 0.38      0

Now, we can do the fourth step and compute the estimates.

``` r
# Mean outcomes in the two worlds
pred_A1 <- mean(Y_A1); pred_A0 <- mean(Y_A0)

# Marginal odds ratio
MOR_gcomp <- (pred_A1*(1 - pred_A0))/((1 - pred_A1)*pred_A0)
# ATE (risk difference)
RD_gcomp <- pred_A1 - pred_A0
c(MOR_gcomp, RD_gcomp) |> round(digits=3)
```

    ## [1] 2.568 0.231

As before when using IPW, we obtain unbiased estimates of the two causal
effects.

To quantify the variance of these estimates, we must rely on
bootstrapping, which we illustrate at the end of this document.

## Doubly-robust standardisation

Next, we illustrate the recipe provided in Box 4.

Doubly-robust standardisation is a doubly-robust estimator combining IPW
and g-computation. It can be more robust to potential misspecification
because it requires only one model to be correctly specified, this
giving us twice the chance to get it right.

This estimator begins (like IPW) with fitting g(C):

``` r
# Correctly specified action model
g <- glm(A ~ x2 + x4 + x2*x4, data = ObsData, family=binomial)$fitted.values
# Misspecified action model
gmis <- glm(A ~ x2 + x4, data = ObsData, family=binomial)$fitted.values
```

We also fit a second model, which is misspecified because it omits the
interaction between `x2` and `x4`. This will later allow us to
demonstrate the doubly robust property of the estimator.

As for IPW, we can check the balance at this step. However, because it’s
a doubly robust estimator, small imbalances are less punishing.

Second, we compute the weights that match the causal effect of interest
(here again the unstabilised ATE):

``` r
# Weights from the correctly specified action model
omega <- ifelse(ObsData$A==1, 1/g, 1/(1-g))
summary(omega)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##   1.037   1.385   1.625   1.999   2.313  17.559

``` r
# Weights from the misspecified action model
omegamis <- ifelse(ObsData$A==1, 1/gmis, 1/(1-gmis))
summary(omegamis)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##   1.040   1.397   1.659   2.003   2.274  21.454

The `omega` weights are identical of those from IPW because it is
exactly the same procedure up to this point. The `omegamis` weights
differ due to the misspecification of g(C).

Now, we start the g-computation part of doubly-robust standardisation.
We fit the outcome model Q(A,C), but this time it is weighted by the
weights `omega`.

``` r
Q <- glm(Y ~ A + x2 + x4 + x2*x4, weights = omega, data = ObsData, family=binomial)
```

We could again obtain the *conditional OR* through the coefficient
related to `A` in `Q`, but we are more interested in the marginal OR
which describes the effect across the whole population.

We also compute various misspecified Q(A,C) models:

``` r
# Outcome model misspecified
Qmis <- glm(Y ~ A + x2 + x4, weights = omega, data = ObsData, family=binomial)
# Outcome model correct but action model (weights) misspecified
Qomis <- glm(Y ~ A + x2 + x4 + x2*x4, weights = omegamis, data = ObsData, family=binomial)
# Both outcome model and action model misspecified
Qmismis <- glm(Y ~ A + x2 + x4, weights = omegamis, data = ObsData, family=binomial)
```

Where `Qmis` is only misspecified in Q(A,C), `Qomis` is only
misspecified in g(C), and `Qmismis` is misspecified in both Q(A,C) and
g(C).

Fourth step, generating the counterfactuals dataset:

``` r
# Duplicate data twice
A1Data <- A0Data <- ObsData
# Set action values
A1Data$A <- 1; A0Data$A <- 0
head(ObsData); head(A1Data); head(A0Data)
```

    ##   x1 x2         x3         x4 A Y Y.1 Y.0
    ## 1  0  1 -0.2909924  0.8374205 1 1   1   0
    ## 2  1  1 -0.1598913 -1.6984448 0 0   1   0
    ## 3  1  0 -0.5454640  1.6010621 0 0   1   0
    ## 4  0  1 -0.4733918 -0.8286665 0 0   1   0
    ## 5  1  1 -0.8476024  1.1711469 1 1   1   1
    ## 6  1  0 -1.2864879  1.5281848 0 0   1   0

    ##   x1 x2         x3         x4 A Y Y.1 Y.0
    ## 1  0  1 -0.2909924  0.8374205 1 1   1   0
    ## 2  1  1 -0.1598913 -1.6984448 1 0   1   0
    ## 3  1  0 -0.5454640  1.6010621 1 0   1   0
    ## 4  0  1 -0.4733918 -0.8286665 1 0   1   0
    ## 5  1  1 -0.8476024  1.1711469 1 1   1   1
    ## 6  1  0 -1.2864879  1.5281848 1 0   1   0

    ##   x1 x2         x3         x4 A Y Y.1 Y.0
    ## 1  0  1 -0.2909924  0.8374205 0 1   1   0
    ## 2  1  1 -0.1598913 -1.6984448 0 0   1   0
    ## 3  1  0 -0.5454640  1.6010621 0 0   1   0
    ## 4  0  1 -0.4733918 -0.8286665 0 0   1   0
    ## 5  1  1 -0.8476024  1.1711469 0 1   1   1
    ## 6  1  0 -1.2864879  1.5281848 0 0   1   0

We are now ready for the fifth step: counterfactual predictions.

``` r
Y_A1 <- predict(Q, A1Data, type="response") 
Y_A0 <- predict(Q, A0Data, type="response") 
data.frame(Y_A1=head(Y_A1), Y_A0=head(Y_A0), TRUE_Y=head(ObsData$Y)) |> round(digits = 2)
```

    ##   Y_A1 Y_A0 TRUE_Y
    ## 1 0.77 0.55      1
    ## 2 0.42 0.21      0
    ## 3 0.63 0.39      0
    ## 4 0.55 0.31      0
    ## 5 0.80 0.60      1
    ## 6 0.63 0.38      0

And we repeat the procedure for the misspecified models.

Finally, we compute the estimates of the doubly-robust standardisation:

``` r
pred_A1 <- mean(Y_A1); pred_A0 <- mean(Y_A0)

# Marginal odds ratio
MOR_DRS <- (pred_A1*(1 - pred_A0))/((1 - pred_A1)*pred_A0)
# ATE (risk difference)
RD_DRS <- pred_A1 - pred_A0
c(MOR_DRS, RD_DRS) |> round(digits=3)
```

    ## [1] 2.564 0.231

Again, the estimates are unbiased since we have included all the
confounder and correctly specified the models. But what if the model(s)
are misspecified?

``` r
pred_A1mis <- mean(Y_A1mis); pred_A0mis <- mean(Y_A0mis)
MOR_DRSmis <- (pred_A1mis*(1 - pred_A0mis))/((1 - pred_A1mis)*pred_A0mis)
RD_DRSmis <- pred_A1mis - pred_A0mis

pred_A1omis <- mean(Y_A1omis); pred_A0omis <- mean(Y_A0omis)
MOR_DRSomis <- (pred_A1omis*(1 - pred_A0omis))/((1 - pred_A1omis)*pred_A0omis)
RD_DRSomis <- pred_A1omis - pred_A0omis

pred_A1mismis <- mean(Y_A1mismis); pred_A0mismis <- mean(Y_A0mismis)
MOR_DRSmismis <- (pred_A1mismis*(1 - pred_A0mis))/((1 - pred_A1mismis)*pred_A0mismis)
RD_DRSmismis <- pred_A1mismis - pred_A0mismis

data.frame(`Q_mis`=c(MOR_DRSmis, RD_DRSmis), `g_mis`=c(MOR_DRSomis, RD_DRSomis), `Q_mis_g_mis`=c(MOR_DRSmismis, RD_DRSmismis))
```

    ##       Q_mis    g_mis Q_mis_g_mis
    ## 1 2.5643738 2.564584    2.611692
    ## 2 0.2311518 0.231171    0.236282

In the first two scenarios (only one model misspecified), there is no
bias. However, when both models are misspecified, we can observe a bias
(which is small in this simulated example but could of course be much
larger in actual data).

You can try to chance the IPW and the g-computation codes by yourself to
see that these procedures lack this doubly-robust property. Maybe you
also want to try omitting a confounder of changing the functional form
(*e.g.*, cubic relationship for `x4`) to get a feel for how the methods
behave.

## Bootstrapping to compute the variance

Bootstrapping is a powerful tool to obtain the variance of these
estimators. The underlying idea is to resample with replacement, so that
we end up with a slightly different sample.

``` r
# Small bootstrapping demonstration
test_db <- LETTERS[1:5]
test_db 
```

    ## [1] "A" "B" "C" "D" "E"

``` r
for(i in 1:3){
  db_boot <- test_db[sample(1:length(test_db), size=length(test_db), replace=TRUE)]
  print(paste0('Bootstrap sample #', i)); print(db_boot)
}
```

    ## [1] "Bootstrap sample #1"
    ## [1] "D" "A" "D" "B" "D"
    ## [1] "Bootstrap sample #2"
    ## [1] "D" "E" "C" "B" "D"
    ## [1] "Bootstrap sample #3"
    ## [1] "C" "E" "A" "D" "C"

By drawing several bootstrap samples and then applying the full
estimation process on each sample, we can obtain a fair estimate of the
variance, taking into account the whole uncertainty in the process
(uncertainty in the estimated weights for IPW; uncertainty in the model
Q for g-computation; uncertainty in both for doubly-robust
standardisation).

Let’s apply the bootstrap to g-computation:

``` r
# A bit of setup before we can start the resamplig

# We will need an empty vector to store the results
boot_MOR <- boot_ATE <- c()

# Number of bootstrap samples, usually 500 or 100
B <- 20 

for (i in 1:B){
  # We repeat everything in this loop B times
  
  # Draw the sample
  db_boot <- ObsData[sample(1:nrow(ObsData), size = nrow(ObsData), replace = TRUE),]
  
  # Step 1: fit Q(A,C) on db_boot (instead of ObsData)
  Q <- glm(Y ~ A + x2 + x4 + x2*x4, data = db_boot, family=binomial)
  
  # Step 2: counterfactual (bootstrap) datasets
  A1Data <- A0Data <- db_boot
  A1Data$A <- 1; A0Data$A <- 0
  
  # Step 3: Counterfactual predictions
  Y_A1 <- predict(Q, A1Data, type="response")
  Y_A0 <- predict(Q, A0Data, type="response")
  
  # Step 4: Estimates
  pred_A1 <- mean(Y_A1); pred_A0 <- mean(Y_A0)
  
  boot_MOR[i] <- (pred_A1*(1 - pred_A0))/((1 - pred_A1)*pred_A0)
  boot_ATE[i] <- pred_A1 - pred_A0
  
}
head(boot_MOR); head(boot_ATE)
```

    ## [1] 2.564340 2.583766 2.555410 2.561102 2.576301 2.575973

    ## [1] 0.2311627 0.2329416 0.2303259 0.2308432 0.2322557 0.2322273

Each bootstrap sample results in an estimate of the causal effect and in
the end, we have a vector of causal effect estimates. When working with
actual data we should run at least 500 bootstrap iterations to be
somewhat confident in the results.

Here, we have only 20 results; the results happen to be pretty similar
across bootstrap samples because of the huge sample size and the simple
data-generating process.

Once we have the result vector(s), we can summarize them to compute the
standard error or confidence intervals.

``` r
# Standard error (SE) of the marginal OR
sd(boot_MOR)
```

    ## [1] 0.0158117

``` r
# 95% Confidence Interval
quantile(boot_MOR, probs=c(0.025,0.975), na.rm=TRUE)
```

    ##     2.5%    97.5% 
    ## 2.529537 2.585768

``` r
# SE of the ATE
sd(boot_ATE)
```

    ## [1] 0.001462938

``` r
# 95% Confidence Interval
quantile(boot_ATE, probs=c(0.025,0.975), na.rm=TRUE)
```

    ##      2.5%     97.5% 
    ## 0.2279146 0.2331152

# Illustration of the impact of the weighting schemes on the pseudo-sample characteristics

In the previous sections, we have used the unstabilised ATE weights for
IPW and the doubly robust standardisation. However, we can use other
weighting schemes (presented in Table 2 of Chatton & Rohrer, 2023;
reproduced below) which result in different pseudo-samples.

| Name             |  Weight if A=1  |   Weight if A=0   | Target population |
|------------------|:---------------:|:-----------------:|:-----------------:|
| Unstabilised ATE |     1/g(C)      |   1/\[1-g(C)\]    |   Whole sample    |
| Stabilised ATE   |   P(A=1)/g(C)   | P(A=0)/\[1-g(C)\] |   Whole sample    |
| Unstabilised ATT |        1        |  g(C)/\[1-g(C)\]  |      Treated      |
| Unstabilised ATU | \[1-g(C)\]/g(C) |         1         |     Untreated     |
| Overlap          |     1-g(C)      |       g(C)        |      Unclear      |

Let’s take a look at how using these weights affects the actual
(simulated) sample characteristics.

``` r
## Construct the table 
tab <- tableone::CreateTableOne(vars = c("x1", "x2", "x3", "x4"), strata = "A", data = ObsData, test = FALSE, addOverall = TRUE)

## Show table without SMD (not relevant here)
print(tab, smd = FALSE)
```

    ##                 Stratified by A
    ##                  Overall       0             1            
    ##   n              500000        248680        251320       
    ##   x1 (mean (SD))   0.50 (0.50)   0.50 (0.50)   0.50 (0.50)
    ##   x2 (mean (SD))   0.65 (0.48)   0.50 (0.50)   0.80 (0.40)
    ##   x3 (mean (SD))   0.00 (1.00)  -0.55 (0.84)   0.54 (0.84)
    ##   x4 (mean (SD))   0.00 (1.00)  -0.22 (0.98)   0.22 (0.97)

We see that without any weighting, only `x1` is balanced between the two
groups.

Now take a look at the pseudo-sample that results when we weight the
data using the unstabilised ATE weights:

``` r
## Recompute the weights from g(C) for pedagogical purpose
omega <- ifelse(ObsData$A==1, 1/g, 1/(1-g))
summary(omega)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##   1.037   1.385   1.625   1.999   2.313  17.559

``` r
## Weighted data (pseudo-sample)
pseudo <- survey::svydesign(ids = ~ 1, data = ObsData, weights = ~ omega)

## Construct the table (This is quite slow)
tabWeighted <- tableone::svyCreateTableOne(vars = c("x1", "x2", "x3", "x4"), strata = "A", data = pseudo, test = FALSE, addOverall = TRUE)

## Show table without SMD
print(tabWeighted, smd = FALSE)
```

    ##                 Stratified by A
    ##                  Overall          0                1               
    ##   n              999696.74        499708.33        499988.41       
    ##   x1 (mean (SD))      0.50 (0.50)      0.50 (0.50)      0.50 (0.50)
    ##   x2 (mean (SD))      0.65 (0.48)      0.65 (0.48)      0.65 (0.48)
    ##   x3 (mean (SD))      0.00 (1.05)     -0.65 (0.83)      0.64 (0.83)
    ##   x4 (mean (SD))      0.00 (1.00)      0.00 (1.00)      0.00 (1.00)

Now, in the two groups in the pseudo-sample, `x2`and `x4` (the
covariates necessary to achieve conditional exchangeability) have the
same distribution as in the actual sample; `x3` differs between the
groups – it was not included when calculating the weights as it was not
necessary to achieve conditional exchangeability. Thus, the ATE weights
target the population represented by the whole sample. We can also see
that the pseudo-sample size is twice the actual sample size. Stabilised
weights correct this issue and produce less extreme weights.

``` r
## Compute the weights from g(C) 
omega <- ifelse(ObsData$A==1, mean(ObsData$A)/g, (1-mean(ObsData$A))/(1-g))
summary(omega)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##  0.5211  0.6923  0.8137  0.9997  1.1550  8.7332

``` r
## Weighted data (pseudo-sample)
pseudo <- survey::svydesign(ids = ~ 1, data = ObsData, weights = ~ omega)

## Construct the table (again slow)
tabWeighted <- tableone::svyCreateTableOne(vars = c("x1", "x2", "x3", "x4"), strata = "A", data = pseudo, test = FALSE, addOverall = TRUE)

## Show table without SMD
print(tabWeighted, smd = FALSE)
```

    ##                 Stratified by A
    ##                  Overall          0                1               
    ##   n              499849.11        248534.93        251314.18       
    ##   x1 (mean (SD))      0.50 (0.50)      0.50 (0.50)      0.50 (0.50)
    ##   x2 (mean (SD))      0.65 (0.48)      0.65 (0.48)      0.65 (0.48)
    ##   x3 (mean (SD))      0.00 (1.05)     -0.65 (0.83)      0.64 (0.83)
    ##   x4 (mean (SD))      0.00 (1.00)      0.00 (1.00)      0.00 (1.00)

We see we have approximate more closely the original sample size with
stabilised weights as expected.

What happens if we use ATT weights instead?

``` r
## Compute the weights from g(C) 
omega <- ifelse(ObsData$A==1, 1, g/(1-g))
summary(omega)
```

    ##     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
    ##  0.08435  0.62160  1.00000  1.00470  1.00000 16.55910

``` r
## Weighted data (pseudo-sample)
pseudo <- survey::svydesign(ids = ~ 1, data = ObsData, weights = ~ omega)

## Construct the table (anew slow)
tabWeighted <- tableone::svyCreateTableOne(vars = c("x1", "x2", "x3", "x4"), strata = "A", data = pseudo, test = FALSE, addOverall = TRUE)

## Show table without SMD
print(tabWeighted, smd = FALSE)
```

    ##                 Stratified by A
    ##                  Overall          0                1               
    ##   n              502348.33        251028.33        251320.00       
    ##   x1 (mean (SD))      0.50 (0.50)      0.50 (0.50)      0.50 (0.50)
    ##   x2 (mean (SD))      0.80 (0.40)      0.80 (0.40)      0.80 (0.40)
    ##   x3 (mean (SD))     -0.10 (1.05)     -0.75 (0.81)      0.54 (0.84)
    ##   x4 (mean (SD))      0.21 (0.97)      0.21 (0.97)      0.22 (0.97)

Here, we see that the distribution of `x2`and `x4`in the two groups no
longer matches the actual sample distribution. Instead, they have the
same distribution as in the attendee group in our original data. Thus,
ATT weights target the “treated” population instead of the whole
population.

ATU weights do “the opposite” and target the untreated population:

``` r
## Compute the weights from g(C) 
omega <- ifelse(ObsData$A==1, (1-g)/g, 1)
summary(omega)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##  0.0368  0.6272  1.0000  0.9947  1.0000 10.4176

``` r
## Weighted data (pseudo-sample)
pseudo <- survey::svydesign(ids = ~ 1, data = ObsData, weights = ~ omega)

## Construct the table (still slow)
tabWeighted <- tableone::svyCreateTableOne(vars = c("x1", "x2", "x3", "x4"), strata = "A", data = pseudo, test = FALSE, addOverall = TRUE)

## Show table without SMD
print(tabWeighted, smd = FALSE)
```

    ##                 Stratified by A
    ##                  Overall          0                1               
    ##   n              497348.41        248680.00        248668.41       
    ##   x1 (mean (SD))      0.50 (0.50)      0.50 (0.50)      0.50 (0.50)
    ##   x2 (mean (SD))      0.50 (0.50)      0.50 (0.50)      0.50 (0.50)
    ##   x3 (mean (SD))      0.10 (1.05)     -0.55 (0.84)      0.74 (0.81)
    ##   x4 (mean (SD))     -0.21 (0.98)     -0.22 (0.98)     -0.21 (0.98)

Lastly, there are overlap weights. They are designed to target a
population in which positivity is respected. Let’s take a look:

``` r
## Compute the weights from g(C) 
omega <- ifelse(ObsData$A==1, 1-g, g)
summary(omega)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ## 0.03549 0.27814 0.38464 0.42482 0.56763 0.94305

``` r
## Weighted data (pseudo-sample)
pseudo <- survey::svydesign(ids = ~ 1, data = ObsData, weights = ~ omega)

## Construct the table (always slow)
tabWeighted <- tableone::svyCreateTableOne(vars = c("x1", "x2", "x3", "x4"), strata = "A", data = pseudo, test = FALSE, addOverall = TRUE)

## Show table without SMD
print(tabWeighted, smd = FALSE)
```

    ##                 Stratified by A
    ##                  Overall          0                1               
    ##   n              212409.38        106204.69        106204.69       
    ##   x1 (mean (SD))      0.50 (0.50)      0.50 (0.50)      0.50 (0.50)
    ##   x2 (mean (SD))      0.67 (0.47)      0.67 (0.47)      0.67 (0.47)
    ##   x3 (mean (SD))      0.00 (1.05)     -0.65 (0.82)      0.64 (0.82)
    ##   x4 (mean (SD))     -0.03 (0.95)     -0.03 (0.95)     -0.03 (0.95)

Here, the pseudo-sample is similar to the one obtained with the help of
ATE weights. This is a “benign” scenario, because positivity is
respected in our simulated data (and thus, overlap weights would not be
necessary to begin with).

But what happens when there is a lack of positivity?

``` r
## Introduce a positivity violation in the simulated data
ObsData$x4[ObsData$x4>0.5 & ObsData$A==1] <- 0.5 # No attendee can have x4 > 0.5, but non-attendees can

## Compute the propensity score g(C)
g <- glm(A ~ x2 + x4 + x2*x4, data = ObsData, family=binomial)$fitted.values

## Compute the weights from g(C) 
omega <- ifelse(ObsData$A==1, (1-g)/g, 1)
summary(omega)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##  0.4517  0.5907  1.0000  0.9914  1.0000  3.7720

``` r
## Weighted data (pseudo-sample)
pseudo <- survey::svydesign(ids = ~ 1, data = ObsData, weights = ~ omega)

## Construct the table (over and over slow)
tabWeighted <- tableone::svyCreateTableOne(vars = c("x1", "x2", "x3", "x4"), strata = "A", data = pseudo, test = FALSE, addOverall = TRUE)

## Show table without SMD
print(tabWeighted, smd = FALSE)
```

    ##                 Stratified by A
    ##                  Overall          0                1               
    ##   n              495715.10        248680.00        247035.10       
    ##   x1 (mean (SD))      0.50 (0.50)      0.50 (0.50)      0.50 (0.50)
    ##   x2 (mean (SD))      0.50 (0.50)      0.50 (0.50)      0.50 (0.50)
    ##   x3 (mean (SD))      0.08 (1.04)     -0.55 (0.84)      0.71 (0.82)
    ##   x4 (mean (SD))     -0.20 (0.87)     -0.22 (0.98)     -0.17 (0.74)

In this scenario, the pseudo-sample distribution maps the non-attendee
group of the actual sample. Therefore, the target population is the
“untreated” population. This is due to the particular positivity
violation that we simulated.

More generally, the population targeted by the overlap weights always
lies somewhere between the populations targeted by the ATT and the ATU.
In the most benign scenario, it will happen to target the whole sample
(as it did before we simulated a positivity violation), but usually it
does not. Thus, we often end up with an ill-defined target population –
especially in settings where positivity is violated, and thus especially
when the statistical properties of these weights would add the most
value (Austin, 2023).

# References

Ali M.S., Groenwold R.H.H., Belitser S.V., Pestman W.R., Hoes A.W., Roes
K.C.B., de Boer A. & Klungel O.H. (2015) Reporting of covariate
selection and balance assessment in propensity score analysis is
suboptimal: A systematic review. *Journal of Clinical Epidemiology*,
68(2), 122‑131.

Austin P.C. (2023). Differences in target estimands between different
propensity score-based weights. *Pharmacoepidemiology and Drug Safety*,
Published ahead of print.

Chatton A. & Rohrer JM. (2023) The causal cookbook: Recipes for
propensity scores, g-computation and doubly robust standardisation. PsyArXiv: 10.31234/osf.io/k2gzp

Danelian G., Foucher Y., Léger M., Le Borgne F. & Chatton A. (2023)
Identifying in-sample positivity violations through regression trees:
the PoRT algorithm. Accepted in *Journal of Causal Inference*

Yoshida K. & Bartel A. (2022). tableone: Create ‘Table 1’ to Describe
Baseline Characteristics with or without Propensity Score Weights. *R
package*, <https://CRAN.R-project.org/package=tableone>

Westreich D. & Greenland S. (2013). The Table 2 Fallacy: Presenting and
Interpreting Confounder and Modifier Coefficients. *American Journal of
Epidemiology*, 177(4), 292‑298.
