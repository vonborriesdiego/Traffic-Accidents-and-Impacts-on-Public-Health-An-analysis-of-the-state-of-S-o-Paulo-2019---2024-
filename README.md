# Traffic Accidents and Impacts on Public Health: An analysis of the state of São Paulo 2019 - 2024

## Overview

This project investigates traffic accidents in the state of São Paulo, Brazil, using Bayesian statistical models to study accident rates and injury severity across municipalities.

The analysis uses traffic accident data reported by **DETRAN-SP** for the period **2019–2024**, covering the 645 municipalities of São Paulo. Socioeconomic and geographic characteristics such as **HDI, GDP per capita, territorial area, population density, and year** were incorporated into the statistical models.

The project applies two complementary Bayesian approaches:

* **Beta Regression** — to model the accident rate relative to the municipal population.
* **ALR Compositional Regression** — to analyze the relative proportions of fatal, severe, mild, and unreported injuries.

## Objective

The main objectives of this project are:

1. Describe the traffic accident scenario in São Paulo between 2019 and 2024.
2. Investigate how socioeconomic and geographic characteristics are associated with municipal accident rates.
3. Analyze the composition of accident outcomes according to injury severity.
4. Apply Bayesian regression models to quantify uncertainty in the estimated effects.

## Dataset

The dataset contains traffic accidents reported by DETRAN-SP between **2019 and 2024**, with information including:

* Municipality
* Year
* Road type
* Victim gender
* Victim age
* Injury severity
* Type of victim
* Accident counts

The study aggregates the data at the **municipality-year level**.

The socioeconomic variables were obtained from **IBGE** and include:

| Variable           | Description                       |
| ------------------ | --------------------------------- |
| Year               | Year of accident occurrence       |
| HDI                | Municipal Human Development Index |
| GDP per capita     | Municipal GDP per capita          |
| Area               | Municipal territorial area        |
| Population         | Municipal population              |
| Population density | Population density                |

The manuscript reports **939,355 traffic accidents** for the 2019–2024 period and considers the 645 municipalities of São Paulo.

## Outcomes

Two main types of outcomes were analyzed.

### 1. Accident Rate

The first outcome is the number of accidents divided by the municipal population:

**accident rate = accident count / population**

This outcome was modeled using **Beta regression**, allowing the relationship between the accident rate and municipal characteristics to be investigated.

### 2. Injury Composition

The accident counts were divided into four categories:

* Fatal
* Severe
* Mild
* Unreported

Because these proportions sum to one, they were treated as **compositional data**.

An **Additive Log-Ratio (ALR)** transformation was used, taking the unreported category as the reference:

* Fatal / Unreported
* Severe / Unreported
* Mild / Unreported

This allows separate regression equations to investigate how municipal characteristics affect the relative composition of injury severity.

## Methodology

### Exploratory Analysis

The initial analysis investigates accident counts according to:

* Age
* Gender
* Road type
* Injury severity
* Year
* Month

The temporal analysis also compares accident patterns during the **2019–2021** and **2022–2024** periods.

A decrease in total accidents was observed during 2019 and 2020, coinciding with the COVID-19 pandemic period. The manuscript also reports differences in the proportions of injury severity between periods.

### Multicollinearity Analysis

Before fitting the Bayesian models, the explanatory variables were evaluated for multicollinearity using the **Variance Inflation Factor (VIF)**.

The analysis did not indicate problematic multicollinearity among the selected covariates.

### Bayesian Beta Regression

A hierarchical Bayesian **Beta regression** was used to model the municipal accident rate.

The model incorporates:

* Year
* HDI
* Log GDP per capita
* Log territorial area
* Population density

A municipality-specific random intercept was included to account for additional variation between municipalities.

The model was implemented in **PyMC** using the **No-U-Turn Sampler (NUTS)**, with four independent chains and posterior sampling after tuning.

### Bayesian ALR Compositional Regression

The second model analyzes the composition of accident severity.

The ALR transformation uses the unreported category as the reference and produces three response variables:

```text
log(Fatal / Unreported)
log(Severe / Unreported)
log(Mild / Unreported)
```

Each transformed outcome is modeled as a function of:

* Year
* HDI
* GDP per capita
* Territorial area
* Population density

The model was also implemented in PyMC using NUTS and four independent chains.

### Bayesian Inference

Bayesian inference was used to obtain posterior distributions for the regression coefficients.

The analysis uses **Markov Chain Monte Carlo (MCMC)** methods, allowing posterior means, standard deviations, and credible intervals to be obtained for the parameters of interest.

## Main Results

### Accident Rate

The Bayesian Beta regression identified significant associations for:

* **HDI:** negative effect
* **GDP per capita:** negative effect
* **Territorial area:** positive effect
* **Population density:** positive effect

The estimated HDI coefficient was **−5.694**, while the GDP per capita coefficient was **−0.177**. Area and population density presented positive coefficients.

### Fatal Injury Ratio

For the fatal-to-unreported ratio:

* **HDI:** −7.098
* **GDP per capita:** 0.542
* **Population density:** −0.830

The manuscript reports that these coefficients had 95% credible intervals that did not include zero.

### Severe Injury Ratio

For the severe-to-unreported ratio:

* **HDI:** 4.330
* **Territorial area:** −0.409
* **Population density:** −1.492

These variables were reported as having credible intervals excluding zero.

### Mild Injury Ratio

For the mild-to-unreported ratio:

* **HDI:** 13.995
* **GDP per capita:** 0.304
* **Territorial area:** −1.437
* **Population density:** −2.751

HDI presented the largest estimated coefficient among the variables in this component of the model.

## Interpretation

The results suggest that socioeconomic and geographic characteristics are associated differently with accident frequency and injury composition.

In particular, the analysis identifies **HDI and population density** as important variables across several outcomes. The manuscript discusses the negative association between population density and fatal injury ratios as potentially related to differences in traffic speed, healthcare access, and urban characteristics.

The temporal analysis also suggests a reduction in accident counts during the COVID-19 period, while the proportion of severe injuries was slightly higher during 2019–2021. These findings are discussed in relation to changes in mobility and driving behavior during the pandemic.

## Computational Implementation

The complete analysis was implemented in **Python 3.12.2**.

Main libraries:

* **NumPy** — numerical computation
* **Pandas** — data manipulation
* **Matplotlib** — visualization
* **PyMC** — Bayesian modeling
* **ArviZ** — Bayesian diagnostics and posterior visualization

The implementation includes posterior summaries, trace plots, forest plots, and diagnostic analyses.

## Project Structure

```text
├── Municipios.py
├── README.md
├── ModelWithUniformDist.jpg
├── TraceModelWithNormalDist.jpg
├── ForestPlotNormal.jpg
├── CompositionalModelTrace.jpg
├── ForestPlotCompositionalM.jpg
└── ...
```

`Municipios.py` contains the complete Python implementation, including:

* Data retrieval and preprocessing
* Multicollinearity analysis
* Bayesian Beta regression
* Prior sensitivity/model development
* Bayesian ALR compositional regression
* MCMC sampling
* Posterior summaries
* Trace plots
* Forest plots

The code also documents the development process used to select the final Beta regression specification, including experimentation with different prior distributions.

## Bayesian Model Development

An important part of the project was the comparison of prior specifications for the Beta regression.

The initial model used Uniform priors for the regression coefficients. However, the posterior distribution for the HDI coefficient reached the boundary of the Uniform prior, suggesting that the prior was restricting the parameter.

The final model therefore used weakly informative **Normal(0, 1)** priors for the regression coefficients, a **Half-Normal** prior for the Beta precision parameter, and an **Exponential** prior for the random-effect precision.

## Diagnostics

MCMC convergence was evaluated using posterior diagnostics and trace plots.

For the final Beta regression, the regression coefficients achieved:

* **R-hat ≈ 1.00**
* **ESS > 4,000**

The random-effect precision parameter showed some mixing difficulty, with **R-hat = 1.02** and ESS = 244, while the manuscript/code reports that this did not affect the regression coefficient estimates.

## References

Key methodological references include:

* Aitchison, J. (1982). *The Statistical Analysis of Compositional Data*.
* Ferrari, S. L. P., & Cribari-Neto, F. (2004). *Beta Regression for Modelling Rates and Proportions*.
* Gelman, A., et al. (2013). *Bayesian Data Analysis*.
* de Oliveira, R. P., & Achcar, J. A. (2020). *Compositional statistical models under a Bayesian approach: An application to traffic accident data in federal highways in Brazil*.
* DETRAN-SP. (2025). *Infosiga SP*.

## Author

**Diego von Borries**

Biomedical Engineer | MSc in Public Health | Statistics & Data Science

Research interests: Bayesian statistics, statistical modeling, epidemiology, machine learning, and data science.
