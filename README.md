# Context
This project is part of a Data Challenge provided by ENS. Each year, the school 
organize machine learning challenges from data provided by public services, companies or 
laboratories. These challenges are free and open to anyone.
You can find this data challenge [here]("https://challengedata.ens.fr/challenges/89/")

# Table of contents
- [Goal](#goal)
- [Dataset](#dataset)
- [Usage](#usage)

# Goal
The aim of this challenge is to give SNCF-Transilien 
(Railway network operator covering the Paris region) the tools to
provide an accurate train occupancy rate forecasting. This tool will allow
the travelers to know how busy the train will be when they board it. This challenge
focus solely on forecasting the occupacy rates at the next station.

# Dataset
The data comes from infra-red sensors located above each door of the 
rolling stocks in Île-de-France, measuring the number of alighting and 
boarding passengers per door.
            
The columns, i.e., the features, are split into 6 contextual variables and 6 lag variables:
                 
Context Variables
- date: date of train passage
- train: id of the train (unique by day)
- station: station id
- hour: time slot
- way: wether the train is going toward Paris (way is 0) or suburb (way is 1)
- composition: number of train unit
            
Lags variables
- p1q0: Occupancy rate of the previous train at the same station
- p2q0: Occupancy rate of the second previous train at the same station
- p3q0: Occupancy rate of the third train at the same station
- p0q1: Occupancy rate of the same train at the previous station
- p0q2: Occupancy rate of the same train at the second previous station
- p0q3: Occupancy rate of the same train at the third previous station

# Usage
This project is a streamlit app that could be used to predict the occupancy rate of 
the train at the next station. You can launch the app by clicking [here]().

# Commit Policy
Please, respect the following commit policy:
```
[START]     first commit describing the branch you created
[ADD]       commit describing new elements or functionalities
[UP]        commit describing elements or functionalities updated
[FIX]       commit describing the error and how it has been fixed
[DEL]       commit describing elements or functionalities deleted
[PR]        pull request
```

Feel free to fork this repository and suggest a pull request if you respect the above commit policy.