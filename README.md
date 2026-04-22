# STAT405 Group Project — U.S. Housing Market Analysis

## Overview
This project analyzes the HouseTS dataset, a ZIP-code-level spatiotemporal panel 
covering 6,000 ZIP codes across 30 U.S. metro areas from March 2012 to December 
2023. We investigate two questions: (1) How did home value recovery after COVID-19 
differ across metro areas? (2) Do ZIP codes with more amenities show higher or 
faster-growing home values?

## Dataset
HouseTS: https://www.kaggle.com/datasets/shengkunwang/housets-dataset

Variables include housing-market indicators (sale/list prices, inventory, listings),
points of interest (restaurants, schools, parks, supermarkets), and socioeconomic 
indicators (income, employment, population, demographics).

**Key line to read data:**
head -1 HouseTS.csv | tr ',' '\n'

## Statistical Methods
**COVID-19 Recovery:** Time-series decomposition, percent change calculation, 
K-means clustering, and ANOVA to compare recovery rates across metros.

**Amenities vs. Home Values:** PCA to reduce amenity dimensions, correlation 
analysis, and multiple linear regression modeling ZHVI growth as a function of 
amenity counts controlling for income and location.

## Computational Steps
Data acquisition, cleaning, temporal alignment, exploratory analysis, feature 
construction, statistical modeling, and visualization. Parallel computation on 
HPC/CHTC splits jobs by metro area to efficiently run decompositions, clustering, 
and regression across 30 cities.

## Clone This Repo
git clone https://github.com/jzhang2752/STAT405-Project_Group-3.git

## Group Members
[Eleanor Zhang]
[Siqi Qiu]
[Charith Reddy Pareddy]
[Mihir Narayan]
[Nikhil Ashokan]
