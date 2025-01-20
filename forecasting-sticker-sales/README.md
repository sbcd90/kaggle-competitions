forecasting-sticker-sales
=========================

Problem
=======
https://www.kaggle.com/competitions/playground-series-s5e1/overview

High Level Design
================

Simple MLP based regression model.

## Original fields
```aiignore
id,date,country,store,product,num_sold
```

## Features include

- Categorial features
```aiignore
country,store,product,season(derived from date)
```

- Numerical features(all derived from date)
```aiignore
month, year, day, day_of_week, week_of_year, quarter, day_sin, day_cos, month_sin, month_cos
```

DL Model Architecture
=====================

