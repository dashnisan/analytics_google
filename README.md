# analytics_google
Samples of activities for Certificate Advanced Data Analytics by Google 

In this repository you find samples of activities for the [Google Advanced Data Analytics Professional Certificate](https://www.coursera.org/professional-certificates/google-advanced-data-analytics) on [Coursera](https://www.coursera.org/)

If not specified otherwise, all the code included in this repository makes reference to the following data set:

This project uses a dataset called waze_dataset.csv. It contains synthetic data created for this project in partnership with Waze. 

The dataset contains:

14,999 rows – each row represents one unique user 

13 columns

| Column name  | Type | Description |
|:------------:|:----:|:------------|
| ID | int | A sequential numbered index |
| label | obj | Binary target variable (“retained” vs “churned”) for if a user has churned anytime during the course of the month |
| sessions | int | The number of occurrence of a user opening the app during the month |
| drives | int | An occurrence of driving at least 1 km during the month 
|device | obj | The type of device a user starts a session with |
| total_sessions | float | A model estimate of the total number of sessions since a user has onboarded |
| n_days_after_onboarding | int | The number of days since a user signed up for the app |
| total_navigations_fav1 | int | Total navigations since onboarding to the user’s favorite place 1 |
| total_navigations_fav2 | int | Total navigations since onboarding to the user’s favorite place 2 |
| driven_km_drives | float | Total kilometers driven during the month |
| duration_minutes_drives | float | Total duration driven in minutes during the month |
| activity_days | int | Number of days the user opens the app during the month  |
| driving_days | int | Number of days the user drives (at least 1 km) during the month |
