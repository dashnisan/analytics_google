# analytics_google
Samples of activities for Certificate Advanced Data Analytics by Google 

In this repository you find samples of activities for the [Google Advanced Data Analytics Professional Certificate](https://www.coursera.org/professional-certificates/google-advanced-data-analytics) on [Coursera](https://www.coursera.org/)


Each folder in this repository corresponds to the final project of one course in the certification program.

The final project of the whole program is found under "0_Final_Project". This project uses a dataset called HR_capstone_dataset.csv. It represents 10 columns of self-reported information from employees of a fictitious multinational vehicle manufacturing corporation. 

The dataset contains:

14,999 rows – each row is a different employee’s self-reported information

10 columns

| Column name  | Type | Description |
|:------------:|:----:|:------------|
| satisfaction_level| int | The employee’s self-reported satisfaction level [0-1] |
| last_evalueation| | int | Score of employee's last performance review [0–1] |
| number_project | int | Number of projects employee contributes to |
| average_monthly_hours | int | Average number of hours employee worked per month |
| time_spend_company | int | How long the employee has been with the company (years) |
| work_accident | int | Whether or not the employee experienced an accident while at work |
| left | int | Whether or not the employee left the company |
|promotion_last_5years | int | Whether or not the employee was promoted in the last 5 years |
| department | string | 200~The employee's department |
| salary | str | The employee's salary (low, medium, or high) |


For the rest of the courses (folders 1 through 4) refer to a dataset called waze_dataset.csv. It contains synthetic data created for this project in partnership with Waze. 

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
