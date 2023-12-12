# User Study Results
We used two objective metrics to analyze the outcomes of closing the loop on robot learning.

1. `Correct Prediction`: records how frequently users were able to correctly predict which chair leg the robot was assembling.

2. `Error`: the robot's error in assembling the chair legs after receiving corrections from the human.


### Plot Objective Results
To plot `Correct Prediction` run this command:
```
python plot_prediction.py
```
To plot `Error` run this command:
```
python plot_error.py
```


### Plot Subjective Results
To plot `Subjective Measures Results`, you will need an excel file with your results (*subjective_responses.xslx*). To plot, run this command:
```
python subjective_data_plotting.py
```
Note: the data structure in *subjective_responses.xslx* is based on how **Google Forms** creates a spreadsheet out of our survey form. The data structure may be different for your survey.

---

All plots will be saved in this directory (`/plots`).
