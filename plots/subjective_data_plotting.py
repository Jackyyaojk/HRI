import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def comp_score(pos_score, neg_score):
  # Calculating composite score for Likert scale
  # pos_score - positive aspect score
  # neg_score - negative aspect score

  score = (pos_score + (8-neg_score))/2
  
  return score



data = pd.read_excel('subjective_responses.xlsx')

# Create an empty DataFrame with specified columns to store question scores
columns = ['Method', 'Participant', 'Question', 'Score']
results = pd.DataFrame(columns=columns)


results_list = []
# preference_list = []

# this next for loop would iterate through rows (participants)
for index, row in data.iterrows(): 
  # this other one iterates through methods
  for number in range(1,4):
    method_string = 'method_' + str(number)
    method = data.at[index, method_string]
    if index == 0:
      continue
    else:
      for i in range(1,6):
        pos_string = 'm' + str(number) + '_q' + str(i) + '_pos'
        neg_string = 'm' + str(number) + '_q' + str(i) + '_neg'
        pref_string = 'm' + str(number) + '_method_score'
        
        pos_score = data.at[index, pos_string]
        neg_score = data.at[index, neg_string]
        method_score = data.at[index, pref_string]

        score = comp_score(pos_score, neg_score)
        # print(f'Method: {method}, Participant: {index}, Question: {i}, score: {score}')

        # Append the question results to a list
        results_list.append([method, index, i, score])
        # Add the results to the 'results' DataFrame
        results = pd.DataFrame(results_list, columns=columns)

      # Append the preference results to a list
      results_list.append([method, index, 6, method_score])
      # Add the results to the 'preference_results' DataFrame
      results = pd.DataFrame(results_list, columns=columns)

      # print(f'Method score: {method_score}')

# Create a dictionary to map numeric questions to your desired names
question_mapping = {1: 'Learned', 
                    2: 'Trust', 
                    3: 'Adapt', 
                    4: 'Intuitive', 
                    5: 'Easy', 
                    6:'Prefer'
                    }

# # Map the numeric question column to your desired names
results['Question'] = results['Question'].map(question_mapping)



# Plotting Question Scores by Method
sns.set(style='whitegrid')

# Create a question scores plot by method
plt.figure(figsize=(12, 6))
sns.barplot(x='Question', y='Score', hue='Method', data=results, palette='viridis')
plt.title('Question Scores by Method')
plt.xlabel('Question')
plt.ylabel('Score')
# Save the plot as a PNG file
plt.savefig('subjective_scores_plot.svg')
plt.savefig('subjective_scores_plot.png')
plt.show()


# Calculate summary statistics for each question by method
summary_stats = results.groupby(['Question', 'Method']).agg({'Score': ['mean', 'std', 'min', 'max']})
print("\nSummary Statistics for Each Question by Method:")
print(summary_stats)
