 Bank-Marketing-ML
	a. Problem statement 
	      Title: Bank Term Deposit Subscription Prediction Dataset
      Subtitle: Predict whether a client will subscribe to a bank term deposit
      Description:
This dataset contains information about clients of a banking institution. The goal is to predict whether a client will subscribe to a bank term deposit (variable y). The data was obtained from a marketing campaign, and each entry corresponds to a single client.
	
	b. Dataset description 
The dataset contains 45,211 entries with 17 attributes. The attributes represent client information and campaign details, 
and they include both categorical and numerical data. 
•	age: Age of the client (numeric) 
•	job: Type of job (categorical: "admin.", "blue-collar", "entrepreneur", etc.)
•	marital: Marital status (categorical: "married", "single", "divorced")
•	education: Level of education (categorical: "primary", "secondary", "tertiary", "unknown")
•	default: Has credit in default? (categorical: "yes", "no")
•	balance: Average yearly balance in euros (numeric)
•	housing: Has a housing loan? (categorical: "yes", "no")
•	loan: Has a personal loan? (categorical: "yes", "no")
•	contact: Type of communication contact (categorical: "unknown", "telephone", "cellular")
•	day: Last contact day of the month (numeric, 1-31)
•	month: Last contact month of the year (categorical: "jan", "feb", "mar", …, "dec")
•	duration: Last contact duration in seconds (numeric)
•	campaign: Number of contacts performed during this campaign (numeric)
•	pdays: Number of days since the client was last contacted from a previous campaign (numeric; -1 means the client was not previously contacted)
•	previous: Number of contacts performed before this campaign (numeric)
•	poutcome: Outcome of the previous marketing campaign (categorical: "unknown", "other", "failure", "success")
y: The target variable, whether the client subscribed to a term deposit (binary: "yes", "no")

	c. Github Repository Link 
https://github.com/sumitmarghade19/Bank-Marketing-ML/


d. Make a Comparison Table with the evaluation metrics calculated for all the 5 models as below: 

ML Model Name 	- Logistic Regression	
Accuracy - 87.40%
AUC  - 77.18%
PRECISION - 45.96%
RECAL - 43.57%
F1 - 44.74%
MCC - 0.377

ML Model Name 	- Decision Tree
Accuracy - 87.36%
AUC  - 76.09%
PRECISION - 46.02%
RECAL - 46.50%
F1 - 46.26%
MCC - 0.391


KNN						
ML Model Name 	- KNN
Accuracy - 84.05%
AUC  - 74.18%
PRECISION - 36.97%
RECAL - 51.51%
F1 - 43.05%
MCC - 0.347
									
ML Model Name 	- Naive Bayes
Accuracy - 87.08%
AUC  - 75.05%
PRECISION - 43.97%
RECAL - 37.90%
F1 - 40.71%
MCC - 0.336

					
ML Model Name 	- Support Vector Machine
Accuracy - 87.02%
AUC  - 77.13%
PRECISION - 44.46%
RECAL - 43.95%
F1 - 44.20%
MCC - 0.369


Add your observations on the performance of each model on the chosen dataset
ML Model Name 	Observation about model performance
1. Logistic Regression -	Best AUC in the experiment including duration. A strong baseline classifier.

2. Decision Tree-	Highest recall when duration is included. Also achieves the best F1 and MCC without duration.

3. KNN -	Highest accuracy in both experiments, but its recall is relatively poor. It misses many actual subscribers.

4. Naive Bayes -	Fastest model, but its overall discrimination performance is comparatively lower.

5. Support Vector Machine -	Best F1 and MCC with duration, and very close to Logistic Regression in AUC.

Overall, Winner for your dataset?	My recommendation:
For predicting before a marketing call: We can go with Decision Tree and Logistic Regression further.
Improvement need in both 
a. For identifying as many subscribers as possible: evaluate recall and adjust the classification threshold.
b.  For minimizing unnecessary marketing calls: optimize precision.

