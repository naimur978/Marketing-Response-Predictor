

# Deploying a Machine Learning Model on AWS

## Project Focus
The focus of this project is on **deploying** a machine learning model to AWS. Although we do make predictions, the main objective is to set up all the necessary AWS resources (S3, SageMaker, Lambda, API Gateway) to host my model in a scalable, serverless environment.

---

## AWS Services Used
- **Amazon S3** – for storing the training data (`train.csv`).
- **Amazon SageMaker** – for training my XGBoost model on the bank dataset and hosting a real-time inference endpoint.
- **AWS Lambda** – for providing inference logic (calls the SageMaker endpoint).
- **Amazon API Gateway** – for creating an HTTP endpoint that triggers the Lambda function.

---

## Step-by-Step Process

### 1. Train and Deploy the Model (SageMaker)
1. **Clone or download** this project.
2. **Open a terminal** in the project folder (where `train_and_deploy.py` is located).
3. **Install necessary libraries** (e.g., `boto3`, `sagemaker`, `pandas`, etc.) if not already available.
4. **Run the script** to train and deploy:
   ```bash
   python3 train_and_deploy.py
   ```
   - This script will:
     - Create an S3 bucket (if it does not exist).
     - Upload `train.csv` to S3.
     - Launch an XGBoost training job in SageMaker.
     - Deploy the resulting model to a SageMaker **endpoint**.
     - Print out the SageMaker **endpoint name** at the end (e.g., `xgboost-2025-01-26-16-02-04-923`).

### 2. Create a Lambda Function
1. **Go to the AWS Lambda Console** and create a new function (e.g., the default one is given as `lambda_function.py`).
2. **Paste the following code** into the Lambda console. Make sure to replace the `ENDPOINT_NAME` with the actual endpoint name printed by `train_and_deploy.py` if it differs.

   ```python
   import json
   import boto3

   # Replace with your actual SageMaker endpoint name
   ENDPOINT_NAME = "xgboost-2025-01-26-16-02-04-923"

   def lambda_handler(event, context):
       # 1. Parse all input features from queryStringParameters
       query_params = event.get('queryStringParameters', {})
       
       # List of all required feature names in the correct order
       required_features = [
           "age", "campaign", "pdays", "previous", "no_previous_contact", "not_working",
           "job_admin.", "job_blue-collar", "job_entrepreneur", "job_housemaid",
           "job_management", "job_retired", "job_self-employed", "job_services",
           "job_student", "job_technician", "job_unemployed", "job_unknown",
           "marital_divorced", "marital_married", "marital_single", "marital_unknown",
           "education_basic.4y", "education_basic.6y", "education_basic.9y", "education_high.school",
           "education_illiterate", "education_professional.course", "education_university.degree",
           "education_unknown", "default_no", "default_unknown", "default_yes",
           "housing_no", "housing_unknown", "housing_yes", "loan_no", "loan_unknown",
           "loan_yes", "contact_cellular", "contact_telephone", "month_apr", "month_aug",
           "month_dec", "month_jul", "month_jun", "month_mar", "month_may", "month_nov",
           "month_oct", "month_sep", "day_of_week_fri", "day_of_week_mon", "day_of_week_thu",
           "day_of_week_tue", "day_of_week_wed", "poutcome_failure", "poutcome_nonexistent",
           "poutcome_success"
       ]

       # Extract values for required features and ensure all are present
       data = [float(query_params.get(feature, 0)) for feature in required_features]

       # 2. Format as CSV (single row)
       payload = ",".join(map(str, data))

       # 3. Invoke SageMaker endpoint
       runtime = boto3.client('sagemaker-runtime')
       try:
           response = runtime.invoke_endpoint(
               EndpointName=ENDPOINT_NAME,
               ContentType='text/csv',
               Body=payload
           )

           # 4. Parse prediction output
           result = response['Body'].read().decode('utf-8')
           prediction = float(result.strip())

           return {
               "statusCode": 200,
               "body": json.dumps({
                   "prediction": prediction
               }),
               "headers": {
                   "Content-Type": "application/json"
               }
           }

       except Exception as e:
           # Handle errors gracefully
           return {
               "statusCode": 500,
               "body": json.dumps({"error": str(e)})
           }
   ```
3. **Deploy** the Lambda function changes (click "**Deploy**" in the Lambda console).

### 3. Set Up an API Gateway
1. **Create an HTTP API** in the Amazon API Gateway console.
2. **Add a Lambda Integration**:
   - Choose "**Lambda**" as the integration type.
   - Select the Lambda function (`BankInferenceLambda`).
3. **Define a Route**:
   - For example, `GET /BankInferenceLambda`.
   - Attach the newly created Lambda integration to that route.
4. **Obtain the Invoke URL** shown in the API Gateway console.  
   For example:  
   ```
   https://ou38vl9wcj.execute-api.us-east-1.amazonaws.com/BankInferenceLambda
   ```

### 4. Test the Endpoint
In a browser or using `curl`, **append query parameters** for each feature. For example:

```
https://ou38vl9wcj.execute-api.us-east-1.amazonaws.com/BankInferenceLambda
    ?age=56
    &campaign=1
    &pdays=999
    &previous=0
    &no_previous_contact=1
    &not_working=0
    &job_admin.=0
    &job_blue-collar=0
    &job_entrepreneur=1
    &job_housemaid=0
    &job_management=0
    &job_retired=0
    &job_self-employed=0
    &job_services=0
    &job_student=0
    &job_technician=0
    &job_unemployed=0
    &job_unknown=0
    &marital_divorced=1
    &marital_married=0
    &marital_single=0
    &marital_unknown=0
    &education_basic.4y=1
    &education_basic.6y=0
    &education_basic.9y=0
    &education_high.school=0
    &education_illiterate=0
    &education_professional.course=0
    &education_university.degree=0
    &education_unknown=1
    &default_no=0
    &default_unknown=0
    &default_yes=1
    &housing_no=0
    &housing_unknown=0
    &housing_yes=1
    &loan_no=0
    &loan_unknown=0
    &loan_yes=0
    &contact_cellular=0
    &contact_telephone=1
    &month_apr=0
    &month_aug=0
    &month_dec=0
    &month_jul=0
    &month_jun=0
    &month_mar=0
    &month_may=1
    &month_nov=0
    &month_oct=0
    &month_sep=0
    &day_of_week_fri=0
    &day_of_week_mon=0
    &day_of_week_thu=0
    &day_of_week_tue=0
    &day_of_week_wed=1
    &poutcome_failure=0
    &poutcome_nonexistent=0
    &poutcome_success=0
```

When you visit this URL, you should see a **JSON** response like:
```json
{
  "prediction": 0.123456
}
```
(The numerical value depends on the model’s prediction.)

---

## Conclusion
1. **Trained** the model by running `python3 train_and_deploy.py`, which deployed a SageMaker endpoint.
2. **Set Up a Lambda Function** to handle incoming requests and call the endpoint with CSV-formatted features.
3. **Exposed** that Lambda function via **API Gateway** to produce a public URL.
4. **Tested** the live endpoint by passing query parameters in a browser or via `curl`.

The project demonstrates how to deploy an ML model in a serverless AWS environment, emphasizing how these AWS services connect rather than focusing solely on the prediction results. 
