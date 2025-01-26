#!/usr/bin/env python
# train_and_deploy.py

import boto3, re, sys, math, json, os, sagemaker, urllib.request
from sagemaker import get_execution_role
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image
from IPython.display import display
from time import gmtime, strftime
from sagemaker.serializers import CSVSerializer  # for newer usage

# Step 1: Import / define IAM role
role = 'arn:aws:iam::195966153534:role/AmazonSageMaker-ExecutionRole'
prefix = 'sagemaker/DEMO-xgboost-dm'
containers = {
    'us-west-2': '433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest',
    'us-east-1': '811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest',
    'us-east-2': '825641698319.dkr.ecr.us-east-2.amazonaws.com/xgboost:latest',
    'eu-west-1': '685385470294.dkr.ecr.eu-west-1.amazonaws.com/xgboost:latest'
}
my_region = 'us-east-1'  # set the region of the instance
print("Success - the MySageMakerInstance is in the " + my_region + " region. "
      "You will use the " + containers[my_region] + " container for your SageMaker endpoint.")

# Step 2: Create S3 bucket (reusing your original variable name)
bucket_name = 'irisbucket234'  # <--- Same as your original script
s3 = boto3.resource('s3')
try:
    if my_region == 'us-east-1':
        s3.create_bucket(Bucket=bucket_name)
    else:
        s3.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={'LocationConstraint': my_region}
        )
    print('S3 bucket created successfully')
except Exception as e:
    print('S3 error: ', e)

# Step 3: Download and read data
try:
    urllib.request.urlretrieve(
        "https://d1.awsstatic.com/tmt/build-train-deploy-machine-learning-model-sagemaker/"
        "bank_clean.27f01fbbdf43271788427f3682996ae29ceca05d.csv",
        "bank_clean.csv"
    )
    print('Success: downloaded bank_clean.csv.')
except Exception as e:
    print('Data load error: ', e)

try:
    model_data = pd.read_csv('./bank_clean.csv', index_col=0)
    print('Success: Data loaded into dataframe.')
except Exception as e:
    print('Data load error: ', e)

# Step 4: Split data
train_data, test_data = np.split(
    model_data.sample(frac=1, random_state=1729),
    [int(0.7 * len(model_data))]
)
print(train_data.shape, test_data.shape)

# Step 5: Save train.csv locally and upload to S3
pd.concat(
    [train_data['y_yes'], train_data.drop(['y_no', 'y_yes'], axis=1)], axis=1
).to_csv('train.csv', index=False, header=False)

boto3.Session().resource('s3').Bucket(bucket_name).Object(
    os.path.join(prefix, 'train/train.csv')
).upload_file('train.csv')

s3_input_train = sagemaker.inputs.TrainingInput(
    s3_data='s3://{}/{}/train'.format(bucket_name, prefix),
    content_type='csv'
)

# Step 6: Define the XGBoost Estimator
sess = sagemaker.Session()
xgb = sagemaker.estimator.Estimator(
    image_uri=containers[my_region],
    role=role,
    instance_count=1,
    instance_type='ml.m4.xlarge',
    output_path='s3://{}/{}/output'.format(bucket_name, prefix),
    sagemaker_session=sess
)
xgb.set_hyperparameters(
    max_depth=5,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    subsample=0.8,
    silent=0,
    objective='binary:logistic',
    num_round=100
)

# Step 7: Train the model
xgb.fit({'train': s3_input_train})

# Step 8: Deploy the model
xgb_predictor = xgb.deploy(
    initial_instance_count=1,
    instance_type='ml.m4.xlarge'
)
endpoint_name = xgb_predictor.endpoint_name

# Step 9: Test predictions on the test set
test_data_array = test_data.drop(['y_no', 'y_yes'], axis=1).values
xgb_predictor.serializer = CSVSerializer()  # or csv_serializer if older SDK
predictions = xgb_predictor.predict(test_data_array).decode('utf-8')
predictions_array = np.fromstring(predictions[1:], sep=',')
print(predictions_array.shape)

# Step 10: Print out quick classification metrics
cm = pd.crosstab(
    index=test_data['y_yes'],
    columns=np.round(predictions_array),
    rownames=['Observed'],
    colnames=['Predicted']
)
tn = cm.iloc[0, 0]
fn = cm.iloc[1, 0]
tp = cm.iloc[1, 1]
fp = cm.iloc[0, 1]
p = (tp + tn) / (tp + tn + fp + fn) * 100
print("\n{0:<20}{1:<4.1f}%\n".format("Overall Classification Rate: ", p))
print("{0:<15}{1:<15}{2:>8}".format("Predicted", "No Purchase", "Purchase"))
print("Observed")
print("{0:<15}{1:<2.0f}% ({2:<}){3:>6.0f}% ({4:<})".format(
    "No Purchase", tn/(tn+fn)*100, tn, fp/(tp+fp)*100, fp
))
print("{0:<16}{1:<1.0f}% ({2:<}){3:>7.0f}% ({4:<}) \n".format(
    "Purchase", fn/(tn+fn)*100, fn, tp/(tp+fp)*100, tp
))

print("\nEndpoint name: ", endpoint_name)
