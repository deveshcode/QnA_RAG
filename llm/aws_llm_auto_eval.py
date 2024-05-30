import boto3
import json
import os
# Create a Bedrock client
bedrock = boto3.client(service_name='bedrock', region_name='us-east-1')

# Define the evaluation job parameters
job_name = 'titan-text-lite-evaluation'
job_description = 'Evaluating the performance of the Amazon Titan Text Lite model on a text generation task'
role_arn = os.getenv('BEDROCK_ROLE_ARN')
evaluation_config = {
    'automated': {
        'datasetMetricConfigs': [
            {
                'taskType': 'GENERATION',
                'dataset': {
                    'name': 'eval_data.jsonl',
                    'datasetLocation': {
                        's3Uri': 's3://qna-eval-bucket/eval_data.jsonl'
                    }
                },
                'metricNames': [
                    'Builtin.Accuracy',
                    'Builtin.Robustness'
                ]
            }
        ]
    }
}
inference_config = {
    'models': [
        {
            'bedrockModel': {
                'modelIdentifier': 'amazon.titan-text-lite-v1',
                'inferenceParams': json.dumps({
                    'maxTokenCount': 100,
                    'stopSequences': [],
                    'temperature': 0.5,
                    'topP': 0.9
                })
            }
        }
    ]
}
output_data_config = {
    's3Uri': 's3://qna-eval-bucket/eval_results'
}

# Create the evaluation job
response = bedrock.create_evaluation_job(
    jobName=job_name,
    jobDescription=job_description,
    roleArn=role_arn,
    evaluationConfig=evaluation_config,
    inferenceConfig=inference_config,
    outputDataConfig=output_data_config
)

# Print the evaluation job ARN
print(response['jobArn'])
