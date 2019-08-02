# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import json
from urllib.parse import urlparse

import boto3


def lambda_handler(event, context):
    """Post labeling lambda function for custom labeling jobs"""

    # Event received
    print("Received event: " + json.dumps(event, indent=2))

    consolidated_labels = []

    parsed_url = urlparse(event['payload']['s3Uri'])

    s3 = boto3.client('s3')
    textFile = s3.get_object(Bucket=parsed_url.netloc, Key=parsed_url.path[1:])
    print(textFile)
    filecont = textFile['Body'].read()
    annotations = json.loads(filecont)

    for dataset in annotations:
        for annotation in dataset['annotations']:
            new_annotation = json.loads(
                annotation['annotationData']['content'])
            label = {
                'datasetObjectId': dataset['datasetObjectId'],
                'consolidatedAnnotation': {
                    'content': {
                        event['labelAttributeName']: {
                            'workerId': annotation['workerId'],
                            'result': new_annotation,
                            'labeledContent': dataset['dataObject']
                        }
                    }
                }
            }
            consolidated_labels.append(label)

    # Response sending
    print("Response: " + json.dumps(consolidated_labels))

    return consolidated_labels
