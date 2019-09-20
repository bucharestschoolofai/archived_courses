#!/bin/bash

# create bucket
gsutil mb -l us-east1 gs://my-awesome-bucket/

# copy files
gsutil cp inception_saved_model/1553809955 gs://my-awesome-bucket

# create model
gcloud ml-engine models create MODEL_NAME --description=DESCRIPTION

# create version
gcloud ml-engine versions create VERSION --model=MODEL --description=DESCRIPTION --origin=ORIGIN

# predict
gcloud ml-engine predict --model bsoai_test --version v1 --json-instances request.json
