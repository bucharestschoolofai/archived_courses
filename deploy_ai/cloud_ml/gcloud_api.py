import googleapiclient.discovery
from keras.applications.inception_v3 import decode_predictions
import numpy as np
import json

def predict_json(project, model, instances, version=None):
    service = googleapiclient.discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']


with open('request.json', 'rt') as f:
    instance = json.load(f)

preds = predict_json('canvas-abacus-235513', 'bsoai_test', instance, 'v1')[0]['predictions']
predictions = decode_predictions(np.array(preds).reshape(1, -1), top=3)[0]

print('Predicted:', )
for _, _class, score in predictions:
    print(score, _class)

