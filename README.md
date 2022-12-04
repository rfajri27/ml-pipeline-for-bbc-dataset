![Python](https://img.shields.io/badge/python-3.8+-blue.svg) [![TFX](https://img.shields.io/badge/TFX-1.7-orange)](https://www.tensorflow.org/tfx)
# Build a ML Pipeline for BBC Dataset using TensorFlow Extended

<div align="center">
  <img src="https://www.tensorflow.org/static/tfx/guide/images/prog_fin.png" width="100%" />

  <h4>
  Machine learning pipeline built with TensorFlow Extended (TFX) and Apache Beam as a pipeline orchestrator in order to handle BBC dataset.
  </h4>

</div>

## Project structure
```
project
├───data
├───pipeline
│   └───bbc-pipeline
│       ├───CsvExampleGen
│       ├───Evaluator
│       ├───ExampleValidator
│       ├───Pusher
│       ├───SchemaGen
│       ├───StatisticsGen
│       ├───Trainer
│       ├───Transform
│       ├───Tuner
│       └───_wheels
├───images
├───modules
├───monitoring
└───serving_model_dir
    └───bbc-model
```

## Dataset
I used a public dataset from the BBC comprised of 2225 articles, each labeled under one of 5 categories: business, entertainment, politics, sport, and tech. The data can be found [here](http://mlg.ucd.ie/datasets/bbc.html).

## Machine Learning Pipeline
Used TFX to build a machine learning pipeline & Apache Beam as a pipeline orchestrator. This pipeline consists of 9 components with specific tasks. The following table describes the components of this pipeline in order:

| Pipeline component | Description |
| ----------- | ----------- |
| CsvExampleGen | |
| StatisticsGen | |
| SchemaGen | |
| ExampleValidator | |
| Transform | |
| Tuner | |
| Trainer | |
| Evaluator | |
| Pusher | |

## Deployment
Build model serving using [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) that runs on the [Cloud Run](https://cloud.google.com/run). The model serving can be found [here](https://bbc-prediction-shzbom57sq-et.a.run.app/v1/models/bbc-model). 

## Monitoring
Used Prometheus to monitor the model serving.