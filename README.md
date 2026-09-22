# ML Platform Repetition Lab

A hands-on MLOps / AI Platform engineering project that follows a machine-learning model from raw data all the way to a running cloud API.

The project was intentionally built as a repetition lab. Instead of learning unrelated tools separately, I repeatedly used the same architecture and added another production layer each time.

```text
MODEL LIFECYCLE
      ↓
SERVING
      ↓
PRODUCTION
```

The progression was:

```text
Project 1
Basic ML System
      ↓
Project 2
Docker
      ↓
Project 3
MLflow
      ↓
Project 4
Kubernetes
      ↓
Project 5
AWS ECR / ECS / Fargate / CloudWatch
```

The purpose of this repository is to demonstrate understanding of the full production ML lifecycle, including architecture, deployment, monitoring, model versioning, troubleshooting, rollback, and cloud operations.

---

# Business Problem

The project predicts whether a piece of equipment is likely to fail.

## Features

The model uses four input features:

```text
temperature
vibration
machine_age
error_count
```

## Target

The prediction target is:

```text
failure = 0
Machine is not predicted to fail

failure = 1
Machine is predicted to fail
```

The ML problem was intentionally kept simple so the main focus could remain on MLOps and platform engineering.

---

# Final Architecture

```text
                         MODEL LIFECYCLE

Historical Data
      │
      ▼
Data Validation
      │
      ▼
Train / Test Split
      │
      ▼
Model Training
model.fit()
      │
      ▼
Model Evaluation
Accuracy
Precision
Recall
F1
      │
      ▼
MLflow Experiment
      │
      ▼
MLflow Run
      │
      ▼
Registered Model
      │
      ▼
Model Version
      │
      ▼
Candidate
      │
      ▼
Approval
      │
      ▼
Champion


                            SERVING

Approved Model
      │
      ▼
Load Model
      │
      ▼
Uvicorn
      │
      ▼
FastAPI
      │
      ▼
Pydantic Validation
      │
      ▼
model.predict()
      │
      ▼
Prediction Response


                           PACKAGING

FastAPI
+
Model
+
Python
+
Dependencies
      │
      ▼
Dockerfile
      │
      ▼
Docker Image


                     KUBERNETES DEPLOYMENT

Docker Image
      │
      ▼
Deployment
      │
      ▼
ReplicaSet
      │
      ▼
Pods
      │
      ▼
Service
      │
      ▼
Health Probes
Logs
Scaling
Rolling Updates
Rollback


                       AWS CLOUD DEPLOYMENT

Docker Image
      │
      ▼
Amazon ECR
      │
      ▼
Amazon ECS
      │
      ▼
Task Definition
      │
      ▼
ECS Service
      │
      ▼
AWS Fargate
      │
      ▼
FastAPI Prediction API
      │
      ▼
Amazon CloudWatch Logs
```

---

# Project 1 — Basic Production ML System

The first repetition built the smallest complete ML system.

```text
Data
 ↓
Validate
 ↓
Train
 ↓
Evaluate
 ↓
Save Model Artifact
 ↓
FastAPI
 ↓
Pydantic
 ↓
Prediction
 ↓
Health / Logs / Metrics
```

## Step 1 — Data

For the lab, a synthetic equipment dataset was generated.

In a real company, an MLOps or AI Platform engineer would normally receive feature definitions and data requirements from sources such as:

- Data scientists
- Data engineers
- Business/domain teams
- SQL databases
- Kafka
- Data warehouses
- Data lakes
- Feature platforms
- Data contracts

The platform engineer should not randomly invent business rules for production data.

---

# Step 2 — Data Validation

Before training, the data was validated.

Validation checked things such as:

- Required columns
- Numeric data types
- Missing values
- Valid ranges
- Valid target values

A deliberately bad row was inserted into the dataset.

The validation pipeline correctly rejected it.

```text
Bad Data
   ↓
Validation
   ↓
FAIL
   ↓
Training does not run
```

This demonstrated an important production principle:

> Bad data should fail early instead of silently reaching model training.

---

# Step 3 — Training

The model used:

```text
X = features

y = target
```

Training happened using:

```python
model.fit(X_train, y_train)
```

Conceptually:

```text
Historical Examples
        ↓
Features + Known Answers
        ↓
model.fit()
        ↓
Trained Model
```

---

# model.fit() vs model.predict()

```text
model.fit()
→ training
→ model learns from historical examples

model.predict()
→ inference
→ trained model predicts on new inputs
```

This distinction became one of the most important concepts in the project.

---

# Step 4 — Train/Test Split

The dataset was divided into approximately:

```text
80% training data

20% test data
```

The model trained on the training data.

The model was then evaluated on data it had not already seen.

The reason is that evaluating on the same data used for training can make the model appear better than it really is.

A useful analogy:

> Giving a student the exact exam questions and answers before testing them does not prove that they learned the subject.

---

# Step 5 — Model Evaluation

The first simple model produced accuracy around the mid-60% range.

One important lesson was:

> 65% accuracy does not automatically mean a model is good.

Model quality depends on more than accuracy.

Metrics considered included:

```text
Accuracy
Precision
Recall
F1
```

The correct metric depends on the business problem.

For equipment failure prediction, recall became particularly important because missing a real equipment failure may be expensive.

```text
High Recall
→ fewer real failures missed
```

---

# Step 6 — Save the Model Artifact

The trained model was saved using Joblib.

```text
Training
   ↓
Trained Model
   ↓
joblib.dump()
   ↓
equipment_failure_model.joblib
```

The serving application later loaded it using:

```python
joblib.load(...)
```

This is important because we do not want to retrain the model every time a client asks for a prediction.

---

# Training vs Inference

```text
TRAINING

Historical Data
      ↓
model.fit()
      ↓
Model Artifact


INFERENCE

New Data
   ↓
Loaded Model
   ↓
model.predict()
   ↓
Prediction
```

---

# Serving Layer

The model was exposed through FastAPI.

The request path became:

```text
Client
  │
  │ HTTP POST
  ▼
Uvicorn
  │
  ▼
FastAPI
  │
  ▼
Pydantic
  │
  ▼
model.predict()
  │
  ▼
Prediction
  │
  ▼
HTTP Response
  │
  ▼
Client
```

---

# Uvicorn

Uvicorn runs the FastAPI ASGI application.

It listens for network requests and passes them to FastAPI.

Example:

```text
Uvicorn
listening on port 8000
```

---

# FastAPI

FastAPI exposes the model as an HTTP API.

Two important endpoints were created:

```text
GET /health

POST /predict
```

---

# Pydantic

Pydantic validates incoming request data.

Example:

```text
Expected:

temperature = number

Received:

temperature = "abc"

Result:

Request rejected
```

The request is rejected before the bad data reaches the ML model.

---

# Where Does Inference Happen?

Inference happens here:

```python
model.predict(...)
```

Pydantic validation is not inference.

FastAPI routing is not inference.

Loading the model is not inference.

The prediction call itself is inference.

---

# /health Endpoint

The health endpoint answers questions such as:

```text
Is the application alive?

Did the model load?

Which model version is running?

Which application release is running?
```

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1"
}
```

---

# /predict Endpoint

The predict endpoint performs inference.

Example request:

```json
{
  "temperature": 90,
  "vibration": 4.2,
  "machine_age": 12,
  "error_count": 7
}
```

Example response:

```json
{
  "failure": 1
}
```

---

# Logs and Metrics

Production observability was introduced early.

Logs can answer:

```text
What happened?

When did it happen?

Which endpoint was called?

Was the request successful?

What error occurred?
```

Metrics answer questions such as:

```text
How many requests?

How long do requests take?

How many errors?

How much traffic?
```

---

# Project 2 — Docker

The FastAPI application was then containerized.

```text
Source Code
+
Model
+
Python
+
Dependencies
      ↓
Dockerfile
      ↓
Docker Image
      ↓
Docker Container
```

---

# Docker Image vs Container

```text
Docker Image
→ packaged application blueprint

Docker Container
→ running instance of that image
```

An image does not become a running application until a container is started from it.

---

# Dockerfile

The Dockerfile defined:

- Python version
- Working directory
- Python dependencies
- Source code
- Model artifact
- Application port
- Uvicorn startup command

---

# Why Docker?

Docker provides a consistent environment.

Instead of:

```text
Works on my laptop
```

we package:

```text
Same Python
Same dependencies
Same source code
Same model
Same startup command
```

---

# Docker Troubleshooting

## docker exec

`docker exec` requires:

```text
container + command
```

Example:

```bash
docker exec equipment-api ls
```

It does not execute commands against an image.

---

# docker ps vs docker ps -a

```bash
docker ps
```

shows running containers.

```bash
docker ps -a
```

shows running and stopped containers.

---

# Docker Build Context Problem

One Docker build failed because Docker could not find:

```text
models/
```

The problem was the Docker build context.

Important lesson:

> Docker COPY paths are relative to the build context, not simply the location of the Dockerfile.

---

# Docker BuildKit Cache Failure

Later a build failed with a missing parent snapshot error.

This was not an application-code error.

The Docker builder cache had become stale/corrupted.

The solution involved:

```text
Prune BuildKit cache
        ↓
Rebuild without cache
```

This demonstrated that infrastructure tooling itself can fail.

---

# Project 3 — MLflow

The next repetition introduced model lifecycle management.

```text
Train
 ↓
Evaluate
 ↓
MLflow Experiment
 ↓
MLflow Run
 ↓
Registered Model
 ↓
Model Version
 ↓
Candidate
 ↓
Approval
 ↓
Champion
 ↓
Serving
```

---

# MLflow Experiment

An experiment groups related training attempts.

Example:

```text
equipment-failure experiment
        │
        ├── Run 1
        ├── Run 2
        ├── Run 3
        └── Run 4
```

---

# MLflow Run

A run represents one execution of training.

A run can record:

```text
Parameters

Metrics

Model artifact

Tags

Runtime information
```

---

# Registered Model

The registered model provides a stable logical name.

Example:

```text
equipment-failure-model
```

---

# Model Versions

Every registered trained model creates another model version.

Example:

```text
equipment-failure-model
       │
       ├── Version 1
       ├── Version 2
       ├── Version 3
       └── Version 4
```

---

# Candidate

A newly trained model was treated as:

```text
candidate
```

This was an important concept:

```text
New model
≠ automatically production model
```

---

# Approval

Before a model should be promoted to production, a company may require:

- Model validation
- Metric review
- Business approval
- Change ticket
- Compliance review
- Testing
- Peer review

The lab simulated that approval boundary.

---

# Champion Alias

The selected production model was assigned:

```text
champion
```

Instead of hard-coding:

```text
version 4
```

the application can conceptually request:

```text
equipment-failure-model@champion
```

This allows the production model to change without rewriting the application code.

---

# Model Promotion

Example:

```text
Version 1
   ↓
Champion

Version 4
   ↓
Candidate
   ↓
Approval
   ↓
Champion
```

---

# Model Rollback

If the newer model performs badly:

```text
Version 4
Champion
   ↓
Problem
   ↓
Move champion alias
   ↓
Version 1
Champion
```

---

# Model Rollback vs Application Rollback

These are different.

```text
MLflow rollback
→ changes ML model

Kubernetes rollback
→ changes application/container release
```

The model can fail while the API works.

The API can fail while the model remains valid.

---

# MLflow Troubleshooting

## Serialization Security Issue

MLflow/Skops initially rejected a Scikit-learn tree type as untrusted.

The model had been created locally and the specific required type was reviewed and explicitly trusted.

Lesson:

> Serialized ML models should be treated as security-sensitive artifacts.

---

# MLflow Connection Refused

The model promotion script failed with:

```text
Connection refused
127.0.0.1:5000
```

Root cause:

```text
MLflow server was not running
```

The promotion script depended on the MLflow registry service.

This was an infrastructure availability issue, not a model-quality issue.

---

# Champion Changed but API Still Served Old Version

The MLflow champion alias was changed.

However, the running FastAPI process still had the old model loaded in memory.

The application had to be restarted.

Important lesson:

```text
Registry state
≠
running process state
```

---

# Project 4 — Kubernetes

The Dockerized application was deployed into a local Kind Kubernetes cluster.

```text
Docker Image
      ↓
Deployment
      ↓
ReplicaSet
      ↓
Pods
      ↓
Service
```

---

# Why Kubernetes?

Docker runs containers.

Kubernetes manages desired state.

Example:

```yaml
replicas: 2
```

means:

> Kubernetes should keep two copies of the workload running.

---

# Deployment

The Deployment defines the desired application state and rollout behavior.

---

# ReplicaSet

The ReplicaSet maintains the required number of pods.

---

# Pod

The Pod is the Kubernetes workload unit that contains the application container.

---

# Service

Clients should not directly depend on changing Pod IP addresses.

A Service provides a stable network endpoint.

```text
Client
   ↓
Service
   ↓
Pods
```

---

# ClusterIP

ClusterIP provides internal Kubernetes networking for the service.

---

# Port Forwarding

During local development:

```text
localhost
   ↓
kubectl port-forward
   ↓
Kubernetes Service
   ↓
Pod
   ↓
FastAPI
```

---

# Kubernetes Self-Healing

A Pod was manually deleted.

Before:

```text
Desired = 2
Actual  = 2
```

After deletion:

```text
Desired = 2
Actual  = 1
```

Kubernetes detected the mismatch and created a replacement.

```text
Desired State
      ↓
Kubernetes Controller
      ↓
Actual State corrected
```

---

# Kubernetes Scaling

The number of replicas was changed.

Example:

```text
2 Pods
  ↓
3 Pods
```

This demonstrated horizontal scaling.

---

# Rolling Update

A new application release was deployed gradually.

```text
Old Pods
   ↓
New Pod starts
   ↓
New Pod becomes ready
   ↓
Old Pod removed
   ↓
Repeat
```

This reduces downtime during deployment.

---

# Kubernetes Rollback

If a new release is broken:

```text
k8s-v2
   ↓
Problem
   ↓
kubectl rollout undo
   ↓
k8s-v1
```

---

# Readiness vs Liveness

```text
Readiness
→ should this Pod receive traffic?

Liveness
→ should Kubernetes restart this container?
```

---

# Kubernetes Troubleshooting Commands

Important commands practiced:

```bash
kubectl get deployments

kubectl get pods

kubectl get services

kubectl describe pod <pod>

kubectl logs <pod>

kubectl logs <pod> --previous

kubectl rollout status deployment/<deployment>

kubectl rollout history deployment/<deployment>
```

---

# Model Version vs Release Version

The project tracked two separate concepts.

```text
model_version
→ trained ML model version

release_version
→ API/container/application version
```

Example:

```text
model_version   = 4

release_version = k8s-v2
```

This makes troubleshooting easier.

---

# Project 5 — AWS Cloud Deployment

The final repetition moved the containerized API into AWS.

```text
Docker Image
      ↓
Amazon ECR
      ↓
Amazon ECS
      ↓
Task Definition
      ↓
ECS Service
      ↓
AWS Fargate
      ↓
FastAPI API
      ↓
CloudWatch Logs
```

---

# AWS Authentication

AWS CLI was installed in GitHub Codespaces.

Initially, root credentials were being used.

A safer workflow was configured using:

```text
IAM Identity Center
      ↓
User
      ↓
Permission Set
      ↓
MFA
      ↓
SSO
      ↓
Temporary CLI Credentials
```

The CLI profile used:

```text
ml-platform-lab
```

---

# Amazon ECR

ECR stored the Docker image.

```text
Docker Image
      ↓
docker push
      ↓
Amazon ECR
      ↓
equipment-failure-api:cloud-v1
```

Important distinction:

```text
ECR
→ stores images

ECR
→ does NOT run applications
```

---

# ECR Authentication Problem

An ECR login initially failed and Docker attempted to contact:

```text
registry-1.docker.io
```

Root cause:

```text
$ECR_REGISTRY
was empty
```

The variable was populated and verified before retrying authentication.

Lesson:

> Check resolved environment variables instead of assuming they are populated.

---

# ECS

Amazon ECS was used for AWS-native container orchestration.

Comparison:

```text
Kubernetes                  Amazon ECS

Deployment              →   ECS Service

Pod                     →   ECS Task

replicas                →   desiredCount

Container configuration →   Task Definition

kubectl logs            →   CloudWatch Logs
```

---

# ECS Task Definition

The Task Definition describes how the container should run.

It included:

- Docker image
- CPU
- Memory
- Port
- Environment variables
- Execution role
- Logging
- Fargate compatibility
- Networking mode

Conceptually:

```text
Task Definition
=
Container Runtime Blueprint
```

---

# AWS Fargate

Fargate provides managed compute.

```text
ECS
→ orchestration

Fargate
→ compute
```

This allowed the application to run without manually managing EC2 worker instances.

---

# AWS Networking

The training environment used:

- Default VPC
- Default subnet
- Security group
- Public IPv4
- Port 8000
- awsvpc network mode

For a production system, a stronger architecture might include:

- Private subnets
- Application Load Balancer
- TLS
- Restricted inbound rules
- Least-privilege IAM
- Autoscaling

---

# First AWS Cloud Prediction

The Fargate task successfully responded to:

```text
GET /health
```

and:

```text
POST /predict
```

Example request:

```json
{
  "temperature": 90,
  "vibration": 4.2,
  "machine_age": 12,
  "error_count": 7
}
```

Example response:

```json
{
  "failure": 1,
  "model_version": "1",
  "release_version": "cloud-v1"
}
```

This proved the complete cloud path:

```text
Client
  ↓
Public IP
  ↓
Fargate Network Interface
  ↓
Container
  ↓
Uvicorn
  ↓
FastAPI
  ↓
Pydantic
  ↓
model.predict()
  ↓
Prediction Response
```

---

# ECS Service

The first Fargate deployment was a one-off task.

It was then replaced by an ECS Service.

```text
desiredCount = 1
```

The service reached:

```text
Desired = 1

Running = 1

Pending = 0
```

This is similar to Kubernetes desired replicas.

---

# CloudWatch Logs

The application sent logs to Amazon CloudWatch.

CloudWatch showed:

- Model loading
- Uvicorn startup
- Health requests
- Prediction requests
- Status codes
- Request duration
- Application shutdown

Logging progression throughout the project:

```text
Local Python
→ terminal

Docker
→ docker logs

Kubernetes
→ kubectl logs

AWS
→ CloudWatch Logs
```

---

# AWS Region Problem

An ECS Task Definition was initially created in:

```text
us-east-1
```

while the intended cloud environment was:

```text
us-east-2
```

The Task Definition and ECS Cluster were verified/recreated in the correct region.

Important lesson:

> AWS resources are often regional. Always verify the region when a resource appears missing or inconsistent.

---

# AWS SSO Expiration

Later, AWS CLI returned an OAuth authorization error.

Root cause:

```text
temporary SSO authentication expired
```

The session was refreshed using:

```bash
aws sso login \
  --profile ml-platform-lab \
  --use-device-code
```

Important lesson:

```text
AWS workload health
≠
engineer's local CLI authentication state
```

The ECS application could still be healthy even if the local CLI session had expired.

---

# AWS Cost Control

At the end of the lab, the ECS Service was scaled to:

```text
Desired = 0

Running = 0

Pending = 0
```

This stopped ECS from maintaining a running Fargate task before final resource cleanup.

---

# Production Troubleshooting Strategy

Example incident:

> A user says the prediction API is not working.

My investigation process:

```text
1. Determine scope

Is this one user?
Multiple users?
Everyone?


2. Check health

Does /health respond?


3. Check platform state

Docker container running?
Kubernetes Pod running?
ECS Task running?


4. Check logs

What error occurred?
When?
Which endpoint?


5. Check networking

Can the client reach the service?


6. Check request validation

Is Pydantic rejecting input?


7. Check model

Did the model load?
Does inference work?


8. Check versions

Which model version is running?

Which application release is running?


9. Roll back

Is there a known-good model or application release?
```

---

# Repository Structure

```text
ml-platform-repetition-lab/
│
├── data/
│
├── models/
│
├── src/
│   ├── generate_data.py
│   ├── validate.py
│   ├── train.py
│   ├── train_mlflow.py
│   ├── promote_model.py
│   ├── api.py
│   └── api_k8s.py
│
├── k8s/
│   ├── namespace.yaml
│   ├── deployment.yaml
│   └── service.yaml
│
├── tests/
│
├── Dockerfile
├── Dockerfile.k8s
├── requirements.txt
├── requirements-serving.txt
└── README.md
```

---

# Important Interview Questions

## What is the difference between training and inference?

```text
Training
→ model learns from historical data

Inference
→ trained model predicts on new data
```

---

## Why validate data before training?

To prevent invalid or malformed data from producing unreliable models.

---

## Why do we split training and test data?

To evaluate the model on data it did not already see during training.

---

## Why is accuracy alone not enough?

Because accuracy can hide problems such as class imbalance and may not reflect the business cost of different mistakes.

---

## If missing a machine failure is expensive, what metric becomes important?

Recall for the failure class.

---

## Why save the trained model?

So inference can use the trained artifact without retraining for every request.

---

## What does Pydantic do?

Validates API input before it reaches the ML model.

---

## Where does inference happen?

At:

```python
model.predict(...)
```

---

## What does Uvicorn do?

Runs the FastAPI ASGI application.

---

## Why Docker?

To package the runtime consistently.

---

## Image vs Container?

```text
Image
→ package

Container
→ running instance
```

---

## Why MLflow?

For:

- Experiment tracking
- Metrics
- Parameters
- Artifacts
- Model versions
- Promotion
- Rollback
- Registry governance

---

## What is the difference between an MLflow Experiment and Run?

```text
Experiment
→ group of related runs

Run
→ one training execution
```

---

## Registered Model vs Model Version?

```text
Registered Model
→ stable model name

Model Version
→ one specific model artifact
```

---

## Candidate vs Champion?

```text
Candidate
→ being evaluated

Champion
→ currently approved model alias
```

---

## Why Kubernetes?

Docker runs a container.

Kubernetes maintains desired application state and supports:

- self-healing
- replicas
- networking
- scaling
- rolling updates
- rollback

---

## Deployment vs ReplicaSet vs Pod?

```text
Deployment
→ manages application rollout

ReplicaSet
→ maintains desired Pod count

Pod
→ runs application container
```

---

## What does Kubernetes Service do?

Provides a stable network endpoint for Pods.

---

## Readiness vs Liveness?

```text
Readiness
→ receive traffic?

Liveness
→ restart container?
```

---

## ECR vs ECS vs Fargate?

```text
Amazon ECR
→ stores container images

Amazon ECS
→ orchestrates containers

AWS Fargate
→ provides compute to run ECS Tasks
```

---

## What is an ECS Task Definition?

A blueprint describing how the container should run.

---

## ECS Task vs ECS Service?

```text
Task
→ one running workload instance

Service
→ maintains desired number of Tasks
```

---

# Security Lessons

Never commit:

- AWS Access Keys
- Secret Keys
- AWS Session Tokens
- Passwords
- MFA codes
- SSO device codes
- `.aws/` credential files
- `.env` secrets

AWS access for this project used IAM Identity Center and temporary SSO credentials.

---

# Future Enhancements

The core project is complete.

Possible future additions:

- ECS self-healing test
- ECS scaling
- ECS deployment rollback
- Application Load Balancer
- Private subnets
- Autoscaling
- Least-privilege IAM
- GitHub Actions CI/CD
- Terraform
- Managed MLflow infrastructure
- S3 artifact storage
- Prometheus / Grafana
- Production alerting
- Model drift monitoring
- Data drift monitoring

---

# Interview Summary

A concise explanation of this project:

> I built an equipment-failure prediction platform and repeatedly extended the same production architecture. I started with data validation, model training, evaluation, Joblib serialization, and FastAPI serving. I containerized the service with Docker, added MLflow experiment tracking and model registry promotion, deployed it to Kubernetes with health probes, desired-state management, scaling, rolling updates and rollback, and finally pushed the container to Amazon ECR and ran it on ECS/Fargate with CloudWatch logging. I also deliberately troubleshot failures involving invalid data, Docker build contexts, BuildKit cache, MLflow connectivity, Kubernetes workloads, AWS SSO authentication, ECR configuration, AWS networking, and region mismatches.

---

# Main Learning Outcome

The main lesson was not memorizing commands.

The reusable engineering thought process is:

```text
MODEL LIFECYCLE

Where does the data come from?

How is it validated?

How is the model trained?

How is it evaluated?

How is it approved?

How is it versioned?


SERVING

How does the client reach the model?

How is input validated?

Where does inference happen?

How is the service packaged?


PRODUCTION

How do we know it is healthy?

Where are the logs?

Which model version is running?

Which application release is running?

How does the platform recover?

How does it scale?

How do we roll back?
```

The same architecture was repeated from local Python through Docker, MLflow, Kubernetes, and finally AWS ECS/Fargate.

