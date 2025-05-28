#!/bin/bash
set -e

# Enable required services
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# Configure Docker authentication
gcloud auth configure-docker us-central1-docker.pkg.dev

# Get project ID
export PROJECT_ID=$(gcloud config get-value project)
echo "Using project: $PROJECT_ID"

# Create the repository if it doesn't exist
gcloud artifacts repositories create aita-vm-image \
    --repository-format=docker \
    --location=us-central1 \
    --description="Docker repository for AITA VM images" || echo "Repository already exists"

# Grant permissions to Cloud Build service account
# Get the project number for the Cloud Build service account
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
CLOUD_BUILD_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${CLOUD_BUILD_SA}" \
    --role="roles/artifactregistry.writer"

# Submit the build
gcloud builds submit --tag us-central1-docker.pkg.dev/$PROJECT_ID/aita-vm-image/gpt:v1

echo "Docker image pushed to: us-central1-docker.pkg.dev/$PROJECT_ID/aita-vm-image/gpt:v1"