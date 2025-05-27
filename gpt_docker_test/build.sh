gcloud services enable artifactregistry.googleapis.com\
gcloud auth configure-docker us-central1-docker.pkg.dev
export PROJECT_ID=$(gcloud config get-value project)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:836454816267@cloudbuild.gserviceaccount.com" \
    --role="roles/artifactregistry.writer"
gcloud builds submit --tag us-central1-docker.pkg.dev/$PROJECT_ID/aita-vm-image/gpt:v1
echo "Docker image pushed to: us-central1-docker.pkg.dev/$PROJECT_ID/aita-vm-image/gpt:v1"