gcloud services enable artifactregistry.googleapis.com
# gcloud artifacts repositories create aita-vm-image \
#     --repository-format=docker \
#     --location=us-central1 # Choose your region
#     --description="Repository for LLM translation service"
gcloud auth configure-docker us-central1-docker.pkg.dev
export PROJECT_ID=$(gcloud config get-value project)
docker build -t us-central1-docker.pkg.dev/$PROJECT_ID/aita-vm-image/gpt:v1 .
docker push us-central1-docker.pkg.dev/$PROJECT_ID/aita-vm-image/gpt:v1
echo "Docker image pushed to: us-central1-docker.pkg.dev/$PROJECT_ID/aita-vm-image/gpt:v1"
# gcloud projects add-iam-policy-binding $PROJECT_ID \
#     --member="serviceAccount:goog-sc-aita-load-balanced-161@snb-ai-translation-agent.iam.gserviceaccount.com" \
#     --role="roles/artifactregistry.reader"