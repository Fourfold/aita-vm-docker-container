#!/bin/bash
set -e

rm -rf buildspec.yml

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "AWS CLI not found. Please install it first."
    exit 1
fi

# Using Docker Hub without authentication (subject to rate limits)
# If you need authenticated access, you can add credentials here:
# export DOCKERHUB_USERNAME="your-dockerhub-username" 
# export DOCKERHUB_TOKEN="your-dockerhub-access-token"

# Get AWS account ID and region
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=${AWS_REGION:-us-east-1}
echo "Using AWS account: $AWS_ACCOUNT_ID in region: $AWS_REGION"

# Repository names
REPO_NAME="paddle-vm-image"
BASE_IMAGE_REPO="nvidia-cuda-base"
IMAGE_NAME="paddle_server"
IMAGE_TAG="v1"
BASE_IMAGE_TAG="11.8.0-cudnn8-runtime-ubuntu20.04"
ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME:$IMAGE_TAG"
BASE_ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$BASE_IMAGE_REPO:$BASE_IMAGE_TAG"

# Create ECR repositories if they don't exist
echo "Creating ECR repositories..."
aws ecr create-repository \
    --repository-name $REPO_NAME \
    --region $AWS_REGION \
    --image-scanning-configuration scanOnPush=true || echo "Main repository already exists"

aws ecr create-repository \
    --repository-name $BASE_IMAGE_REPO \
    --region $AWS_REGION \
    --image-scanning-configuration scanOnPush=true || echo "Base image repository already exists"

# Create buildspec.yml for CodeBuild if it doesn't exist
if [ ! -f "buildspec.yml" ]; then
    echo "Creating buildspec.yml for CodeBuild..."
    cat > buildspec.yml << EOF
version: 0.2

phases:
  pre_build:
    commands:
      - echo Logging in to Amazon ECR...
      - aws ecr get-login-password --region \$AWS_DEFAULT_REGION | docker login --username AWS --password-stdin \$AWS_ACCOUNT_ID.dkr.ecr.\$AWS_DEFAULT_REGION.amazonaws.com
      - echo Checking if base image needs to be pulled from Docker Hub...
      - aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws
      - echo Checking if base image exists in private ECR...
      - |
        if ! aws ecr describe-images --repository-name \$BASE_IMAGE_REPO --image-ids imageTag=\$BASE_IMAGE_TAG --region \$AWS_DEFAULT_REGION >/dev/null 2>&1; then
          echo "Base image not found in private ECR, pulling and pushing..."
          docker pull nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
          docker tag nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 \$AWS_ACCOUNT_ID.dkr.ecr.\$AWS_DEFAULT_REGION.amazonaws.com/\$BASE_IMAGE_REPO:\$BASE_IMAGE_TAG
          docker push \$AWS_ACCOUNT_ID.dkr.ecr.\$AWS_DEFAULT_REGION.amazonaws.com/\$BASE_IMAGE_REPO:\$BASE_IMAGE_TAG
          echo "Base image pushed to private ECR"
        else
          echo "Base image already exists in private ECR"
        fi
      - pip install --upgrade pip
  build:
    commands:
      - echo Build started on \`date\`
      - echo Building the Docker image using private ECR NVIDIA CUDA base image...
      - docker build --build-arg AWS_ACCOUNT_ID=\$AWS_ACCOUNT_ID --build-arg AWS_REGION=\$AWS_DEFAULT_REGION -t \$IMAGE_REPO_NAME:\$IMAGE_TAG .
      - docker tag \$IMAGE_REPO_NAME:\$IMAGE_TAG \$AWS_ACCOUNT_ID.dkr.ecr.\$AWS_DEFAULT_REGION.amazonaws.com/\$IMAGE_REPO_NAME:\$IMAGE_TAG
  post_build:
    commands:
      - echo Build completed on \`date\`
      - echo Pushing the Docker image...
      - docker push \$AWS_ACCOUNT_ID.dkr.ecr.\$AWS_DEFAULT_REGION.amazonaws.com/\$IMAGE_REPO_NAME:\$IMAGE_TAG
EOF
fi

# Create or update CodeBuild project
PROJECT_NAME="paddle-vm-image-build"
echo "Creating/updating CodeBuild project..."

# Create service role for CodeBuild if it doesn't exist
ROLE_NAME="CodeBuildServiceRole-$PROJECT_NAME"
aws iam create-role \
    --role-name $ROLE_NAME \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "codebuild.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }' || echo "Role already exists"

# Attach necessary policies
aws iam attach-role-policy \
    --role-name $ROLE_NAME \
    --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess || true

aws iam attach-role-policy \
    --role-name $ROLE_NAME \
    --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser || true

# Create and attach S3 policy for the source bucket
S3_POLICY_NAME="CodeBuildS3Access-$PROJECT_NAME"
cat > s3-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:GetObjectVersion"
            ],
            "Resource": "arn:aws:s3:::codebuild-source-$AWS_ACCOUNT_ID-$AWS_REGION/*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket"
            ],
            "Resource": "arn:aws:s3:::codebuild-source-$AWS_ACCOUNT_ID-$AWS_REGION"
        }
    ]
}
EOF

# Create the policy and attach it to the role
aws iam create-policy \
    --policy-name $S3_POLICY_NAME \
    --policy-document file://s3-policy.json || echo "Policy already exists"

aws iam attach-role-policy \
    --role-name $ROLE_NAME \
    --policy-arn arn:aws:iam::$AWS_ACCOUNT_ID:policy/$S3_POLICY_NAME || true

# Clean up policy file
rm -f s3-policy.json

# Handle Git LFS files before creating archive
echo "Checking for Git LFS files..."
if [ -f "gemmax2_9b_finetuned/adapter_model.safetensors" ]; then
    ADAPTER_SIZE=$(stat -c%s "gemmax2_9b_finetuned/adapter_model.safetensors" 2>/dev/null || stat -f%z "gemmax2_9b_finetuned/adapter_model.safetensors" 2>/dev/null || echo "0")
    echo "Current adapter_model.safetensors size: $ADAPTER_SIZE bytes"
    
    if [ "$ADAPTER_SIZE" -lt 1000000 ]; then  # Less than 1MB suggests it's a pointer file
        echo "Detected small file - likely a Git LFS pointer. Attempting to download actual file..."
        
        # Check if git-lfs is installed
        if command -v git-lfs >/dev/null 2>&1; then
            echo "Git LFS is installed, attempting to pull files..."
            git lfs pull
            
            # Check size again
            NEW_SIZE=$(stat -c%s "gemmax2_9b_finetuned/adapter_model.safetensors" 2>/dev/null || stat -f%z "gemmax2_9b_finetuned/adapter_model.safetensors" 2>/dev/null || echo "0")
            echo "New adapter_model.safetensors size: $NEW_SIZE bytes"
            
            if [ "$NEW_SIZE" -lt 1000000 ]; then
                echo "WARNING: Git LFS pull did not download the file. The build may fail."
                echo "Please ensure you have access to the Git LFS storage and run 'git lfs pull' manually."
            else
                echo "Successfully downloaded Git LFS files!"
            fi
        else
            echo "WARNING: Git LFS not installed. Installing..."
            # Try to install git-lfs
            if command -v brew >/dev/null 2>&1; then
                brew install git-lfs
            elif command -v apt-get >/dev/null 2>&1; then
                sudo apt-get update && sudo apt-get install -y git-lfs
            elif command -v yum >/dev/null 2>&1; then
                sudo yum install -y git-lfs
            else
                echo "ERROR: Cannot install git-lfs automatically. Please install it manually and run 'git lfs pull'"
                exit 1
            fi
            
            git lfs install
            git lfs pull
        fi
    else
        echo "File size looks correct (>1MB), proceeding with build..."
    fi
else
    echo "adapter_model.safetensors not found. This may cause build issues."
fi

# Create a zip file of the current directory for upload
echo "Creating source archive..."
zip -r source.zip . -x "*.git*" "*.zip"

# Upload to S3 bucket (create bucket if needed)
BUCKET_NAME="codebuild-source-$AWS_ACCOUNT_ID-$AWS_REGION"
aws s3 mb s3://$BUCKET_NAME --region $AWS_REGION || echo "Bucket already exists"
aws s3 cp source.zip s3://$BUCKET_NAME/paddle-vm-source.zip

# Create source configuration JSON file for S3 source
cat > source-config.json << EOF
{
    "type": "S3",
    "location": "$BUCKET_NAME/paddle-vm-source.zip",
    "buildspec": "buildspec.yml"
}
EOF

# Check if project exists first
echo "Checking if project $PROJECT_NAME exists..."
if aws codebuild describe-projects --names $PROJECT_NAME >/dev/null 2>&1; then
    echo "Project exists, updating..."
    if ! aws codebuild update-project \
        --name $PROJECT_NAME \
        --source file://source-config.json \
        --artifacts type=NO_ARTIFACTS \
        --environment type=LINUX_CONTAINER,image=aws/codebuild/amazonlinux2-x86_64-standard:3.0,computeType=BUILD_GENERAL1_LARGE,privilegedMode=true \
        --service-role arn:aws:iam::$AWS_ACCOUNT_ID:role/$ROLE_NAME; then
        echo "Error: Failed to update CodeBuild project"
        exit 1
    fi
    echo "Project updated successfully"
else
    echo "Creating new project..."
    if ! aws codebuild create-project \
        --name $PROJECT_NAME \
        --source file://source-config.json \
        --artifacts type=NO_ARTIFACTS \
        --environment type=LINUX_CONTAINER,image=aws/codebuild/amazonlinux2-x86_64-standard:3.0,computeType=BUILD_GENERAL1_LARGE,privilegedMode=true \
        --service-role arn:aws:iam::$AWS_ACCOUNT_ID:role/$ROLE_NAME; then
        echo "Error: Failed to create CodeBuild project"
        echo "This might be because the project already exists. Trying to update instead..."
        
        # Try updating if create failed due to existing project
        if ! aws codebuild update-project \
            --name $PROJECT_NAME \
            --source file://source-config.json \
            --artifacts type=NO_ARTIFACTS \
            --environment type=LINUX_CONTAINER,image=aws/codebuild/amazonlinux2-x86_64-standard:3.0,computeType=BUILD_GENERAL1_LARGE,privilegedMode=true \
            --service-role arn:aws:iam::$AWS_ACCOUNT_ID:role/$ROLE_NAME; then
            echo "Error: Failed to both create and update CodeBuild project"
            exit 1
        fi
        echo "Project updated successfully (after failed create)"
    else
        echo "Project created successfully"
    fi
fi

# Start the build
echo "Starting CodeBuild..."
BUILD_ID=$(aws codebuild start-build \
    --project-name $PROJECT_NAME \
    --environment-variables-override name=AWS_DEFAULT_REGION,value=$AWS_REGION name=AWS_ACCOUNT_ID,value=$AWS_ACCOUNT_ID name=IMAGE_REPO_NAME,value=$REPO_NAME name=IMAGE_TAG,value=$IMAGE_TAG name=BASE_IMAGE_REPO,value=$BASE_IMAGE_REPO name=BASE_IMAGE_TAG,value=$BASE_IMAGE_TAG \
    --query 'build.id' --output text)

# Clean up local files
rm -f source.zip source-config.json

echo "Build started with ID: $BUILD_ID"
echo "You can monitor the build at: https://console.aws.amazon.com/codesuite/codebuild/projects/$PROJECT_NAME/build/$BUILD_ID"
echo "Or run: aws codebuild batch-get-builds --ids $BUILD_ID"

# Optionally wait for build to complete and show status
echo "Waiting for build to complete..."
aws codebuild batch-get-builds --ids $BUILD_ID --query 'builds[0].buildStatus' --output text

echo "Docker image will be available at: $ECR_URI"
