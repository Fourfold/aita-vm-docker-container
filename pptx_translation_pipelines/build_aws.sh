#!/bin/bash
set -e

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "AWS CLI not found. Please install it first."
    exit 1
fi

# # Set Docker Hub credentials
# export DOCKERHUB_USERNAME="your-username"
# export DOCKERHUB_TOKEN="your-access-token"

# Get AWS account ID and region
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=${AWS_REGION:-us-east-1}
echo "Using AWS account: $AWS_ACCOUNT_ID in region: $AWS_REGION"

# Repository name
REPO_NAME="aita-vm-image"
IMAGE_NAME="pptx_translation_pipelines"
IMAGE_TAG="v1"
ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME:$IMAGE_TAG"

# Create ECR repository if it doesn't exist
echo "Creating ECR repository..."
aws ecr create-repository \
    --repository-name $REPO_NAME \
    --region $AWS_REGION \
    --image-scanning-configuration scanOnPush=true || echo "Repository already exists"

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
      - echo Logging in to Docker Hub to avoid rate limits...
      - |
        if [ ! -z "\$DOCKERHUB_USERNAME" ] && [ ! -z "\$DOCKERHUB_TOKEN" ]; then
          echo "Authenticating with Docker Hub..."
          echo \$DOCKERHUB_TOKEN | docker login --username \$DOCKERHUB_USERNAME --password-stdin
        else
          echo "Docker Hub credentials not provided. Using public ECR for NVIDIA images..."
          aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws
        fi
  build:
    commands:
      - echo Build started on \`date\`
      - echo Building the Docker image...
      - |
        if [ ! -z "\$DOCKERHUB_USERNAME" ] && [ ! -z "\$DOCKERHUB_TOKEN" ]; then
          BASE_IMAGE="nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04"
          echo "Using Docker Hub NVIDIA image..."
          # Use original Dockerfile with Docker Hub base image
          docker build -t \$IMAGE_REPO_NAME:\$IMAGE_TAG .
        else
          BASE_IMAGE="public.ecr.aws/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04"
          echo "Using AWS Public ECR NVIDIA image..."
          # Modify only the base image in the original Dockerfile
          sed "s|nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04|public.ecr.aws/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04|g" Dockerfile > Dockerfile.aws
          
          # Add better apt handling at the beginning
          sed -i '/^FROM /a\\nENV DEBIAN_FRONTEND=noninteractive\n\n# Configure apt sources for better reliability\nRUN echo "deb http://us.archive.ubuntu.com/ubuntu/ focal main restricted universe multiverse" > /etc/apt/sources.list && \\\n    echo "deb http://us.archive.ubuntu.com/ubuntu/ focal-updates main restricted universe multiverse" >> /etc/apt/sources.list && \\\n    echo "deb http://us.archive.ubuntu.com/ubuntu/ focal-backports main restricted universe multiverse" >> /etc/apt/sources.list && \\\n    echo "deb http://security.ubuntu.com/ubuntu focal-security main restricted universe multiverse" >> /etc/apt/sources.list' Dockerfile.aws
          
          # Improve apt-get commands with retry logic
          sed -i 's/RUN apt-get update && apt-get install -y --no-install-recommends/RUN for i in 1 2 3; do apt-get update --fix-missing \&\& apt-get install -y --no-install-recommends --fix-missing/g' Dockerfile.aws
          sed -i 's/&& rm -rf \/var\/lib\/apt\/lists\/\*/\&\& break || sleep 10; done \&\& apt-get clean \&\& rm -rf \/var\/lib\/apt\/lists\/*/g' Dockerfile.aws
          
          docker build -f Dockerfile.aws -t \$IMAGE_REPO_NAME:\$IMAGE_TAG .
          rm -f Dockerfile.aws
        fi
      - docker tag \$IMAGE_REPO_NAME:\$IMAGE_TAG \$AWS_ACCOUNT_ID.dkr.ecr.\$AWS_DEFAULT_REGION.amazonaws.com/\$IMAGE_REPO_NAME:\$IMAGE_TAG
  post_build:
    commands:
      - echo Build completed on \`date\`
      - echo Pushing the Docker image...
      - docker push \$AWS_ACCOUNT_ID.dkr.ecr.\$AWS_DEFAULT_REGION.amazonaws.com/\$IMAGE_REPO_NAME:\$IMAGE_TAG
EOF
fi

# Create or update CodeBuild project
PROJECT_NAME="aita-vm-image-build"
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

# Create a zip file of the current directory for upload
echo "Creating source archive..."
zip -r source.zip . -x "*.git*" "*.zip"

# Upload to S3 bucket (create bucket if needed)
BUCKET_NAME="codebuild-source-$AWS_ACCOUNT_ID-$AWS_REGION"
aws s3 mb s3://$BUCKET_NAME --region $AWS_REGION || echo "Bucket already exists"
aws s3 cp source.zip s3://$BUCKET_NAME/aita-vm-source.zip

# Create source configuration JSON file for S3 source
cat > source-config.json << EOF
{
    "type": "S3",
    "location": "$BUCKET_NAME/aita-vm-source.zip",
    "buildspec": "buildspec.yml"
}
EOF

# Check if project exists first
PROJECT_EXISTS=$(aws codebuild describe-projects --names $PROJECT_NAME --query 'projects[0].name' --output text 2>/dev/null)

if [ "$PROJECT_EXISTS" = "$PROJECT_NAME" ]; then
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
    --environment-variables-override name=AWS_DEFAULT_REGION,value=$AWS_REGION name=AWS_ACCOUNT_ID,value=$AWS_ACCOUNT_ID name=IMAGE_REPO_NAME,value=$REPO_NAME name=IMAGE_TAG,value=$IMAGE_TAG \
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
