#!/bin/bash
set -e

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "AWS CLI not found. Please install it first."
    exit 1
fi

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
  build:
    commands:
      - echo Build started on \`date\`
      - echo Building the Docker image...
      - docker build -t \$IMAGE_REPO_NAME:\$IMAGE_TAG .
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

# Create the CodeBuild project
aws codebuild create-project \
    --name $PROJECT_NAME \
    --source type=NO_SOURCE,buildspec=buildspec.yml \
    --artifacts type=NO_ARTIFACTS \
    --environment type=LINUX_CONTAINER,image=aws/codebuild/amazonlinux2-x86_64-standard:3.0,computeType=BUILD_GENERAL1_LARGE,privilegedMode=true \
    --service-role arn:aws:iam::$AWS_ACCOUNT_ID:role/$ROLE_NAME || echo "Project already exists, updating..."

# If project exists, update it
aws codebuild update-project \
    --name $PROJECT_NAME \
    --source type=NO_SOURCE,buildspec=buildspec.yml \
    --artifacts type=NO_ARTIFACTS \
    --environment type=LINUX_CONTAINER,image=aws/codebuild/amazonlinux2-x86_64-standard:3.0,computeType=BUILD_GENERAL1_LARGE,privilegedMode=true \
    --service-role arn:aws:iam::$AWS_ACCOUNT_ID:role/$ROLE_NAME || true

# Create a zip file of the current directory for upload
echo "Creating source archive..."
zip -r source.zip . -x "*.git*" "*.zip"

# Upload to S3 bucket (create bucket if needed)
BUCKET_NAME="codebuild-source-$AWS_ACCOUNT_ID-$AWS_REGION"
aws s3 mb s3://$BUCKET_NAME --region $AWS_REGION || echo "Bucket already exists"
aws s3 cp source.zip s3://$BUCKET_NAME/aita-vm-source.zip

# Start the build with source override
echo "Starting CodeBuild..."
BUILD_ID=$(aws codebuild start-build \
    --project-name $PROJECT_NAME \
    --source-type-override S3 \
    --source-location-override $BUCKET_NAME/aita-vm-source.zip \
    --environment-variables-override name=AWS_DEFAULT_REGION,value=$AWS_REGION name=AWS_ACCOUNT_ID,value=$AWS_ACCOUNT_ID name=IMAGE_REPO_NAME,value=$REPO_NAME name=IMAGE_TAG,value=$IMAGE_TAG \
    --query 'build.id' --output text)

# Clean up local zip file
rm -f source.zip

echo "Build started with ID: $BUILD_ID"
echo "You can monitor the build at: https://console.aws.amazon.com/codesuite/codebuild/projects/$PROJECT_NAME/build/$BUILD_ID"
echo "Or run: aws codebuild batch-get-builds --ids $BUILD_ID"

# Optionally wait for build to complete and show status
echo "Waiting for build to complete..."
aws codebuild batch-get-builds --ids $BUILD_ID --query 'builds[0].buildStatus' --output text

echo "Docker image will be available at: $ECR_URI"
