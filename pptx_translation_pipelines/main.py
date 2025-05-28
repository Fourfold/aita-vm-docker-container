from fastapi import FastAPI, HTTPException, Response
from contextlib import asynccontextmanager
from pipeline_pro import PipelinePro
from pipeline_public import PipelinePublic
from paddle_classifier import LayoutClassifier
from pipeline_utilities import init_firebase

app = FastAPI()
pipeline_pro = None
pipeline_public = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    print("App is starting up...")
    init_firebase()
    LayoutClassifier.initialize()
    global pipeline_pro
    pipeline_pro = PipelinePro()
    global pipeline_public
    pipeline_public = PipelinePublic()
    yield
    # Shutdown code
    print("App is shutting down...")


# --- API Endpoints ---
# GET request allowed for health check, works in browser
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# POST request for actual API call, does not work in browser
@app.post("/pro")
async def pro(request: dict):
    try:
        is_success = pipeline_pro.run_translation(request)
        if is_success:
            return {"result": "success"}
        else:
            raise HTTPException(status_code=500, detail="Translation failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/public")
async def gpt(request: dict):
    try:
        is_success = pipeline_pro.run_translation(request)
        if is_success:
            return {"result": "success"}
        else:
            raise HTTPException(status_code=500, detail="Translation failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

