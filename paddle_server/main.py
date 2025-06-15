from fastapi import FastAPI, HTTPException, Response
from contextlib import asynccontextmanager
import asyncio
from paddle_classifier import LayoutClassifier
from pipeline_utilities import init_firebase


paddle_classifier = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    print("App is starting up...")
    init_firebase()

    global paddle_classifier
    paddle_classifier = LayoutClassifier()
    
    print("App startup complete - ready to handle requests")
    yield
    # Shutdown code
    print("App is shutting down...")

app = FastAPI(lifespan=lifespan)

# --- API Endpoints ---
# GET request allowed for health check, works in browser
@app.get("/health")
def health_check():
    """
    Basic health check - always responds quickly for load balancer
    """
    return {"status": "healthy"}

@app.post("/layout")
async def layout(request: dict):
    try:
        result = await asyncio.to_thread(paddle_classifier.run_classification, request)
        
        if result:
            return result
        else:
            raise HTTPException(status_code=500, detail="Layout classification failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

