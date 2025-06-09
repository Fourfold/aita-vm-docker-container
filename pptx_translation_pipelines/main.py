from fastapi import FastAPI, HTTPException, Response
from contextlib import asynccontextmanager
import asyncio
import threading
from pipeline_pro import PipelinePro
from pipeline_public import PipelinePublic
from paddle_classifier import LayoutClassifier
from pipeline_utilities import init_firebase

# Global instances
pipeline_pro = None
pipeline_public = None

# Concurrency control for GPU-bound operations
_pro_lock = threading.Lock()  # Serialize PipelinePro requests
# _pro_semaphore = threading.Semaphore(2)  # Allow 2 concurrent PipelinePro requests with batching

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    print("App is starting up...")
    init_firebase()
    LayoutClassifier.initialize()
    
    global pipeline_pro, pipeline_public
    pipeline_pro = PipelinePro()
    pipeline_public = PipelinePublic()
    
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

@app.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check that verifies actual service functionality
    Use this for monitoring, not for load balancer health checks
    """
    health_status = {
        "status": "healthy",
        "timestamp": asyncio.get_event_loop().time(),
        "services": {
            "paddle_classifier": "unknown",
            "pipeline_pro": "unknown", 
            "pipeline_public": "unknown"
        },
        "gpu": {
            "available": False,
            "device_count": 0,
            "current_device": "none"
        }
    }
    
    try:
        # Check GPU status
        import torch
        if torch.cuda.is_available():
            health_status["gpu"]["available"] = True
            health_status["gpu"]["device_count"] = torch.cuda.device_count()
            health_status["gpu"]["current_device"] = torch.cuda.get_device_name()
        
        # Check PaddleX classifier
        try:
            classifier = LayoutClassifier()
            if hasattr(classifier, '_failed_init') and classifier._failed_init:
                health_status["services"]["paddle_classifier"] = "fallback_mode"
            elif classifier.model is not None:
                health_status["services"]["paddle_classifier"] = "healthy"
            else:
                health_status["services"]["paddle_classifier"] = "unavailable"
        except Exception as e:
            health_status["services"]["paddle_classifier"] = f"error: {str(e)[:100]}"
        
        # Check PipelinePro (GPU model)
        try:
            if pipeline_pro is not None and hasattr(pipeline_pro, 'model'):
                health_status["services"]["pipeline_pro"] = "healthy"
            else:
                health_status["services"]["pipeline_pro"] = "not_initialized"
        except Exception as e:
            health_status["services"]["pipeline_pro"] = f"error: {str(e)[:100]}"
        
        # Check PipelinePublic (OpenAI)
        try:
            if pipeline_public is not None:
                health_status["services"]["pipeline_public"] = "healthy"
            else:
                health_status["services"]["pipeline_public"] = "not_initialized"
        except Exception as e:
            health_status["services"]["pipeline_public"] = f"error: {str(e)[:100]}"
            
        # Determine overall status
        service_states = list(health_status["services"].values())
        if any("error" in state for state in service_states):
            health_status["status"] = "degraded"
        elif any(state in ["unavailable", "not_initialized"] for state in service_states):
            health_status["status"] = "partial"
            
    except Exception as e:
        health_status["status"] = "error"
        health_status["error"] = str(e)[:200]
    
    return health_status

@app.get("/health/ready")
async def readiness_check():
    """
    Readiness check - verifies app can handle requests
    Returns 200 only if core services are ready
    """
    try:
        # Quick check of essential services
        if pipeline_pro is None or pipeline_public is None:
            raise HTTPException(status_code=503, detail="Services not initialized")
            
        # Check if we're overwhelmed with requests
        # You could add more sophisticated checks here
        
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Not ready: {str(e)}")

@app.get("/health/live")
async def liveness_check():
    """
    Liveness check - verifies app is alive and responding
    Should never block, used by orchestrators to restart unhealthy containers
    """
    return {"status": "alive", "timestamp": asyncio.get_event_loop().time()}

# POST request for actual API call, does not work in browser
@app.post("/pro")
async def pro(request: dict):
    """
    GPU-intensive pipeline with controlled parallel access
    """
    try:
        # Use semaphore to limit concurrent GPU requests while allowing parallelism
        def run_pro_pipeline():
            # with _pro_semaphore:  # Allow up to 2 concurrent requests
            with _pro_lock:  # Ensure only one PipelinePro request at a time
                return pipeline_pro.run_translation(request)
        
        # Run in thread pool to avoid blocking the event loop
        is_success = await asyncio.to_thread(run_pro_pipeline)
        
        if is_success:
            return {"result": "success"}
        else:
            raise HTTPException(status_code=500, detail="Translation failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/public")
async def public(request: dict):
    """
    GPT-based pipeline with better concurrency support
    """
    try:
        # PipelinePublic has internal thread safety, so we can allow more concurrency
        # but still use asyncio.to_thread to avoid blocking
        is_success = await asyncio.to_thread(pipeline_public.run_translation, request)
        
        if is_success:
            return {"result": "success"}
        else:
            raise HTTPException(status_code=500, detail="Translation failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

