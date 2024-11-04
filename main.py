from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from utils import JobMatchAnalyzer
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(
    title="CV-Job Matching API",
    description="API for analyzing match between CV and job postings",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for CV and Job data
stored_data = {
    "cv_data": None,
    "job_data": None
}

# Initialize analyzer
analyzer = JobMatchAnalyzer()

class CVData(BaseModel):
    Data: List[Dict[str, Any]]
    success: bool = True

class JobData(BaseModel):
    id: int
    jobTitle: str
    jobDescription: str
    experienceRequired: int
    qualificationRequired: str
    skillSets: List[str]
    companyName: str
    jobType: Dict[str, Any]

@app.get("/")
async def root():
    return {
        "message": "CV-Job Matching API",
        "status": "active",
        "device": str(analyzer.device)
    }

@app.post("/upload_cv/")
async def upload_cv(cv_data: CVData):
    """Upload CV data"""
    try:
        stored_data["cv_data"] = cv_data.dict()
        return {
            "success": True,
            "message": "CV data stored successfully",
            "candidate_name": cv_data.Data[0]['personal']['name'][0] if cv_data.Data else None
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "error_type": type(e).__name__
            }
        )

@app.post("/upload_job/")
async def upload_job(job_data: JobData):
    """Upload Job Post data"""
    try:
        stored_data["job_data"] = job_data.dict()
        return {
            "success": True,
            "message": "Job data stored successfully",
            "job_title": job_data.jobTitle
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "error_type": type(e).__name__
            }
        )

@app.post("/analyze_match/")
async def analyze_match():
    """Analyze match using stored CV and Job data"""
    try:
        if not stored_data["cv_data"] or not stored_data["job_data"]:
            raise HTTPException(
                status_code=400,
                detail="Both CV and Job data must be uploaded first"
            )
            
        result = analyzer.analyze_match(
            cv_data=stored_data["cv_data"],
            job_data=stored_data["job_data"]
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "error_type": type(e).__name__
            }
        )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "loaded" if hasattr(analyzer, 'model') else "not_loaded",
        "device": str(analyzer.device),
        "stored_data": {
            "has_cv": stored_data["cv_data"] is not None,
            "has_job": stored_data["job_data"] is not None
        }
    }

@app.delete("/clear_data/")
async def clear_data():
    """Clear stored CV and Job data"""
    stored_data["cv_data"] = None
    stored_data["job_data"] = None
    return {
        "success": True,
        "message": "All stored data cleared"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5000,
        reload=True
    )