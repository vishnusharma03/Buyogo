from fastapi import FastAPI, HTTPException, APIRouter
from agent_rag import execute_query
from database import write_to_db
from utils import create_db_engine
from pydantic import BaseModel
import logging
import pandas as pd
from typing import List, Dict, Any
from embed import main

from analytics.revenue import analyze_revenue_trends
from analytics.cancellation import analyze_cancellation_rates
from analytics.geographical import analyze_geographical_distribution
from analytics.booking_lead import analyze_lead_time_distribution
from analytics.satisfaction import analyze_customer_satisfaction


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()
router = APIRouter()
conn = create_db_engine()

class QueryRequest(BaseModel):
    query: str

class DataFrameInput(BaseModel):
    data: List[Dict[Any, Any]]
    table_name: str = 'booking'

class AnalyticsRequest(BaseModel):
    analysis_type: str

@router.post("/ask")
async def execute_sql_query(query_request: QueryRequest):
    try:
        # Log incoming request
        logger.info(f"Processing query: {query_request.query}")
        
        # Execute the query
        result = execute_query(query_request.query)
        
        # Return successful response
        return {"result": result}
    except Exception as e:
        # Log the error
        logger.error(f"Error processing query: {str(e)}")
        
        # Return appropriate error response
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )


@router.post("/analytics")
async def get_analytics(request: AnalyticsRequest):
    try:
        # Log the analytics request
        logger.info(f"Processing analytics request: {request.analysis_type}")
        
        
        # Map analysis_type to the appropriate function
        analysis_functions = {
            'revenue': analyze_revenue_trends,
            'cancellation': analyze_cancellation_rates,
            'geographical': analyze_geographical_distribution,
            'booking_lead': analyze_lead_time_distribution,
            'satisfaction': analyze_customer_satisfaction
        }
        
        # Check if the requested analysis type exists
        if request.analysis_type not in analysis_functions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid analysis type. Available types: {', '.join(analysis_functions.keys())}"
            )
        
        # Call the appropriate analysis function
        image_base64, explanation = analysis_functions[request.analysis_type](params)
        
        # Return the analysis results
        return {
            "image": image_base64,
            "explanation": explanation,
            "analysis_type": request.analysis_type
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in analytics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analytics error: {str(e)}"
        )

@router.get("/health")
async def health_check():
    try:
        # You could add more comprehensive health checks here
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="Service unavailable"
        )

@router.get("/")
async def root():
    return {"message": "Welcome to the Hotel Booking Management System"}

@router.post("/update")
async def update_database(df_input: DataFrameInput):
    try:
        # Convert input JSON to pandas DataFrame
        df = pd.DataFrame(df_input.data)
        
        # Log the update attempt
        logger.info(f"Attempting to update table {df_input.table_name} with {len(df)} records")
        
        # Write to database using the imported function
        write_to_db(df, df_input.table_name, conn)
        main(custom_df=df)
        return {
            "status": "success",
            "message": f"Successfully updated table {df_input.table_name} with {len(df)} records"
        }
        
    except Exception as e:
        logger.error(f"Error updating database: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update database: {str(e)}"
        )

app.include_router(router)