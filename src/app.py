from fastapi import FastAPI, HTTPException, APIRouter
from src.agent_rag import execute_query
from src.database import write_to_db
from src.utils import create_db_engine
from pydantic import BaseModel
import logging
import pandas as pd
from typing import List, Dict, Any
from src.analytics import run_all_analytics


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


@router.get("/analytics")
async def get_analytics():
    try:
        # Call the appropriate analysis function
        things =  run_all_analytics()
        
        # Return the analysis results
        return things
        
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