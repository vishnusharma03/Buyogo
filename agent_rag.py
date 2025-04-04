from smolagents import CodeAgent
from utils import load_api_keys, create_db_engine
from smolagents import tool, LiteLLMModel
from sqlalchemy import text


engine = create_db_engine()


@tool
def sql_engine(query: str) -> str:
    """
    Allows you to perform SQL queries on the table. Returns a string representation of the result.
    The table is named 'booking'. Its description is as follows:
        Columns:
        - index: INTEGER
        - hotel: TEXT
        - is_canceled: INTEGER
        - lead_time: INTEGER
        - arrival_date_year: INTEGER
        - arrival_date_month: TEXT
        - arrival_date_week_number: INTEGER
        - arrival_date_day_of_month: INTEGER
        - stays_in_weekend_nights: INTEGER
        - stays_in_week_nights: INTEGER
        - adults: INTEGER
        - children: REAL
        - babies: INTEGER
        - meal: TEXT
        - country: TEXT
        - market_segment: TEXT
        - distribution_channel: TEXT
        - is_repeated_guest: INTEGER
        - previous_cancellations: INTEGER
        - previous_bookings_not_canceled: INTEGER
        - reserved_room_type: TEXT
        - assigned_room_type: TEXT
        - booking_changes: INTEGER
        - deposit_type: TEXT
        - agent: REAL
        - company: REAL
        - days_in_waiting_list: INTEGER
        - customer_type: TEXT
        - adr: REAL
        - required_car_parking_spaces: INTEGER
        - total_of_special_requests: INTEGER
        - reservation_status: TEXT
        - reservation_status_date: TEXT

    Args:
        query: The query to perform. This should be correct SQL.
    """
    output = ""
    with engine.connect() as con:
        rows = con.execute(text(query))
        for row in rows:
            output += "\n" + str(row)
    return output

    
def create_llm_model():
    """Create and return a LiteLLM model instance."""
    api_keys = load_api_keys()
    return LiteLLMModel(
        model_id="gemini/gemini-2.0-flash-exp", # "gemini-2.0-flash-exp",  # Specify the Gemini model ID
        api_key=api_keys['gemini'],  # Use your API key from environment variables
        project_id="lambda4110"
    )



def create_agent():
    """Create and return a CodeAgent instance with SQL tools."""
    return CodeAgent(
        tools=[sql_engine],
        model=model
    )

engine = create_db_engine()
model = create_llm_model()
agent = create_agent()


# def execute_query(query: str, agent: CodeAgent, engine) -> str:
#     """
#     Execute a SQL query using the provided agent, model, and database engine.
    
#     Args:
#         query: SQL query string to execute
#         agent: CodeAgent instance
#         model: LiteLLM model instance
#         engine: SQLAlchemy engine instance
        
#     Returns:
#         str: Query execution results
#     """
#     try:
#         # Set the engine globally for sql_engine tool
#         # Move global declaration to top of function
#         global agent, engine
#         agent_to_use = agent_instance if agent_instance is not None else agent
#         engine_to_use = engine_instance if engine_instance is not None else engine
        
#         # Execute the query through the agent
#         result = agent.run(f"Execute this SQL query: {query}")
#         return natural_response(query, result, agent)
#     except Exception as e:
#         return f"Error executing query: {str(e)}"

def execute_query(query: str, agent_instance=None, engine_instance=None) -> str:
    """
    Execute a SQL query using the provided agent, model, and database engine.
    
    Args:
        query: SQL query string to execute
        agent_instance: CodeAgent instance (defaults to global agent)
        engine_instance: SQLAlchemy engine instance (defaults to global engine)
        
    Returns:
        str: Query execution results
    """
    try:
        # Use global variables if no instances are provided
        global agent
        agent_to_use = agent_instance if agent_instance is not None else agent
        
        
        # Execute the query through the agent
        result = agent_to_use.run(f"Execute this SQL query: {query}")
        return natural_response(query, result, agent_to_use)
    except Exception as e:
        return f"Error executing query: {str(e)}"


def natural_response(query: str, result: str, agent: CodeAgent) -> str:
    """
    Convert SQL query results into natural language response using LLM.
    
    Args:
        query: The original SQL query
        result: The raw query result string
        model: LiteLLM model instance
        
    Returns:
        str: Natural language interpretation of the results
    """
    try:
        # Prepare prompt for LLM
        prompt = f"""
        Given this SQL query: {query}
        And these results: {result}
        
        Please provide a natural language summary of these results. 
        Focus on:
        1. What was queried (the main objective)
        2. Key findings from the results
        3. Any notable patterns or insights
        
        Format the response in a clear, concise & conversational way that a non-technical person would understand.
        """
        
        # Get LLM response
        llm_response = agent.run(prompt)
        
        # If LLM fails, fall back to basic formatting
        if not llm_response or "error" in llm_response.lower():
            # Extract key information from query
            query_lower = query.lower()
            
            # Initialize response components
            action = "retrieved"
            if "count" in query_lower:
                action = "counted"
            elif "avg" in query_lower or "average" in query_lower:
                action = "calculated the average of"
            elif "sum" in query_lower:
                action = "calculated the total of"
                
            # Build basic response
            if not result.strip():
                return "No results found for this query."
                
            response = f"I have {action} the following information:\n"
            
            # Format results
            results_list = result.strip().split("\n")
            for row in results_list:
                cleaned_row = row.strip("()").replace("'", "")
                response += f"• {cleaned_row}\n"
                
            return response
            
        return llm_response
        
    except Exception as e:
        return f"I encountered an error while processing the results: {str(e)}"

