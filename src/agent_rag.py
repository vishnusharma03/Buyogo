from smolagents import CodeAgent
from src.utils import load_api_keys, create_db_engine
from smolagents import tool, LiteLLMModel
from sqlalchemy import text
from smolagents.memory import ActionStep


engine = create_db_engine()


@tool
def sql_engine(query: str) -> str:
    """
    Allows you to perform SQL queries on the table. Returns a string representation of the result.
    The table is named 'hotel_bookings'. Its description is as follows:
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


@tool
def natural_to_sql(question: str) -> str:
    """
    Converts a natural language question into a SQL query for the hotel_bookings table.
    
    Args:
        question: A natural language question about hotel bookings data
        
    Returns:
        str: A valid SQL query that answers the question
    """
    # The table schema is provided to help with query generation
    table_schema = """
    Table: hotel_bookings
    Columns:
    - index: INTEGER
    - hotel: TEXT (Resort Hotel or City Hotel)
    - is_canceled: INTEGER (1 if canceled, 0 if not)
    - lead_time: INTEGER (days between booking and arrival)
    - arrival_date_year: INTEGER
    - arrival_date_month: TEXT
    - arrival_date_week_number: INTEGER
    - arrival_date_day_of_month: INTEGER
    - stays_in_weekend_nights: INTEGER
    - stays_in_week_nights: INTEGER
    - adults: INTEGER
    - children: REAL
    - babies: INTEGER
    - meal: TEXT (meal package: BB, HB, FB, etc.)
    - country: TEXT (country of origin)
    - market_segment: TEXT (market segment designation)
    - distribution_channel: TEXT (booking distribution channel)
    - is_repeated_guest: INTEGER (1 if repeated guest, 0 if not)
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
    - adr: REAL (average daily rate)
    - required_car_parking_spaces: INTEGER
    - total_of_special_requests: INTEGER
    - reservation_status: TEXT
    - reservation_status_date: TEXT
    """
    
    # Prompt for the LLM to convert natural language to SQL
    prompt = f"""
    Given the following database schema:
    {table_schema}
    
    Convert this natural language question to a valid SQL query:
    "{question}"
    
    Return ONLY the SQL query without any explanation or additional text.
    The query should be valid SQL that can be executed directly against the hotel_bookings table.
    """
    
    # Use the agent to generate the SQL query
    global agent
    sql_query = agent.run(prompt)
    
    # Clean up the response to ensure it's just the SQL query
    sql_query = sql_query.strip()
    
    # Remove any markdown code block formatting if present
    if sql_query.startswith("```sql"):
        sql_query = sql_query.split("```sql")[1]
    if sql_query.startswith("```"):
        sql_query = sql_query.split("```")[1]
    if sql_query.endswith("```"):
        sql_query = sql_query.split("```")[0]
    
    return sql_query.strip()


@tool
def is_sql_query(query: str) -> bool:
    """
    Determine if a query is SQL or natural language.
    
    Args:
        query: The query string to check
        
    Returns:
        bool: True if the query appears to be SQL, False if it appears to be natural language
    """
    # Normalize the query
    query = query.strip().lower()
    
    # Common SQL keywords that indicate a SQL query
    sql_keywords = [
        "select", "from", "where", "group by", "order by", 
        "having", "join", "inner join", "left join", "right join",
        "limit", "offset", "union", "insert", "update", "delete"
    ]
    
    # Check if the query starts with common SQL keywords
    for keyword in sql_keywords:
        if query.startswith(keyword):
            return True
    
    # Check if the query contains SQL-specific patterns
    sql_patterns = [
        "select .* from", 
        "from .* where",
        "group by .*",
        "order by .*"
    ]
    
    import re
    for pattern in sql_patterns:
        if re.search(pattern, query):
            return True
    
    # If no SQL patterns are found, assume it's natural language
    return False

    
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
        tools=[sql_engine, natural_to_sql, is_sql_query],
        model=model,
        provide_run_summary=True
    )

engine = create_db_engine()
model = create_llm_model()
agent = create_agent()


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

        thought_process = []
        if hasattr(agent_to_use, 'memory') and hasattr(agent_to_use.memory, 'steps'):
            for i, step in enumerate(agent_to_use.memory.steps):
                step_info = {}
                
                if isinstance(step, ActionStep):
                    # Extract the thought process from model_output
                    thought = step.model_output.split("<end_code>")[0] if step.model_output else ""
                    
                    step_info = {
                        "step_number": i + 1,
                        "thought": thought.strip(),
                        "observation": step.observations if step.observations else "",
                        "error": step.error if step.error else None
                    }
                elif hasattr(step, 'task'):
                    # For TaskStep, just include the task
                    step_info = {
                        "step_number": i + 1,
                        "task": step.task
                    }
                
                if step_info:
                    thought_process.append(step_info)

        # print("Agent memory structure:")
        # if hasattr(agent_to_use, 'memory'):
        #     print(f"Memory type: {type(agent_to_use.memory)}")
        #     print(f"Memory attributes: {dir(agent_to_use.memory)}")
            
        #     if hasattr(agent_to_use.memory, 'steps'):
        #         print(f"Steps type: {type(agent_to_use.memory.steps)}")
        #         print(f"Steps length: {len(agent_to_use.memory.steps)}")
                
        #         if len(agent_to_use.memory.steps) > 0:
        #             first_step = agent_to_use.memory.steps[0]
        #             print(f"First step type: {type(first_step)}")
        #             print(f"First step attributes: {dir(first_step)}")
        #             print(f"First step repr: {repr(first_step)}")
        
        # # Extract thought process from agent memory
        # thought_process = []
        # if hasattr(agent_to_use, 'memory') and hasattr(agent_to_use.memory, 'steps'):
        #     for i, step in enumerate(agent_to_use.memory.steps):
        #         # Create a more detailed representation of the step
        #         step_info = {
        #             "step_number": i + 1,
        #             "step_type": str(type(step)),
        #             "step_dir": str(dir(step)),
        #             "step_repr": str(repr(step))
        #         }
                
        #         # Try different ways to access step data
        #         if hasattr(step, 'to_dict'):
        #             step_data = step.to_dict()
        #             step_info.update(step_data)
        #         elif hasattr(step, '__dict__'):
        #             step_info.update(step.__dict__)
                
        #         # Add common attributes we expect to find
        #         for attr in ['thought', 'action', 'action_input', 'observation', 'message']:
        #             if hasattr(step, attr):
        #                 step_info[attr] = getattr(step, attr)
                
        #         thought_process.append(step_info)
                
        ## print(thought_process)
        nl_response = natural_response(query, result, agent_to_use)
        return {
            "result": nl_response,
            "thought_process": thought_process
        }

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
        1. Key findings from the results
        
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

