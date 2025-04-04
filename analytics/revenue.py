import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import sys
import os
import base64
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now you can potentially use an absolute import
import utils

def analyze_revenue_trends(params=None):
    """
    Analyze revenue trends over time from hotel bookings data.
    
    Args:
        params (dict): Parameters for filtering the data (e.g., time period, hotel type)
        
    Returns:
        tuple: (base64 encoded image, explanation text)
    """
    # Default parameters if none provided
    if params is None:
        params = {}
    
    # Get database connection
    conn = utils.create_db_engine()
    
    # Check if table exists, if not, create it from CSV
    ensure_table_exists(conn)
    
    # Extract filter parameters
    hotel_type = params.get('hotel_type', None)
    year = params.get('year', None)
    month = params.get('month', None)
    
    # Build the SQL query - using the correct table name (booking)
    query = """
    SELECT 
        arrival_date_year, 
        arrival_date_month, 
        hotel,
        SUM((stays_in_weekend_nights + stays_in_week_nights) * adr) as revenue
    FROM booking
    WHERE is_canceled = 0
    """
    
    # Add filters based on parameters
    if hotel_type:
        query += f" AND hotel = '{hotel_type}'"
    if year:
        query += f" AND arrival_date_year = {year}"
    if month:
        query += f" AND arrival_date_month = '{month}'"
    
    # Group by time period and hotel type
    query += """
    GROUP BY arrival_date_year, arrival_date_month, hotel
    ORDER BY arrival_date_year, 
        CASE arrival_date_month
            WHEN 'January' THEN 1
            WHEN 'February' THEN 2
            WHEN 'March' THEN 3
            WHEN 'April' THEN 4
            WHEN 'May' THEN 5
            WHEN 'June' THEN 6
            WHEN 'July' THEN 7
            WHEN 'August' THEN 8
            WHEN 'September' THEN 9
            WHEN 'October' THEN 10
            WHEN 'November' THEN 11
            WHEN 'December' THEN 12
        END
    """
    
    try:
        # Execute the query and load into DataFrame
        df = pd.read_sql(query, conn)
        
        # Create month-year column for better visualization
        df['month_year'] = df['arrival_date_month'] + ' ' + df['arrival_date_year'].astype(str)
        
        # Create the visualization
        plt.figure(figsize=(12, 6))
        
        if hotel_type:
            # Single hotel type - show monthly trend
            sns.lineplot(data=df, x='month_year', y='revenue')
            plt.title(f'Revenue Trends for {hotel_type} Hotels')
        else:
            # Multiple hotel types - show comparison
            sns.lineplot(data=df, x='month_year', y='revenue', hue='hotel')
            plt.title('Revenue Trends by Hotel Type')
        
        plt.xlabel('Month-Year')
        plt.ylabel('Total Revenue (€)')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Generate explanation
        explanation = generate_explanation(df, params)
        
        # Convert plot to base64 encoded image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return image_base64, explanation
    
    except Exception as e:
        print(f"Error executing query: {e}")
        # Return a placeholder image and error message
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', fontsize=12)
        plt.axis('off')
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return image_base64, f"An error occurred: {str(e)}"

def ensure_table_exists(conn):
    """Check if the booking table exists, if not create it from CSV."""
    from sqlalchemy import inspect, text
    
    inspector = inspect(conn)
    if not inspector.has_table('booking'):
        # Table doesn't exist, try to create it from CSV
        try:
            csv_path = os.path.join(project_root, 'data', 'hotel_bookings.csv')
            if os.path.exists(csv_path):
                print(f"Creating booking table from CSV: {csv_path}")
                df = pd.read_csv(csv_path)
                df.to_sql('booking', conn, if_exists='replace', index=False)
                print("Table 'booking' created successfully")
            else:
                print(f"CSV file not found: {csv_path}")
        except Exception as e:
            print(f"Error creating table: {e}")

def generate_explanation(df, params):
    """Generate a textual explanation of the revenue trends."""
    if df.empty:
        return "No data available for the selected parameters."
    
    hotel_type = params.get('hotel_type', None)
    year = params.get('year', None)
    month = params.get('month', None)
    
    # Start with a basic explanation
    if hotel_type:
        explanation = f"This chart shows the revenue trends for {hotel_type} hotels"
    else:
        explanation = "This chart shows the revenue trends across different hotel types"
    
    if year and month:
        explanation += f" in {month} {year}."
    elif year:
        explanation += f" throughout {year}."
    elif month:
        explanation += f" during {month} across all years."
    else:
        explanation += " over the entire period."
    
    # Add insights about the data
    total_revenue = df['revenue'].sum()
    avg_revenue = df['revenue'].mean()
    max_revenue = df['revenue'].max()
    max_revenue_period = df.loc[df['revenue'].idxmax()]
    
    explanation += f"\n\nTotal revenue: €{total_revenue:,.2f}"
    explanation += f"\nAverage monthly revenue: €{avg_revenue:,.2f}"
    explanation += f"\nHighest revenue (€{max_revenue:,.2f}) was recorded in {max_revenue_period['month_year']}"
    
    if not hotel_type:
        # Add hotel type comparison
        hotel_revenues = df.groupby('hotel')['revenue'].sum()
        top_hotel = hotel_revenues.idxmax()
        explanation += f"\n\n{top_hotel} hotels generated the highest total revenue (€{hotel_revenues[top_hotel]:,.2f})"
        
        # Calculate percentage difference between hotel types
        if len(hotel_revenues) > 1:
            hotel_types = hotel_revenues.index.tolist()
            percentages = [f"{hotel}: {hotel_revenues[hotel]/total_revenue*100:.1f}%" for hotel in hotel_types]
            explanation += f"\n\nRevenue distribution: {', '.join(percentages)}"
    
    # Add trend analysis
    if len(df) > 1 and 'arrival_date_year' in df.columns:
        years = df['arrival_date_year'].unique()
        if len(years) > 1:
            yearly_revenue = df.groupby('arrival_date_year')['revenue'].sum()
            trend_direction = "increasing" if yearly_revenue.iloc[-1] > yearly_revenue.iloc[0] else "decreasing"
            explanation += f"\n\nThe overall revenue trend is {trend_direction} over the years."
    
    return explanation


