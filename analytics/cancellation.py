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

def analyze_cancellation_rates(params=None):
    """
    Analyze cancellation rates as percentage of total bookings.
    
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
    
    # Build the SQL query for cancellation analysis
    query = """
    SELECT 
        arrival_date_year,
        arrival_date_month,
        hotel,
        COUNT(*) as total_bookings,
        SUM(CASE WHEN is_canceled = 1 THEN 1 ELSE 0 END) as canceled_bookings,
        CAST(SUM(CASE WHEN is_canceled = 1 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100 as cancellation_rate
    FROM booking
    """
    
    # Add filters based on parameters
    where_clauses = []
    if hotel_type:
        where_clauses.append(f"hotel = '{hotel_type}'")
    if year:
        where_clauses.append(f"arrival_date_year = {year}")
    if month:
        where_clauses.append(f"arrival_date_month = '{month}'")
    
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    
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
            plt.bar(df['month_year'], df['cancellation_rate'])
            plt.title(f'Cancellation Rates for {hotel_type} Hotels')
        else:
            # Create a grouped bar chart for hotel comparison
            ax = sns.barplot(x='month_year', y='cancellation_rate', hue='hotel', data=df)
            plt.title('Cancellation Rates by Hotel Type')
            
            # Add value labels on top of bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f%%', padding=3)
        
        plt.xlabel('Month-Year')
        plt.ylabel('Cancellation Rate (%)')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
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
    """Generate a textual explanation of the cancellation rates."""
    if df.empty:
        return "No data available for the selected parameters."
    
    hotel_type = params.get('hotel_type', None)
    year = params.get('year', None)
    month = params.get('month', None)
    
    # Start with a basic explanation
    if hotel_type:
        explanation = f"This chart shows the cancellation rates for {hotel_type} hotels"
    else:
        explanation = "This chart shows the cancellation rates across different hotel types"
    
    if year and month:
        explanation += f" in {month} {year}."
    elif year:
        explanation += f" throughout {year}."
    elif month:
        explanation += f" during {month} across all years."
    else:
        explanation += " over the entire period."
    
    # Add insights about the data
    total_bookings = df['total_bookings'].sum()
    total_cancellations = df['canceled_bookings'].sum()
    overall_rate = (total_cancellations / total_bookings) * 100
    
    explanation += f"\n\nTotal bookings: {total_bookings:,}"
    explanation += f"\nTotal cancellations: {total_cancellations:,}"
    explanation += f"\nOverall cancellation rate: {overall_rate:.2f}%"
    
    # Find highest and lowest cancellation rates
    max_rate_row = df.loc[df['cancellation_rate'].idxmax()]
    min_rate_row = df.loc[df['cancellation_rate'].idxmin()]
    
    explanation += f"\n\nHighest cancellation rate: {max_rate_row['cancellation_rate']:.2f}% ({max_rate_row['month_year']}, {max_rate_row['hotel']})"
    explanation += f"\nLowest cancellation rate: {min_rate_row['cancellation_rate']:.2f}% ({min_rate_row['month_year']}, {min_rate_row['hotel']})"
    
    if not hotel_type:
        # Compare cancellation rates between hotel types
        # Fix for deprecation warning - use agg instead of apply
        hotel_stats = df.groupby('hotel').agg({
            'canceled_bookings': 'sum',
            'total_bookings': 'sum'
        })
        hotel_rates = (hotel_stats['canceled_bookings'] / hotel_stats['total_bookings']) * 100
        hotel_bookings = hotel_stats['total_bookings']
        
        for hotel in hotel_rates.index:
            explanation += f"\n\n{hotel} hotels:"
            explanation += f"\n  - Cancellation rate: {hotel_rates[hotel]:.2f}%"
            explanation += f"\n  - Total bookings: {hotel_bookings[hotel]:,}"
        
        # Identify which hotel type has higher cancellation rate
        if len(hotel_rates) > 1:
            higher_rate_hotel = hotel_rates.idxmax()
            lower_rate_hotel = hotel_rates.idxmin()
            rate_diff = hotel_rates[higher_rate_hotel] - hotel_rates[lower_rate_hotel]
            
            explanation += f"\n\n{higher_rate_hotel} hotels have a {rate_diff:.2f}% higher cancellation rate than {lower_rate_hotel} hotels."
    
    # Add trend analysis if applicable
    if len(df) > 1 and 'arrival_date_year' in df.columns:
        years = df['arrival_date_year'].unique()
        if len(years) > 1:
            # Fix for deprecation warning - use agg instead of apply
            yearly_stats = df.groupby('arrival_date_year').agg({
                'canceled_bookings': 'sum',
                'total_bookings': 'sum'
            })
            yearly_rates = (yearly_stats['canceled_bookings'] / yearly_stats['total_bookings']) * 100
            
            trend_direction = "increasing" if yearly_rates.iloc[-1] > yearly_rates.iloc[0] else "decreasing"
            explanation += f"\n\nThe overall cancellation rate trend is {trend_direction} over the years."
    
    return explanation



