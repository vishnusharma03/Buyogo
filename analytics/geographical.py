import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import sys
import os
import base64
from matplotlib.colors import LinearSegmentedColormap
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now you can potentially use an absolute import
import utils

def analyze_geographical_distribution(params=None):
    """
    Analyze geographical distribution of users doing the bookings.
    
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
    
    # Build the SQL query for geographical analysis
    query = """
    SELECT 
        country,
        hotel,
        COUNT(*) as booking_count
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
    
    # Group by country and hotel type
    query += """
    GROUP BY country, hotel
    ORDER BY booking_count DESC
    """
    
    try:
        # Execute the query and load into DataFrame
        df = pd.read_sql(query, conn)
        
        # Handle null or empty country values
        df['country'] = df['country'].fillna('Unknown')
        df.loc[df['country'] == '', 'country'] = 'Unknown'
        
        # Aggregate by country for total counts
        country_totals = df.groupby('country')['booking_count'].sum().reset_index()
        country_totals = country_totals.sort_values('booking_count', ascending=False)
        
        # Limit to top 15 countries for better visualization
        top_countries = country_totals.head(15)
        
        # Create the visualization
        plt.figure(figsize=(12, 8))
        
        if hotel_type:
            # Single hotel type - show country distribution
            ax = sns.barplot(x='booking_count', y='country', data=top_countries, 
                           palette='viridis', orient='h')
            plt.title(f'Geographical Distribution of Bookings for {hotel_type} Hotels')
            
            # Add value labels
            for i, v in enumerate(top_countries['booking_count']):
                ax.text(v + 0.1, i, f"{v:,}", va='center')
                
        else:
            # Multiple hotel types - show stacked bars
            # Pivot the data for stacked bar chart
            pivot_df = df[df['country'].isin(top_countries['country'])]
            pivot_df = pivot_df.pivot_table(
                index='country', 
                columns='hotel', 
                values='booking_count',
                aggfunc='sum',
                fill_value=0
            ).reset_index()
            
            # Sort by total bookings
            pivot_df['total'] = pivot_df.sum(axis=1, numeric_only=True)
            pivot_df = pivot_df.sort_values('total', ascending=False)
            pivot_df = pivot_df.drop('total', axis=1)
            
            # Plot stacked bar chart
            ax = pivot_df.set_index('country').plot(
                kind='barh', 
                stacked=True,
                figsize=(12, 8),
                colormap='viridis'
            )
            plt.title('Geographical Distribution of Bookings by Hotel Type')
            
            # Add legend
            plt.legend(title='Hotel Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.xlabel('Number of Bookings')
        plt.ylabel('Country')
        plt.grid(True, linestyle='--', alpha=0.7, axis='x')
        plt.tight_layout()
        
        # Generate explanation
        explanation = generate_explanation(df, country_totals, params)
        
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

def generate_explanation(df, country_totals, params):
    """Generate a textual explanation of the geographical distribution."""
    if df.empty:
        return "No data available for the selected parameters."
    
    hotel_type = params.get('hotel_type', None)
    year = params.get('year', None)
    month = params.get('month', None)
    
    # Start with a basic explanation
    if hotel_type:
        explanation = f"This chart shows the geographical distribution of bookings for {hotel_type} hotels"
    else:
        explanation = "This chart shows the geographical distribution of bookings across different hotel types"
    
    if year and month:
        explanation += f" in {month} {year}."
    elif year:
        explanation += f" throughout {year}."
    elif month:
        explanation += f" during {month} across all years."
    else:
        explanation += " over the entire period."
    
    # Add insights about the data
    total_bookings = df['booking_count'].sum()
    total_countries = df['country'].nunique()
    
    explanation += f"\n\nTotal bookings: {total_bookings:,}"
    explanation += f"\nNumber of countries: {total_countries}"
    
    # Top countries
    top_5_countries = country_totals.head(5)
    explanation += "\n\nTop 5 countries by number of bookings:"
    
    for _, row in top_5_countries.iterrows():
        percentage = (row['booking_count'] / total_bookings) * 100
        explanation += f"\n  - {row['country']}: {row['booking_count']:,} bookings ({percentage:.1f}% of total)"
    
    # Concentration analysis
    top_5_percentage = top_5_countries['booking_count'].sum() / total_bookings * 100
    explanation += f"\n\nThe top 5 countries account for {top_5_percentage:.1f}% of all bookings."
    
    # Hotel type distribution by country if applicable
    if not hotel_type and len(df['hotel'].unique()) > 1:
        # Get the most popular hotel type for the top country
        top_country = top_5_countries.iloc[0]['country']
        country_data = df[df['country'] == top_country]
        top_hotel = country_data.groupby('hotel')['booking_count'].sum().idxmax()
        top_hotel_count = country_data.groupby('hotel')['booking_count'].sum().max()
        top_hotel_percentage = (top_hotel_count / country_data['booking_count'].sum()) * 100
        
        explanation += f"\n\nIn {top_country} (the top country), {top_hotel} is the most popular hotel type with {top_hotel_count:,} bookings ({top_hotel_percentage:.1f}% of bookings from this country)."
    
    # International vs. domestic analysis if possible
    if 'PRT' in df['country'].values:  # Assuming PRT is Portugal, the location of the hotels
        domestic_bookings = df[df['country'] == 'PRT']['booking_count'].sum()
        domestic_percentage = (domestic_bookings / total_bookings) * 100
        international_percentage = 100 - domestic_percentage
        
        explanation += f"\n\nDomestic bookings (from Portugal) account for {domestic_percentage:.1f}% of all bookings, while international bookings account for {international_percentage:.1f}%."
    
    return explanation


