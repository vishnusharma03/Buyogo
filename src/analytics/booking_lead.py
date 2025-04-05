import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import sys
import os
import base64
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now you can potentially use an absolute import
import utils

def analyze_lead_time_distribution(params=None):
    """
    Analyze booking lead time distribution (days between booking and arrival).
    
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
    
    # Build the SQL query for lead time analysis
    query = """
    SELECT 
        lead_time,
        hotel,
        arrival_date_year,
        arrival_date_month,
        is_canceled
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
    
    try:
        # Execute the query and load into DataFrame
        df = pd.read_sql(query, conn)
        
        # Create lead time categories for better visualization
        bins = [0, 7, 30, 90, 180, 365, float('inf')]
        labels = ['0-7 days', '8-30 days', '31-90 days', '91-180 days', '181-365 days', '365+ days']
        df['lead_time_category'] = pd.cut(df['lead_time'], bins=bins, labels=labels)
        
        # Create the visualization
        plt.figure(figsize=(12, 8))
        
        # Create a subplot layout
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot 1: Distribution by category
        if hotel_type:
            # Single hotel type - show lead time distribution
            lead_time_counts = df['lead_time_category'].value_counts().sort_index()
            ax1.bar(lead_time_counts.index, lead_time_counts.values, color='skyblue')
            
            # Add percentage labels
            total = lead_time_counts.sum()
            for i, count in enumerate(lead_time_counts):
                percentage = (count / total) * 100
                ax1.text(i, count + 5, f"{percentage:.1f}%", ha='center')
                
            ax1.set_title(f'Lead Time Distribution for {hotel_type} Hotels')
        else:
            # Multiple hotel types - show comparison
            # Fixed FutureWarning by adding observed=True parameter
            lead_time_by_hotel = df.groupby(['lead_time_category', 'hotel'], observed=True).size().unstack()
            lead_time_by_hotel.plot(kind='bar', ax=ax1, colormap='viridis')
            ax1.set_title('Lead Time Distribution by Hotel Type')
            ax1.legend(title='Hotel Type')
        
        ax1.set_xlabel('Lead Time')
        ax1.set_ylabel('Number of Bookings')
        ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Plot 2: Box plot of lead time distribution
        sns.boxplot(x='hotel', y='lead_time', data=df, ax=ax2)
        ax2.set_title('Lead Time Distribution (Box Plot)')
        ax2.set_xlabel('Hotel Type')
        ax2.set_ylabel('Lead Time (days)')
        ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Add median line and annotation
        if hotel_type:
            median_lead_time = df['lead_time'].median()
            ax2.axhline(median_lead_time, color='red', linestyle='--', alpha=0.7)
            ax2.text(0, median_lead_time + 5, f'Median: {median_lead_time:.0f} days', color='red')
        
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
    """Generate a textual explanation of the lead time distribution."""
    if df.empty:
        return "No data available for the selected parameters."
    
    hotel_type = params.get('hotel_type', None)
    year = params.get('year', None)
    month = params.get('month', None)
    
    # Start with a basic explanation
    if hotel_type:
        explanation = f"This chart shows the lead time distribution for {hotel_type} hotels"
    else:
        explanation = "This chart shows the lead time distribution across different hotel types"
    
    if year and month:
        explanation += f" in {month} {year}."
    elif year:
        explanation += f" throughout {year}."
    elif month:
        explanation += f" during {month} across all years."
    else:
        explanation += " over the entire period."
    
    # Add insights about the data
    mean_lead_time = df['lead_time'].mean()
    median_lead_time = df['lead_time'].median()
    min_lead_time = df['lead_time'].min()
    max_lead_time = df['lead_time'].max()
    
    explanation += f"\n\nBooking lead time statistics:"
    explanation += f"\n  - Average lead time: {mean_lead_time:.1f} days"
    explanation += f"\n  - Median lead time: {median_lead_time:.0f} days"
    explanation += f"\n  - Minimum lead time: {min_lead_time:.0f} days"
    explanation += f"\n  - Maximum lead time: {max_lead_time:.0f} days"
    
    # Lead time category distribution
    lead_time_dist = df['lead_time_category'].value_counts(normalize=True) * 100
    lead_time_dist = lead_time_dist.sort_index()
    
    explanation += "\n\nLead time distribution:"
    for category, percentage in lead_time_dist.items():
        explanation += f"\n  - {category}: {percentage:.1f}%"
    
    # Most common booking window
    most_common_category = df['lead_time_category'].value_counts().idxmax()
    most_common_percentage = (df['lead_time_category'].value_counts(normalize=True) * 100).max()
    
    explanation += f"\n\nThe most common booking window is {most_common_category} ({most_common_percentage:.1f}% of bookings)."
    
    # Compare hotel types if applicable
    if not hotel_type and len(df['hotel'].unique()) > 1:
        explanation += "\n\nLead time comparison by hotel type:"
        
        for hotel in df['hotel'].unique():
            hotel_data = df[df['hotel'] == hotel]
            hotel_mean = hotel_data['lead_time'].mean()
            hotel_median = hotel_data['lead_time'].median()
            
            explanation += f"\n  - {hotel}:"
            explanation += f"\n    - Average lead time: {hotel_mean:.1f} days"
            explanation += f"\n    - Median lead time: {hotel_median:.0f} days"
        
        # Identify which hotel type has longer lead times
        hotel_means = df.groupby('hotel')['lead_time'].mean()
        longer_lead_hotel = hotel_means.idxmax()
        shorter_lead_hotel = hotel_means.idxmin()
        lead_diff = hotel_means[longer_lead_hotel] - hotel_means[shorter_lead_hotel]
        
        explanation += f"\n\n{longer_lead_hotel} bookings have on average {lead_diff:.1f} days longer lead time than {shorter_lead_hotel} bookings."
    
    # Relationship between lead time and cancellation
    if 'is_canceled' in df.columns:
        canceled_mean = df[df['is_canceled'] == 1]['lead_time'].mean()
        not_canceled_mean = df[df['is_canceled'] == 0]['lead_time'].mean()
        
        explanation += "\n\nRelationship between lead time and cancellation:"
        explanation += f"\n  - Average lead time for canceled bookings: {canceled_mean:.1f} days"
        explanation += f"\n  - Average lead time for non-canceled bookings: {not_canceled_mean:.1f} days"
        
        if canceled_mean > not_canceled_mean:
            explanation += f"\n\nBookings with longer lead times are more likely to be canceled."
        else:
            explanation += f"\n\nBookings with shorter lead times are more likely to be canceled."
    
    return explanation


