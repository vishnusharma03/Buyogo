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

def analyze_customer_satisfaction(params=None):
    """
    Analyze customer satisfaction based on available metrics.
    
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
    
    # Build the SQL query for satisfaction analysis
    # Since direct satisfaction scores might not be available, we'll use proxies like:
    # - Repeated guests (customer_type = 'Repeated Guest')
    # - Special requests (total_of_special_requests)
    # - Previous cancellations and no-shows
    query = """
    SELECT 
        hotel,
        arrival_date_year,
        arrival_date_month,
        customer_type,
        total_of_special_requests,
        previous_cancellations,
        previous_bookings_not_canceled,
        reservation_status,
        reserved_room_type,
        assigned_room_type,
        booking_changes
    FROM booking
    WHERE is_canceled = 0  -- Only analyze stays that actually happened
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
        if "WHERE" in query:
            query += " AND " + " AND ".join(where_clauses)
        else:
            query += " WHERE " + " AND ".join(where_clauses)
    
    try:
        # Execute the query and load into DataFrame
        df = pd.read_sql(query, conn)
        
        # Create derived satisfaction metrics
        # 1. Room type match (got the room type they reserved)
        df['got_reserved_room'] = (df['reserved_room_type'] == df['assigned_room_type']).astype(int)
        
        # 2. Repeat customer ratio
        df['is_repeat_guest'] = (df['customer_type'] == 'Repeat Guest').astype(int)
        
        # 3. Previous loyalty (previous bookings not canceled)
        df['has_previous_stays'] = (df['previous_bookings_not_canceled'] > 0).astype(int)
        
        # Create the visualization - multiple plots to show different satisfaction metrics
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # Plot 1: Room Type Match Rate by Hotel
        if hotel_type:
            room_match_rate = df['got_reserved_room'].mean() * 100
            axes[0, 0].bar(['Room Type Match Rate'], [room_match_rate], color='skyblue')
            axes[0, 0].set_ylim(0, 100)
            axes[0, 0].set_ylabel('Percentage (%)')
            axes[0, 0].set_title(f'Room Type Match Rate for {hotel_type}')
            axes[0, 0].text(0, room_match_rate + 2, f"{room_match_rate:.1f}%", ha='center')
        else:
            room_match_by_hotel = df.groupby('hotel')['got_reserved_room'].mean() * 100
            room_match_by_hotel.plot(kind='bar', ax=axes[0, 0], color='skyblue')
            axes[0, 0].set_ylabel('Percentage (%)')
            axes[0, 0].set_title('Room Type Match Rate by Hotel')
            axes[0, 0].set_ylim(0, 100)
            
            # Add percentage labels
            for i, v in enumerate(room_match_by_hotel):
                axes[0, 0].text(i, v + 2, f"{v:.1f}%", ha='center')
        
        # Plot 2: Repeat Guest Percentage
        if hotel_type:
            repeat_rate = df['is_repeat_guest'].mean() * 100
            axes[0, 1].bar(['Repeat Guest Rate'], [repeat_rate], color='lightgreen')
            axes[0, 1].set_ylim(0, max(repeat_rate * 1.2, 10))  # Ensure some space above the bar
            axes[0, 1].set_ylabel('Percentage (%)')
            axes[0, 1].set_title(f'Repeat Guest Rate for {hotel_type}')
            axes[0, 1].text(0, repeat_rate + 0.5, f"{repeat_rate:.1f}%", ha='center')
        else:
            repeat_by_hotel = df.groupby('hotel')['is_repeat_guest'].mean() * 100
            repeat_by_hotel.plot(kind='bar', ax=axes[0, 1], color='lightgreen')
            axes[0, 1].set_ylabel('Percentage (%)')
            axes[0, 1].set_title('Repeat Guest Rate by Hotel')
            
            # Add percentage labels
            for i, v in enumerate(repeat_by_hotel):
                axes[0, 1].text(i, v + 0.5, f"{v:.1f}%", ha='center')
        
        # Plot 3: Special Requests Distribution
        sns.histplot(data=df, x='total_of_special_requests', hue='hotel' if not hotel_type else None, 
                    multiple='stack', discrete=True, ax=axes[1, 0])
        axes[1, 0].set_title('Distribution of Special Requests')
        axes[1, 0].set_xlabel('Number of Special Requests')
        axes[1, 0].set_ylabel('Count')
        
        # Plot 4: Booking Changes Distribution
        sns.histplot(data=df, x='booking_changes', hue='hotel' if not hotel_type else None, 
                    multiple='stack', discrete=True, ax=axes[1, 1])
        axes[1, 1].set_title('Distribution of Booking Changes')
        axes[1, 1].set_xlabel('Number of Booking Changes')
        axes[1, 1].set_ylabel('Count')
        
        # Use subplots_adjust instead of tight_layout to have more control
        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.25, wspace=0.15)
        
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
    """Generate a textual explanation of the customer satisfaction metrics."""
    if df.empty:
        return "No data available for the selected parameters."
    
    hotel_type = params.get('hotel_type', None)
    year = params.get('year', None)
    month = params.get('month', None)
    
    # Start with a basic explanation
    if hotel_type:
        explanation = f"This analysis shows customer satisfaction metrics for {hotel_type} hotels"
    else:
        explanation = "This analysis shows customer satisfaction metrics across different hotel types"
    
    if year and month:
        explanation += f" in {month} {year}."
    elif year:
        explanation += f" throughout {year}."
    elif month:
        explanation += f" during {month} across all years."
    else:
        explanation += " over the entire period."
    
    # Add insights about the data
    total_bookings = len(df)
    room_match_rate = df['got_reserved_room'].mean() * 100
    repeat_guest_rate = df['is_repeat_guest'].mean() * 100
    avg_special_requests = df['total_of_special_requests'].mean()
    
    explanation += f"\n\nTotal completed stays analyzed: {total_bookings:,}"
    explanation += f"\n\nKey satisfaction indicators:"
    explanation += f"\n  - Room type match rate: {room_match_rate:.1f}% (guests received their reserved room type)"
    explanation += f"\n  - Repeat guest rate: {repeat_guest_rate:.1f}% (returning customers)"
    explanation += f"\n  - Average special requests per booking: {avg_special_requests:.2f}"
    
    # Special requests distribution
    special_req_dist = df['total_of_special_requests'].value_counts(normalize=True).sort_index() * 100
    
    explanation += "\n\nSpecial requests distribution:"
    for req, percentage in special_req_dist.items():
        explanation += f"\n  - {req} requests: {percentage:.1f}%"
    
    # Compare hotel types if applicable
    if not hotel_type and len(df['hotel'].unique()) > 1:
        explanation += "\n\nSatisfaction comparison by hotel type:"
        
        for hotel in df['hotel'].unique():
            hotel_data = df[df['hotel'] == hotel]
            hotel_room_match = hotel_data['got_reserved_room'].mean() * 100
            hotel_repeat_rate = hotel_data['is_repeat_guest'].mean() * 100
            hotel_special_req = hotel_data['total_of_special_requests'].mean()
            
            explanation += f"\n\n{hotel}:"
            explanation += f"\n  - Room type match rate: {hotel_room_match:.1f}%"
            explanation += f"\n  - Repeat guest rate: {hotel_repeat_rate:.1f}%"
            explanation += f"\n  - Average special requests: {hotel_special_req:.2f}"
        
        # Compare which hotel has better satisfaction metrics
        room_match_by_hotel = df.groupby('hotel')['got_reserved_room'].mean()
        repeat_by_hotel = df.groupby('hotel')['is_repeat_guest'].mean()
        
        better_room_match = room_match_by_hotel.idxmax()
        better_repeat_rate = repeat_by_hotel.idxmax()
        
        explanation += f"\n\n{better_room_match} hotels have a higher room type match rate."
        explanation += f"\n{better_repeat_rate} hotels have a higher repeat guest rate."
    
    # Additional insights
    has_booking_changes = (df['booking_changes'] > 0).mean() * 100
    explanation += f"\n\n{has_booking_changes:.1f}% of bookings had changes after the initial reservation."
    
    # Room type mismatch analysis
    if (df['got_reserved_room'] == 0).any():
        mismatch_df = df[df['got_reserved_room'] == 0]
        common_reserved = mismatch_df['reserved_room_type'].value_counts().idxmax()
        common_assigned = mismatch_df['assigned_room_type'].value_counts().idxmax()
        
        explanation += f"\n\nWhen room types don't match, guests most commonly reserved room type '{common_reserved}' "
        explanation += f"but were assigned room type '{common_assigned}'."
    
    return explanation


