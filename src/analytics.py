import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import sys
import os
import base64
import numpy as np
from sqlalchemy import inspect, text

# Get project root path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import utility functions
from src.utils import create_db_engine

def ensure_table_exists(conn):
    """Check if the booking table exists."""
    inspector = inspect(conn)
    try:
        if inspector.has_table('hotel_bookings'):
            pass
    except Exception as e:
        print(f"Error creating table: {e}")

def analyze_revenue_trends(conn):
    """Analyze revenue trends over time from hotel bookings data."""
    
    # Build the SQL query
    query = """
    SELECT 
        arrival_date_year, 
        arrival_date_month, 
        hotel,
        SUM((stays_in_weekend_nights + stays_in_week_nights) * adr) as revenue
    FROM hotel_bookings
    WHERE is_canceled = 0
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
        
        # Create a categorical type with specific order for months
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                      'July', 'August', 'September', 'October', 'November', 'December']
        df['arrival_date_month'] = pd.Categorical(df['arrival_date_month'], categories=month_order, ordered=True)
        
        # Sort the dataframe by year and month for proper time series display
        df = df.sort_values(['arrival_date_year', 'arrival_date_month'])
        
        # Create a proper categorical order for month_year
        all_month_years = []
        for year in sorted(df['arrival_date_year'].unique()):
            for month in month_order:
                all_month_years.append(f"{month} {year}")
        
        # Filter to only include month_years that exist in the data
        existing_month_years = [my for my in all_month_years if my in df['month_year'].values]
        df['month_year'] = pd.Categorical(df['month_year'], categories=existing_month_years, ordered=True)
        
        # Create the visualization
        plt.figure(figsize=(12, 6))
        
        # Multiple hotel types - show comparison
        sns.lineplot(data=df, x='month_year', y='revenue', hue='hotel')
        plt.title('Revenue Trends by Hotel Type')
        
        plt.xlabel('Month-Year')
        plt.ylabel('Total Revenue (€)')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Generate explanation
        explanation = generate_revenue_explanation(df)
        
        # Convert plot to base64 encoded image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return {
            "heading": "Revenue Trends Analysis",
            "image": image_base64,
            "explanation": explanation
        }
    
    except Exception as e:
        print(f"Error executing revenue query: {e}")
        return {
            "heading": "Revenue Trends Analysis",
            "image": generate_error_image(str(e)),
            "explanation": f"An error occurred: {str(e)}"
        }

def analyze_cancellation_rates(conn):
    """Analyze cancellation rates as percentage of total bookings."""
    
    # Build the SQL query for cancellation analysis
    query = """
    SELECT 
        arrival_date_year,
        arrival_date_month,
        hotel,
        COUNT(*) as total_bookings,
        SUM(CASE WHEN is_canceled = 1 THEN 1 ELSE 0 END) as canceled_bookings,
        CAST(SUM(CASE WHEN is_canceled = 1 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100 as cancellation_rate
    FROM hotel_bookings
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
        
        # Create a categorical type with specific order for months
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                      'July', 'August', 'September', 'October', 'November', 'December']
        df['arrival_date_month'] = pd.Categorical(df['arrival_date_month'], categories=month_order, ordered=True)
        
        # Sort the dataframe by year and month for proper time series display
        df = df.sort_values(['arrival_date_year', 'arrival_date_month'])
        
        # Create a proper categorical order for month_year
        all_month_years = []
        for year in sorted(df['arrival_date_year'].unique()):
            for month in month_order:
                all_month_years.append(f"{month} {year}")
        
        # Filter to only include month_years that exist in the data
        existing_month_years = [my for my in all_month_years if my in df['month_year'].values]
        df['month_year'] = pd.Categorical(df['month_year'], categories=existing_month_years, ordered=True)
        
        # Create the visualization
        plt.figure(figsize=(12, 6))
        
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
        explanation = generate_cancellation_explanation(df)
        
        # Convert plot to base64 encoded image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return {
            "heading": "Cancellation Rates Analysis",
            "image": image_base64,
            "explanation": explanation
        }
    
    except Exception as e:
        print(f"Error executing cancellation query: {e}")
        return {
            "heading": "Cancellation Rates Analysis",
            "image": generate_error_image(str(e)),
            "explanation": f"An error occurred: {str(e)}"
        }

def analyze_geographical_distribution(conn):
    """Analyze geographical distribution of users doing the bookings."""
    
    # Build the SQL query for geographical analysis
    query = """
    SELECT 
        country,
        hotel,
        COUNT(*) as booking_count
    FROM hotel_bookings
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
        explanation = generate_geographical_explanation(df, country_totals)
        
        # Convert plot to base64 encoded image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return {
            "heading": "Geographical Distribution Analysis",
            "image": image_base64,
            "explanation": explanation
        }
    
    except Exception as e:
        print(f"Error executing geographical query: {e}")
        return {
            "heading": "Geographical Distribution Analysis",
            "image": generate_error_image(str(e)),
            "explanation": f"An error occurred: {str(e)}"
        }

def analyze_lead_time_distribution(conn):
    """Analyze booking lead time distribution (days between booking and arrival)."""
    
    # Build the SQL query for lead time analysis
    query = """
    SELECT 
        lead_time,
        hotel,
        arrival_date_year,
        arrival_date_month,
        is_canceled
    FROM hotel_bookings
    """
    
    try:
        # Execute the query and load into DataFrame
        df = pd.read_sql(query, conn)
        
        # Create lead time categories for better visualization
        bins = [0, 7, 30, 90, 180, 365, float('inf')]
        labels = ['0-7 days', '8-30 days', '31-90 days', '91-180 days', '181-365 days', '365+ days']
        df['lead_time_category'] = pd.cut(df['lead_time'], bins=bins, labels=labels)
        
        # Ensure lead_time_category is treated as a proper categorical type with the right order
        df['lead_time_category'] = pd.Categorical(df['lead_time_category'], categories=labels, ordered=True)
        
        # Create the visualization
        plt.figure(figsize=(12, 8))
        
        # Create a subplot layout
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot 1: Distribution by category
        # Multiple hotel types - show comparison
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
        
        plt.tight_layout()
        
        # Generate explanation
        explanation = generate_lead_time_explanation(df)
        
        # Convert plot to base64 encoded image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return {
            "heading": "Booking Lead Time Analysis",
            "image": image_base64,
            "explanation": explanation
        }
    
    except Exception as e:
        print(f"Error executing lead time query: {e}")
        return {
            "heading": "Booking Lead Time Analysis",
            "image": generate_error_image(str(e)),
            "explanation": f"An error occurred: {str(e)}"
        }

def analyze_customer_satisfaction(conn):
    """Analyze customer satisfaction based on available metrics."""
    
    # Build the SQL query for satisfaction analysis
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
    FROM hotel_bookings
    WHERE is_canceled = 0  -- Only analyze stays that actually happened
    """
    
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
        room_match_by_hotel = df.groupby('hotel')['got_reserved_room'].mean() * 100
        room_match_by_hotel.plot(kind='bar', ax=axes[0, 0], color='skyblue')
        axes[0, 0].set_ylabel('Percentage (%)')
        axes[0, 0].set_title('Room Type Match Rate by Hotel')
        axes[0, 0].set_ylim(0, 100)
        
        # Add percentage labels
        for i, v in enumerate(room_match_by_hotel):
            axes[0, 0].text(i, v + 2, f"{v:.1f}%", ha='center')
        
        # Plot 2: Repeat Guest Percentage
        repeat_by_hotel = df.groupby('hotel')['is_repeat_guest'].mean() * 100
        repeat_by_hotel.plot(kind='bar', ax=axes[0, 1], color='lightgreen')
        axes[0, 1].set_ylabel('Percentage (%)')
        axes[0, 1].set_title('Repeat Guest Rate by Hotel')
        
        # Add percentage labels
        for i, v in enumerate(repeat_by_hotel):
            axes[0, 1].text(i, v + 0.5, f"{v:.1f}%", ha='center')
        
        # Plot 3: Special Requests Distribution
        sns.histplot(data=df, x='total_of_special_requests', hue='hotel', 
                    multiple='stack', discrete=True, ax=axes[1, 0])
        axes[1, 0].set_title('Distribution of Special Requests')
        axes[1, 0].set_xlabel('Number of Special Requests')
        axes[1, 0].set_ylabel('Count')
        
        # Plot 4: Booking Changes Distribution
        sns.histplot(data=df, x='booking_changes', hue='hotel', 
                    multiple='stack', discrete=True, ax=axes[1, 1])
        axes[1, 1].set_title('Distribution of Booking Changes')
        axes[1, 1].set_xlabel('Number of Booking Changes')
        axes[1, 1].set_ylabel('Count')
        
        # Use subplots_adjust instead of tight_layout to have more control
        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.25, wspace=0.15)
        
        # Generate explanation
        explanation = generate_satisfaction_explanation(df)
        
        # Convert plot to base64 encoded image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return {
            "heading": "Customer Satisfaction Analysis",
            "image": image_base64,
            "explanation": explanation
        }
    
    except Exception as e:
        print(f"Error executing satisfaction query: {e}")
        return {
            "heading": "Customer Satisfaction Analysis",
            "image": generate_error_image(str(e)),
            "explanation": f"An error occurred: {str(e)}"
        }

def generate_error_image(error_message):
    """Generate a placeholder image with error message."""
    plt.figure(figsize=(8, 4))
    plt.text(0.5, 0.5, f"Error: {error_message}", ha='center', va='center', fontsize=12)
    plt.axis('off')
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return image_base64

def generate_revenue_explanation(df):
    """Generate a textual explanation of the revenue trends."""
    if df.empty:
        return "No data available for the analysis."
    
    # Add insights about the data
    total_revenue = df['revenue'].sum()
    avg_revenue = df['revenue'].mean()
    max_revenue = df['revenue'].max()
    max_revenue_period = df.loc[df['revenue'].idxmax()]
    
    explanation = "This chart shows the revenue trends across different hotel types over the entire period."
    explanation += f"\n\nTotal revenue: €{total_revenue:,.2f}"
    explanation += f"\nAverage monthly revenue: €{avg_revenue:,.2f}"
    explanation += f"\nHighest revenue (€{max_revenue:,.2f}) was recorded in {max_revenue_period['month_year']}"
    
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

def generate_cancellation_explanation(df):
    """Generate a textual explanation of the cancellation rates."""
    if df.empty:
        return "No data available for the analysis."
    
    # Add insights about the data
    total_bookings = df['total_bookings'].sum()
    total_cancellations = df['canceled_bookings'].sum()
    overall_rate = (total_cancellations / total_bookings) * 100
    
    explanation = "This chart shows the cancellation rates across different hotel types over the entire period."
    explanation += f"\n\nTotal bookings: {total_bookings:,}"
    explanation += f"\nTotal cancellations: {total_cancellations:,}"
    explanation += f"\nOverall cancellation rate: {overall_rate:.2f}%"
    
    # Find highest and lowest cancellation rates
    max_rate_row = df.loc[df['cancellation_rate'].idxmax()]
    min_rate_row = df.loc[df['cancellation_rate'].idxmin()]
    
    explanation += f"\n\nHighest cancellation rate: {max_rate_row['cancellation_rate']:.2f}% ({max_rate_row['month_year']}, {max_rate_row['hotel']})"
    explanation += f"\nLowest cancellation rate: {min_rate_row['cancellation_rate']:.2f}% ({min_rate_row['month_year']}, {min_rate_row['hotel']})"
    
    # Compare cancellation rates between hotel types
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
            yearly_stats = df.groupby('arrival_date_year').agg({
                'canceled_bookings': 'sum',
                'total_bookings': 'sum'
            })
            yearly_rates = (yearly_stats['canceled_bookings'] / yearly_stats['total_bookings']) * 100
            
            trend_direction = "increasing" if yearly_rates.iloc[-1] > yearly_rates.iloc[0] else "decreasing"
            explanation += f"\n\nThe overall cancellation rate trend is {trend_direction} over the years."
    
    return explanation

def generate_geographical_explanation(df, country_totals):
    """Generate a textual explanation of the geographical distribution."""
    if df.empty:
        return "No data available for the analysis."
    
    # Add insights about the data
    total_bookings = df['booking_count'].sum()
    total_countries = df['country'].nunique()
    
    explanation = "This chart shows the geographical distribution of bookings across different hotel types over the entire period."
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
    
    # Hotel type distribution by country
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

def generate_lead_time_explanation(df):
    """Generate a textual explanation of the lead time distribution."""
    if df.empty:
        return "No data available for the analysis."
    
    # Add insights about the data
    mean_lead_time = df['lead_time'].mean()
    median_lead_time = df['lead_time'].median()
    min_lead_time = df['lead_time'].min()
    max_lead_time = df['lead_time'].max()
    
    explanation = "This chart shows the lead time distribution across different hotel types over the entire period."
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
    
    # Compare hotel types
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

def generate_satisfaction_explanation(df):
    """Generate a textual explanation of the customer satisfaction metrics."""
    if df.empty:
        return "No data available for the analysis."
    
    # Add insights about the data
    total_bookings = len(df)
    room_match_rate = df['got_reserved_room'].mean() * 100
    repeat_guest_rate = df['is_repeat_guest'].mean() * 100
    avg_special_requests = df['total_of_special_requests'].mean()
    
    explanation = "This analysis shows customer satisfaction metrics across different hotel types over the entire period."
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
    
    # Compare hotel types
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

def run_all_analytics():
    """Run all analytics functions and return results in a structured format."""

    # Get database connection
    conn = create_db_engine()
    
    # Check if table exists, if not, create it from CSV
    ensure_table_exists(conn)

    # Run all analytics functions
    try:
        # Run all analytics functions
        revenue_analysis = analyze_revenue_trends(conn)
        cancellation_analysis = analyze_cancellation_rates(conn)
        geographical_analysis = analyze_geographical_distribution(conn)
        lead_time_analysis = analyze_lead_time_distribution(conn)
        satisfaction_analysis = analyze_customer_satisfaction(conn)
        
        # Combine all results into a single dictionary
        all_results = {
            "revenue": revenue_analysis,
            "cancellation": cancellation_analysis,
            "geographical": geographical_analysis,
            "booking_lead": lead_time_analysis,
            "satisfaction": satisfaction_analysis
        }
        
        return all_results
    
    except Exception as e:
        print(f"Error running analytics: {e}")
        return {
            "error": str(e)
        }

# if __name__ == "__main__":
#     # Run all analytics and print results
#     results = run_all_analytics()
#     print("Analytics completed successfully!")
#     print(f"Generated {len(results)} analytics reports.")

    
#     # You can save the results to a JSON file if needed
#     import json
#     with open('analytics_results.json', 'w') as f:
#         json.dump(results, f, indent=2)