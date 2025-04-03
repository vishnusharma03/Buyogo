# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from matplotlib.ticker import PercentFormatter

# %%
# Set style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)

data_path = "data/hotel_bookings.csv"
df = pd.read_csv(data_path)
df.head(5)

# 1. Cancellation Rate as Percentage of Total Bookings
print("\n--- Cancellation Rate Analysis ---\n")

# Calculate cancellation rate
total_bookings = len(df)
cancelled_bookings = df['is_canceled'].sum()
cancellation_rate = (cancelled_bookings / total_bookings) * 100

print(f"Total Bookings: {total_bookings:,}")
print(f"Cancelled Bookings: {cancelled_bookings:,}")
print(f"Cancellation Rate: {cancellation_rate:.2f}%")

# Create a pie chart to visualize cancellation rate
labels = ['Confirmed', 'Cancelled']
sizes = [(total_bookings - cancelled_bookings), cancelled_bookings]
colors = ['#66b3ff', '#ff9999']
explode = (0, 0.1)  # explode the 2nd slice (Cancelled)

plt.figure(figsize=(10, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90, textprops={'fontsize': 14})
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.title('Booking Status Distribution', fontsize=16, pad=20)
plt.tight_layout()
plt.show()

# Create a bar chart to visualize cancellation rate by hotel type
hotel_cancel = df.groupby('hotel')['is_canceled'].mean() * 100
plt.figure(figsize=(10, 6))
sns.barplot(x=hotel_cancel.index, y=hotel_cancel.values, palette='viridis')
plt.title('Cancellation Rate by Hotel Type', fontsize=16, pad=20)
plt.xlabel('Hotel Type', fontsize=14)
plt.ylabel('Cancellation Rate (%)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.tight_layout()
plt.show()

# 2. Geographical Distribution of Users
print("\n--- Geographical Distribution Analysis ---\n")

# Check if 'country' column exists in the dataset
if 'country' in df.columns:
    country_column = 'country'
else:
    # Use 'country_of_origin' or similar column if available
    potential_columns = ['country_of_origin', 'nationality', 'origin_country']
    for col in potential_columns:
        if col in df.columns:
            country_column = col
            break
    else:
        # If no direct country column, use market segment as a proxy
        country_column = 'market_segment'
        print("No direct country information found. Using market segment as a proxy for geographical distribution.")

# Count bookings by country/market segment
country_counts = df[country_column].value_counts().reset_index()
country_counts.columns = [country_column, 'count']
print(f"Top 10 {country_column} by number of bookings:")
print(country_counts.head(10))

# Create a bar chart for geographical distribution
plt.figure(figsize=(12, 8))
top_countries = country_counts.head(15)  # Show top 15 countries/segments
sns.barplot(x='count', y=country_column, data=top_countries, palette='viridis')
plt.title(f'Top 15 {country_column} by Number of Bookings', fontsize=16, pad=20)
plt.xlabel('Number of Bookings', fontsize=14)
plt.ylabel(country_column.replace('_', ' ').title(), fontsize=14)
plt.tight_layout()
plt.show()

# Create a pie chart for top 5 countries/segments
plt.figure(figsize=(10, 8))
top5 = country_counts.head(5)
others = pd.DataFrame({country_column: ['Others'], 
                      'count': [country_counts['count'][5:].sum()]})
pie_data = pd.concat([top5, others])

plt.pie(pie_data['count'], labels=pie_data[country_column], autopct='%1.1f%%', 
        startangle=90, shadow=True, explode=[0.05]*len(pie_data),
        textprops={'fontsize': 12}, colors=sns.color_palette('viridis', len(pie_data)))
plt.axis('equal')
plt.title(f'Distribution of Bookings by {country_column.replace("_", " ").title()}', fontsize=16, pad=20)
plt.tight_layout()
plt.show()

# 3. Booking Lead Time Distribution
print("\n--- Lead Time Distribution Analysis ---\n")

# Basic statistics for lead time
lead_time_stats = df['lead_time'].describe()
print("Lead Time Statistics (days):")
print(lead_time_stats)

# Create a histogram for lead time distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['lead_time'], bins=50, kde=True)
plt.title('Distribution of Booking Lead Time', fontsize=16, pad=20)
plt.xlabel('Lead Time (days)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.axvline(df['lead_time'].median(), color='red', linestyle='--', label=f'Median: {df["lead_time"].median():.0f} days')
plt.axvline(df['lead_time'].mean(), color='green', linestyle='--', label=f'Mean: {df["lead_time"].mean():.0f} days')
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# Create a box plot for lead time by hotel type
plt.figure(figsize=(10, 6))
sns.boxplot(x='hotel', y='lead_time', data=df)
plt.title('Lead Time Distribution by Hotel Type', fontsize=16, pad=20)
plt.xlabel('Hotel Type', fontsize=14)
plt.ylabel('Lead Time (days)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# Create a violin plot for a more detailed view
plt.figure(figsize=(12, 6))
sns.violinplot(x='hotel', y='lead_time', data=df, inner='quartile')
plt.title('Lead Time Distribution by Hotel Type (Violin Plot)', fontsize=16, pad=20)
plt.xlabel('Hotel Type', fontsize=14)
plt.ylabel('Lead Time (days)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# Additional analysis: Lead time by market segment
plt.figure(figsize=(14, 8))
sns.boxplot(x='market_segment', y='lead_time', data=df)
plt.title('Lead Time Distribution by Market Segment', fontsize=16, pad=20)
plt.xlabel('Market Segment', fontsize=14)
plt.ylabel('Lead Time (days)', fontsize=14)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# Summary of findings
print("\n--- Summary of Findings ---\n")
print(f"1. Cancellation Rate: {cancellation_rate:.2f}% of all bookings were cancelled.")
print(f"2. Geographical Distribution: The top source of bookings is from {country_counts.iloc[0][country_column]}.")
print(f"3. Lead Time: The average booking lead time is {df['lead_time'].mean():.0f} days, with a median of {df['lead_time'].median():.0f} days.")
