# Buyogo - Hotel Booking Analytics System

## Overview

Buyogo is a comprehensive hotel booking analytics system that combines traditional data analysis with modern RAG (Retrieval Augmented Generation) capabilities. The system allows users to query hotel booking data using natural language, generate various analytics reports, and visualize booking trends.

## Features

- **Natural Language Querying**: Ask questions about hotel booking data in plain English
- **Advanced Analytics**: Generate insights on revenue, cancellations, geographical distribution, and more
- **Interactive Visualizations**: View data trends through automatically generated charts and graphs
- **Vector Database Integration**: Uses ChromaDB for efficient similarity search and retrieval
- **RESTful API**: Access all functionality through a FastAPI-based interface

## Tech Stack

- **Backend**: Python 3.11+ with FastAPI
- **Database**: SQLite for relational data, ChromaDB for vector embeddings
- **AI/ML**: Gemini AI for natural language processing
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Containerization**: Docker

## Installation

### Prerequisites

- Docker and Docker Compose
- API keys for Gemini AI

### Environment Setup

1. Clone the repository
2. Create a `.env` file in the project root with the following variables:

```
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
HUGGINGFACEHUB_API_KEY=your_huggingface_api_key
```

### Running with Docker

```bash
# Build and start the containers
docker-compose up -d

# Check logs
docker-compose logs -f
```

### Manual Installation

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn main:app --host 0.0.0.0 --port 9000 --reload
```

## Usage

### API Endpoints

#### Natural Language Queries

```http
POST /ask
Content-Type: application/json

{
  "query": "What was the average length of stay for resort hotel bookings in August?"
}
```

#### Analytics Reports

```http
POST /analytics
Content-Type: application/json

{
  "analysis_type": "revenue"
}
```

Available analysis types:
- `revenue`: Revenue trends over time
- `cancellation`: Cancellation rates analysis
- `geographical`: Geographical distribution of bookings
- `booking_lead`: Lead time distribution analysis
- `satisfaction`: Customer satisfaction analysis

## Project Structure

```
├── analytics/              # Analytics modules for different metrics
│   ├── booking_lead.py     # Lead time distribution analysis
│   ├── cancellation.py     # Cancellation rates analysis
│   ├── geographical.py     # Geographical distribution analysis
│   ├── revenue.py          # Revenue trends analysis
│   └── satisfaction.py     # Customer satisfaction analysis
├── chroma_db/              # ChromaDB persistence directory
├── data/                   # Data files
│   └── hotel_bookings.csv  # Hotel bookings dataset
├── agent_rag.py            # RAG implementation with CodeAgent
├── compose.yaml            # Docker Compose configuration
├── config.yaml             # Application configuration
├── database.py             # Database setup and operations
├── Dockerfile              # Docker image definition
├── embed.py                # Vector embedding functionality
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── run.sh                  # Application startup script
└── utils.py                # Utility functions
```

## Configuration

The application can be configured through the `config.yaml` file. Key configuration options include:

- **HTTP Server Settings**:
  - `port`: The port on which the server listens (default: 8000)
  - `listen_address`: The address to bind to (default: "0.0.0.0")
  - `max_payload_size_bytes`: Maximum allowed payload size (default: 41943040)

- **ChromaDB Settings**:
  - `persist_path`: Directory for ChromaDB persistence (default: "./chroma")
  - `allow_reset`: Whether to allow database reset (default: false)

## Data Model

The system uses a hotel bookings dataset with the following key fields:

- `hotel`: Hotel type (Resort Hotel or City Hotel)
- `is_canceled`: Whether the booking was canceled (1) or not (0)
- `lead_time`: Number of days between booking and arrival
- `arrival_date_*`: Arrival date components (year, month, day)
- `stays_in_weekend_nights`: Number of weekend nights
- `stays_in_week_nights`: Number of weekday nights
- `adults`, `children`, `babies`: Number of guests by age category
- `meal`: Type of meal package
- `country`: Country of origin
- `market_segment`: Market segment designation
- `distribution_channel`: Booking distribution channel
- `reserved_room_type`: Code of room type reserved
- `assigned_room_type`: Code of room type assigned
- `booking_changes`: Number of changes to the booking
- `deposit_type`: Type of deposit made
- `adr`: Average Daily Rate
- `reservation_status`: Reservation status (Canceled, Check-Out, No-Show)

## Docker Support

The application is containerized using Docker. The `Dockerfile` sets up a Python environment with all necessary dependencies and configures the application for production use.

Key Docker features:
- Based on the ChromaDB official image
- Includes all required Python packages
- Configures environment variables
- Sets up health checks
- Exposes port 9000

## License

[Specify your license here]

## Contributing

[Add contribution guidelines here]