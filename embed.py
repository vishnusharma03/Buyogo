import os
from tqdm import tqdm
import pandas as pd
import sqlite3 as sql
from google import genai
import chromadb as cdb
from utils import load_api_keys, create_db_engine
import chromadb.utils.embedding_functions as embedding_functions
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='embed.log'
)
logger = logging.getLogger('hotel_embedding')

def safe_value(val):
    """Return the value if not null, otherwise return 'missing'."""
    return str(val) if pd.notnull(val) else "missing"

def format_date(year, month, day):
    """Format date components into a string, handling missing values."""
    if "missing" in (str(year), str(month), str(day)):
        return "missing"
    return f"{year}-{month}-{day}"

def calculate_total_nights(weekend_nights, week_nights):
    """Calculate total nights, handling missing values."""
    if weekend_nights == "missing" or week_nights == "missing":
        return "missing"
    try:
        return int(float(weekend_nights)) + int(float(week_nights))
    except (ValueError, TypeError):
        return "missing"

def create_document_string(row):
    """Create a document string from a DataFrame row."""
    arrival_year = safe_value(row.get('arrival_date_year'))
    arrival_month = safe_value(row.get('arrival_date_month'))
    arrival_day = safe_value(row.get('arrival_date_day_of_month'))
    arrival_date_str = format_date(arrival_year, arrival_month, arrival_day)
    
    reservation_status_date = safe_value(row.get('reservation_status_date'))
    
    stays_weekend = safe_value(row.get('stays_in_weekend_nights'))
    stays_week = safe_value(row.get('stays_in_week_nights'))
    total_nights = calculate_total_nights(stays_weekend, stays_week)
    
    document = f"""
        Hotel: {safe_value(row.get('hotel'))}
        is_canceled: {safe_value(row.get('is_canceled'))}
        lead_time: {safe_value(row.get('lead_time'))}
        arrival_date: {arrival_date_str}
        stays_in_weekend_nights: {stays_weekend}
        stays_in_week_nights: {stays_week}
        total_nights: {total_nights}
        adults: {safe_value(row.get('adults'))}
        children: {safe_value(row.get('children'))}
        babies: {safe_value(row.get('babies'))}
        meal: {safe_value(row.get('meal'))}
        country: {safe_value(row.get('country'))}
        market_segment: {safe_value(row.get('market_segment'))}
        distribution_channel: {safe_value(row.get('distribution_channel'))}
        is_repeated_guest: {safe_value(row.get('is_repeated_guest'))}
        previous_cancellations: {safe_value(row.get('previous_cancellations'))}
        previous_bookings_not_canceled: {safe_value(row.get('previous_bookings_not_canceled'))}
        reserved_room_type: {safe_value(row.get('reserved_room_type'))}
        assigned_room_type: {safe_value(row.get('assigned_room_type'))}
        booking_changes: {safe_value(row.get('booking_changes'))}
        deposit_type: {safe_value(row.get('deposit_type'))}
        agent: {safe_value(row.get('agent'))}
        company: {safe_value(row.get('company'))}
        days_in_waiting_list: {safe_value(row.get('days_in_waiting_list'))}
        customer_type: {safe_value(row.get('customer_type'))}
        adr: {safe_value(row.get('adr'))}
        required_car_parking_spaces: {safe_value(row.get('required_car_parking_spaces'))}
        total_of_special_requests: {safe_value(row.get('total_of_special_requests'))}
        reservation_status: {safe_value(row.get('reservation_status'))}
        reservation_status_date: {reservation_status_date}
    """
    return document.strip()

def create_metadata_dict(idx, row):
    """Create a metadata dictionary from a DataFrame row."""
    arrival_year = safe_value(row.get('arrival_date_year'))
    arrival_month = safe_value(row.get('arrival_date_month'))
    arrival_day = safe_value(row.get('arrival_date_day_of_month'))
    arrival_date_str = format_date(arrival_year, arrival_month, arrival_day)
    
    reservation_status_date = safe_value(row.get('reservation_status_date'))
    
    stays_weekend = safe_value(row.get('stays_in_weekend_nights'))
    stays_week = safe_value(row.get('stays_in_week_nights'))
    total_nights = calculate_total_nights(stays_weekend, stays_week)
    
    return {
        'booking_id': str(idx),
        'hotel': safe_value(row.get('hotel')),
        'is_canceled': safe_value(row.get('is_canceled')),
        'lead_time': safe_value(row.get('lead_time')),
        'arrival_date': arrival_date_str,
        'stays_in_weekend_nights': stays_weekend,
        'stays_in_week_nights': stays_week,
        'total_nights': total_nights,
        'adults': safe_value(row.get('adults')),
        'children': safe_value(row.get('children')),
        'babies': safe_value(row.get('babies')),
        'meal': safe_value(row.get('meal')),
        'country': safe_value(row.get('country')),
        'market_segment': safe_value(row.get('market_segment')),
        'distribution_channel': safe_value(row.get('distribution_channel')),
        'is_repeated_guest': safe_value(row.get('is_repeated_guest')),
        'previous_cancellations': safe_value(row.get('previous_cancellations')),
        'previous_bookings_not_canceled': safe_value(row.get('previous_bookings_not_canceled')),
        'reserved_room_type': safe_value(row.get('reserved_room_type')),
        'assigned_room_type': safe_value(row.get('assigned_room_type')),
        'booking_changes': safe_value(row.get('booking_changes')),
        'deposit_type': safe_value(row.get('deposit_type')),
        'agent': safe_value(row.get('agent')),
        'company': safe_value(row.get('company')),
        'days_in_waiting_list': safe_value(row.get('days_in_waiting_list')),
        'customer_type': safe_value(row.get('customer_type')),
        'adr': safe_value(row.get('adr')),
        'required_car_parking_spaces': safe_value(row.get('required_car_parking_spaces')),
        'total_of_special_requests': safe_value(row.get('total_of_special_requests')),
        'reservation_status': safe_value(row.get('reservation_status')),
        'reservation_status_date': reservation_status_date
    }

def process_dataframe(df):
    """Process DataFrame and return documents and metadatas."""
    if df is None or df.empty:
        logger.warning("Empty DataFrame provided")
        return [], []
        
    required_columns = [
        'arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month',
        'stays_in_weekend_nights', 'stays_in_week_nights'
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    documents = []
    metadatas = []
    
    for idx, row in df.iterrows():
        try:
            document = create_document_string(row)
            metadata = create_metadata_dict(idx, row)
            documents.append(document)
            metadatas.append(metadata)
        except Exception as e:
            logger.error(f"Error processing row {idx}: {str(e)}")
            continue
    
    return documents, metadatas

def save_upload_progress(start_batch, collection_name="hotel_collection", progress_file="upload_progress.txt"):
    """Save the current upload progress to a file."""
    try:
        progress_dir = os.path.dirname(progress_file)
        if progress_dir and not os.path.exists(progress_dir):
            os.makedirs(progress_dir)
            
        with open(progress_file, 'w') as f:
            f.write(f"{start_batch},{collection_name}")
        logger.info(f"Saved upload progress: batch {start_batch}")
        return True
    except Exception as e:
        logger.error(f"Error saving upload progress: {str(e)}")
        return False

def load_upload_progress(progress_file="upload_progress.txt", default_batch=0, default_collection="hotel_collection"):
    """Load the upload progress from a file."""
    try:
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                content = f.read().strip()
                if content:
                    parts = content.split(',')
                    batch = int(parts[0]) if parts else default_batch
                    collection = parts[1] if len(parts) > 1 else default_collection
                    logger.info(f"Loaded upload progress: resuming from batch {batch}")
                    return batch, collection
        return default_batch, default_collection
    except Exception as e:
        logger.error(f"Error loading upload progress: {str(e)}")
        return default_batch, default_collection

def initialize_chroma_collection(api_key, collection_name="hotel_collection", path="./Database", reset=False):
    """Initialize ChromaDB client and create collection or load if it exists."""
    try:
        embedding_collection = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=api_key,
            model_name="models/text-embedding-004"
        )
        
        client = cdb.PersistentClient(path=path)
        
        if reset:
            try:
                client.delete_collection(collection_name)
                logger.info(f"Deleted existing collection: {collection_name}")
            except Exception:
                logger.info("No existing collection to delete")
            
            collection = client.create_collection(
                name=collection_name,
                embedding_function=embedding_collection
            )
            logger.info(f"Created new collection: {collection_name}")
        else:
            try:
                collection = client.get_collection(
                    name=collection_name,
                    embedding_function=embedding_collection
                )
                doc_count = collection.count()
                logger.info(f"Loaded existing collection: {collection_name} with {doc_count} documents")
            except Exception:
                collection = client.create_collection(
                    name=collection_name,
                    embedding_function=embedding_collection
                )
                logger.info(f"Created new collection: {collection_name}")
        
        return collection
    except Exception as e:
        logger.error(f"Error initializing ChromaDB collection: {str(e)}")
        raise

def add_documents_to_collection(collection, documents, metadatas, batch_size=1000, start_batch=None, 
                              progress_file="upload_progress.txt", save_interval=5):
    """Add documents to ChromaDB collection in batches with progress tracking and resumption."""
    try:
        if not documents or not metadatas:
            logger.warning("No documents to add")
            return True
            
        existing_count = collection.count()
        logger.info(f"Current document count in collection: {existing_count}")
        
        num_batches = (len(documents) + batch_size - 1) // batch_size
        
        if start_batch is None:
            start_batch, _ = load_upload_progress(progress_file, default_batch=0)
            
        logger.info(f"Starting upload from batch {start_batch}/{num_batches}")
        
        last_successful_batch = start_batch - 1
        doc_ids = [f"doc_{j}" for j in range(len(documents))]
        
        for i in tqdm(range(start_batch, num_batches)):
            try:
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(documents))
                
                collection.add(
                    documents=documents[start_idx:end_idx],
                    metadatas=metadatas[start_idx:end_idx],
                    ids=doc_ids[start_idx:end_idx]
                )
                
                last_successful_batch = i
                logger.info(f"Added batch {i+1}/{num_batches}")
                
                if i % save_interval == 0:
                    save_upload_progress(i + 1, collection.name, progress_file)
                
            except Exception as e:
                logger.error(f"Error adding batch {i+1}: {str(e)}")
                save_upload_progress(last_successful_batch + 1, collection.name, progress_file)
                
                if "Connection" in str(e) or "Timeout" in str(e):
                    logger.error("Connection issue detected. Stopping upload process.")
                    return False
                continue
        
        save_upload_progress(num_batches, collection.name, progress_file)
        logger.info("All documents have been added to the vector database")
        return True
        
    except Exception as e:
        logger.error(f"Error adding documents to collection: {str(e)}")
        raise

def main(batch_size=1000, start_batch=None, progress_file="upload_progress.txt", 
         custom_df=None, reset=False):
    """Main function to run the embedding process."""
    engine = None
    try:
        api_keys = load_api_keys()
        engine = create_db_engine()
        
        gemini_api_key = api_keys.get('gemini')
        if not gemini_api_key:
            raise ValueError("Gemini API key not found")
        
        try:
            if custom_df is not None:
                df = custom_df
                logger.info(f"Using provided DataFrame with {len(df)} rows")
            else:
                df = pd.read_sql_query("SELECT * FROM booking", engine)
                logger.info(f"Loaded {len(df)} rows from database")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
        
        documents, metadatas = process_dataframe(df)
        logger.info(f"Processed {len(documents)} documents")
        
        collection = initialize_chroma_collection(gemini_api_key, reset=reset)
        
        success = add_documents_to_collection(
            collection, 
            documents, 
            metadatas, 
            batch_size=batch_size,
            start_batch=start_batch,
            progress_file=progress_file
        )
        
        if success:
            logger.info("Embedding process completed successfully")
        else:
            logger.warning("Embedding process completed with some issues")
            
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise
    finally:
        if engine:
            engine.dispose()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Embed hotel booking data into ChromaDB')
    parser.add_argument('--batch-size', type=int, default=1000, 
                       help='Batch size for uploading documents')
    parser.add_argument('--start-batch', type=int, default=None, 
                       help='Batch to start from (overrides progress file)')
    parser.add_argument('--progress-file', type=str, default='upload_progress.txt', 
                       help='File to save/load progress')
    parser.add_argument('--csv-file', type=str, help='Optional CSV file to use instead of database')
    parser.add_argument('--reset', action='store_true', 
                       help='Reset the collection before adding documents')
    
    args = parser.parse_args()
    
    custom_df = None
    if args.csv_file:
        try:
            custom_df = pd.read_csv(args.csv_file)
            logger.info(f"Loaded CSV file: {args.csv_file}")
        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            raise
    
    main(
        batch_size=args.batch_size,
        start_batch=args.start_batch,
        progress_file=args.progress_file,
        custom_df=custom_df,
        reset=args.reset
    )










# import os
# from tqdm import tqdm
# import pandas as pd
# import sqlite3 as sql
# from google import genai
# import chromadb as cdb
# from utils import load_api_keys, create_db_engine
# import chromadb.utils.embedding_functions as embedding_functions
# import logging

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     filename='embed.log'
# )
# logger = logging.getLogger('hotel_embedding')

# def safe_value(val):
#     """Return the value if not null, otherwise return 'missing'."""
#     return val if pd.notnull(val) else "missing"

# def format_date(year, month, day):
#     """Format date components into a string, handling missing values."""
#     if "missing" in (year, month, day):
#         return "missing"
#     return f"{year}-{month}-{day}"

# def calculate_total_nights(weekend_nights, week_nights):
#     """Calculate total nights, handling missing values."""
#     if weekend_nights == "missing" or week_nights == "missing":
#         return "missing"
#     return int(weekend_nights) + int(week_nights)

# def create_document_string(row):
#     """Create a document string from a DataFrame row."""
#     # Process date components
#     arrival_year = safe_value(row['arrival_date_year'])
#     arrival_month = safe_value(row['arrival_date_month'])
#     arrival_day = safe_value(row['arrival_date_day_of_month'])
#     arrival_date_str = format_date(arrival_year, arrival_month, arrival_day)
    
#     reservation_status = safe_value(row['reservation_status_date'])
    
#     # Calculate total nights
#     stays_weekend = safe_value(row['stays_in_weekend_nights'])
#     stays_week = safe_value(row['stays_in_week_nights'])
#     total_nights = calculate_total_nights(stays_weekend, stays_week)
    
#     # Create document string
#     document = f"""
#         Hotel: {safe_value(row['hotel'])}
#         is_canceled: {safe_value(row['is_canceled'])}
#         lead_time: {safe_value(row['lead_time'])}
#         arrival_date_year: {arrival_year}
#         arrival_date_month: {arrival_month}
#         arrival_date_week_number: {safe_value(row['arrival_date_week_number'])}
#         arrival_date_day_of_month: {arrival_day}
#         stays_in_weekend_nights: {stays_weekend}
#         stays_in_week_nights: {stays_week}
#         adults: {safe_value(row['adults'])}
#         children: {safe_value(row['children'])}
#         babies: {safe_value(row['babies'])}
#         meal: {safe_value(row['meal'])}
#         country: {safe_value(row['country'])}
#         market_segment: {safe_value(row['market_segment'])}
#         distribution_channel: {safe_value(row['distribution_channel'])}
#         is_repeated_guest: {safe_value(row['is_repeated_guest'])}
#         previous_cancellations: {safe_value(row['previous_cancellations'])}
#         previous_bookings_not_canceled: {safe_value(row['previous_bookings_not_canceled'])}
#         reserved_room_type: {safe_value(row['reserved_room_type'])}
#         assigned_room_type: {safe_value(row['assigned_room_type'])}
#         booking_changes: {safe_value(row['booking_changes'])}
#         deposit_type: {safe_value(row['deposit_type'])}
#         agent: {safe_value(row['agent'])}
#         company: {safe_value(row['company'])}
#         days_in_waiting_list: {safe_value(row['days_in_waiting_list'])}
#         customer_type: {safe_value(row['customer_type'])}
#         adr: {safe_value(row['adr'])}
#         required_car_parking_spaces: {safe_value(row['required_car_parking_spaces'])}
#         total_of_special_requests: {safe_value(row['total_of_special_requests'])}
#         Combined arrival_date: {arrival_date_str}
#         reservation_status_date: {reservation_status}
#     """
#     return document.strip()

# def create_metadata_dict(idx, row):
#     """Create a metadata dictionary from a DataFrame row."""
#     # Process date components
#     arrival_year = safe_value(row['arrival_date_year'])
#     arrival_month = safe_value(row['arrival_date_month'])
#     arrival_day = safe_value(row['arrival_date_day_of_month'])
#     arrival_date_str = format_date(arrival_year, arrival_month, arrival_day)
    
#     reservation_status = safe_value(row['reservation_status_date'])
    
#     # Calculate total nights
#     stays_weekend = safe_value(row['stays_in_weekend_nights'])
#     stays_week = safe_value(row['stays_in_week_nights'])
#     total_nights = calculate_total_nights(stays_weekend, stays_week)
    
#     # Create metadata dictionary
#     return {
#         'booking_id': str(idx),
#         'hotel': safe_value(row['hotel']),
#         'is_canceled': safe_value(row['is_canceled']),
#         'lead_time': safe_value(row['lead_time']),
#         'arrival_date_year': safe_value(row['arrival_date_year']),
#         'arrival_date_month': safe_value(row['arrival_date_month']),
#         'arrival_date_week_number': safe_value(row['arrival_date_week_number']),
#         'arrival_date_day_of_month': safe_value(row['arrival_date_day_of_month']),
#         'stays_in_weekend_nights': safe_value(row['stays_in_weekend_nights']),
#         'stays_in_week_nights': safe_value(row['stays_in_week_nights']),
#         'adults': safe_value(row['adults']),
#         'children': safe_value(row['children']),
#         'babies': safe_value(row['babies']),
#         'meal': safe_value(row['meal']),
#         'country': safe_value(row['country']),
#         'market_segment': safe_value(row['market_segment']),
#         'distribution_channel': safe_value(row['distribution_channel']),
#         'is_repeated_guest': safe_value(row['is_repeated_guest']),
#         'previous_cancellations': safe_value(row['previous_cancellations']),
#         'previous_bookings_not_canceled': safe_value(row['previous_bookings_not_canceled']),
#         'reserved_room_type': safe_value(row['reserved_room_type']),
#         'assigned_room_type': safe_value(row['assigned_room_type']),
#         'booking_changes': safe_value(row['booking_changes']),
#         'deposit_type': safe_value(row['deposit_type']),
#         'agent': safe_value(row['agent']),
#         'company': safe_value(row['company']),
#         'days_in_waiting_list': safe_value(row['days_in_waiting_list']),
#         'customer_type': safe_value(row['customer_type']),
#         'adr': safe_value(row['adr']),
#         'required_car_parking_spaces': safe_value(row['required_car_parking_spaces']),
#         'total_of_special_requests': safe_value(row['total_of_special_requests']),
#         'reservation_status': safe_value(row['reservation_status']),
#         'arrival_date': arrival_date_str,
#         'reservation_status_date': reservation_status,
#         'total_nights': total_nights
#     }

# def process_dataframe(df):
#     """Process DataFrame and return documents and metadatas."""
#     documents = []
#     metadatas = []
    
#     try:
#         # Iterate over each row in the DataFrame
#         for idx, row in df.iterrows():
#             try:
#                 document = create_document_string(row)
#                 metadata = create_metadata_dict(idx, row)
                
#                 documents.append(document)
#                 metadatas.append(metadata)
#             except Exception as e:
#                 logger.error(f"Error processing row {idx}: {str(e)}")
#                 continue
        
#         return documents, metadatas
#     except Exception as e:
#         logger.error(f"Error processing DataFrame: {str(e)}")
#         raise


# def initialize_chroma_collection(api_key, collection_name="hotel_collection", path="./Database", reset=False):
#     """Initialize ChromaDB client and create collection or load if it exists."""
#     try:
#         # Initialize embedding function
#         embedding_collection = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
#             api_key=api_key,
#             model_name="models/text-embedding-004"
#         )
        
#         # Initialize ChromaDB client
#         client = cdb.PersistentClient(path=path)
        
#         # Handle collection initialization
#         if reset:
#             try:
#                 client.delete_collection(collection_name)
#                 logger.info(f"Deleted existing collection: {collection_name}")
#             except Exception as e:
#                 logger.info(f"No existing collection to delete: {str(e)}")
            
#             collection = client.create_collection(
#                 name=collection_name,
#                 embedding_function=embedding_collection
#             )
#             logger.info(f"Created new collection: {collection_name}")
#         else:
#             try:
#                 collection = client.get_collection(
#                     name=collection_name,
#                     embedding_function=embedding_collection
#                 )
#                 doc_count = collection.count()
#                 logger.info(f"Loaded existing collection: {collection_name} with {doc_count} documents")
#             except Exception as e:
#                 logger.info(f"Collection not found, creating new: {str(e)}")
#                 collection = client.create_collection(
#                     name=collection_name,
#                     embedding_function=embedding_collection
#                 )
#                 logger.info(f"Created new collection: {collection_name}")
        
#         return collection
#     except Exception as e:
#         logger.error(f"Error initializing ChromaDB collection: {str(e)}")
#         raise

# def add_documents_to_collection(collection, documents, metadatas, batch_size=1000, start_batch=None, 
#                               progress_file="upload_progress.txt", save_interval=5):
#     """
#     Add documents to ChromaDB collection in batches with progress tracking and resumption.
    
#     Args:
#         collection: ChromaDB collection
#         documents: List of documents to add
#         metadatas: List of metadata dictionaries
#         batch_size: Size of each batch
#         start_batch: Batch to start from (if None, will try to load from progress file)
#         progress_file: File to save/load progress
#         save_interval: How often to save progress (every N batches)
#     """
#     try:
#         # Get existing document count
#         existing_count = collection.count()
#         logger.info(f"Current document count in collection: {existing_count}")
        
#         # Calculate number of batches
#         num_batches = (len(documents) + batch_size - 1) // batch_size
        
#         # If start_batch is not provided, try to load from progress file
#         if start_batch is None:
#             start_batch, _ = load_upload_progress(progress_file, default_batch=0)
            
#         logger.info(f"Starting upload from batch {start_batch}/{num_batches} with batch size {batch_size}")
        
#         # Track successful batches
#         last_successful_batch = start_batch - 1
        
#         # Generate all document IDs first
#         doc_ids = [f"doc_{j}" for j in range(len(documents))]
        
#         for i in tqdm(range(start_batch, num_batches)):
#             try:
#                 # Get the current batch
#                 start_idx = i * batch_size
#                 end_idx = min((i + 1) * batch_size, len(documents))
                
#                 current_docs = documents[start_idx:end_idx]
#                 current_metadatas = metadatas[start_idx:end_idx]
#                 current_ids = doc_ids[start_idx:end_idx]
                
#                 # Add the batch to the vector database
#                 collection.add(
#                     documents=current_docs,
#                     metadatas=current_metadatas,
#                     ids=current_ids
#                 )
                
#                 last_successful_batch = i
#                 logger.info(f"Added batch {i+1}/{num_batches} to vector database")
                
#                 # Save progress at intervals
#                 if i % save_interval == 0:
#                     save_upload_progress(i + 1, collection.name, progress_file)
                
#             except Exception as e:
#                 logger.error(f"Error adding batch {i+1}: {str(e)}")
#                 save_upload_progress(last_successful_batch + 1, collection.name, progress_file)
#                 logger.info(f"Saved progress at batch {last_successful_batch + 1}. You can resume from this point.")
                
#                 if "Connection" in str(e) or "Timeout" in str(e):
#                     logger.error("Connection issue detected. Stopping upload process.")
#                     return False
#                 else:
#                     logger.warning("Continuing to next batch despite error...")
#                     continue
        
#         # Save final progress
#         save_upload_progress(num_batches, collection.name, progress_file)
#         logger.info("All documents have been added to the vector database")
#         return True
        
#     except Exception as e:
#         logger.error(f"Error adding documents to collection: {str(e)}")
#         raise

# def save_upload_progress(start_batch, collection_name="hotel_collection", progress_file="upload_progress.txt"):
#     """Save the current upload progress to a file."""
#     try:
#         with open(progress_file, 'w') as f:
#             f.write(f"{start_batch},{collection_name}")
#         logger.info(f"Saved upload progress: batch {start_batch}")
#         return True
#     except Exception as e:
#         logger.error(f"Error saving upload progress: {str(e)}")
#         return False

# def load_upload_progress(progress_file="upload_progress.txt", default_batch=0, default_collection="hotel_collection"):
#     """Load the upload progress from a file."""
#     try:
#         if os.path.exists(progress_file):
#             with open(progress_file, 'r') as f:
#                 content = f.read().strip()
#                 if content:
#                     parts = content.split(',')
#                     if len(parts) >= 1:
#                         batch = int(parts[0])
#                         collection = parts[1] if len(parts) > 1 else default_collection
#                         logger.info(f"Loaded upload progress: resuming from batch {batch}")
#                         return batch, collection
#         return default_batch, default_collection
#     except Exception as e:
#         logger.error(f"Error loading upload progress: {str(e)}")
#         return default_batch, default_collection

# def add_documents_to_collection(collection, documents, metadatas, batch_size=1000, start_batch=None, 
#                                progress_file="upload_progress.txt", save_interval=5):
#     """
#     Add documents to ChromaDB collection in batches with progress tracking and resumption.
    
#     Args:
#         collection: ChromaDB collection
#         documents: List of documents to add
#         metadatas: List of metadata dictionaries
#         batch_size: Size of each batch
#         start_batch: Batch to start from (if None, will try to load from progress file)
#         progress_file: File to save/load progress
#         save_interval: How often to save progress (every N batches)
#     """
#     try:
#         # Calculate number of batches
#         num_batches = (len(documents) + batch_size - 1) // batch_size
        
#         # If start_batch is not provided, try to load from progress file
#         if start_batch is None:
#             start_batch, _ = load_upload_progress(progress_file, default_batch=0)
        
#         logger.info(f"Starting upload from batch {start_batch}/{num_batches} with batch size {batch_size}")
        
#         # Track successful batches
#         last_successful_batch = start_batch - 1
        
#         for i in tqdm(range(start_batch, num_batches)):
#             try:
#                 # Get the current batch
#                 start_idx = i * batch_size
#                 end_idx = min((i + 1) * batch_size, len(documents))
                
#                 current_docs = documents[start_idx:end_idx]
#                 current_metadatas = metadatas[start_idx:end_idx]
                
#                 # Add the batch to the vector database
#                 collection.add(
#                     documents=current_docs,
#                     metadatas=current_metadatas,
#                     ids=[f"doc_{j}" for j in range(start_idx, end_idx)]
#                 )
                
#                 last_successful_batch = i
#                 logger.info(f"Added batch {i+1}/{num_batches} to vector database")
                
#                 # Save progress at intervals
#                 if i % save_interval == 0:
#                     save_upload_progress(i + 1, collection.name, progress_file)
                
#             except Exception as e:
#                 logger.error(f"Error adding batch {i+1}: {str(e)}")
#                 # Save progress at the last successful batch
#                 save_upload_progress(last_successful_batch + 1, collection.name, progress_file)
#                 logger.info(f"Saved progress at batch {last_successful_batch + 1}. You can resume from this point.")
                
#                 # Decide whether to continue or break based on the error
#                 if "Connection" in str(e) or "Timeout" in str(e):
#                     logger.error("Connection issue detected. Stopping upload process.")
#                     return False
#                 else:
#                     logger.warning("Continuing to next batch despite error...")
#                     continue
        
#         # Save final progress
#         save_upload_progress(num_batches, collection.name, progress_file)
#         logger.info("All documents have been added to the vector database")
#         return True
        
#     except Exception as e:
#         logger.error(f"Error adding documents to collection: {str(e)}")
#         raise


# def main(batch_size=1000, start_batch=None, progress_file="upload_progress.txt", custom_df=None):
#     """
#     Main function to run the embedding process.
    
#     Args:
#         batch_size: Size of each batch for uploading
#         start_batch: Batch to start from (if None, will try to load from progress file)
#         progress_file: File to save/load progress
#         custom_df: Optional DataFrame to use instead of loading from database
#     """
#     try:
#         # Load API keys and create database engine
#         api_keys = load_api_keys()
#         engine = create_db_engine()
        
#         # Get Gemini API key
#         gemini_api_key = api_keys.get('gemini')
#         if not gemini_api_key:
#             logger.error("Gemini API key not found")
#             raise ValueError("Gemini API key not found")
        
#         # Load data from provided DataFrame or database
#         try:
#             if custom_df is not None:
#                 df = custom_df
#                 logger.info(f"Using provided DataFrame with {len(df)} rows")
#             else:
#                 df = pd.read_sql_query("SELECT * FROM booking", engine)
#                 logger.info(f"Loaded {len(df)} rows from database")
#         except Exception as e:
#             logger.error(f"Error loading data: {str(e)}")
#             raise
        
#         # Process DataFrame
#         documents, metadatas = process_dataframe(df)
#         logger.info(f"Processed {len(documents)} documents")
        
#         # Initialize ChromaDB collection
#         collection = initialize_chroma_collection(gemini_api_key)
        
#         # Add documents to collection with resumption capability
#         success = add_documents_to_collection(
#             collection, 
#             documents, 
#             metadatas, 
#             batch_size=batch_size,
#             start_batch=start_batch,
#             progress_file=progress_file
#         )
        
#         if success:
#             logger.info("Embedding process completed successfully")
#         else:
#             logger.warning("Embedding process completed with some issues")
            
#     except Exception as e:
#         logger.error(f"Error in main function: {str(e)}")
#         raise

# if __name__ == "__main__":
#     import argparse
    
#     # Parse command line arguments
#     parser = argparse.ArgumentParser(description='Embed hotel booking data into ChromaDB')
#     parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for uploading documents')
#     parser.add_argument('--start-batch', type=int, default=None, help='Batch to start from (overrides progress file)')
#     parser.add_argument('--progress-file', type=str, default='upload_progress.txt', help='File to save/load progress')
#     parser.add_argument('--csv-file', type=str, help='Optional CSV file to use instead of database')
    
#     args = parser.parse_args()
    
#     # If CSV file is provided, load it as custom_df
#     custom_df = None
#     if args.csv_file:
#         try:
#             custom_df = pd.read_csv(args.csv_file)
#             logger.info(f"Loaded CSV file: {args.csv_file}")
#         except Exception as e:
#             logger.error(f"Error loading CSV file: {str(e)}")
#             raise
    
#     # Run main function with parsed arguments
#     main(
#         batch_size=args.batch_size,
#         start_batch=args.start_batch,
#         progress_file=args.progress_file,
#         custom_df=custom_df
#     )





# # def main(batch_size=1000, start_batch=None, progress_file="upload_progress.txt"):
# #     """
# #     Main function to run the embedding process.
    
# #     Args:
# #         batch_size: Size of each batch for uploading
# #         start_batch: Batch to start from (if None, will try to load from progress file)
# #         progress_file: File to save/load progress
# #     """
# #     try:
# #         # Load API keys and create database engine
# #         api_keys = load_api_keys()
# #         engine = create_db_engine()
        
# #         # Get Gemini API key
# #         gemini_api_key = api_keys.get('gemini')
# #         if not gemini_api_key:
# #             logger.error("Gemini API key not found")
# #             raise ValueError("Gemini API key not found")
        
# #         # Load data from database
# #         try:
# #             df = pd.read_sql_query("SELECT * FROM booking", engine)
# #             logger.info(f"Loaded {len(df)} rows from database")
# #         except Exception as e:
# #             logger.error(f"Error loading data from database: {str(e)}")
# #             raise
        
# #         # Process DataFrame
# #         documents, metadatas = process_dataframe(df)
# #         logger.info(f"Processed {len(documents)} documents")
        
# #         # Initialize ChromaDB collection
# #         collection = initialize_chroma_collection(gemini_api_key)
        
# #         # Add documents to collection with resumption capability
# #         success = add_documents_to_collection(
# #             collection, 
# #             documents, 
# #             metadatas, 
# #             batch_size=batch_size,
# #             start_batch=start_batch,
# #             progress_file=progress_file
# #         )
        
# #         if success:
# #             logger.info("Embedding process completed successfully")
# #         else:
# #             logger.warning("Embedding process completed with some issues")
            
# #     except Exception as e:
# #         logger.error(f"Error in main function: {str(e)}")
# #         raise

# # if __name__ == "__main__":
# #     import argparse
    
# #     # Parse command line arguments
# #     parser = argparse.ArgumentParser(description='Embed hotel booking data into ChromaDB')
# #     parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for uploading documents')
# #     parser.add_argument('--start-batch', type=int, default=None, help='Batch to start from (overrides progress file)')
# #     parser.add_argument('--progress-file', type=str, default='upload_progress.txt', help='File to save/load progress')
    
# #     args = parser.parse_args()
    
# #     # Run main function with parsed arguments
# #     main(
# #         batch_size=args.batch_size,
# #         start_batch=args.start_batch,
# #         progress_file=args.progress_file
# #     )







# # import os
# # from tqdm import tqdm
# # import pandas as pd
# # import sqlite3 as sql
# # from google import genai
# # import chromadb as cdb
# # from utlis import load_api_keys, create_db_engine
# # import chromadb.utils.embedding_functions as embedding_functions



# # api_keys = load_api_keys()
# # engine = create_db_engine()

# # df = pd.read_sql_query("SELECT * FROM booking", engine)

# # def safe_value(val):
# #     return val if pd.notnull(val) else "missing"

# # # Prepare empty lists to store the documents and metadatas.
# # documents = []
# # metadatas = []

# # # Iterate over each row in the DataFrame.
# # for idx, row in df.iterrows():
# #     # For arrival and reservation dates, check each component.
# #     arrival_year = safe_value(row['arrival_date_year'])
# #     arrival_month = safe_value(row['arrival_date_month'])
# #     arrival_day = safe_value(row['arrival_date_day_of_month'])
# #     if "missing" in (arrival_year, arrival_month, arrival_day):
# #         arrival_date_str = "missing"
# #     else:
# #         arrival_date_str = f"{arrival_year}-{arrival_month}-{arrival_day}"
    
# #     res_year = safe_value(row['reservation_status_date_year'])
# #     res_month = safe_value(row['reservation_status_date_month'])
# #     res_day = safe_value(row['reservation_status_date_day'])
# #     if "missing" in (res_year, res_month, res_day):
# #         reservation_status_date_str = "missing"
# #     else:
# #         reservation_status_date_str = f"{res_year}-{res_month}-{res_day}"
    
# #     # Compute total nights, ensuring that if either value is missing, we mark as missing.
# #     stays_weekend = safe_value(row['stays_in_weekend_nights'])
# #     stays_week = safe_value(row['stays_in_week_nights'])
# #     if stays_weekend == "missing" or stays_week == "missing":
# #         total_nights = "missing"
# #     else:
# #         total_nights = int(stays_weekend) + int(stays_week)
    
# #     # Create a document string that includes all columns in a readable format.
# #     document = f"""
# #         Hotel: {safe_value(row['hotel'])}
# #         is_canceled: {safe_value(row['is_canceled'])}
# #         lead_time: {safe_value(row['lead_time'])}
# #         arrival_date_year: {arrival_year}
# #         arrival_date_month: {arrival_month}
# #         arrival_date_week_number: {safe_value(row['arrival_date_week_number'])}
# #         arrival_date_day_of_month: {arrival_day}
# #         stays_in_weekend_nights: {stays_weekend}
# #         stays_in_week_nights: {stays_week}
# #         adults: {safe_value(row['adults'])}
# #         children: {safe_value(row['children'])}
# #         babies: {safe_value(row['babies'])}
# #         meal: {safe_value(row['meal'])}
# #         country: {safe_value(row['country'])}
# #         market_segment: {safe_value(row['market_segment'])}
# #         distribution_channel: {safe_value(row['distribution_channel'])}
# #         is_repeated_guest: {safe_value(row['is_repeated_guest'])}
# #         previous_cancellations: {safe_value(row['previous_cancellations'])}
# #         previous_bookings_not_canceled: {safe_value(row['previous_bookings_not_canceled'])}
# #         reserved_room_type: {safe_value(row['reserved_room_type'])}
# #         assigned_room_type: {safe_value(row['assigned_room_type'])}
# #         booking_changes: {safe_value(row['booking_changes'])}
# #         deposit_type: {safe_value(row['deposit_type'])}
# #         agent: {safe_value(row['agent'])}
# #         company: {safe_value(row['company'])}
# #         days_in_waiting_list: {safe_value(row['days_in_waiting_list'])}
# #         customer_type: {safe_value(row['customer_type'])}
# #         adr: {safe_value(row['adr'])}
# #         required_car_parking_spaces: {safe_value(row['required_car_parking_spaces'])}
# #         total_of_special_requests: {safe_value(row['total_of_special_requests'])}
# #         reservation_status: {safe_value(row['reservation_status'])}
# #         reservation_status_date_year: {safe_value(row['reservation_status_date_year'])}
# #         reservation_status_date_month: {safe_value(row['reservation_status_date_month'])}
# #         reservation_status_date_day: {safe_value(row['reservation_status_date_day'])}
# #         Combined arrival_date: {arrival_date_str}
# #         Combined reservation_status_date: {reservation_status_date_str}
# #     """
# #     documents.append(document.strip())
    
# #     # Create the metadata dictionary including all columns.
# #     metadata = {
# #         'booking_id': str(idx),
# #         'hotel': safe_value(row['hotel']),
# #         'is_canceled': safe_value(row['is_canceled']),
# #         'lead_time': safe_value(row['lead_time']),
# #         'arrival_date_year': safe_value(row['arrival_date_year']),
# #         'arrival_date_month': safe_value(row['arrival_date_month']),
# #         'arrival_date_week_number': safe_value(row['arrival_date_week_number']),
# #         'arrival_date_day_of_month': safe_value(row['arrival_date_day_of_month']),
# #         'stays_in_weekend_nights': safe_value(row['stays_in_weekend_nights']),
# #         'stays_in_week_nights': safe_value(row['stays_in_week_nights']),
# #         'adults': safe_value(row['adults']),
# #         'children': safe_value(row['children']),
# #         'babies': safe_value(row['babies']),
# #         'meal': safe_value(row['meal']),
# #         'country': safe_value(row['country']),
# #         'market_segment': safe_value(row['market_segment']),
# #         'distribution_channel': safe_value(row['distribution_channel']),
# #         'is_repeated_guest': safe_value(row['is_repeated_guest']),
# #         'previous_cancellations': safe_value(row['previous_cancellations']),
# #         'previous_bookings_not_canceled': safe_value(row['previous_bookings_not_canceled']),
# #         'reserved_room_type': safe_value(row['reserved_room_type']),
# #         'assigned_room_type': safe_value(row['assigned_room_type']),
# #         'booking_changes': safe_value(row['booking_changes']),
# #         'deposit_type': safe_value(row['deposit_type']),
# #         'agent': safe_value(row['agent']),
# #         'company': safe_value(row['company']),
# #         'days_in_waiting_list': safe_value(row['days_in_waiting_list']),
# #         'customer_type': safe_value(row['customer_type']),
# #         'adr': safe_value(row['adr']),
# #         'required_car_parking_spaces': safe_value(row['required_car_parking_spaces']),
# #         'total_of_special_requests': safe_value(row['total_of_special_requests']),
# #         'reservation_status': safe_value(row['reservation_status']),
# #         'reservation_status_date_year': safe_value(row['reservation_status_date_year']),
# #         'reservation_status_date_month': safe_value(row['reservation_status_date_month']),
# #         'reservation_status_date_day': safe_value(row['reservation_status_date_day']),
# #         'arrival_date': arrival_date_str,
# #         'reservation_status_date': reservation_status_date_str,
# #         'total_nights': total_nights
# #     }
# #     metadatas.append(metadata)




# # embedding_collection = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
# #     api_key=gemini_api_key,
# #     model_name="models/text-embedding-004"  # You can change the model as needed
# # )

# # # Initialize ChromaDB client and create collection
# # client = cdb.PersistentClient(path="./Database")
# # # If collection already exists, delete it to avoid errors
# # try:
# #     client.delete_collection("hotel_collection")
# #     print("Deleted existing collection")
# # except:
# #     pass
# # collection = client.create_collection(
# #     name="hotel_collection",
# #     embedding_function=embedding_collection
# # )



# # batch_size = 1000
# # num_batches = (len(documents) + batch_size - 1) // batch_size

# # # Start from batch 31 since the first 31,000 documents (batches 0 to 30) have been uploaded
# # start_batch = 49

# # for i in tqdm(range(start_batch, num_batches)):
# #     # Get the current batch
# #     start_idx = i * batch_size
# #     end_idx = min((i + 1) * batch_size, len(documents))
    
# #     current_docs = documents[start_idx:end_idx]
# #     current_metadatas = metadatas[start_idx:end_idx]
    
# #     # Add the batch to the vector database
# #     collection.add(
# #         documents=current_docs,
# #         metadatas=current_metadatas,
# #         ids=[f"doc_{j}" for j in range(start_idx, end_idx)]
# #     )
    
# #     print(f"Added batch {i+1}/{num_batches} to vector database")

# # print("All documents have been added to the vector database")