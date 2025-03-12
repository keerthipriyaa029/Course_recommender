from pymongo import MongoClient, ASCENDING
import sys
from datetime import datetime

def init_database():
    try:
        # Connect to MongoDB
        client = MongoClient('mongodb://localhost:27017/')
        
        # Create or get the database
        db = client['course_recommender']
        
        # Create or get the collections
        user_collection = db['user_details']
        preferences_collection = db['user_preferences']
        interactions_collection = db['user_interactions']
        
        # Create indexes
        user_collection.create_index('username', unique=True)
        preferences_collection.create_index('user_id')
        
        # Create compound indexes for user_interactions
        interactions_collection.create_index([
            ('user_id', ASCENDING),
            ('course_id', ASCENDING)
        ], unique=True)  # Ensures unique user-course pairs
        
        interactions_collection.create_index([
            ('user_id', ASCENDING),
            ('interaction_type', ASCENDING)
        ])  # For quick filtering by interaction type
        
        interactions_collection.create_index('created_at')  # For time-based queries
        
        print("Successfully connected to MongoDB!")
        print(f"Database 'course_recommender' initialized")
        print(f"Collections and indexes created successfully")
        
        # List all databases to verify
        print("\nAvailable databases:")
        print(client.list_database_names())
        
        return True
        
    except Exception as e:
        print(f"Error: Could not connect to MongoDB. Make sure MongoDB is running.")
        print(f"Error details: {str(e)}")
        return False

def create_interaction(db, user_id, course_id, course_name, platform, course_link, interaction_type):
    """Helper function to create or update user interaction"""
    try:
        current_time = datetime.utcnow()
        
        # Try to update existing interaction
        result = db.user_interactions.update_one(
            {
                'user_id': user_id,
                'course_id': course_id
            },
            {
                '$set': {
                    'interaction_type': interaction_type,
                    'updated_at': current_time,
                    'course_name': course_name,
                    'platform': platform,
                    'course_link': course_link
                }
            },
            upsert=True  # Create if doesn't exist
        )
        
        if result.upserted_id:
            # This was a new interaction
            db.user_interactions.update_one(
                {'_id': result.upserted_id},
                {
                    '$set': {
                        'created_at': current_time
                    }
                }
            )
        
        return True
    except Exception as e:
        print(f"Error creating/updating interaction: {str(e)}")
        return False

def remove_interaction(db, user_id, course_id):
    """Helper function to remove user interaction"""
    try:
        db.user_interactions.delete_one({
            'user_id': user_id,
            'course_id': course_id
        })
        return True
    except Exception as e:
        print(f"Error removing interaction: {str(e)}")
        return False

if __name__ == "__main__":
    init_database() 