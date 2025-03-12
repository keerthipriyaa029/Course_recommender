from pymongo import MongoClient
import bcrypt
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['course_recommender']
admin_collection = db['admin_users']

def create_admin_user(username, password):
    # Hash the password
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    
    # Create admin user document
    admin_user = {
        'username': username,
        'password': hashed_password,
        'created_at': datetime.utcnow(),
        'role': 'admin',
        'permissions': [
            'manage_users',
            'manage_courses',
            'view_analytics',
            'manage_admins',
            'system_settings'
        ]
    }
    
    # Insert the admin user
    result = admin_collection.insert_one(admin_user)
    
    if result.inserted_id:
        print(f"Admin user '{username}' created successfully!")
    else:
        print("Failed to create admin user.")

if __name__ == "__main__":
    # Get admin credentials from user input
    username = input("Enter admin username: ")
    password = input("Enter admin password: ")
    
    # Create the admin user
    create_admin_user(username, password) 