from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from functools import wraps
from pymongo import MongoClient
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
import bcrypt
from jose import JWTError, jwt
import os
from dotenv import load_dotenv
from bson.objectid import ObjectId

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', ' 010a1884f9c60ccd15febd8b4ecd2e5319b5a67fe38c294bdc680f9ab78add8a')

# JWT settings
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# MongoDB connection and collections
try:
    client = MongoClient('mongodb://localhost:27017/')
    db = client['course_recommender']
    user_details = db['user_details']
    user_preferences = db['user_preferences']
    user_interactions = db['user_interactions']
    admin_collection = db['admin_users']
    
    # Test the connection
    db.command('ping')
    print("Successfully connected to MongoDB!")
except Exception as e:
    print(f"Error connecting to MongoDB: {str(e)}")
    raise e

# Course data path
COURSE_DATA_PATH = r'D:\datasets\data1\final_cleaned_course_data.csv'

# Domain name mapping
domain_full_names = {
    'Data Engineering': 'Data Engineering',
    'Data Science': 'Data Science',
    'Machine Learning': 'Machine Learning',
    'Artificial Intelligence': 'Artificial Intelligence',
    'Multidisciplinary': 'Multidisciplinary',
    'None': 'No Preference'
}

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, app.secret_key, algorithm=ALGORITHM)
    return encoded_jwt

def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password)

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_token' not in session:
            flash('Please login first!', 'error')
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    return redirect(url_for('admin_login'))

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        admin_user = admin_collection.find_one({"username": username})
        if admin_user and verify_password(password, admin_user['password']):
            token = create_access_token({"sub": username})
            session['admin_token'] = token
            flash('Successfully logged in!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid credentials!', 'error')
    
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    session.clear()
    flash('Successfully logged out!', 'success')
    return redirect(url_for('admin_login'))

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    # Get user statistics
    total_users = user_details.count_documents({})
    week_ago = datetime.now() - timedelta(days=7)
    active_users = len(set(
        doc['user_id'] for doc in user_interactions.find({
            'created_at': {'$gte': week_ago}
        })
    ))
    
    # Calculate engagement rate
    engagement_rate = (active_users / total_users * 100) if total_users > 0 else 0
    
    # Calculate average interactions per user
    total_interactions = user_interactions.count_documents({})
    avg_interactions = total_interactions / total_users if total_users > 0 else 0
    
    return render_template('admin_dashboard.html',
                         total_users=total_users,
                         active_users=active_users,
                         engagement_rate=round(engagement_rate, 1),
                         avg_interactions=round(avg_interactions, 1))

@app.route('/admin/user_management')
@admin_required
def user_management():
    try:
        # Debug: Print total users in user_details collection
        users = list(user_details.find())
        print(f"Total users found: {len(users)}")
        
        user_data = []
        domain_counts = {}
        skill_counts = {}
        
        for user in users:
            try:
                user_id = str(user['_id'])
                print(f"\nProcessing user: {user_id}")
                
                # Get username
                username = user.get('username')
                print(f"Username: {username}")
                
                # Get preferences with user_id as string
                preferences = user_preferences.find_one({'user_id': user_id})
                if not preferences:
                    # Try with ObjectId
                    preferences = user_preferences.find_one({'user_id': ObjectId(user_id)})
                print(f"User preferences: {preferences}")
                
                # Get interactions with user_id as string
                interactions = list(user_interactions.find({'user_id': user_id}))
                if not interactions:
                    # Try with ObjectId
                    interactions = list(user_interactions.find({'user_id': ObjectId(user_id)}))
                print(f"User interactions: {len(interactions)}")
                
                # Handle domain
                domain = 'No Preference'
                if preferences and 'domain' in preferences:
                    domain = preferences['domain']
                print(f"Domain: {domain}")
                
                # Handle skills
                skills_list = []
                if preferences and 'skills' in preferences:
                    if isinstance(preferences['skills'], list):
                        skills_list = preferences['skills']
                    elif isinstance(preferences['skills'], str):
                        skills_list = [preferences['skills']]
                skills = ', '.join(skills_list) if skills_list else 'Not set'
                print(f"Skills: {skills}")
                
                # Update domain counts
                if domain != 'No Preference':
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
                
                # Update skill counts
                for skill in skills_list:
                    skill_counts[skill] = skill_counts.get(skill, 0) + 1
                
                # Calculate interaction counts
                completed_courses = sum(1 for i in interactions if i.get('interaction_type') == 'completed')
                enrolled_courses = sum(1 for i in interactions if i.get('interaction_type') == 'enroll')
                liked_courses = sum(1 for i in interactions if i.get('interaction_type') == 'like')
                
                print(f"Interactions - Completed: {completed_courses}, Enrolled: {enrolled_courses}, Liked: {liked_courses}")
                
                user_info = {
                    'username': username or str(user['_id']),
                    'domain': domain,
                    'skills': skills,
                    'completed': completed_courses,
                    'enrolled': enrolled_courses,
                    'liked': liked_courses,
                    'total_activity': len(interactions)
                }
                user_data.append(user_info)
                print(f"Added user info: {user_info}")
                
            except Exception as e:
                print(f"Error processing user {user_id}: {str(e)}")
                continue
        
        # Debug: Print final data
        print(f"\nTotal users processed: {len(user_data)}")
        print(f"Domain counts: {domain_counts}")
        print(f"Skill counts: {skill_counts}")
        
        # Pre-format chart data
        chart_data = {
            'domain_values': list(domain_counts.values()),
            'domain_labels': list(domain_counts.keys()),
            'skills_x': list(dict(sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:10]).keys()),
            'skills_y': list(dict(sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:10]).values())
        }
        
        print(f"Chart data: {chart_data}")
        
        return render_template('user_management.html',
                             users=user_data,
                             chart_data=chart_data)
                             
    except Exception as e:
        print(f"Major error in user_management: {str(e)}")
        return f"Error: {str(e)}", 500

@app.route('/admin/course_analytics')
@admin_required
def course_analytics():
    df = pd.read_csv(COURSE_DATA_PATH)
    
    total_courses = len(df)
    avg_rating = df['Rating'].mean()
    
    platform_dist = df['platform'].value_counts().to_dict()
    
    level_dist = {}
    if 'Levels' in df.columns:
        valid_levels = ['All Levels', 'Beginner', 'Intermediate', 'Expert']
        level_dist = df[df['Levels'].isin(valid_levels)]['Levels'].value_counts().to_dict()
    
    domain_counts = {
        'AI': 0, 'ML': 0, 'DS': 0, 'DE': 0
    }
    
    domain_mapping = {
        'AI': ['artificial intelligence', 'ai', 'computer vision', 'nlp'],
        'ML': ['machine learning', 'ml', 'deep learning', 'neural networks'],
        'DS': ['data science', 'data analytics', 'statistics', 'visualization'],
        'DE': ['data engineering', 'data pipeline', 'etl', 'data warehouse']
    }
    
    if 'skills' in df.columns:
        for _, row in df.iterrows():
            if pd.isna(row['skills']):
                continue
            skills = str(row['skills']).lower()
            for domain, keywords in domain_mapping.items():
                if any(keyword in skills for keyword in keywords):
                    domain_counts[domain] += 1
    
    return render_template('course_analytics.html',
                         total_courses=total_courses,
                         avg_rating=round(avg_rating, 2),
                         platform_dist=platform_dist,
                         level_dist=level_dist,
                         domain_counts=domain_counts)

@app.route('/admin/user_activity')
@admin_required
def user_activity():
    start_date = request.args.get('start_date', 
                                (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
    end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    selected_user = request.args.get('username', '')
    
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
    
    # Base query
    query = {
        "created_at": {
            "$gte": start_datetime,
            "$lt": end_datetime
        }
    }
    
    # Add user filter if selected
    if selected_user:
        query["user_id"] = selected_user
    
    activities = list(user_interactions.find(query))
    
    # Get unique usernames for dropdown
    all_users = list(user_details.find({}, {"username": 1}))
    usernames = [user.get('username', str(user['_id'])) for user in all_users]
    
    # Calculate statistics
    total_activities = len(activities)
    unique_users = len(set(act['user_id'] for act in activities))
    avg_activities_per_user = total_activities / unique_users if unique_users > 0 else 0
    
    # Prepare activity data
    daily_activity = {}
    type_counts = {}
    hour_counts = [0] * 24
    
    for activity in activities:
        # Daily activity
        date = activity['created_at'].date()
        daily_activity[date] = daily_activity.get(date, 0) + 1
        
        # Activity types - using interaction_type field
        act_type = activity.get('interaction_type', 'view')
        type_counts[act_type] = type_counts.get(act_type, 0) + 1
        
        # Hour distribution
        hour = activity['created_at'].hour
        hour_counts[hour] += 1
    
    # Sort daily activity by date
    daily_activity = dict(sorted(daily_activity.items()))
    
    # Prepare data for charts
    activity_data = {
        'dates': list(daily_activity.keys()),
        'counts': list(daily_activity.values()),
        'type_labels': list(type_counts.keys()),
        'type_counts': list(type_counts.values()),
        'hours': list(range(24)),
        'hour_counts': hour_counts
    }
    
    return render_template('user_activity.html',
                         start_date=start_date,
                         end_date=end_date,
                         total_activities=total_activities,
                         unique_users=unique_users,
                         avg_activities_per_user=round(avg_activities_per_user, 1),
                         activity_data=activity_data,
                         usernames=usernames)

@app.route('/admin/user_activity/details/<username>')
@admin_required
def user_activity_details(username):
    user = user_details.find_one({"username": username})
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    # Get user interactions
    interactions = list(user_interactions.find({"user_id": str(user['_id'])}))
    
    # Calculate metrics using interaction_type field
    enrolled_courses = len([i for i in interactions if i.get('interaction_type') == 'enroll'])
    liked_courses = len([i for i in interactions if i.get('interaction_type') == 'like'])
    
    # Get last login
    last_login = max([i['created_at'] for i in interactions]) if interactions else None
    last_login_str = last_login.strftime('%Y-%m-%d %H:%M') if last_login else 'Never'
    
    # Calculate time spent (simplified version)
    time_spent = len(interactions) * 5  # Assuming 5 minutes per interaction
    time_spent_str = f"{time_spent}m"
    
    return jsonify({
        'enrolled_courses': enrolled_courses,
        'liked_courses': liked_courses,
        'last_login': last_login_str,
        'time_spent': time_spent_str
    })

@app.route('/admin/system_stats')
@admin_required
def system_stats():
    total_users = user_details.count_documents({})
    total_interactions = user_interactions.count_documents({})
    total_preferences = user_preferences.count_documents({})
    
    # Get new users in last 24h using ObjectId timestamp
    yesterday = datetime.now() - timedelta(days=1)
    new_users_24h = user_details.count_documents({
        'created_at': {'$gte': yesterday}
    })
    
    new_interactions_24h = user_interactions.count_documents({
        'created_at': {'$gte': yesterday}
    })
    
    # Calculate retention for different time periods
    retention_data = []
    for days in [1, 7, 30, 90]:
        period_start = datetime.now() - timedelta(days=days)
        active_users = len(set([
            doc['user_id'] for doc in user_interactions.find({
                'created_at': {'$gte': period_start}
            })
        ]))
        
        retention_data.append({
            'period': f'Last {days} days',
            'active_users': active_users,
            'retention_rate': f"{(active_users / total_users * 100):.1f}%" if total_users > 0 else "0%"
        })
    
    return render_template('system_stats.html',
                         total_users=total_users,
                         total_interactions=total_interactions,
                         total_preferences=total_preferences,
                         new_users_24h=new_users_24h,
                         new_interactions_24h=new_interactions_24h,
                         retention_data=retention_data)

if __name__ == '__main__':
    app.run(debug=True, port=5001) 