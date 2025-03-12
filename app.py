from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
import os
from dotenv import load_dotenv
from datetime import datetime, timezone, UTC
from init_db import init_database
from model import CourseRecommender
import pandas as pd
from chatbot import CourseBot
from dashboard import DashboardAnalytics

# Load environment variables
load_dotenv()
print("Environment variables loaded in Flask app")  # Debug line
print(f"OpenRouter API Key status: {'Present' if os.getenv('OPENROUTER_API_KEY') else 'Missing'}")  # Debug line

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', '0044000fcb1aab5244197969b51469c20a729c3a375ecd2c5df626ea5615be7b')
app.config['MONGO_URI'] = 'mongodb://localhost:27017/course_recommender'

# Initialize MongoDB
mongo = PyMongo(app)

# Initialize Login Manager
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Ensure database is initialized
init_database()

# Initialize the recommender model
recommender = CourseRecommender()
try:
    recommender.load_model()
except:
    print("No saved model found. Training new model...")
    recommender.preprocess_data()
    recommender.save_model()

# User Model
class User(UserMixin):
    def __init__(self, user_data):
        self.user_data = user_data
        self.id = str(user_data['_id'])

    def get_id(self):
        return str(self.user_data['_id'])

@login_manager.user_loader
def load_user(user_id):
    user_data = mongo.db.user_details.find_one({'_id': ObjectId(user_id)})
    return User(user_data) if user_data else None

# Routes
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user_data = mongo.db.user_details.find_one({'username': username})
        
        if user_data and check_password_hash(user_data['password'], password):
            user = User(user_data)
            login_user(user)
            
            # Check if user has preferences
            user_preferences = mongo.db.user_preferences.find_one({
                'user_id': ObjectId(user_data['_id'])
            })
            
            # If user has preferences, go to recommendations
            if user_preferences and user_preferences.get('domain') and user_preferences.get('skills'):
                flash('Login successful!', 'success')
                return redirect(url_for('recommendations'))
            # If no preferences, go to domain selection
            else:
                flash('Please set your preferences to get started!', 'info')
                return redirect(url_for('domain_selection'))
        
        flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Check if username already exists
        if mongo.db.user_details.find_one({'username': username}):
            flash('Username already exists', 'danger')
            return redirect(url_for('signup'))
        
        try:
            # Create new user
            hashed_password = generate_password_hash(password)
            user_data = {
                'username': username,
                'password': hashed_password,
                'created_at': datetime.utcnow()
            }
            
            # Insert into database
            result = mongo.db.user_details.insert_one(user_data)
            
            # Log the new user in
            user = User(user_data)
            login_user(user)
            
            # Set session flag for first-time user
            session['is_first_time'] = True
            
            flash('Account created successfully! Please select your preferences.', 'success')
            return redirect(url_for('domain_selection'))
            
        except Exception as e:
            flash('An error occurred during signup. Please try again.', 'danger')
            print(f"Signup error: {str(e)}")
            return redirect(url_for('signup'))
        
    return render_template('signup.html')

@app.route('/domain-selection', methods=['GET', 'POST'])
@login_required
def domain_selection():
    if request.method == 'POST':
        selected_domain = request.form.get('domain')
        
        if not selected_domain:
            flash('Please select a domain', 'danger')
            return redirect(url_for('domain_selection'))
        
        try:
            # Check if user already has preferences
            existing_preferences = mongo.db.user_preferences.find_one({
                'user_id': ObjectId(current_user.get_id())
            })
            
            if existing_preferences:
                # Update existing preferences
                mongo.db.user_preferences.update_one(
                    {'user_id': ObjectId(current_user.get_id())},
                    {'$set': {'domain': selected_domain}}
                )
            else:
                # Create new preferences
                preferences_data = {
                    'user_id': ObjectId(current_user.get_id()),
                    'domain': selected_domain,
                    'skills': []  # Will be updated in the skills selection page
                }
                mongo.db.user_preferences.insert_one(preferences_data)
            
            flash(f'Domain "{selected_domain}" selected successfully!', 'success')
            return redirect(url_for('skill_selection'))
            
        except Exception as e:
            flash('An error occurred while saving your preferences. Please try again.', 'danger')
            print(f"Error saving domain preference: {str(e)}")
            return redirect(url_for('domain_selection'))
    
    return render_template('domain_selection.html')

@app.route('/skill-selection', methods=['GET', 'POST'])
@login_required
def skill_selection():
    # Define domain icons
    domain_icons = {
        'Data Science': 'fas fa-chart-bar',
        'Data Engineering': 'fas fa-database',
        'Machine Learning': 'fas fa-brain',
        'Artificial Intelligence': 'fas fa-robot',
        'No Preference': 'fas fa-globe'
    }

    # Define skills for each domain
    all_skills = {
        'Data Science': [
            'Python', 'SQL', 'Pandas', 'Statistics', 'Data Visualization',
            'R Programming', 'MATLAB', 'Exploratory Data Analysis',
            'Feature Selection', 'Data Cleaning'
        ],
        'Data Engineering': [
            'Apache Spark', 'ETL Pipelines', 'Cloud Platforms',
            'Data Warehousing', 'Apache Kafka', 'Airflow',
            'BigQuery', 'Hadoop', 'NoSQL Databases', 'Distributed Computing'
        ],
        'Machine Learning': [
            'Machine Learning Basics', 'Scikit-learn', 'TensorFlow',
            'PyTorch', 'Feature Engineering', 'Hyperparameter Tuning',
            'Deep Learning', 'Ensemble Learning', 'Cross-Validation',
            'Transfer Learning'
        ],
        'Artificial Intelligence': [
            'Deep Learning', 'NLP', 'Computer Vision',
            'Reinforcement Learning', 'Generative AI',
            'Graph Neural Networks', 'Attention Mechanisms',
            'AI Ethics & Fairness', 'Speech Recognition',
            'Autonomous Systems'
        ]
    }

    if request.method == 'POST':
        selected_skills = request.form.getlist('skills')
        
        if not selected_skills:
            flash('Please select at least one skill', 'danger')
            return redirect(url_for('skill_selection'))
        
        try:
            # Update skills in user_preferences
            mongo.db.user_preferences.update_one(
                {'user_id': ObjectId(current_user.get_id())},
                {'$set': {'skills': selected_skills}}
            )
            flash('Skills updated successfully!', 'success')
            return redirect(url_for('recommendations'))
        except Exception as e:
            flash('An error occurred while saving your skills. Please try again.', 'danger')
            print(f"Error saving skills: {str(e)}")
            return redirect(url_for('skill_selection'))
    
    # Get user's selected domain
    user_preferences = mongo.db.user_preferences.find_one({
        'user_id': ObjectId(current_user.get_id())
    })
    
    if not user_preferences:
        flash('Please select a domain first', 'warning')
        return redirect(url_for('domain_selection'))
    
    selected_domain = user_preferences.get('domain')
    
    # Get skills based on domain
    if selected_domain == 'No Preference':
        return render_template('skill_selection.html',
                            selected_domain=selected_domain,
                            all_skills=all_skills,
                            domain_icons=domain_icons)
    else:
        skills = all_skills.get(selected_domain, [])
        return render_template('skill_selection.html',
                            selected_domain=selected_domain,
                            skills=skills,
                            domain_icons=domain_icons)

@app.route('/recommendations')
@login_required
def recommendations():
    try:
        # Get user preferences
        user_preferences = mongo.db.user_preferences.find_one({
            'user_id': ObjectId(current_user.get_id())
        })
        
        if not user_preferences:
            flash('Please select your domain and skills first', 'warning')
            return redirect(url_for('domain_selection'))
        
        # Get user interactions
        user_interactions = list(mongo.db.user_interactions.find())
        user_interactions_df = pd.DataFrame(user_interactions) if user_interactions else pd.DataFrame()
        
        # Get user skills and normalize them
        user_skills = user_preferences.get('skills', [])
        
        # Get initial recommendations with normalized skills
        recommended_courses = recommender.get_knowledge_based_recommendations(
            user_domain=user_preferences.get('domain', []),
            user_skills=user_skills,  # The normalization happens in the recommender
            user_level='Beginner',
            n_recommendations=10,
            offset=0
        )
        
        if recommended_courses is None or recommended_courses.empty:
            flash('No recommendations available for your preferences.', 'info')
            return render_template('recommendations.html', recommendations=[], interactions=[])
        
        # Get similar courses based on liked courses
        similar_courses = recommender.get_recommendations_for_liked_courses(
            user_interactions_df,
            current_user.get_id(),
            n_recommendations=6,
            exclude_course_ids=recommended_courses['Course ID'].tolist()
        )
        
        # Get trending courses
        trending_courses = recommender.get_trending_recommendations(
            user_interactions_df,
            n_recommendations=10,
            exclude_course_ids=recommended_courses['Course ID'].tolist() + 
                            ([] if similar_courses.empty else similar_courses['Course ID'].tolist())
        )
        
        # Convert recommendations to list of dictionaries
        recommendations = recommended_courses.to_dict('records')
        similar_recommendations = [] if similar_courses.empty else similar_courses.to_dict('records')
        trending_recommendations = [] if trending_courses.empty else trending_courses.to_dict('records')
        
        # Add matched skills information to the recommendations
        for course in recommendations:
            if 'matched_skills' in course:
                course['matched_skills_str'] = ', '.join(course['matched_skills'])
        
        # Convert ObjectId to string for JSON serialization
        for interaction in user_interactions:
            if 'course_id' in interaction:
                interaction['course_id'] = str(interaction['course_id'])
            interaction['user_id'] = str(interaction['user_id'])
            interaction['_id'] = str(interaction['_id'])
        
        return render_template(
            'recommendations.html',
            recommendations=recommendations,
            similar_recommendations=similar_recommendations,
            trending_recommendations=trending_recommendations,
            interactions=user_interactions,
            has_more=len(recommendations) == 10
        )
        
    except Exception as e:
        print(f"Error generating recommendations: {str(e)}")
        import traceback
        traceback.print_exc()
        flash('An error occurred while generating recommendations. Please try again.', 'danger')
        return redirect(url_for('domain_selection'))

@app.route('/api/recommendations/load-more')
@login_required
def load_more_recommendations():
    try:
        offset = int(request.args.get('offset', 0))
        if offset >= 50:  # Maximum limit reached
            return jsonify({'courses': [], 'has_more': False})
        
        # Get user preferences
        user_preferences = mongo.db.user_preferences.find_one({
            'user_id': ObjectId(current_user.get_id())
        })
        
        if not user_preferences:
            return jsonify({'error': 'User preferences not found'}), 404
        
        # Get user skills and normalize them
        user_skills = user_preferences.get('skills', [])
        
        # Get next batch of recommendations with normalized skills
        recommended_courses = recommender.get_knowledge_based_recommendations(
            user_domain=user_preferences.get('domain', []),
            user_skills=user_skills,  # The normalization happens in the recommender
            user_level='Beginner',
            n_recommendations=10,
            offset=offset
        )
        
        if recommended_courses is None or recommended_courses.empty:
            return jsonify({'courses': [], 'has_more': False})
        
        # Convert to list of dictionaries
        courses = recommended_courses.to_dict('records')
        
        # Add matched skills information
        for course in courses:
            if 'matched_skills' in course:
                course['matched_skills_str'] = ', '.join(course['matched_skills'])
        
        # Get user interactions for these courses
        current_interactions = list(mongo.db.user_interactions.find({
            'user_id': ObjectId(current_user.get_id()),
            'course_id': {'$in': [str(course.get('Course ID', '')) for course in courses]}
        }))
        
        # Convert ObjectId to string for JSON serialization
        for course in courses:
            if '_id' in course:
                course['_id'] = str(course['_id'])
            if 'Course ID' in course:
                course['Course ID'] = str(course['Course ID'])
        
        for interaction in current_interactions:
            interaction['_id'] = str(interaction['_id'])
            interaction['user_id'] = str(interaction['user_id'])
            interaction['course_id'] = str(interaction['course_id'])
        
        return jsonify({
            'courses': courses,
            'interactions': current_interactions,
            'has_more': len(courses) == 10 and offset + 10 < 50
        })
        
    except Exception as e:
        print(f"Error loading more recommendations: {str(e)}")
        return jsonify({'error': 'Failed to load more recommendations'}), 500

@app.route('/api/interactions', methods=['POST'])
@login_required
def create_interaction():
    try:
        data = request.json
        required_fields = ['course_id', 'course_name', 'platform', 'course_link', 'interaction_type']
        
        # Validate required fields
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
            
        # Validate interaction type
        valid_interactions = ['like', 'enroll', 'completed']
        if data['interaction_type'] not in valid_interactions:
            return jsonify({'error': 'Invalid interaction type'}), 400
            
        # Create or update interaction
        current_time = datetime.now(timezone.utc)
        
        # Try to update existing interaction
        result = mongo.db.user_interactions.update_one(
            {
                'user_id': ObjectId(current_user.get_id()),
                'course_id': str(data['course_id'])  # Ensure course_id is string
            },
            {
                '$set': {
                    'interaction_type': data['interaction_type'],
                    'updated_at': current_time,
                    'course_name': data['course_name'],
                    'platform': data['platform'],
                    'course_link': data['course_link']
                }
            },
            upsert=True
        )
        
        # If this was a new interaction, add created_at
        if result.upserted_id:
            mongo.db.user_interactions.update_one(
                {'_id': result.upserted_id},
                {'$set': {'created_at': current_time}}
            )
        
        return jsonify({'message': 'Interaction saved successfully'}), 200
            
    except Exception as e:
        print(f"Error in create_interaction: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/interactions/<course_id>', methods=['DELETE'])
@login_required
def remove_interaction(course_id):
    try:
        result = mongo.db.user_interactions.delete_one({
            'user_id': ObjectId(current_user.get_id()),
            'course_id': str(course_id)  # Ensure course_id is string
        })
        
        if result.deleted_count > 0:
            return jsonify({'message': 'Interaction removed successfully'}), 200
        else:
            return jsonify({'error': 'Interaction not found'}), 404
            
    except Exception as e:
        print(f"Error in remove_interaction: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/interactions/<course_id>', methods=['GET'])
@login_required
def get_interaction(course_id):
    try:
        interaction = mongo.db.user_interactions.find_one({
            'user_id': ObjectId(current_user.get_id()),
            'course_id': course_id
        })
        
        if interaction:
            # Convert ObjectId to string for JSON serialization
            interaction['_id'] = str(interaction['_id'])
            interaction['user_id'] = str(interaction['user_id'])
            return jsonify(interaction), 200
        else:
            return jsonify({'message': 'No interaction found'}), 404
            
    except Exception as e:
        print(f"Error in get_interaction: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/interactions', methods=['GET'])
@login_required
def list_interactions():
    try:
        # Get optional query parameters
        interaction_type = request.args.get('type')
        platform = request.args.get('platform')
        
        # Build query
        query = {'user_id': ObjectId(current_user.get_id())}
        if interaction_type:
            query['interaction_type'] = interaction_type
        if platform:
            query['platform'] = platform
            
        # Get interactions
        interactions = list(mongo.db.user_interactions.find(query).sort('created_at', -1))
        
        # Convert ObjectId to string for JSON serialization
        for interaction in interactions:
            interaction['_id'] = str(interaction['_id'])
            interaction['user_id'] = str(interaction['user_id'])
            
        return jsonify(interactions), 200
        
    except Exception as e:
        print(f"Error in list_interactions: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/api/search')
@login_required
def search_courses():
    query = request.args.get('q', '').strip().lower()
    platform = request.args.get('platform', 'all')
    duration = request.args.get('duration', 'all')
    
    if not query:
        return jsonify({'courses': [], 'interactions': [], 'total_results': 0})
    
    try:
        # Extract search components
        search_terms = query.split()
        
        # Define abbreviations and their expansions
        abbreviations = {
            'ai': ['artificial intelligence', 'ai'],
            'ml': ['machine learning', 'ml'],
            'dl': ['deep learning', 'dl'],
            'ds': ['data science', 'ds'],
            'da': ['data analytics', 'data analysis', 'da'],
            'py': ['python', 'py'],
            'js': ['javascript', 'js'],
            'fe': ['frontend', 'front-end', 'front end'],
            'be': ['backend', 'back-end', 'back end'],
            'fs': ['fullstack', 'full-stack', 'full stack'],
            'db': ['database', 'db'],
            'sql': ['sql', 'structured query language'],
            'nlp': ['natural language processing', 'nlp'],
            'cv': ['computer vision', 'cv'],
            'rl': ['reinforcement learning', 'rl'],
            'devops': ['development operations', 'devops'],
            'aws': ['amazon web services', 'aws'],
            'gcp': ['google cloud platform', 'gcp'],
            'azure': ['microsoft azure', 'azure'],
            'oop': ['object oriented programming', 'object-oriented programming', 'oop']
        }
        
        # Enhanced institution mapping
        institution_mapping = {
            'google': ['google', 'google cloud', 'google analytics', 'google data studio', 'google bigquery', 'tensorflow', 'google ai', 'google cloud platform', 'gcp'],
            'ibm': ['ibm', 'ibm watson', 'ibm cloud', 'ibm data science', 'ibm analytics', 'cognos', 'ibm certified'],
            'microsoft': ['microsoft', 'azure', 'microsoft azure', 'power bi', 'powerbi', 'microsoft power bi', 'microsoft certified'],
            'aws': ['aws', 'amazon web services', 'amazon aws', 'aws cloud', 'aws certified', 'amazon certified'],
            'databricks': ['databricks', 'databricks certified', 'databricks spark', 'databricks platform', 'databricks sql'],
            'coursera': ['coursera', 'coursera project', 'coursera certificate', 'coursera specialization'],
            'udacity': ['udacity', 'udacity nanodegree', 'udacity project', 'udacity certification'],
            'udemy': ['udemy', 'udemy course', 'udemy project', 'udemy certificate'],
            'stanford': ['stanford', 'stanford university', 'stanford online', 'stanford mooc'],
            'mit': ['mit', 'massachusetts institute of technology', 'mit opencourseware', 'mit online']
        }
        
        platform_keywords = list(institution_mapping.keys()) + ['edx']
        duration_keywords = {
            'quick': ['short', 'quick', 'brief', '1hour', '2hour', '3hour', '4hour', '5hour'],
            'short': ['medium', 'intermediate', 'week', '2week', '3week', 'month'],
            'full': ['long', 'complete', 'comprehensive', 'bootcamp', 'specialization', 'certification']
        }
        
        # Initialize search criteria
        platform_filter = None
        duration_filter = None
        expanded_terms = []
        institution_terms = []
        
        # Process search terms
        for term in search_terms:
            # Check for institution/platform mentions
            institution_found = False
            for institution, variations in institution_mapping.items():
                if term.lower() in [v.lower() for v in variations]:
                    institution_terms.extend(variations)
                    platform_filter = institution
                    institution_found = True
                    break
            if institution_found:
                continue
            
            # Check for platform mentions
            if term in platform_keywords:
                platform_filter = term
                continue
                
            # Check for duration mentions
            duration_found = False
            for dur_type, keywords in duration_keywords.items():
                if term in keywords:
                    duration_filter = dur_type
                    duration_found = True
                    break
            if duration_found:
                continue
            
            # Expand abbreviations
            if term in abbreviations:
                expanded_terms.extend(abbreviations[term])
            else:
                expanded_terms.append(term)
        
        # If platform specified in URL params, it overrides the search term
        if platform != 'all':
            platform_filter = platform
            # Add institution variations if it's an institution
            if platform.lower() in institution_mapping:
                institution_terms.extend(institution_mapping[platform.lower()])
        
        # If duration specified in URL params, it overrides the search term
        if duration != 'all':
            duration_filter = duration
        
        # Build the base query with expanded terms
        search_conditions = []
        search_text = ' '.join(expanded_terms)
        
        # Add institution-specific search conditions
        if institution_terms:
            for term in institution_terms:
                search_conditions.extend([
                    {'platform': {'$regex': f"\\b{term}\\b", '$options': 'i'}},
                    {'Instructors/University': {'$regex': f"\\b{term}\\b", '$options': 'i'}},
                    {'Course Name': {'$regex': f"\\b{term}\\b", '$options': 'i'}},
                    {'Course Description': {'$regex': f"\\b{term}\\b", '$options': 'i'}}
                ])
        
        # Add exact match conditions for other terms
        for term in expanded_terms:
            search_conditions.extend([
                {'Course Name': {'$regex': f"\\b{term}\\b", '$options': 'i'}},
                {'Course Description': {'$regex': f"\\b{term}\\b", '$options': 'i'}},
                {'Instructors/University': {'$regex': f"\\b{term}\\b", '$options': 'i'}},
                {'skills': {'$regex': f"\\b{term}\\b", '$options': 'i'}},
                {'domain': {'$regex': f"\\b{term}\\b", '$options': 'i'}}
            ])
        
        # Add partial match conditions with lower weight
        search_conditions.extend([
            {'Course Name': {'$regex': search_text, '$options': 'i'}},
            {'Course Description': {'$regex': search_text, '$options': 'i'}},
            {'Instructors/University': {'$regex': search_text, '$options': 'i'}},
            {'skills': {'$regex': search_text, '$options': 'i'}},
            {'domain': {'$regex': search_text, '$options': 'i'}}
        ])
        
        base_query = {'$or': search_conditions}
        
        # Add platform filter if specified
        if platform_filter and not institution_terms:  # Skip if we're already using institution terms
            base_query['platform'] = {'$regex': platform_filter, '$options': 'i'}
            
        # Add duration filter if specified
        if duration_filter:
            duration_ranges = {
                'quick': {'$lt': 5},
                'short': {'$gte': 5, '$lte': 20},
                'full': {'$gt': 20}
            }
            if duration_filter in duration_ranges:
                base_query['duration_in_hours'] = duration_ranges[duration_filter]
        
        # Search in the courses collection with distinct Course IDs
        pipeline = [
            {'$match': base_query},
            {'$group': {
                '_id': '$Course ID',
                'doc': {'$first': '$$ROOT'}
            }},
            {'$replaceRoot': {'newRoot': '$doc'}},
            {'$limit': 20}
        ]
        
        courses = list(mongo.db.courses.aggregate(pipeline))
        
        # Get total count for unique courses
        total_pipeline = [
            {'$match': base_query},
            {'$group': {'_id': '$Course ID'}},
            {'$count': 'total'}
        ]
        total_result = list(mongo.db.courses.aggregate(total_pipeline))
        total_results = total_result[0]['total'] if total_result else 0
        
        # Get user interactions for the found courses
        user_interactions = list(mongo.db.user_interactions.find({
            'user_id': ObjectId(current_user.get_id()),
            'course_id': {'$in': [str(course.get('Course ID', '')) for course in courses]}
        }))
        
        # Convert ObjectId to string for JSON serialization
        for course in courses:
            if '_id' in course:
                course['_id'] = str(course['_id'])
            if 'Course ID' in course:
                course['Course ID'] = str(course['Course ID'])
            
            # Add a snippet of matching content (description, instructor, or platform)
            snippets = []
            
            # Check description match
            if 'Course Description' in course and course['Course Description']:
                desc = course['Course Description'].lower()
                for term in expanded_terms + institution_terms:
                    if term.lower() in desc:
                        pos = desc.find(term.lower())
                        start = max(0, pos - 50)
                        end = min(len(desc), pos + len(term) + 50)
                        prefix = '...' if start > 0 else ''
                        suffix = '...' if end < len(desc) else ''
                        snippet = f"{prefix}{desc[start:end]}{suffix}"
                        if snippet not in snippets:
                            snippets.append(f"Description: {snippet}")
            
            # Check instructor match
            if 'Instructors/University' in course:
                instructor = course['Instructors/University'].lower()
                for term in expanded_terms + institution_terms:
                    if term.lower() in instructor:
                        snippets.append(f"Instructor: {course['Instructors/University']}")
                        break
            
            # Add platform info if it was part of the search
            if (platform_filter or institution_terms) and 'platform' in course:
                snippets.append(f"Platform: {course['platform']}")
            
            # Add duration info if it was part of the search
            if duration_filter and 'Duration' in course:
                snippets.append(f"Duration: {course['Duration']}")
            
            # Join all snippets
            if snippets:
                course['Description_Snippet'] = ' | '.join(snippets)
        
        for interaction in user_interactions:
            interaction['_id'] = str(interaction['_id'])
            interaction['user_id'] = str(interaction['user_id'])
            interaction['course_id'] = str(interaction['course_id'])
        
        return jsonify({
            'courses': courses,
            'interactions': user_interactions,
            'query': query,
            'total_results': total_results
        })
        
    except Exception as e:
        print(f"Search error: {str(e)}")
        return jsonify({'error': 'Search failed', 'message': str(e)}), 500

@app.route('/api/similar-courses/<course_id>')
@login_required
def get_similar_courses(course_id):
    try:
        # Get all recommended course IDs to exclude
        recommended_courses = set()
        
        # Get user interactions for recommendations
        user_interactions = list(mongo.db.user_interactions.find())
        user_interactions_df = pd.DataFrame(user_interactions)
        
        # Get similar courses based on liked courses
        similar_courses = recommender.get_recommendations_for_liked_courses(
            user_interactions_df,
            current_user.get_id(),
            n_recommendations=6,
            exclude_course_ids=list(recommended_courses)
        )
        
        if similar_courses is None or similar_courses.empty:
            return jsonify({'courses': [], 'total': 0})
        
        # Convert to list of dictionaries
        courses = similar_courses.to_dict('records')
        
        # Get user interactions for these courses
        current_interactions = list(mongo.db.user_interactions.find({
            'user_id': ObjectId(current_user.get_id()),
            'course_id': {'$in': [str(course.get('Course ID', '')) for course in courses]}
        }))
        
        # Convert ObjectId to string for JSON serialization
        for course in courses:
            if '_id' in course:
                course['_id'] = str(course['_id'])
            if 'Course ID' in course:
                course['Course ID'] = str(course['Course ID'])
            
            # Add reference to the original course
            course['recommended_because'] = f"Based on your interest in: {course.get('based_on_course_name', '')}"
        
        for interaction in current_interactions:
            interaction['_id'] = str(interaction['_id'])
            interaction['user_id'] = str(interaction['user_id'])
            interaction['course_id'] = str(interaction['course_id'])
        
        return jsonify({
            'courses': courses,
            'interactions': current_interactions,
            'total': len(courses)
        })
        
    except Exception as e:
        print(f"Error fetching similar courses: {str(e)}")
        return jsonify({'error': 'Failed to fetch similar courses'}), 500

@app.route('/api/trending-courses')
@login_required
def get_trending_courses():
    try:
        # Get all recommended and similar course IDs to exclude
        recommended_courses = set()
        
        # Get user interactions for recommendations
        user_interactions = list(mongo.db.user_interactions.find())
        user_interactions_df = pd.DataFrame(user_interactions)
        
        # Get trending courses
        trending_courses = recommender.get_trending_recommendations(
            user_interactions_df,
            n_recommendations=10,
            exclude_course_ids=list(recommended_courses)
        )
        
        if trending_courses is None or trending_courses.empty:
            return jsonify({'courses': [], 'total': 0})
        
        # Convert to list of dictionaries
        courses = trending_courses.to_dict('records')
        
        # Get user interactions for these courses
        current_interactions = list(mongo.db.user_interactions.find({
            'user_id': ObjectId(current_user.get_id()),
            'course_id': {'$in': [str(course.get('Course ID', '')) for course in courses]}
        }))
        
        # Convert ObjectId to string for JSON serialization
        for course in courses:
            if '_id' in course:
                course['_id'] = str(course['_id'])
            if 'Course ID' in course:
                course['Course ID'] = str(course['Course ID'])
            
            # Add trending indicator
            course['trending_indicator'] = '🔥 Trending'
        
        for interaction in current_interactions:
            interaction['_id'] = str(interaction['_id'])
            interaction['user_id'] = str(interaction['user_id'])
            interaction['course_id'] = str(interaction['course_id'])
        
        return jsonify({
            'courses': courses,
            'interactions': current_interactions,
            'total': len(courses)
        })
        
    except Exception as e:
        print(f"Error fetching trending courses: {str(e)}")
        return jsonify({'error': 'Failed to fetch trending courses'}), 500

@app.route('/chatbot')
@login_required
def chatbot_page():
    return render_template('chatbot.html')

@app.route('/chatbot', methods=['POST'])
@login_required
def handle_chat():
    data = request.get_json()
    user_message = data.get('message')
    chat_id = data.get('chat_id')
    
    try:
        # Get user's domain preference
        user_prefs = mongo.db.user_preferences.find_one({'user_id': current_user.id})
        domain = user_prefs.get('domain') if user_prefs else None
        
        # Initialize chatbot with user context
        chatbot = CourseBot(mongo.db, current_user.id, domain)
        
        # Get response and course recommendations
        response_data = chatbot.generate_response(user_message)
        
        return jsonify(response_data)
        
    except Exception as e:
        app.logger.error(f"Error in chatbot: {str(e)}")
        return jsonify({
            'response': 'I apologize, but I encountered an error. Please try again.',
            'recommended_courses': [],
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/chatbot/history', methods=['GET'])
@login_required
def get_chat_history():
    try:
        # Get all chat sessions for the current user
        chats = list(mongo.db.chat_history.find(
            {'user_id': ObjectId(current_user.get_id())},
            {'_id': 1, 'chat_id': 1, 'title': 1, 'created_at': 1}
        ).sort('created_at', -1))
        
        # Convert ObjectId to string for JSON serialization
        for chat in chats:
            chat['_id'] = str(chat['_id'])
            chat['user_id'] = str(chat['user_id'])
        
        return jsonify(chats)
    except Exception as e:
        print(f"Error getting chat history: {str(e)}")
        return jsonify([])

@app.route('/chatbot/messages/<chat_id>', methods=['GET'])
@login_required
def get_chat_messages(chat_id):
    try:
        # Get all messages for the specified chat
        messages = list(mongo.db.chat_messages.find(
            {
                'user_id': ObjectId(current_user.get_id()),
                'chat_id': chat_id
            }
        ).sort('timestamp', 1))
        
        # Convert ObjectId to string for JSON serialization
        for message in messages:
            message['_id'] = str(message['_id'])
            message['user_id'] = str(message['user_id'])
        
        return jsonify(messages)
    except Exception as e:
        print(f"Error getting chat messages: {str(e)}")
        return jsonify([])

@app.route('/chatbot/save', methods=['POST'])
@login_required
def save_chat_message():
    try:
        data = request.get_json()
        chat_id = data.get('chat_id')
        message = data.get('message')
        is_user = data.get('is_user', False)
        recommended_courses = data.get('recommended_courses', [])
        
        # Save message to database with timezone-aware datetime
        mongo.db.chat_messages.insert_one({
            'user_id': ObjectId(current_user.get_id()),
            'chat_id': chat_id,
            'message': message,
            'is_user': is_user,
            'recommended_courses': recommended_courses,
            'timestamp': datetime.now(UTC)
        })
        
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"Error saving chat message: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/chatbot/history', methods=['POST'])
@login_required
def create_chat_session():
    try:
        data = request.get_json()
        chat_id = data.get('chat_id')
        title = data.get('title', 'New Chat')
        
        # Create new chat session with timezone-aware datetime
        mongo.db.chat_history.insert_one({
            'user_id': ObjectId(current_user.get_id()),
            'chat_id': chat_id,
            'title': title,
            'created_at': datetime.now(UTC)
        })
        
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"Error creating chat session: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/chatbot/history/<chat_id>', methods=['DELETE'])
@login_required
def delete_chat_session(chat_id):
    try:
        # Delete chat session and all its messages
        mongo.db.chat_history.delete_one({
            'user_id': ObjectId(current_user.get_id()),
            'chat_id': chat_id
        })
        mongo.db.chat_messages.delete_many({
            'user_id': ObjectId(current_user.get_id()),
            'chat_id': chat_id
        })
        
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"Error deleting chat session: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/dashboard')
@login_required
def dashboard():
    try:
        # Initialize dashboard analytics
        dashboard = DashboardAnalytics(mongo.db, current_user.get_id())
        
        # Get all analytics data
        analytics = dashboard.get_all_analytics()
        
        return render_template(
            'dashboard.html',
            analytics=analytics,
            title="My Dashboard"
        )
    except Exception as e:
        flash('Error loading dashboard analytics', 'error')
        return redirect(url_for('recommendations'))

@app.route('/api/dashboard/refresh')
@login_required
def refresh_dashboard():
    """API endpoint to refresh dashboard data"""
    try:
        dashboard = DashboardAnalytics(mongo.db, current_user.get_id())
        analytics = dashboard.get_all_analytics()
        return jsonify(analytics)
    except Exception as e:
        return jsonify({
            'error': 'Failed to refresh dashboard data',
            'message': str(e)
        }), 500

@app.route('/update-preferences')
@login_required
def update_preferences():
    # Set a flag to indicate this is an update, not first-time setup
    session['is_updating_preferences'] = True
    return redirect(url_for('domain_selection'))

@app.route('/api/interactions/move-to-enrolled', methods=['POST'])
@login_required
def move_to_enrolled():
    try:
        data = request.get_json()
        required_fields = ['course_id', 'course_name', 'platform', 'course_link']
        
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
            
        current_time = datetime.now(UTC)
        
        # Update or insert into enrolled list
        mongo.db.user_interactions.update_one(
            {
                'user_id': ObjectId(current_user.get_id()),
                'course_id': data['course_id']
            },
            {
                '$set': {
                    'course_name': data['course_name'],
                    'platform': data['platform'],
                    'course_link': data['course_link'],
                    'interaction_type': 'enroll',
                    'updated_at': current_time
                },
                '$setOnInsert': {
                    'created_at': current_time
                }
            },
            upsert=True
        )
        
        # Remove from favorites list if it exists
        mongo.db.user_interactions.delete_one({
            'user_id': ObjectId(current_user.get_id()),
            'course_id': data['course_id'],
            'interaction_type': 'like'
        })
        
        # Get updated lists
        favorites = list(mongo.db.user_interactions.find({
            'user_id': ObjectId(current_user.get_id()),
            'interaction_type': 'like'
        }))
        enrolled = list(mongo.db.user_interactions.find({
            'user_id': ObjectId(current_user.get_id()),
            'interaction_type': 'enroll'
        }))
        
        # Convert ObjectId to string for JSON serialization
        for item in favorites + enrolled:
            item['_id'] = str(item['_id'])
            item['user_id'] = str(item['user_id'])
        
        return jsonify({
            'favorites': favorites,
            'enrolled': enrolled
        })
    except Exception as e:
        app.logger.error(f"Error in move_to_enrolled: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/interactions/move-to-completed', methods=['POST'])
@login_required
def move_to_completed():
    try:
        data = request.get_json()
        required_fields = ['course_id', 'course_name', 'platform', 'course_link']
        
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
            
        current_time = datetime.now(UTC)
        
        # Update or insert into completed list
        mongo.db.user_interactions.update_one(
            {
                'user_id': ObjectId(current_user.get_id()),
                'course_id': data['course_id']
            },
            {
                '$set': {
                    'course_name': data['course_name'],
                    'platform': data['platform'],
                    'course_link': data['course_link'],
                    'interaction_type': 'completed',
                    'updated_at': current_time
                },
                '$setOnInsert': {
                    'created_at': current_time
                }
            },
            upsert=True
        )
        
        # Remove from enrolled list if it exists
        mongo.db.user_interactions.delete_one({
            'user_id': ObjectId(current_user.get_id()),
            'course_id': data['course_id'],
            'interaction_type': 'enroll'
        })
        
        # Get updated lists
        enrolled = list(mongo.db.user_interactions.find({
            'user_id': ObjectId(current_user.get_id()),
            'interaction_type': 'enroll'
        }))
        completed = list(mongo.db.user_interactions.find({
            'user_id': ObjectId(current_user.get_id()),
            'interaction_type': 'completed'
        }))
        
        # Convert ObjectId to string for JSON serialization
        for item in enrolled + completed:
            item['_id'] = str(item['_id'])
            item['user_id'] = str(item['user_id'])
        
        return jsonify({
            'enrolled': enrolled,
            'completed': completed
        })
    except Exception as e:
        app.logger.error(f"Error in move_to_completed: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 