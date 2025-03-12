from flask import redirect, url_for, flash, render_template, session
from flask_login import login_user, login_required, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from bson import ObjectId
from .models import User

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if not username or not password:
            flash('Please fill all fields', 'danger')
            return redirect(url_for('signup'))

        # Check if username already exists
        if mongo.db.users.find_one({'username': username}):
            flash('Username already exists', 'danger')
            return redirect(url_for('signup'))

        # Create new user
        hashed_password = generate_password_hash(password)
        user_id = mongo.db.users.insert_one({
            'username': username,
            'password': hashed_password,
            'category': 'general'  # Default category
        }).inserted_id

        # Log in the new user
        user = User(str(user_id), username)
        login_user(user)
        
        # Set session state for new user
        session['user_id'] = str(user_id)
        session['username'] = username
        session['is_first_time'] = True
        
        # New users always go to domain selection first
        return redirect(url_for('domain_selection'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Find user in database
        user_data = mongo.db.users.find_one({'username': username})
        
        # Verify credentials
        if user_data and check_password_hash(user_data['password'], password):
            user = User(str(user_data['_id']), username)
            login_user(user)
            
            # Set session state
            session['user_id'] = str(user_data['_id'])
            session['username'] = username
            session['is_first_time'] = False
            
            # Existing users ALWAYS go directly to recommendations
            return redirect(url_for('recommendations'))
        
        flash('Invalid credentials', 'danger')
        return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()  # Clear all session data
    return redirect(url_for('login')) 