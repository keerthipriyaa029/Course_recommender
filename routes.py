@app.route('/update-preferences')
@login_required
def update_preferences():
    # Set a flag to indicate this is an update, not first-time setup
    session['is_updating_preferences'] = True
    return redirect(url_for('domain_selection'))

@app.route('/domain-selection', methods=['GET', 'POST'])
@login_required
def domain_selection():
    # Allow access if it's first time or updating preferences
    if not session.get('is_first_time', False) and not session.get('is_updating_preferences', False):
        return redirect(url_for('recommendations'))

    if request.method == 'POST':
        selected_domain = request.form.get('domain')
        if not selected_domain:
            flash('Please select a domain', 'danger')
            return redirect(url_for('domain_selection'))

        # Save or update domain preference
        mongo.db.user_preferences.update_one(
            {'user_id': ObjectId(current_user.get_id())},
            {
                '$set': {
                    'user_id': ObjectId(current_user.get_id()),
                    'domain': selected_domain
                }
            },
            upsert=True
        )

        # Redirect to skills selection
        return redirect(url_for('skills_selection'))

    # Get current preferences if updating
    if session.get('is_updating_preferences', False):
        user_preferences = mongo.db.user_preferences.find_one({
            'user_id': ObjectId(current_user.get_id())
        })
        return render_template('domain_selection.html', current_domain=user_preferences.get('domain') if user_preferences else None)

    return render_template('domain_selection.html')

@app.route('/skills-selection', methods=['GET', 'POST'])
@login_required
def skills_selection():
    # Allow access if it's first time or updating preferences
    if not session.get('is_first_time', False) and not session.get('is_updating_preferences', False):
        return redirect(url_for('recommendations'))

    # Check if domain is selected
    user_preferences = mongo.db.user_preferences.find_one({
        'user_id': ObjectId(current_user.get_id())
    })
    
    if not user_preferences or not user_preferences.get('domain'):
        flash('Please select your domain first', 'warning')
        return redirect(url_for('domain_selection'))

    if request.method == 'POST':
        selected_skills = request.form.getlist('skills')
        if not selected_skills:
            flash('Please select at least one skill', 'danger')
            return redirect(url_for('skills_selection'))

        # Update user preferences with selected skills
        mongo.db.user_preferences.update_one(
            {'user_id': ObjectId(current_user.get_id())},
            {
                '$set': {
                    'skills': selected_skills
                }
            }
        )

        # Clear the update flag and first-time flag
        session['is_first_time'] = False
        session.pop('is_updating_preferences', None)
        
        flash('Your preferences have been updated successfully!', 'success')
        return redirect(url_for('recommendations'))

    return render_template('skills_selection.html', 
                         domain=user_preferences.get('domain'),
                         current_skills=user_preferences.get('skills', []))

@app.route('/recommendations')
@login_required
def recommendations():
    # Check if this is a first-time user
    if session.get('is_first_time', False):
        return redirect(url_for('domain_selection'))
        
    try:
        # Get user interactions
        user_interactions = list(mongo.db.user_interactions.find({
            'user_id': ObjectId(current_user.get_id())
        }))
        
        # Get user preferences if they exist
        user_preferences = mongo.db.user_preferences.find_one({
            'user_id': ObjectId(current_user.get_id())
        })
        
        # For returning users without preferences, show empty state
        if not user_preferences or not user_preferences.get('domain') or not user_preferences.get('skills'):
            flash('Please update your preferences to get personalized recommendations', 'info')
            return render_template('recommendations.html', 
                                recommendations=[], 
                                similar_recommendations=[],
                                trending_recommendations=[],
                                interactions=[],
                                has_more=False)
        
        # Get recommendations based on user preferences
        recommended_courses = recommender.get_knowledge_based_recommendations(
            user_domain=user_preferences.get('domain', []),
            user_skills=user_preferences.get('skills', []),
            user_level='Beginner',
            n_recommendations=10,
            offset=0
        )
        
        if recommended_courses is None or recommended_courses.empty:
            return render_template('recommendations.html', 
                                recommendations=[], 
                                similar_recommendations=[],
                                trending_recommendations=[],
                                interactions=[],
                                has_more=False)
        
        # Get similar courses based on liked courses
        similar_courses = recommender.get_recommendations_for_liked_courses(
            pd.DataFrame(user_interactions) if user_interactions else pd.DataFrame(),
            current_user.get_id(),
            n_recommendations=6,
            exclude_course_ids=recommended_courses['Course ID'].tolist()
        )
        
        # Get trending courses
        trending_courses = recommender.get_trending_recommendations(
            pd.DataFrame(user_interactions) if user_interactions else pd.DataFrame(),
            n_recommendations=10,
            exclude_course_ids=recommended_courses['Course ID'].tolist() + 
                            ([] if similar_courses.empty else similar_courses['Course ID'].tolist())
        )
        
        # Convert recommendations to list of dictionaries
        recommendations = recommended_courses.to_dict('records')
        similar_recommendations = [] if similar_courses.empty else similar_courses.to_dict('records')
        trending_recommendations = [] if trending_courses.empty else trending_courses.to_dict('records')
        
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
        flash('An error occurred while generating recommendations', 'danger')
        return render_template('recommendations.html', 
                            recommendations=[], 
                            similar_recommendations=[],
                            trending_recommendations=[],
                            interactions=[],
                            has_more=False) 