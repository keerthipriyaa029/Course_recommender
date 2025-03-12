from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd
from collections import Counter
from bson.objectid import ObjectId
from datetime import timezone

class DashboardAnalytics:
    def __init__(self, db, user_id):
        self.db = db
        self.user_id = ObjectId(user_id)
        
    def get_course_engagement_summary(self) -> Dict[str, int]:
        """Get summary of user's course interactions"""
        try:
            interactions = list(self.db.user_interactions.find({
                'user_id': self.user_id
            }))
            
            summary = {
                'liked_courses': sum(1 for x in interactions if x.get('interaction_type') == 'like'),
                'enrolled_courses': sum(1 for x in interactions if x.get('interaction_type') == 'enroll'),
                'completed_courses': sum(1 for x in interactions if x.get('interaction_type') == 'completed')
            }
            
            return summary
        except Exception as e:
            print(f"Error getting course engagement summary: {str(e)}")
            return {'liked_courses': 0, 'enrolled_courses': 0, 'completed_courses': 0}
            
    def get_completion_analytics(self) -> Dict[str, Any]:
        """Get course completion analytics"""
        try:
            # Get all enrolled courses
            interactions = list(self.db.user_interactions.find({
                'user_id': self.user_id
            }))
            
            total_enrolled = len(interactions)
            completed = sum(1 for x in interactions if x.get('interaction_type') == 'completed')
            
            # Get course details for duration analysis
            completed_course_ids = [
                x['course_id'] for x in interactions 
                if x.get('interaction_type') == 'completed'
            ]
            
            completed_courses = list(self.db.courses.find({
                'Course ID': {'$in': completed_course_ids}
            }))
            
            # Categorize by duration
            duration_categories = {'Short': 0, 'Medium': 0, 'Long': 0}
            for course in completed_courses:
                duration = course.get('Duration', '').lower()
                if 'hour' in duration or ('1-' in duration and 'week' in duration):
                    duration_categories['Short'] += 1
                elif 'week' in duration or ('1-' in duration and 'month' in duration):
                    duration_categories['Medium'] += 1
                else:
                    duration_categories['Long'] += 1
            
            return {
                'completion_rate': (completed / total_enrolled * 100) if total_enrolled > 0 else 0,
                'duration_distribution': duration_categories
            }
        except Exception as e:
            print(f"Error getting completion analytics: {str(e)}")
            return {
                'completion_rate': 0,
                'duration_distribution': {'Short': 0, 'Medium': 0, 'Long': 0}
            }
            
    def get_learning_preferences(self) -> Dict[str, Any]:
        """Get user's learning preferences analytics"""
        try:
            # Get all enrolled courses
            interactions = list(self.db.user_interactions.find({
                'user_id': self.user_id
            }))
            
            enrolled_course_ids = [x['course_id'] for x in interactions if x.get('interaction_type') == 'enroll']
            enrolled_courses = list(self.db.courses.find({
                'Course ID': {'$in': enrolled_course_ids}
            }))
            
            # Platform distribution
            platform_counts = Counter(
                course.get('platform', 'Unknown') 
                for course in enrolled_courses
            )
            
            # Get user preferences
            user_prefs = self.db.user_preferences.find_one({'user_id': self.user_id})
            selected_skills = user_prefs.get('skills', []) if user_prefs else []
            
            # Course level distribution (based on course descriptions and titles)
            level_counts = {'Beginner': 0, 'Intermediate': 0, 'Advanced': 0}
            for course in enrolled_courses:
                title = course.get('Course Name', '').lower()
                desc = course.get('Description', '').lower()
                
                if 'advanced' in title or 'advanced' in desc:
                    level_counts['Advanced'] += 1
                elif 'intermediate' in title or 'intermediate' in desc:
                    level_counts['Intermediate'] += 1
                else:
                    level_counts['Beginner'] += 1
            
            return {
                'platform_distribution': dict(platform_counts),
                'level_distribution': level_counts,
                'top_skills': selected_skills[:10]  # Top 10 skills
            }
        except Exception as e:
            print(f"Error getting learning preferences: {str(e)}")
            return {
                'platform_distribution': {},
                'level_distribution': {'Beginner': 0, 'Intermediate': 0, 'Advanced': 0},
                'top_skills': []
            }
            
    def get_interaction_trends(self) -> Dict[str, Any]:
        """Get user's interaction trends over time"""
        try:
            # Get all interactions with timestamps
            interactions = list(self.db.user_interactions.find({
                'user_id': self.user_id
            }))
            
            # Convert interactions to DataFrame for easier analysis
            df = pd.DataFrame(interactions)
            if df.empty:
                return {
                    'monthly_trends': {
                        'likes': [], 'enrollments': [], 'completions': [],
                        'months': []
                    },
                    'weekly_activity': {str(i): 0 for i in range(7)}  # 0 = Monday
                }
            
            # Ensure timestamp field exists
            df['timestamp'] = pd.to_datetime(
                df['timestamp'].apply(lambda x: x if x else datetime.now(timezone.utc))
            )
            
            # Monthly trends (last 6 months)
            six_months_ago = datetime.now(timezone.utc) - timedelta(days=180)
            monthly_data = df[df['timestamp'] >= six_months_ago].copy()
            
            monthly_trends = {
                'months': [],
                'likes': [],
                'enrollments': [],
                'completions': []
            }
            
            for month in pd.date_range(start=six_months_ago, end=datetime.now(timezone.utc), freq='M'):
                month_data = monthly_data[monthly_data['timestamp'].dt.month == month.month]
                monthly_trends['months'].append(month.strftime('%B %Y'))
                monthly_trends['likes'].append(sum(1 for x in month_data.itertuples() if x.interaction_type == 'like'))
                monthly_trends['enrollments'].append(sum(1 for x in month_data.itertuples() if x.interaction_type == 'enroll'))
                monthly_trends['completions'].append(sum(1 for x in month_data.itertuples() if x.interaction_type == 'completed'))
            
            # Weekly activity pattern
            weekly_activity = df['timestamp'].dt.dayofweek.value_counts().to_dict()
            weekly_activity = {str(i): weekly_activity.get(i, 0) for i in range(7)}
            
            return {
                'monthly_trends': monthly_trends,
                'weekly_activity': weekly_activity
            }
        except Exception as e:
            print(f"Error getting interaction trends: {str(e)}")
            return {
                'monthly_trends': {
                    'likes': [], 'enrollments': [], 'completions': [],
                    'months': []
                },
                'weekly_activity': {str(i): 0 for i in range(7)}
            }
    
    def get_all_analytics(self) -> Dict[str, Any]:
        """Get all dashboard analytics in one call"""
        return {
            'engagement_summary': self.get_course_engagement_summary(),
            'completion_analytics': self.get_completion_analytics(),
            'learning_preferences': self.get_learning_preferences(),
            'interaction_trends': self.get_interaction_trends()
        }

# Example usage:
if __name__ == "__main__":
    from pymongo import MongoClient
    
    # Connect to MongoDB
    client = MongoClient('mongodb://localhost:27017/')
    db = client['course_recommender']
    
    # Initialize dashboard analytics
    dashboard = DashboardAnalytics(db, user_id="test_user")
    
    # Get all analytics
    analytics = dashboard.get_all_analytics()
    print("Dashboard Analytics:", analytics) 