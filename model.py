import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import pickle
from datetime import datetime, UTC, timedelta
import os

class CourseRecommender:
    def __init__(self, data_path='D:/datasets/data1/final_cleaned_course_data.csv'):
        """Initialize the recommender system"""
        self.data_path = data_path
        self.df = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.content_similarity_matrix = None
        self.scaler = MinMaxScaler()
        
        # Enhanced skill variations dictionary with institutions and platforms
        self.skill_variations = {
            # Technical Skills
            'powerbi': ['power bi', 'power-bi', 'microsoft power bi', 'power bi desktop', 'powerbi desktop', 'power bi service'],
            'tableau': ['tableau desktop', 'tableau server', 'tableau prep', 'tableau visualization'],
            'python': ['pyhton', 'pythn', 'py', 'python programming', 'python3'],
            'sql': ['mysql', 'postgresql', 'sqlite', 'sql database', 'tsql', 't-sql', 'plsql', 'pl/sql'],
            
            # Institutions and Platforms
            'google': ['google cloud', 'google analytics', 'google data studio', 'google bigquery', 'tensorflow', 'google ai'],
            'microsoft': ['microsoft azure', 'microsoft excel', 'microsoft power bi', 'azure ml', 'microsoft ai'],
            'ibm': ['ibm watson', 'ibm cloud', 'ibm data science', 'ibm analytics', 'cognos'],
            'aws': ['amazon web services', 'amazon aws', 'aws cloud', 'aws certified', 'amazon certified'],
            'databricks': ['databricks certified', 'databricks spark', 'databricks platform', 'databricks sql'],
            'coursera': ['coursera project', 'coursera certificate', 'coursera specialization'],
            'udacity': ['udacity nanodegree', 'udacity project', 'udacity certification'],
            'udemy': ['udemy course', 'udemy project', 'udemy certificate'],
            'stanford': ['stanford university', 'stanford online', 'stanford mooc'],
            'mit': ['massachusetts institute of technology', 'mit opencourseware', 'mit online'],
            
            # Keep existing variations...
            'excel': ['microsoft excel', 'excel analytics', 'advanced excel', 'excel dashboard'],
            'data visualization': ['data viz', 'visualization', 'visual analytics', 'data visualisation', 'business visualization'],
            'dashboard': ['dashboarding', 'dashboard creation', 'dashboard design', 'interactive dashboard'],
            'business intelligence': ['bi', 'business analytics', 'bi tools', 'business reporting'],
            'data analysis': ['data analytics', 'data analyst', 'analytical skills', 'data insights'],
            'reporting': ['report creation', 'report design', 'business reporting', 'analytical reporting'],
            'dax': ['data analysis expressions', 'power bi dax', 'dax formulas', 'dax calculations'],
            'power query': ['m language', 'power bi etl', 'power query editor', 'power bi transformation'],
            'javascript': ['js', 'javascrpt', 'javscript', 'java script'],
            'machine learning': ['ml', 'machine learing', 'machinelearning', 'machine-learning'],
            'artificial intelligence': ['ai', 'artifical intelligence', 'artificial inteligence'],
            'deep learning': ['dl', 'deep learing', 'deeplearning', 'deep-learning'],
            'data science': ['ds', 'datascience', 'data sciense', 'data-science'],
            'natural language processing': ['nlp', 'natural lang processing', 'nlproc'],
            'computer vision': ['cv', 'computervision', 'computer-vision'],
            'statistics': ['stats', 'stat', 'statistical', 'stat analysis'],
            'visualization': ['visualisation', 'viz', 'data viz', 'data visualization'],
            'database': ['db', 'databases', 'databse', 'data base'],
            'analysis': ['analytics', 'analyse', 'analyzing', 'data analysis'],
            'programming': ['coding', 'developement', 'development', 'software development'],
            'algorithms': ['algo', 'algorithims', 'algorythms', 'algorithmic'],
            'web development': ['web dev', 'webdev', 'web programming'],
            'data engineering': ['data eng', 'dataeng', 'data pipeline'],
            'cloud computing': ['cloud', 'aws', 'azure', 'gcp'],
            'big data': ['bigdata', 'hadoop', 'spark'],
            'devops': ['dev ops', 'dev-ops', 'development operations']
        }

        # Add institution weights for search relevance
        self.institution_weights = {
            'google': 1.5,
            'microsoft': 1.5,
            'ibm': 1.5,
            'aws': 1.5,
            'databricks': 1.5,
            'coursera': 1.3,
            'udacity': 1.3,
            'udemy': 1.3,
            'stanford': 1.4,
            'mit': 1.4
        }

    def normalize_skill(self, skill):
        """Normalize a skill name using the variations dictionary"""
        skill = str(skill).lower().strip()
        
        # Check direct matches in variations
        for standard, variations in self.skill_variations.items():
            if skill in [v.lower() for v in variations] or skill == standard.lower():
                return standard
        
        # Check partial matches
        for standard, variations in self.skill_variations.items():
            if standard.lower() in skill or any(var.lower() in skill for var in variations):
                return standard
            
        # If no match found, return the original skill
        return skill

    def normalize_skills_list(self, skills):
        """Normalize a list of skills"""
        if not skills:
            return []
        
        normalized = []
        for skill in skills:
            norm_skill = self.normalize_skill(skill)
            if norm_skill and norm_skill not in normalized:
                normalized.append(norm_skill)
        return normalized

    def preprocess_data(self):
        """Load and preprocess the course data"""
        try:
            self.df = pd.read_csv(self.data_path)
            
            # Clean 'No of reviews' column - extract numbers and convert to float
            self.df['No of reviews'] = self.df['No of reviews'].apply(
                lambda x: float(''.join(filter(str.isdigit, str(x)))) if pd.notnull(x) else None
            )
            
            # Handle missing values
            self.df['Description'] = self.df['Description'].fillna('No description available')
            self.df['Levels'] = self.df['Levels'].fillna('All Levels')
            self.df['Rating'] = self.df['Rating'].fillna(self.df['Rating'].median())
            self.df['No of reviews'] = self.df['No of reviews'].fillna(self.df['No of reviews'].median())
            
            # Convert skills and domain to lists if they're strings
            self.df['skills'] = self.df['skills'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
            self.df['domain'] = self.df['domain'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
            
            # Create combined text for TF-IDF
            self.df['combined_text'] = self.df.apply(self._create_combined_text, axis=1)
            
            # Create TF-IDF matrix
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=5000,
                ngram_range=(1, 2)
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['combined_text'])
            
            # Calculate content similarity matrix
            self.content_similarity_matrix = cosine_similarity(self.tfidf_matrix)
            
            return self.df
            
        except Exception as e:
            print(f"Error in preprocess_data: {str(e)}")
            return None
    
    def _create_combined_text(self, row):
        """Enhanced combined text creation for TF-IDF with institution weighting"""
        # Get instructor/university text and apply institution weights
        instructor_text = str(row.get('Instructors/University', '')).lower()
        platform_text = str(row.get('platform', '')).lower()
        
        # Apply weights to institutional content
        weighted_instructor = instructor_text
        weighted_platform = platform_text
        
        for institution, weight in self.institution_weights.items():
            if institution.lower() in instructor_text:
                weighted_instructor = f"{instructor_text} {' '.join([instructor_text] * int(weight * 10))}"
            if institution.lower() in platform_text:
                weighted_platform = f"{platform_text} {' '.join([platform_text] * int(weight * 10))}"
        
        text_parts = [
            str(row['Course Name']).lower() * 2,  # Double weight for course name
            str(row['Description']).lower(),
            ' '.join(str(s).lower() for s in row.get('skills', '')) * 3,  # Triple weight for skills
            ' '.join(str(d).lower() for d in row.get('domain', '')) * 2,  # Double weight for domain
            str(row.get('Levels', '')).lower(),
            weighted_instructor,  # Weighted instructor/university text
            weighted_platform    # Weighted platform text
        ]
        
        # Add variations of skills and institutions to improve matching
        skills_text = ' '.join(str(s).lower() for s in row.get('skills', ''))
        for term, variations in self.skill_variations.items():
            if term.lower() in skills_text or \
               term.lower() in instructor_text or \
               term.lower() in platform_text:
                text_parts.extend(v.lower() for v in variations * 2)
        
        return ' '.join(text_parts)
    
    def get_content_based_recommendations(self, course_idx, n_recommendations=5):
        """Get content-based recommendations using cosine similarity"""
        if self.content_similarity_matrix is None:
            raise ValueError("Model not initialized. Call preprocess_data() first.")
            
        # Get similarity scores
        sim_scores = list(enumerate(self.content_similarity_matrix[course_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n_recommendations+1]  # Exclude the course itself
        
        course_indices = [i[0] for i in sim_scores]
        return self.df.iloc[course_indices]
    
    def get_knowledge_based_recommendations(self, user_domain, user_skills, user_level='Beginner', n_recommendations=10, offset=0):
        """Enhanced knowledge-based recommendations with improved institution matching"""
        if self.df is None:
            raise ValueError("Model not initialized. Call preprocess_data() first.")
        
        # Convert inputs to lists if they're strings
        if isinstance(user_domain, str):
            user_domain = [user_domain]
        if isinstance(user_skills, str):
            user_skills = [user_skills]
        
        # Normalize user skills and add variations
        normalized_skills = self.normalize_skills_list(user_skills)
        expanded_skills = []
        for skill in normalized_skills:
            expanded_skills.append(skill)
            if skill.lower() in self.skill_variations:
                expanded_skills.extend(self.skill_variations[skill.lower()])
        
        # Calculate domain matches with fuzzy matching
        domain_matches = self.df['domain'].apply(lambda x: len(set(x) & set(user_domain)) > 0)
        
        # Enhanced TF-IDF similarity calculation with expanded skills
        skill_query = ' '.join(expanded_skills)
        skill_query_vector = self.tfidf_vectorizer.transform([skill_query])
        skill_similarities = cosine_similarity(skill_query_vector, self.tfidf_matrix).flatten()
        
        # Enhanced skill and institution matching
        def calculate_match_score(row_idx, course_skills):
            if not course_skills:
                return 0
            
            # Get course information
            course_row = self.df.iloc[row_idx]
            instructor_text = str(course_row['Instructors/University']).lower()
            platform_text = str(course_row['platform']).lower()
            course_name = str(course_row['Course Name']).lower()
            course_desc = str(course_row['Description']).lower()
            
            # Calculate institution match score
            institution_score = 0
            for institution, weight in self.institution_weights.items():
                if institution.lower() in instructor_text or institution.lower() in platform_text:
                    institution_score = max(institution_score, weight)
            
            # Normalize course skills and add variations
            norm_course_skills = self.normalize_skills_list(course_skills)
            expanded_course_skills = []
            for skill in norm_course_skills:
                expanded_course_skills.append(skill)
                if skill.lower() in self.skill_variations:
                    expanded_course_skills.extend(self.skill_variations[skill.lower()])
            
            # Calculate exact match score with expanded skills
            exact_match_score = len(set(expanded_course_skills) & set(expanded_skills)) / max(len(expanded_skills), 1)
            
            # Get TF-IDF similarity score
            tfidf_score = skill_similarities[row_idx]
            
            # Calculate keyword presence score
            keyword_score = sum([
                1 if skill.lower() in course_name else 0.5 if skill.lower() in course_desc else 0
                for skill in normalized_skills
            ]) / len(normalized_skills)
            
            # Combine scores with adjusted weights
            return (0.35 * exact_match_score + 
                   0.25 * tfidf_score + 
                   0.25 * keyword_score +
                   0.15 * institution_score)  # Added institution score weight
        
        # Apply enhanced matching
        match_scores = pd.Series([
            calculate_match_score(idx, skills) 
            for idx, skills in enumerate(self.df['skills'])
        ])
        
        # Define level progression
        level_order = {'Beginner': 0, 'Intermediate': 1, 'Expert': 2, 'All Levels': 1}
        user_level_value = level_order.get(user_level, 0)
        
        # Calculate appropriate level
        level_appropriateness = self.df['Levels'].apply(
            lambda x: 1 if x == 'All Levels' or level_order.get(x, 0) <= user_level_value + 1 else 0
        )
        
        # Calculate final scores with adjusted weights
        scores = (
            domain_matches.astype(float) * 0.2 +
            match_scores * 0.5 +
            level_appropriateness * 0.15 +
            self.df['Rating'].fillna(0) / 5 * 0.15
        )
        
        # Get recommendations with pagination
        top_indices = scores.nlargest(offset + n_recommendations).index[offset:offset + n_recommendations]
        recommendations = self.df.iloc[top_indices].copy()
        
        # Add detailed scoring information
        recommendations['recommendation_score'] = scores.iloc[top_indices].values
        recommendations['matched_skills'] = recommendations.apply(
            lambda row: [
                skill for skill in normalized_skills 
                if any(var.lower() in str(row['Course Name']).lower() or 
                      var.lower() in str(row['Description']).lower() or
                      var.lower() in str(row['Instructors/University']).lower() or
                      var.lower() in str(row['platform']).lower()
                      for var in [skill] + self.skill_variations.get(skill.lower(), []))
            ],
            axis=1
        )
        recommendations['similarity_score'] = skill_similarities[top_indices]
        
        # Sort by recommendation score and similarity score
        recommendations = recommendations.sort_values(
            ['recommendation_score', 'similarity_score'], 
            ascending=[False, False]
        )
        
        return recommendations
    
    def get_collaborative_recommendations(self, user_interactions_df, target_user_id, n_recommendations=5):
        """Get collaborative filtering recommendations based on user interactions"""
        if user_interactions_df.empty:
            return pd.DataFrame()
            
        try:
            # Ensure user_id is string type
            user_interactions_df['user_id'] = user_interactions_df['user_id'].astype(str)
            target_user_id = str(target_user_id)
            
            # Create user-item interaction matrix
            interaction_weights = {'like': 1, 'enroll': 2, 'completed': 3}
            user_course_matrix = pd.pivot_table(
                user_interactions_df,
                values='interaction_type',
                index='user_id',
                columns='course_id',
                aggfunc=lambda x: max(interaction_weights.get(i, 0) for i in x)
            ).fillna(0)
            
            # If target user has no interactions, return empty DataFrame
            if target_user_id not in user_course_matrix.index:
                return pd.DataFrame()
            
            # Calculate user similarity
            user_similarity = cosine_similarity(user_course_matrix)
            user_similarity_df = pd.DataFrame(
                user_similarity,
                index=user_course_matrix.index,
                columns=user_course_matrix.index
            )
            
            # Get similar users
            similar_users = (
                user_similarity_df[target_user_id]
                .sort_values(ascending=False)
                .index[1:6]  # Top 5 similar users, excluding self
            )
            
            # Get courses liked by similar users
            recommended_courses = set()
            for user_id in similar_users:
                user_courses = user_interactions_df[
                    (user_interactions_df['user_id'] == user_id) &
                    (user_interactions_df['interaction_type'].isin(['like', 'enroll', 'completed']))
                ]['course_id'].tolist()
                recommended_courses.update(user_courses)
            
            # Remove courses the target user has already interacted with
            target_user_courses = user_interactions_df[
                user_interactions_df['user_id'] == target_user_id
            ]['course_id'].tolist()
            recommended_courses = list(recommended_courses - set(target_user_courses))
            
            if not recommended_courses:
                return pd.DataFrame()
            
            # Get top N recommendations
            return self.df[self.df['Course ID'].isin(recommended_courses)].head(n_recommendations)
            
        except Exception as e:
            print(f"Error in collaborative filtering: {str(e)}")
            return pd.DataFrame()
    
    def get_hybrid_recommendations(self, user_id, user_domain, user_skills, user_level,
                                 user_interactions_df, n_recommendations=10):
        """Get hybrid recommendations combining all approaches"""
        try:
            # Get recommendations from each approach
            knowledge_recs = self.get_knowledge_based_recommendations(
                user_domain, user_skills, user_level, n_recommendations=n_recommendations
            )
            
            collab_recs = self.get_collaborative_recommendations(
                user_interactions_df, user_id, n_recommendations=n_recommendations
            )
            
            # If we have a recent course interaction, use it for content-based recommendations
            content_recs = pd.DataFrame()
            if not user_interactions_df.empty:
                recent_interaction = user_interactions_df[
                    user_interactions_df['user_id'] == str(user_id)
                ].sort_values('created_at', ascending=False).head(1)
                
                if not recent_interaction.empty:
                    recent_course_id = recent_interaction.iloc[0]['course_id']
                    try:
                        recent_course_idx = self.df[self.df['Course ID'].astype(str) == str(recent_course_id)].index[0]
                        content_recs = self.get_content_based_recommendations(
                            recent_course_idx, n_recommendations=n_recommendations
                        )
                    except (IndexError, KeyError) as e:
                        print(f"Could not find recent course in dataset: {str(e)}")
            
            # Combine recommendations with weights
            all_recommendations = pd.concat([
                knowledge_recs.assign(source='knowledge', weight=0.4),
                collab_recs.assign(source='collaborative', weight=0.3) if not collab_recs.empty else pd.DataFrame(),
                content_recs.assign(source='content', weight=0.3) if not content_recs.empty else pd.DataFrame()
            ])
            
            if all_recommendations.empty:
                return knowledge_recs  # Return knowledge-based recommendations if no other recommendations available
            
            # Remove duplicates, keeping the highest weighted occurrence
            all_recommendations = all_recommendations.sort_values('weight', ascending=False)
            all_recommendations = all_recommendations.drop_duplicates(subset='Course ID', keep='first')
            
            # Calculate final score incorporating rating and number of reviews
            all_recommendations['final_score'] = (
                all_recommendations['weight'] * 0.4 +
                all_recommendations['Rating'].fillna(0) / 5 * 0.3 +
                (all_recommendations['No of reviews'] / all_recommendations['No of reviews'].max()) * 0.3
            )
            
            # Get top N recommendations
            return all_recommendations.nlargest(n_recommendations, 'final_score').drop(
                columns=['source', 'weight', 'final_score']
            )
            
        except Exception as e:
            print(f"Error in hybrid recommendations: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def save_model(self, filepath='course_recommender/models/'):
        """Save the model to disk"""
        os.makedirs(filepath, exist_ok=True)
        model_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'content_similarity_matrix': self.content_similarity_matrix,
            'scaler': self.scaler,
            'df': self.df
        }
        with open(os.path.join(filepath, 'recommender_model.pkl'), 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath='course_recommender/models/recommender_model.pkl'):
        """Load the model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.tfidf_matrix = model_data['tfidf_matrix']
        self.content_similarity_matrix = model_data['content_similarity_matrix']
        self.scaler = model_data['scaler']
        self.df = model_data['df']

    def get_similar_courses(self, course_id, n_similar=3):
        """Get similar courses based on content similarity"""
        try:
            if self.content_similarity_matrix is None:
                raise ValueError("Model not initialized. Call preprocess_data() first.")
            
            # Find the course index
            course_idx = self.df[self.df['Course ID'].astype(str) == str(course_id)].index
            if len(course_idx) == 0:
                return pd.DataFrame()
            
            course_idx = course_idx[0]
            
            # Get similarity scores
            sim_scores = list(enumerate(self.content_similarity_matrix[course_idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get top similar courses (excluding the course itself)
            similar_indices = [i[0] for i in sim_scores[1:n_similar+1]]
            similar_courses = self.df.iloc[similar_indices].copy()
            
            # Add similarity scores
            similar_courses['similarity_score'] = [i[1] for i in sim_scores[1:n_similar+1]]
            
            return similar_courses
            
        except Exception as e:
            print(f"Error in get_similar_courses: {str(e)}")
            return pd.DataFrame()

    def get_trending_recommendations(self, user_interactions_df, n_recommendations=10, exclude_course_ids=None):
        """Get trending course recommendations based on collaborative filtering"""
        try:
            if user_interactions_df.empty:
                return pd.DataFrame()
            
            # Initialize exclude_course_ids if None
            if exclude_course_ids is None:
                exclude_course_ids = set()
            else:
                exclude_course_ids = set(str(cid) for cid in exclude_course_ids)
            
            # Calculate course popularity scores
            course_scores = {}
            recent_timeframe = datetime.now(UTC) - timedelta(days=30)  # Last 30 days
            
            # Filter recent interactions
            recent_interactions = user_interactions_df[
                pd.to_datetime(user_interactions_df['created_at']).dt.tz_localize('UTC') > recent_timeframe
            ]
            
            # Calculate weighted scores for each course
            for course_id in recent_interactions['course_id'].unique():
                if str(course_id) in exclude_course_ids:
                    continue
                    
                course_interactions = recent_interactions[recent_interactions['course_id'] == course_id]
                
                # Calculate weighted score based on interaction types
                score = (
                    len(course_interactions[course_interactions['interaction_type'] == 'like']) * 1 +
                    len(course_interactions[course_interactions['interaction_type'] == 'enroll']) * 2 +
                    len(course_interactions[course_interactions['interaction_type'] == 'completed']) * 3
                )
                
                course_scores[course_id] = score
            
            # Sort courses by score and get top N
            trending_course_ids = sorted(
                course_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:n_recommendations]
            
            # Get course details
            trending_courses = self.df[
                self.df['Course ID'].astype(str).isin([str(c[0]) for c in trending_course_ids])
            ].copy()
            
            # Add trending score
            trending_courses['trending_score'] = trending_courses['Course ID'].astype(str).map(dict(trending_course_ids))
            
            return trending_courses
            
        except Exception as e:
            print(f"Error in get_trending_recommendations: {str(e)}")
            return pd.DataFrame()

    def get_recommendations_for_liked_courses(self, user_interactions_df, user_id, n_recommendations=6, exclude_course_ids=None):
        """Get recommendations based on user's recently liked courses"""
        try:
            if user_interactions_df.empty:
                return pd.DataFrame()
            
            # Get user's recently liked courses (up to 2)
            liked_courses = user_interactions_df[
                (user_interactions_df['user_id'] == str(user_id)) &
                (user_interactions_df['interaction_type'] == 'like')
            ].sort_values('created_at', ascending=False)['course_id'].head(2)
            
            if liked_courses.empty:
                return pd.DataFrame()
            
            # Initialize exclude_course_ids if None
            if exclude_course_ids is None:
                exclude_course_ids = set()
            else:
                exclude_course_ids = set(str(cid) for cid in exclude_course_ids)
            
            # Get similar courses for each liked course
            similar_courses_list = []
            for course_id in liked_courses:
                if str(course_id) in exclude_course_ids:
                    continue
                    
                similar = self.get_similar_courses(course_id, n_similar=3)
                if not similar.empty:
                    similar['based_on_course'] = course_id
                    similar_courses_list.append(similar)
            
            if not similar_courses_list:
                return pd.DataFrame()
            
            # Combine all similar courses
            recommendations = pd.concat(similar_courses_list)
            
            # Remove duplicates and excluded courses
            recommendations = recommendations[
                ~recommendations['Course ID'].astype(str).isin(exclude_course_ids)
            ]
            recommendations = recommendations.sort_values('similarity_score', ascending=False)
            recommendations = recommendations.drop_duplicates(subset=['Course ID'], keep='first')
            
            # Get the original course names for reference
            liked_course_names = self.df[
                self.df['Course ID'].astype(str).isin(liked_courses.astype(str))
            ][['Course ID', 'Course Name']].set_index('Course ID')
            
            # Add the original course name
            recommendations['based_on_course_name'] = recommendations['based_on_course'].map(
                liked_course_names['Course Name']
            )
            
            return recommendations.head(n_recommendations)
            
        except Exception as e:
            print(f"Error in get_recommendations_for_liked_courses: {str(e)}")
            return pd.DataFrame() 