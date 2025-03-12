import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from bson.objectid import ObjectId

# Load environment variables
load_dotenv()

class CourseBot:
    def __init__(self, db, user_id, domain=None):
        # Initialize OpenRouter API for Deepseek
        try:
            self.api_key = os.getenv('OPENROUTER_API_KEY')
            print(f"API Key status: {'Loaded successfully' if self.api_key else 'Not found'}")  # Debug line
            if not self.api_key:
                raise ValueError("OPENROUTER_API_KEY not found in environment variables")
            self.api_url = "https://openrouter.ai/api/v1/chat/completions"
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "http://localhost:5000",  # Updated for local development
                "X-Title": "CourseHub"
            }
            print("API Headers configured successfully")  # Debug line
        except Exception as e:
            print(f"Error initializing OpenRouter client: {str(e)}")
            self.api_key = None

        # Store MongoDB connection and user context
        self.db = db
        self.user_id = user_id
        self.domain = domain
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        
        # Load and prepare course data
        self.prepare_course_data()
        
    def prepare_course_data(self):
        """Load course data from MongoDB and prepare TF-IDF vectors"""
        try:
            # Get courses from MongoDB
            courses = list(self.db.courses.find({}, {
                '_id': 1,
                'Course ID': 1,
                'Course Name': 1,
                'Description': 1,
                'platform': 1,
                'Course Link': 1,
                'Duration': 1,
                'domain': 1
            }))
            
            # Convert to DataFrame
            self.courses_df = pd.DataFrame(courses)
            
            # Handle missing fields
            self.courses_df['Course Name'] = self.courses_df['Course Name'].fillna('')
            self.courses_df['Description'] = self.courses_df['Description'].fillna('')
            self.courses_df['platform'] = self.courses_df['platform'].fillna('Not specified')
            self.courses_df['Duration'] = self.courses_df['Duration'].fillna('Not specified')
            self.courses_df['Course Link'] = self.courses_df['Course Link'].fillna('#')
            self.courses_df['domain'] = self.courses_df['domain'].fillna('Not specified')
            
            # Combine course name and description for better matching
            self.courses_df['combined_text'] = (
                self.courses_df['Course Name'] + ' ' + 
                self.courses_df['Description']
            )
            
            # Create TF-IDF matrix
            self.tfidf_matrix = self.vectorizer.fit_transform(self.courses_df['combined_text'])
            
        except Exception as e:
            print(f"Error preparing course data: {str(e)}")
            # Initialize empty DataFrame with required columns
            self.courses_df = pd.DataFrame(columns=[
                'Course ID', 'Course Name', 'Description', 
                'platform', 'Course Link', 'Duration', 'domain'
            ])
            self.tfidf_matrix = None
        
    def get_course_recommendations(self, query: str, top_n: int = 3, platform: str = None, duration: str = None) -> List[Dict[str, Any]]:
        """Get course recommendations based on the query using TF-IDF and cosine similarity"""
        try:
            if self.tfidf_matrix is None or self.courses_df.empty:
                return []
            
            # Institution mapping for better matching
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
            
            # Check if query is institution-based
            query_lower = query.lower()
            institution_terms = []
            for institution, variations in institution_mapping.items():
                if any(term.lower() in query_lower for term in variations):
                    institution_terms.extend(variations)
                    if not platform:  # Only set platform if not explicitly specified
                        platform = institution
                    break
            
            # Enhance query with institution terms if found
            enhanced_query = query
            if institution_terms:
                enhanced_query = f"{query} {' '.join(institution_terms)}"
                
            # Transform query to TF-IDF vector
            query_vector = self.vectorizer.transform([enhanced_query.lower()])
            
            # Calculate cosine similarity
            cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get all courses sorted by similarity
            indices = cosine_similarities.argsort()[::-1]
            
            # Filter courses based on criteria
            filtered_courses = []
            for idx in indices:
                course = self.courses_df.iloc[idx]
                
                # For institution searches, check multiple fields
                if institution_terms:
                    course_text = ' '.join([
                        str(course.get('platform', '')),
                        str(course.get('Instructors/University', '')),
                        str(course.get('Course Name', '')),
                        str(course.get('Course Description', ''))
                    ]).lower()
                    
                    if not any(term.lower() in course_text for term in institution_terms):
                        continue
                # Regular platform filter
                elif platform and platform.lower() not in str(course.get('platform', '')).lower():
                    continue
                    
                # Apply duration filter if specified (case-insensitive)
                if duration and duration.lower() not in str(course.get('Duration', '')).lower():
                    continue
                    
                # Apply domain filter if specified (case-insensitive)
                if self.domain and self.domain != 'No Preference' and str(course.get('domain', '')).lower() != self.domain.lower():
                    continue
                    
                filtered_courses.append({
                    'course_id': str(course.get('Course ID', '')),
                    'name': course.get('Course Name', 'Untitled Course'),
                    'platform': course.get('platform', 'Not specified'),
                    'duration': course.get('Duration', 'Not specified'),
                    'link': course.get('Course Link', '#'),
                    'similarity_score': float(cosine_similarities[idx])
                })
                
                if len(filtered_courses) >= top_n:
                    break
            
            return filtered_courses
            
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return []

    def generate_response(self, user_message: str) -> Dict[str, Any]:
        """Generate chatbot response using OpenRouter API with Deepseek model"""
        try:
            # Check if API key is available
            if self.api_key is None:
                print("API key is not initialized")
                return {
                    'response': "I apologize, but I am currently unable to process your request due to a configuration issue. Please try again later.",
                    'recommended_courses': [],
                    'status': 'error',
                    'error': 'OpenRouter API key not initialized'
                }

            # Check if message is a greeting
            greetings = ['hi', 'hello', 'hey', 'how are you', "what's up", 'whats up']
            if user_message.lower().strip() in greetings:
                return {
                    'response': "Hi, how may I help you today?",
                    'recommended_courses': [],
                    'status': 'success'
                }
            
            # Extract platform and duration preferences if specified
            platform = None
            duration = None
            message_lower = user_message.lower()
            
            # Check for platform preference
            platforms = ['coursera', 'udemy', 'edx', 'udacity']
            for p in platforms:
                if p in message_lower:
                    platform = p.title()
                    break
                    
            # Check for duration preference
            durations = ['hour', 'hours', 'week', 'weeks', 'month', 'months']
            for d in durations:
                if d in message_lower:
                    duration = d
                    break
            
            # Get course recommendations based on user query and filters
            recommended_courses = self.get_course_recommendations(
                user_message,
                platform=platform,
                duration=duration
            )
            
            if not recommended_courses:
                return {
                    'response': "I couldn't find any courses matching your criteria. Could you please try adjusting your search parameters or exploring different topics?",
                    'recommended_courses': [],
                    'status': 'success'
                }
            
            # Create payload for OpenRouter API
            payload = {
                "model": "deepseek/deepseek-chat:free",
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"""Based on the user's query: "{user_message}"
                    
                    Please provide a helpful response that:
                    1. Addresses their learning goals
                    2. Maintains a friendly and encouraging tone
                    3. Keeps the response separate from course details
                    4. Does not list or describe specific courses in the response
                    
                    Remember to be concise and focus on the user's needs."""}
                ],
                "temperature": 0.7,
                "max_tokens": 800
            }
            
            # Make API call to OpenRouter
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"API call failed with status code {response.status_code}: {response.text}")
            
            # Extract the chatbot's response
            response_data = response.json()
            chatbot_response = response_data['choices'][0]['message']['content']
            
            return {
                'response': chatbot_response,
                'recommended_courses': recommended_courses,
                'status': 'success'
            }
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return {
                'response': "I apologize, but I encountered an error while processing your request. Please try again.",
                'recommended_courses': [],
                'status': 'error',
                'error': str(e)
            }

    # Update system prompt for Deepseek
    system_prompt = """
    You are a highly intelligent and knowledgeable course recommendation assistant. Your task is to help users find the most suitable courses based on their queries and answer their questions about data science, AI, ML, and data engineering education.
    
    You are also a career guidance assistant, specialized in providing career path advice and course recommendations that build the skills needed for specific careers, especially data science, data engineering, Artificial intelligence and machine learning.
    
    Key Rules to Follow:
    
    1. **Greeting Responses**:
       - For greetings like 'hi', 'hello', 'hey', etc., respond with "Hi, how may I help you today?"
       - For messages like 'how are you', 'what's up', etc., respond with "How may I help you today?"
       - Always maintain a helpful, friendly tone while keeping focus on course recommendations.

    2. **Domain Knowledge**:
       - Be precise about domain definitions
       - Data Engineering is about building data infrastructure, pipelines, and ETL processes.
       - Data Science focuses on statistical analysis and insights from data.
       - Machine Learning is about developing algorithms that learn from data.
       - Artificial Intelligence is a broader field including ML and other approaches to mimic human intelligence.
       - Provide accurate information about course content
       - Explain technical concepts clearly

    3. **Comparative Requests**:
       - If the user asks to compare courses across platforms, clearly highlight the differences in: 
         content quality, instructor reputation, course structure, and practical applications.
       - Present a balanced view of courses from different platforms.
       - If the query is about comparing courses, ensure you highlight the key differences.

    4. **Different Query Types**:
        - For CAREER GUIDANCE queries, first provide career path advice and then suggest courses that build those skills.
        - For COMPARATIVE queries, explicitly compare the courses on content quality, instructor reputation, course structure, and practical applications.
        - For BEGINNER/ADVANCED queries, ensure the recommendations match the user's experience level.
        - For PLATFORM-SPECIFIC queries, explain why those platforms might be best for their needs.
        - For PRICE or DURATION focused queries, highlight those aspects in your response.
        - For CERTIFICATION queries, emphasize the credential value.
        - For SKILL-SPECIFIC queries, explain how the course builds that particular skill.
     
     5. **Domain Expertise**:
         - Show domain knowledge specific to the user's query:
         - DATA SCIENCE courses focus on analytics, statistics, and insights
         - DATA ENGINEERING courses focus on data pipelines, ETL, and infrastructure
         - MACHINE LEARNING courses focus on algorithms and model development
         - AI courses focus on broader artificial intelligence concepts
         - NLP courses focus on text processing and language understanding
         - COMPUTER VISION courses focus on image and video analysis

      6. **Learning Paths**:
         - If the user is asking about a learning path or career transition, provide a structured approach with course sequence recommendations.
         - For beginners, suggest foundational courses before advanced ones.
         - For career transitions, acknowledge existing transferable skills.

      7. **Answer User Questions First**:
          - If the user asks a question, first provide a thoughtful answer to their question, then recommend relevant courses.
          - For comparative questions, provide a balanced analysis before recommending courses.
                                               
      8. **Response Format**:
          - For each course recommendation, include:
            Course Name, Platform, Duration, and Link
          - Make recommendations relevant to the user's query
          - Highlight key features of recommended courses

      9. **Course Recommendations**:
          - Atleast provide one relevant coursera platform course while giving recommendation
          - If a user asks specifically for a course from a platform, include that platform in the recommendations.
          - if that platform is not available, provide a brief explanation.
          - Always provide a variety of course recommendations based on the user's query.

      10. **Interaction Style**:
          - Be concise and clear
          - Use a friendly, professional tone
          - Focus on being helpful and informative
      
       Always focus on providing accurate domain information and ensuring the course recommendations match exactly what the user is looking for.
                
       Use a friendly, conversational tone but be concise and focused on answering the user's specific question. Ensure all course links are included correctly.


       Remember: For simple greetings or "how are you" type messages, respond with the exact specified greeting responses.
    """

    def refresh_course_data(self):
        """Refresh course data and TF-IDF vectors"""
        self.prepare_course_data()

# Example usage:
if __name__ == "__main__":
    from pymongo import MongoClient
    
    # Connect to MongoDB
    client = MongoClient('mongodb://localhost:27017/')
    db = client['course_recommender']
    
    # Initialize chatbot
    chatbot = CourseBot(db, user_id="test_user", domain="Data Science")
    
    # Test the chatbot
    test_query = "I want to learn Python programming for data science"
    result = chatbot.generate_response(test_query)
    
    print("User Query:", test_query)
    print("\nChatbot Response:", result['response'])
    print("\nRecommended Courses:")
    for i, course in enumerate(result['recommended_courses'], 1):
        print(f"\n{i}. {course['name']}")
        print(f"   Platform: {course['platform']}")
        print(f"   Duration: {course['duration']}")
        print(f"   Link: {course['link']}") 