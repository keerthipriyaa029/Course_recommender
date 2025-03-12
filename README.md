# Course Recommendation System

A sophisticated course recommendation system that suggests online courses from Udemy and Coursera based on user preferences and skills. The system includes an AI-powered chatbot for interactive course discovery and an admin panel for system management.

## Features

- User Authentication (Login/Signup)
- Domain-based Course Selection
- Skill-based Course Filtering
- Personalized Course Recommendations
- User Dashboard
- AI-powered Chatbot Integration
- Modern and Responsive UI
- Admin Panel with Analytics

## Project Structure

```
course_recommender/
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── main.js
│   └── images/
├── templates/
│   ├── base.html
│   ├── login.html
│   ├── signup.html
│   ├── domain_selection.html
│   ├── skill_selection.html
│   ├── recommendations.html
│   ├── dashboard.html
│   └── chatbot.html
├── admin/
│   ├── templates/
│   │   ├── admin_login.html
│   │   ├── admin_dashboard.html
│   │   ├── user_management.html
│   │   └── course_analytics.html
│   ├── static/
│   │   ├── css/
│   │   └── js/
│   ├── admin.py
│   ├── create_admin.py
│   └── requirements.txt
├── models/
├── data/
├── utils/
├── config/
├── app.py
├── chatbot.py
├── model.py
├── routes.py
├── auth.py
├── dashboard.py
├── init_db.py
└── requirements.txt
```

## Database Structure

The system uses MongoDB with the following collections:

### Main Database: `course_recommender`

1. **user_details Collection**
   - `_id`: ObjectId (Primary Key)
   - `username`: String (Unique)
   - `password`: String (Hashed)
   - `email`: String

2. **user_preferences Collection**
   - `user_id`: ObjectId (References user_details)
   - `domain`: String
   - `skills`: Array[String]
   - Indexed on: `user_id`

3. **user_interactions Collection**
   - `user_id`: ObjectId (References user_details)
   - `course_id`: String
   - `course_name`: String
   - `platform`: String
   - `course_link`: String
   - `interaction_type`: String (enroll/like/completed)
   - `created_at`: DateTime
   - `updated_at`: DateTime
   - Compound Indexes:
     - `(user_id, course_id)` (Unique)
     - `(user_id, interaction_type)`
     - `created_at`

4. **admin_users Collection**
   - `_id`: ObjectId (Primary Key)
   - `username`: String
   - `password`: String (Hashed)

## Admin Panel Features

The admin panel provides the following functionality:

1. **Dashboard**
   - Total user count
   - Active users in last 7 days
   - User engagement rate
   - Average interactions per user

2. **User Management**
   - View all users
   - User preferences
   - User interaction statistics
   - Domain distribution
   - Skills distribution

3. **Course Analytics**
   - Total courses
   - Average ratings
   - Platform distribution
   - Course level distribution
   - Domain-wise analysis

4. **User Activity Monitoring**
   - Real-time user interactions
   - Course completion rates
   - User engagement metrics
   - Detailed activity logs

## Setup Instructions

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - Unix/MacOS:
     ```bash
     source venv/bin/activate
     ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   cd admin && pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in both root and admin directories with:
   ```
   SECRET_KEY=your_secret_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. Start MongoDB:
   ```bash
   mongod
   ```

6. Initialize the database:
   ```bash
   python init_db.py
   ```

7. Create admin user:
   ```bash
   cd admin
   python create_admin.py
   ```

8. Run the applications:
   - Main application:
     ```bash
     python app.py
     ```
   - Admin panel:
     ```bash
     cd admin
     python admin.py
     ```

## Technologies Used

- Backend: Flask (Python)
- Frontend: HTML5, CSS3, JavaScript
- Database: MongoDB
- AI Integration: OpenAI API
- UI Framework: Bootstrap 5
- Additional Libraries: See requirements.txt

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

