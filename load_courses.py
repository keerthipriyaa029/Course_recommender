from pymongo import MongoClient
import pandas as pd

def load_courses():
    try:
        # Read the CSV file
        df = pd.read_csv(r'D:\datasets\data1\final_cleaned_course_data.csv')
        
        # Convert DataFrame to list of dictionaries
        courses = df.to_dict('records')
        
        # Connect to MongoDB
        client = MongoClient('mongodb://localhost:27017/')
        db = client['course_recommender']
        
        # Drop existing courses collection if it exists
        db.courses.drop()
        
        # Insert courses
        result = db.courses.insert_many(courses)
        
        print(f"Successfully loaded {len(result.inserted_ids)} courses into the database")
        
        # Create index on Course Name and Course Description for better search performance
        db.courses.create_index([
            ("Course Name", "text"),
            ("Course Description", "text")
        ])
        
        print("Created text index on Course Name and Course Description")
        
        # Print first few courses as verification
        print("\nFirst few courses loaded:")
        for course in db.courses.find().limit(3):
            print(f"Course Name: {course.get('Course Name')}")
            print(f"Platform: {course.get('platform')}")
            print("---")
        
    except Exception as e:
        print(f"Error loading courses: {str(e)}")
    finally:
        client.close()

if __name__ == "__main__":
    load_courses() 