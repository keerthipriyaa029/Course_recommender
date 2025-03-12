from model import CourseRecommender
import os

def train_and_save_model():
    """Train and save the recommendation model"""
    try:
        # Initialize the recommender
        recommender = CourseRecommender()
        
        # Preprocess data and train the model
        print("Loading and preprocessing data...")
        recommender.preprocess_data()
        
        # Create models directory if it doesn't exist
        models_dir = 'course_recommender/models'
        os.makedirs(models_dir, exist_ok=True)
        
        # Save the model
        print("Saving model...")
        recommender.save_model()
        
        print("Model training and saving completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        return False

if __name__ == "__main__":
    train_and_save_model() 