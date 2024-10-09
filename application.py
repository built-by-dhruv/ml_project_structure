import gradio as gr
import joblib
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

def predict_math_score(gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score):
    # Create a CustomData object
    data = CustomData(
        gender=gender,
        race_ethnicity=race_ethnicity,
        parental_level_of_education=parental_level_of_education,
        lunch=lunch,
        test_preparation_course=test_preparation_course,
        reading_score=reading_score,
        writing_score=writing_score 
    )
    
    # Convert the data to a DataFrame
    pred_df = data.get_data_as_df()
    pipeline = PredictPipeline()

    print(pred_df.columns)
    # Make a prediction
    math_score = pipeline.predict(pred_df)

    print(f"Predicted math score in app.py: {math_score}")
    return int(math_score)

# Create the Gradio interface using the new API
iface = gr.Interface(
    fn=predict_math_score,
    inputs=[
        gr.Dropdown(['male', 'female'], label="Gender"),
        gr.Dropdown(['group B', 'group C', 'group A', 'group D', 'group E'], label="Race/Ethnicity"),
        gr.Dropdown(
            ["bachelor's degree", "some college", "master's degree", "associate's degree", 'high school', 'some high school'],
            label="Parental Level of Education"
        ),
        gr.Dropdown(['standard', 'free/reduced'], label="Lunch"),
        gr.Dropdown(['none', 'completed'], label="Test Preparation Course"),
        gr.Slider(0, 100, label="Reading Score"),
        gr.Slider(0, 100, label="Writing Score")
    ],
    outputs=gr.Number(label="Predicted Math Score"),
    title="Math Score Predictor",
    description="Predict the math score based on various inputs."
)


# Create a WSGI app
app = gr.mount_gradio_app(gr.make_wrappers(), iface, path="/")

# This line is for local testing, Elastic Beanstalk will ignore it
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=8000)