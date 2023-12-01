creator: Made Oka Resia Wedamerta (m.wedamerta@innopolis.university) BS20-AI
# Movie Recommender System

This repository contains the source code and documentation for a movie recommender system. The system is designed to provide personalized movie recommendations using collaborative filtering and demographic information.

## Project Structure

The project is organized into the following directories:

- **data**: Contains the MovieLens 100K dataset and separate directories for the training set (`trainset`) and test set (`testset`).
  
- **models**: Houses trained and serialized models, including the final checkpoint `best_model_with_demographics.pkl`.
  
- **notebooks**: Jupyter notebooks for various stages of the project, e.g., data exploration (`1.initial-data-exploration.ipynb`) and the main recommender system (`2.Movie_Recommendation_System.ipynb`).
  
- **reports**: Holds generated graphics and figures, with the `final_report.pdf` summarizing data exploration, solution exploration, training processes, and evaluation.

- **benchmark**: Includes the test set (`testset.pkl`) used for evaluation and the `evaluate.py` script for performing model evaluation.
- **demo**: Contains `Model_Demo.py`that can be used to deploy the model using stramlit as model visualization.

## Getting Started

To reproduce the results of the movie recommender system, follow the steps below:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/movie-recommender-system.git
   cd movie-recommender-system
   ```
2. **Evaluation:**
  The trained model is saved in the model folder. To evaluate the model, simply run the evaluation.py
3. **Demo/Visualization:**
   Model demonstration available in demo folder. it deployed using streamlit

   To run the streamlit app :
   ```bash
   streamlit run Model_Demo.py
   ```
5. **Reproduce the Training model:**
   The model can be reproduced by running the notebook "2.Movie_Recommendation_System.ipynb"
