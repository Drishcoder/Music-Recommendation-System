# 🎵 Music Recommendation System - Streamlit App

An attractive, theme-based Streamlit application for the Music Recommendation System.

## Features

✨ **Modern UI Design**
- Dark theme with vibrant pink and orange accents
- Responsive layout with custom styling
- Interactive visualizations with Plotly

🎯 **Recommendation Engine**
- AI-powered song recommendations using cosine similarity
- Cultural filtering (Indian vs. Non-Indian songs)
- Customizable number of recommendations
- Similarity score display

📊 **Data Visualizations**
- **PCA Projection**: 2D visualization of the music latent space
- **Audio Features Distribution**: Box plots for different genres
- **Genre Distribution**: Pie chart of available songs

⚙️ **Interactive Controls**
- Song search and selection
- Adjustable recommendation count (1-10)
- Genre and audio feature filtering
- Real-time dataset statistics

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Install required packages:**
```bash
pip install streamlit pandas numpy scikit-learn plotly
```

2. **Navigate to the project directory:**
```bash
cd "c:\Users\user\Desktop\Machine Learning\Music Reccomendation System Rebot"
```

3. **Run the Streamlit app:**
```bash
streamlit run app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`

## Usage

1. **Select a Song**: Use the dropdown menu to search and select a song from the database
2. **Choose Number of Recommendations**: Adjust the slider to get 1-10 song recommendations
3. **Click "Get Recommendations"**: View personalized recommendations with similarity scores
4. **Explore Visualizations**: Check the tabs to view PCA projections, feature distributions, and genre breakdowns

## File Structure

```
Music Reccomendation System Rebot/
├── app.py                 # Main Streamlit application
├── model.ipynb           # Model training and development notebook
├── dataset.csv           # Music dataset
└── README.md            # This file
```

## Technical Details

### Model Components
- **Feature Extraction**: 9 audio features (danceability, energy, loudness, etc.)
- **Dimensionality Reduction**: PCA (2 components)
- **Similarity Metric**: Cosine Similarity
- **Filtering Logic**: Genre-based cultural filtering

### Genres Included
- Classical
- Black Metal
- Dance
- Acoustic
- Hip-Hop
- Indian
- Rock
- Mandopop

## Customization

You can customize the app by modifying:
- **Colors**: Change the CSS variables in the theme section
- **Genres**: Edit the `target_genres` list in the `load_data()` function
- **Audio Features**: Modify the `audio_features` list
- **Layout**: Adjust column widths and spacing

## Requirements

- `streamlit>=1.0.0`
- `pandas>=1.0.0`
- `numpy>=1.19.0`
- `scikit-learn>=0.24.0`
- `plotly>=5.0.0`

## Troubleshooting

**Issue**: "FileNotFoundError: dataset.csv"
- **Solution**: Ensure `dataset.csv` is in the same directory as `app.py`

**Issue**: "ModuleNotFoundError"
- **Solution**: Install missing packages using pip: `pip install <package-name>`

**Issue**: App not opening in browser
- **Solution**: Manually open `http://localhost:8501` in your browser

## Performance Notes

- First run may take a few seconds to load and process the dataset
- Data is cached using `@st.cache_resource` for improved performance
- Subsequent searches will be much faster

## Future Enhancements

- Add user ratings and feedback
- Implement collaborative filtering
- Add playlist generation
- Include audio preview functionality
- Add more audio features for analysis

---

Built with  **Streamlit**, **Scikit-learn**, and **Plotly**
