****Overview****

This project uses a Jupyter notebook (Project-1.ipynb) to predict total sales (sales_total) for mobile games using a RandomForestRegressor. The dataset (mobile games data.csv) contains 16,598 records with features like device_type, launch_year, game_genre, publisher_name, and regional sales. The notebook includes data preprocessing, feature engineering, hyperparameter tuning, model evaluation, and visualizations to improve the R² score from 0.70 to 0.80.

This README provides instructions to set up the environment, install dependencies, and run the notebook.

****Prerequisites****





Operating System: Windows, macOS, or Linux.



Python Version: Python 3.8 or higher (tested with Python 3.11.0, as per notebook metadata).



Jupyter Notebook: Required to run the .ipynb file.



Dataset: Ensure mobile games data.csv is available at d:/gradious/mobile games data.csv (Windows path). Adjust the path in the notebook if using a different location or OS.

Dependencies

The notebook relies on the following Python libraries:





pandas (2.2.2): Data manipulation and analysis.



numpy (1.26.4): Numerical computations.



matplotlib (3.8.4): Plotting visualizations.



seaborn (0.13.2): Enhanced statistical visualizations.



scikit-learn (1.5.0): Machine learning (preprocessing, models, metrics).



xgboost (2.0.3): XGBoost model for comparison.



lightgbm (4.3.0): LightGBM model for comparison.



joblib (1.4.2): Model serialization.

These versions are compatible with Python 3.11.0 and were current as of June 2025. Older versions may work but are not guaranteed.

***Setup Instructions***

1. Clone or Download the Repository





Download the Project-1.ipynb file and mobile games data.csv to your local machine.



Alternatively, clone the repository (if hosted):

git clone <repository-url>
cd <repository-directory>

2. Create a Virtual Environment

To avoid conflicts with other projects, use a virtual environment:

python -m venv venv

Activate the virtual environment:





Windows:

venv\Scripts\activate



macOS/Linux:

source venv/bin/activate

3. Install Dependencies

Install the required libraries using pip. A requirements.txt file is recommended for reproducibility. Create it with the following content:

pandas==2.2.2
numpy==1.26.4
matplotlib==3.8.4
seaborn==0.13.2
scikit-learn==1.5.0
xgboost==2.0.3
lightgbm==4.3.0
joblib==1.4.2
jupyter==1.0.0

Install the dependencies:

pip install -r requirements.txt

Alternatively, install libraries individually:

pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm joblib jupyter

4. Prepare the Dataset





Place mobile games data.csv in the directory d:/gradious/ (Windows). If using a different path or OS, update the file path in the notebook:

df = pd.read_csv('path/to/mobile games data.csv')



Ensure the dataset has the expected columns: title_name, device_type, launch_year, game_genre, publisher_name, sales_usa, sales_europe, sales_asia, sales_misc, sales_total.

5. Launch Jupyter Notebook

Start the Jupyter server:

jupyter notebook





This opens a browser window. Navigate to the directory containing Project-1.ipynb.



Open Project-1.ipynb by clicking it.

6. Run the Notebook





Execute cells sequentially (Shift+Enter) to load data, preprocess, train the model, and generate visualizations.



Key outputs include:





Data summaries (df.head(), df.info(), df.describe()).



Visualizations: histograms, box plots, correlation heatmap, prediction vs. actual scatter, residual plot, feature importance (*.png files saved in the working directory).



Model performance: R², RMSE, MAE for training and test sets.



Saved model: best_random_forest_model.pkl.

7. Verify Outputs





Check the console for metrics (e.g., Test R²: >=0.80).



Inspect saved visualizations (e.g., feature_importances_updated.png) for insights.



Ensure the model file (best_random_forest_model.pkl) is created for deployment.

Troubleshooting





ModuleNotFoundError: Ensure all libraries are installed in the active virtual environment. Verify with pip list.



FileNotFoundError: Confirm the dataset path is correct. Use absolute paths if relative paths fail.



Memory Issues: RandomForest with hyperparameter tuning may be memory-intensive. Reduce n_iter in RandomizedSearchCV (e.g., from 50 to 20) or use a machine with at least 8GB RAM.



Slow Execution: Tuning with n_jobs=-1 uses all CPU cores. If slow, set n_jobs=1 or reduce n_estimators.



Deprecation Warnings: If using older library versions, update to the specified versions to avoid compatibility issues.

Additional Notes





Environment: The notebook was developed with Python 3.11.0 (notebook metadata). Python 3.8–3.11 should be compatible.



Dataset Size: The dataset (16,598 rows) is manageable