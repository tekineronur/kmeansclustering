# Project Overview
This project is centered around analyzing clinical data using Python, focusing on implementing algorithms and programs to gain insights and make predictions.

# Objective
The primary goal is to delve into a dataset comprising clinical cases recorded by Dr. Wolberg. With 699 rows and 11 columns, including patient IDs, attributes ranging from A2 to A10 with values between 1-10, and a class column categorizing cases as benign (2) or malignant (4), the aim is to extract meaningful patterns and predict classes effectively.

# Methodology
## Phase 1: Data Exploration and Preparation
read_to_pandas(): Uploading data into Pandas for processing.
fill(df): Handling null values by replacing them with the mean of respective columns.
stat_data(): Selecting specific columns for analysis, excluding unique identifiers and the class column.
grap(df): Creating histograms to visualize attribute distributions.
main(): Coordinating and executing all programs, ensuring comprehensive results.
## Phase 2: K-Means Clustering for Class Prediction
initial_centroids(): Selecting initial centroids by random selection.
distance(row, mu_2, mu_4): Calculating Euclidean distances to assign predicted classes based on proximity to centroids.
main(): Orchestrating program execution, outputting results, and tracking iterations.
## Phase 3: Refinement and Error Analysis
finding_stats(final_df): Analyzing errors by comparing predicted classes against actual classes.
main(): Conducting final computations, including centroid adjustments and class assignments, to improve predictions.
# Conclusion
The project aims to showcase the application of Python programming and algorithms in processing clinical data. By employing k-means clustering and error refinement techniques, the goal is to efficiently predict and analyze class distinctions within the datase
