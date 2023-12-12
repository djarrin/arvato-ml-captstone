# Arvato ML Captstone
This project is my captstone project for the Udacity AWS ML Engineer course.

## Project Setup & Execution
1. You will need to setup sagemaker studio profile within your AWS account
2. Upload the Arvato training data set csv to your sagemaker default buckets /data directory (naming it train.csv)
3. Create a sagemaker notebook and set this repo as the repo to pull from.
4. Within Sagemaker studio open the clean_train.flow file
   1. First click Run Validation
   2. Then click create job and setup csv export job
   ![Processing Job Creation](visualizations/datawrangler-job-creation.png)

5. In the "Clean Data Train" section (in project_workbook.ipynd) you'll need to replace the train_data = pd.read_csv(f'{s3_data_path}/clean-large.csv') file path with the path of where your data processing job dumps the CSV.
6. Should be able to run all the cells in project_workbook.ipynd and recreate results.




