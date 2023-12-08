import pandas as pd
import os
from autogluon.tabular import TabularPredictor

def train():
    train_data_path = os.environ['SM_CHANNEL_TRAIN']
    print("Path attempting to load data from", train_data_path)

     # Load training data
    all_files = [os.path.join(train_data_path, file) for file in os.listdir(train_data_path) if file.endswith('.csv')]
    train_data = pd.concat([pd.read_csv(file) for file in all_files], ignore_index=True)

    # Create a TabularPredictor instance and train the model
    # predictor = TabularPredictor(label="RESPONSE", verbosity=4, sample_weight='RECORD_WEIGHT', weight_evaluation=True).fit(
    predictor = TabularPredictor(label="RESPONSE", verbosity=4, eval_metric='log_loss').fit(
        train_data=train_data,
        time_limit=600,
        presets="best_quality"
    )

    # Save the model to the output directory (S3 URI)
    output_path = '/opt/ml/model'  # SageMaker default output directory
    model_directory = os.path.join(output_path, 'model')

    predictor.save(model_directory))

if __name__ == '__main__':
    train()
