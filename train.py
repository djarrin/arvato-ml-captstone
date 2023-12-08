import argparse
from autogluon.tabular import TabularPredictor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True, help='Training data location (S3 URI)')
    return parser.parse_args()

def train():
    args = parse_args()

    # Load training data
    train_data = TabularPredictor.load_pd(args.train)

    # Create a TabularPredictor instance and train the model
    predictor = TabularPredictor(label="RESPONSE", verbosity=4, sample_weight='RECORD_WEIGHT', weight_evaluation=True, eval_metric='log_loss').fit(
        train_data=train_data,
        time_limit=600,
        presets="best_quality"
    )

    # Save the model to the output directory (S3 URI)
    output_path = '/opt/ml/model'  # SageMaker default output directory
    predictor.save(output_path)

if __name__ == '__main__':
    train()
