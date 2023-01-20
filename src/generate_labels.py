from fastai.vision.all import *
import argparse


parser = argparse.ArgumentParser(
	description="Parser for creating the initial dataset for Task 1",
	epilog='python generate_labels.py --in_file=data/task1/train_data/annotations.csv --out_file=data/task1/train_data/annotations_labeled.csv'
)

parser.add_argument('--seed', default=42, type=int, help="Seed for reproducibility")
parser.add_argument('--valid_pct', default=0.2, type=int,
					help="Percentage of images for the validation set from the labeled samples")

parser.add_argument('--in_file', type=str, help='Path to the original annotations', required=True)
parser.add_argument('--out_file', type=str, required=True,
					help='Path to the new annotations with train/validation split')


args = parser.parse_args()


def main():
	set_seed(args.seed, reproducible=True)
	splitter = RandomSplitter(valid_pct=args.valid_pct, seed=args.seed)

	df = pd.read_csv(args.in_file)
	idxs = list(range(df.shape[0]))
	_, idxs_valid = splitter(idxs)

	# Creating new column for marking train and validation samples
	df['is_valid'] = False

	# Setting validation indexes according to splitter
	df.loc[idxs_valid, 'is_valid'] = True

	# Save final annotations file
	df.to_csv(args.out_file, index=False)


if __name__ == '__main__':
	main()



