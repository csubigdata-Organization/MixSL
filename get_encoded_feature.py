import argparse
from Encoded_Utils.encoded_dataset import save_encoded_feature

graph_classification_dataset=['DD','MUTAG','PROTEINS','NCI1','NCI109','IMDB-BINARY','REDDIT-BINARY', 'BZR', 'COX2', 'IMDB-MULTI', 'COLLAB', 'REDDIT-MULTI-5K', 'KKI', 'OHSU', 'PROTEINS_full', 'Peking_1', 'github_stargazers']

def arg_parse():
    parser = argparse.ArgumentParser("MixSL.")

    parser.add_argument('--data_name', type=str, default='MUTAG', help='location of the data corpus')

    return parser.parse_args()

args = arg_parse()

args.graph_classification_dataset = graph_classification_dataset

if args.data_name in args.graph_classification_dataset:
    save_encoded_feature(args.data_name)
