import os, pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_summary_path = os.path.join(BASE_DIR, 'summary.pkl')
if not os.path.exists(dataset_summary_path):
  print('summary.pkl not exists')
dataset_summary = pickle.load(open(dataset_summary_path,'r'))
print(dataset_summary)
