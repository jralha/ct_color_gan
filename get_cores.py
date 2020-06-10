#%%
import boto3
import os
from pathlib import Path
from tqdm.auto import tqdm

# %% Download cores from S3 bucket
s3 = boto3.resource('s3')
bucket = s3.Bucket('libraml')

for obj in tqdm(list(bucket.objects.all())):
    if 'cores' in obj.key:
        KEY = obj.key
        name = KEY.split('/')[-1]
        dl_path = str(Path(os.getcwd()).joinpath('datasets\\libra\\tempB\\'+name))
        if not os.path.exists(dl_path):
            bucket.download_file(KEY,dl_path)
# %%
