#%%
import boto3
import os
from pathlib import Path
from tqdm.auto import tqdm

# %% Download cores from S3 bucket
print('Starting AWS session' )
s3 = boto3.resource('s3')
bucket = s3.Bucket('libraml')

obj_list = bucket.objects.all()

test=[]
print('Starting downloads.' )
prefix='cores/' 
for obj in bucket.objects.filter(Prefix=prefix):
    if 'cores' in obj.key:
        KEY = obj.key
        name = KEY.split('/')[-1]
        print('{0}:{1}'.format(bucket.name, obj.key))
        dl_path = str(Path(os.getcwd()).joinpath('datasets\\libra\\tempB\\'+name))
        if not os.path.exists(dl_path):
            bucket.download_file(KEY,dl_path)