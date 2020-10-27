import generate_train_eval 
from utils import upload_pipeline
import sys

resp = upload_pipeline(sys.argv[2], getattr(generate_train_eval, sys.argv[1]))
print(resp)