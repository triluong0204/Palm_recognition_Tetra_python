import os
from smtplib import LMTP
import LMTrP

X_img_path = "dataset_palm/test"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', "bmp"}

for image_file in os.listdir("dataset_palm/test"):
    print(image_file)
    full_file_path = os.path.join("dataset_palm/test", image_file)
    print(format(full_file_path))