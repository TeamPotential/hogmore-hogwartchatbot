import gdown

import os

os.system('pip install torch')

import os.path                                         # 메소드 call을 위한 module불러오기
file = 'poly_16_pytorch_model_48_1.bin'     # 예제 Textfile

if os.path.isfile(file):
    print("Yes. it is a file")
    pass
else:
    url = 'https://drive.google.com/uc?id=1e3eujIm3jjqCLL-nKk8FQmOtK0jARQ_U'
    output = 'poly_16_pytorch_model_48_1.bin'
    gdown.download(url, output, quiet=False)

print(1)