import gdown
import os

import os.path                                         # 메소드 call을 위한 module불러오기
file = 'poly_16_pytorch_model_32.bin'     # 예제 Textfile

if os.path.isfile(file):
    print("Yes. it is a file")
    pass
else:
    url = 'https://drive.google.com/uc?id=15RuDAuWGBT1G4FwgeaAdoa1IgfeKGlya'
    output = 'poly_16_pytorch_model_32.bin'
    gdown.download(url, output, quiet=False)