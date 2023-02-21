import numpy as np
import cv2 as cv

# base_dir = "../drn/datasets/cityscapes/leftImg8bit/demoVideo/stuttgart_00/" 
# fourcc = cv.VideoWriter_fourcc(*'X264')
# out = cv.VideoWriter('./DIS/testwrite.mkv', fourcc, 30, (2048, 1024), True)

# for  i in range(1, 51):
#     frame = cv.imread(base_dir+f"stuttgart_00_000000_{i:06d}_leftImg8bit.png")
#     out.write(frame)
# out.release()


cap = cv.VideoCapture('./DIS/testwrite.mkv')
i = 0
while True:
    ret, frame = cap.read()
    if ret:
        # cv.imshow('frame', frame)
        print(f'./DIS/frames/{i:06d}.png')
        cv.imwrite(f'./DIS/frames/{i:06d}.png', frame)
        i += 1
    else:
        break
cv.destroyAllWindows()