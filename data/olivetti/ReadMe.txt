Source dataset
http://www.cs.nyu.edu/~roweis/data.html
Grayscale faces 8 bit [0-255], a few images of several different people. 40 persons * 10 images/person
400 total images, 64x64 size.
From the Oivetti database at ATT. 

imshow(mat2gray(reshape(faces(:,1),64,64)))

[Olivetti]
# This data set is stemmed from the original ORL(64*64).
dimension: 32 * 32 = 1024
classes: 40 (0~39)
samples: 10 * 40 = 400