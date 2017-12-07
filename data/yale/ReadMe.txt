This subset is stemmed from CroppedYale.zip. 
We filtered the directory such as YaleB16 because it contains some bad pgm files.

The remaining files are of the same size. 64 faces for each person (with yaleB01_P00_Ambient.pgm removed).

31 persons, 64 images for each person, sized by 192*168 (imshow(mat2gray(reshape(group(:,1),192,168))))


[Yale]
# This data set is stemmed from the original Yale(192*168).
dimension: 32 * 32 = 1024
classes: 15 (0~14)
samples: 11 * 15 = 165