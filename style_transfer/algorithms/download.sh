mkdir -p datasets
FILES=(monet2photo cezanne2photo ukiyoe2photo vangogh2photo iphone2dslr_flower)
for FILE in ${FILES[*]}
do
  URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$FILE.zip
  ZIP_FILE=./datasets/$FILE.zip
  TARGET_DIR=./datasets/$FILE/
  wget -N $URL -O $ZIP_FILE
  mkdir $TARGET_DIR
  unzip $ZIP_FILE -d ./datasets/
  rm $ZIP_FILE
done
