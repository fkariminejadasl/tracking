mkdir -p ~/data/data8_v1_old/{train,valid}/{images,labels}
ln -s ~/data/data8_v1/valid/labels/* ~/data/data8_v1_old/valid/labels
ln -s ~/data/data8_v1/valid/images/* ~/data/data8_v1_old/valid/images
ln -s ~/data/data8_v1/train/labels/* ~/data/data8_v1_old/train/labels
ln -s ~/data/data8_v1/train/images/* ~/data/data8_v1_old/train/images
ln -s ~/data/combChromis_norm.v1i.yolov5pytorch/valid/images/* ~/data/data8_v1_old/valid/images
ln -s ~/data/combChromis_norm.v1i.yolov5pytorch/valid/labels/* ~/data/data8_v1_old/valid/labels
ln -s ~/data/combChromis_norm.v1i.yolov5pytorch/train/labels/* ~/data/data8_v1_old/train/labels
ln -s ~/data/combChromis_norm.v1i.yolov5pytorch/train/images/* ~/data/data8_v1_old/train/images

# copy data.yaml file and change the path to
: '
train: ~/data/data8_v1_old/train/images
val: ~/data/data8_v1_old/valid/images

nc: 1
names: ["0"]
'