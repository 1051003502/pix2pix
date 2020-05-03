from PIL import Image
import numpy as np
import os
#合成训练集图片  两种风格的对应图片横向拼接起来
def jointPicture(path1,path2,resultPath):
    image_filenames = []
    image_data1 = []
    image_data2 = []
    final_width = 256 * 2
    final_high = 256
    #root_path = "data/train/"
    dirs=os.listdir(path1)
    for dir in dirs:
        image_filenames.append(dir)
    if(not os.path.exists(resultPath)):
        os.mkdir(resultPath)
    for file in image_filenames:
        image_data1 = (Image.open(path1  + '/'+file))
        image_data2 = (Image.open(path2  + '/'+file))
        target = Image.new('RGB', (final_width, final_high))
        target.paste(image_data1, (0, 0, 256, 256))
        target.paste(image_data2, (256, 0, 256 * 2, 256))
        target.save(resultPath +'/'+ file)
def picToNumpy():
    im = Image.open("pic01.jpg")
    img = np.asarray(im)
    img = img[:, 0:256, :]
    im = Image.fromarray(np.uint8(img))
    im.show()
def imgTonpy(rootDir,npyName):

    dirs=os.listdir(rootDir)
    trainDataList = []
    for dir in dirs:
        im = Image.open(rootDir+'/'+dir)
        imData=np.asarray(im)
        trainDataList.append(imData)
    trainDataNumpy=np.array(trainDataList)
    np.save(npyName,trainDataNumpy)
def test():
    L=[]
    L.append(6)
    L.append(7)

if __name__ == '__main__':
    '''im=Image.open("data\\01\\U_005C3F.jpg")
    print('format:{}'.format(im.format))
    print('mode:{}'.format(im.mode))
    print('size:{}'.format(im.size))
    list_data=list(im.getdata())
    print(len(list_data))'''
    imgTonpy("data/test/res","valData")
    #jointPicture('data/test/黑体','data/test/经黑','data/test/res')
    pass