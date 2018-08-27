from cv2 import imread,imwrite
import os
def fiveCrop(segmentation_dir):
    all_pics = os.listdir(segmentation_dir)
    for pic in all_pics:
        pic_name = pic[:pic.index(".")]
        five_crop_dir = "/home/zf/out/"+pic_name
        if os.path.exists(five_crop_dir):
            pass
        else:
            os.makedirs(five_crop_dir)

        img_matrix = imread(segmentation_dir+pic)
        size = img_matrix.shape

        up_img = img_matrix[0:int(size[0] * 0.2)]
        print(up_img.shape)
        down_img = img_matrix[int(size[0] * 0.8):size[0]]

        center_main = img_matrix[int(size[0] * 0.2):int(size[0] * 0.8)]

        left_img = center_main[:, 0:int(size[1] * 0.2), :]
        center_img = center_main[:, int(size[1] * 0.2):int(size[1] * 0.8), :]
        right_img = center_main[:, int(size[1] * 0.8):size[1], :]
        imwrite(five_crop_dir+"/up.jpg", up_img)
        print(pic_name+"的分割图片写入"+five_crop_dir+"/up.jpg"+"中")
        imwrite(five_crop_dir+"/down.jpg", down_img)
        print(pic_name + "的分割图片写入" + five_crop_dir + "/down.jpg" + "中")
        imwrite(five_crop_dir+"/left.jpg", left_img)
        print(pic_name + "的分割图片写入" + five_crop_dir + "/left.jpg" + "中")
        imwrite(five_crop_dir+"/center.jpg", center_img)
        print(pic_name + "的分割图片写入" + five_crop_dir + "/center.jpg" + "中")
        imwrite(five_crop_dir+"/right.jpg", right_img)
        print(pic_name + "的分割图片写入" + five_crop_dir + "/right.jpg" + "中")


