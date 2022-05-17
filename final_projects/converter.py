import os
import pyheif
from PIL import Image

def convert_images(images_path, output_path):
    """
        이미지들이 존재하는 경로, convert된 이미지가 저장될 경로를 입력하면
        확장자 변경, 이미지 크기 변경하여 저장.
    """
    for img_type in ["rock", "scissors", "paper"]:
        img_path_list = os.listdir(images_path + "/" + img_type)
        img_path_list.remove(".DS_Store")
        print(len(img_path_list))
        for i, img_path in enumerate(img_path_list):
            print(img_path)
            heif_file = pyheif.read(images_path + '/' + img_type + '/' + img_path)
            img = Image.frombytes(ㅎ
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                'raw',
                heif_file.mode,
                heif_file.stride,
            )
            img = img.resize((512, 512))
            img_num = 27 * 30 + i
            converted_img_path = output_path + '/' + img_type + '/' + str(img_num) + ".png"
            img.save(converted_img_path, "PNG")
            print(converted_img_path)
    print(len(os.listdir(images_path)), "개의 이미지. DONE")

if __name__ == "__main__":
    # print(os.listdir("./final_projects/images"))
    convert_images("./final_projects/images", "./final_projects/converted_images")