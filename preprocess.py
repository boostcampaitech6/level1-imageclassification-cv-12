import os
import random
import numpy as np
import PIL
import shutil

from PIL import Image

random.seed(42)

img_dir = "/data/ephemeral/home/removed_background_train"
save_dir = "/data/ephemeral/home/removed_background_train"


def split_profile_by_gender(profiles):
    """
    images의 하위폴더에서 한 사람의 폴더 이름을 확인해 남/여 구분해 따로 list를 만드는 함수

    Args:
        profiles = images의 하위폴더 이름 리스트
    """
    male, female = [], []

    for profile in profiles:
        if "Fake" in profile:
            continue

        id, gender, race, age = profile.split("_")

        if int(age) >= 57:
            if gender == "male":
                male.append(os.path.join(img_dir, profile))
            else:
                female.append(os.path.join(img_dir, profile))

    return [male, female]


def make_folder(save_dir):
    """
    새로 만든 사진을 젖아하는 폴더를 만드는 함수
    """
    if os.path.isdir(save_dir):
        return
    else:
        os.mkdir(save_dir)


def make_images(front_idx, back_idx, save_dir, new_folder_dir, profiles):
    """
    Image a, b를 mixup 해서 새로운 사진을 만들고,
    save_dir/new_folder_dir 위치에 저장합니다.

    Args:
        image_a (int) : 사진의 id
        image_b (ing) : 사진의 id
        save_dir (str) : 만들어진 사진을 저장할 상위 폴더 경로
        new_folder_dir (str) : 새로 만든 7장의 사진을 저장할 폴더의 이름
        profiles (list) : idx로 기존의 사진을 가져오기 위한 리스트
    """
    title = ["incorrect_mask", "mask1", "mask2", "mask3", "mask4", "mask5", "normal"]
    ext = ".png"

    for image in title:
        image_a = (
            np.array(
                Image.open(os.path.join(profiles[front_idx], image + ext)).convert(
                    "RGB"
                )
            )
            // 2
        )
        image_b = (
            np.array(
                Image.open(os.path.join(profiles[back_idx], image + ext)).convert("RGB")
            )
            // 2
        )

        new_image = image_a + image_b

        img = PIL.Image.fromarray(new_image)
        img.save(os.path.join(save_dir, new_folder_dir, image + ext))


def make_img_by_gender(gender, profiles, save_dir=img_dir):
    """
    성별에 따라 새로운 이미지를 생성하는 함수

    Args:
        gender (str) : 성별 - male, female
        profiles (list) : 성별에 따라 사람을 나눈 리스트
        save_dir (str) : 새로 만든 데이터 폴더를 저장할 위치
    """
    cnt = len(profiles) // 2  # 절반의 사진을 가지고

    front_img_idxs = set(random.sample([i for i in range(len(profiles))], cnt))
    back_img_idxs = list(set([i for i in range(len(profiles))]) - front_img_idxs)
    front_img_idxs = list(front_img_idxs)

    id = 0
    ids = sorted(os.listdir(save_dir))

    if ids:
        id = int(sorted(os.listdir(save_dir))[-1].split("_")[0])

    for i in range(len(front_img_idxs)):
        id += 1
        new_folder_dir = f"{id:0>6}_{gender}_Fake_60"
        os.mkdir(os.path.join(img_dir, new_folder_dir))

        make_images(
            front_img_idxs[i], back_img_idxs[i], save_dir, new_folder_dir, profiles
        )


def make_new_data(save_dir=img_dir):
    make_folder(save_dir)
    make_img_by_gender(gender="male", profiles=male, save_dir=img_dir)
    make_img_by_gender(gender="female", profiles=female, save_dir=img_dir)

    print("Done.")


def remove_fake_pic(save_dir=img_dir):
    if not os.path.isdir(save_dir):
        print("no folder")
        return

    for fake in [i for i in os.listdir(save_dir) if "Fake" in i]:
        fake_dir = os.path.join(save_dir, fake)
        shutil.rmtree(fake_dir)

    print("Remove Done")


folders = os.listdir(img_dir)

profiles = [folder for folder in folders if not folder.startswith(".")]
male, female = split_profile_by_gender(profiles)


remove_fake_pic()
make_new_data()
