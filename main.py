from utils import *
import argparse


def run_yolo_augmentor():
    """
    Run the YOLO augmentor on a set of images.

    This function processes each image in the input directory, applies augmentations,
    and saves the augmented images and labels to the output directories.

    """
    imgs = [img for img in os.listdir(CONSTANTS["inp_img_pth"]) if is_image_by_extension(img)]

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.debug:
        imgs = imgs[:5]

    if not os.path.exists(CONSTANTS["out_img_pth"]+"_"+str(str(args.seed))):
        os.makedirs(CONSTANTS["out_img_pth"]+"_"+str(str(args.seed)))
    if not os.path.exists(CONSTANTS["out_lab_pth"]+"_"+str(str(args.seed))):
        os.makedirs(CONSTANTS["out_lab_pth"]+"_"+str(str(args.seed)))

    saved_image_count = 0

    for img_num, img_file in enumerate(imgs):
        print(f"{img_num+1}-image is processing: {img_file}\r", end="")
        image, gt_bboxes, aug_file_name = get_inp_data(img_file)
        aug_img, aug_label = get_augmented_results(image, gt_bboxes, args.seed)
        if len(aug_img):
            saved_image_count += 1
            save_augmentation(aug_img, aug_label, aug_file_name, args.seed)

    print(f"\n{saved_image_count} images were saved.")


if __name__ == "__main__":
    run_yolo_augmentor()