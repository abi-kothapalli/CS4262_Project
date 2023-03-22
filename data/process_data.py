from glob import glob
from tqdm import tqdm
import cv2
import json
import os


def get_metadata(metadata_files):
    metadata = {}

    for metadata_file in metadata_files:
        with open(metadata_file) as f:
            dat = json.load(f)
        metadata.update(dat)
    
    return metadata

def process_images(video_files, metadata):
    labels = {}
    os.makedirs("./processed_images", exist_ok=True)

    vid_num = 0
    for vid in tqdm(video_files):
        vid_name = os.path.basename(vid).split(".")[0]
        video = cv2.VideoCapture(vid)
        frame_num = 0
        vid_num += 1

        while video.isOpened():
            ret, frame = video.read()
            if ret:
                if frame_num % 30 == 0:
                    img_name = f"{vid_name}_{frame_num}"
                    cv2.imwrite(os.path.join("./processed_images", f"{img_name}.jpg"), frame)
                    labels[img_name] = metadata[f"{vid_name}.mp4"]["label"]
                frame_num += 1
            else:
                break
        video.release()

        if vid_num % 500 == 0:
            with open("./processed_images/labels.json", "w") as f:
                json.dump(labels, f)

    with open("./processed_images/labels.json", "w") as f:
        json.dump(labels, f)

def process_data():
    video_files = glob("./raw_data/*/*.mp4")
    metadata_files = glob("./raw_data/*/metadata.json")

    metadata = get_metadata(metadata_files)
    
    process_images(video_files, metadata)


if __name__ == "__main__":
    process_data()