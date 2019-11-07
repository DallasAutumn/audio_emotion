import shutil
import os

src_dir = "/run/media/dallasautumn/data/duan-qiu-yang/2019-2020-现代程序设计技术-大作业-数据/A类问题/A3-闻声知乐/CASIA情感语料库/CASIA情感语料库/不同文本100"
dest_dir = "/run/media/dallasautumn/data/duan-qiu-yang/CASIA_train_test/不同文本100"
emotions = ["angry", "fear", "happy", "normal", "sad", "surprise"]
speakers = ["Chang.Liu", "Quanyin.Zhao", "Zhe.Wang", "Zuoxiang.Zhao"]


def emerge_files(src=src_dir, dest=dest_dir):
    for speaker in speakers:
        for emotion in emotions:
            flag = "emotional" if emotion != "normal" else "neutral"
            cur_dir = os.path.join(src_dir, speaker, emotion)

            for filename in os.listdir(cur_dir):
                if filename.endswith(".wav"):
                    src_path = os.path.join(cur_dir, filename)

                    if flag == "emotional":
                        dest_path = os.path.join(
                            dest_dir,
                            flag + '_' + emotion + '_' + speaker + '_' + filename
                        )
                    else:
                        dest_path = os.path.join(
                            dest_dir,
                            flag + '_' + speaker + '_' + filename
                        )

                    print(src_path, "====>", dest_path)
                    shutil.copy(src_path, dest_path)


if __name__ == "__main__":
    emerge_files()
    print("end")
