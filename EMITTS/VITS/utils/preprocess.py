import argparse
from tqdm import tqdm
import utils.text_utils as text
from utils.utils import load_filelist

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--out_extension", default="cleaned")
#     parser.add_argument("--text_index", default=2, type=int)
#     # parser.add_argument("--filelists", nargs="+", default=["filelists/ljs_audio_text_val_filelist.txt", "filelists/ljs_audio_text_test_filelist.txt"])
#     # parser.add_argument("--filelists", nargs="+", default=["filelists/meld_audio_sid_text_emotion_intensity_train_filelist.txt"])
#     parser.add_argument("--filelists", nargs="+", default=[
#                         "filelists/meld_audio_sid_text_emotion_intensity_test_filelist.txt"])
#     # parser.add_argument("--filelists", nargs="+", default=["filelists/meld_audio_sid_text_emotion_intensity_dev_filelist.txt", "filelists/meld_audio_sid_text_emotion_intensity_test_filelist.txt","filelists/meld_audio_sid_text_emotion_intensity_train_filelist.txt"])
#     parser.add_argument("--text_cleaners", nargs="+",
#                         default=["english_cleaners2"])

#     args = parser.parse_args()

#     for filelist in args.filelists:
#         print("START:", filelist)
#         filepaths_and_text = load_filelist(filelist)
#         for i in tqdm(range(len(filepaths_and_text))):
#             original_text = filepaths_and_text[i][args.text_index]
#             cleaned_text = text._clean_text(original_text, args.text_cleaners)
#             filepaths_and_text[i][args.text_index] = cleaned_text

#         new_filelist = filelist + "." + args.out_extension
#         with open(new_filelist, "w", encoding="utf-8") as f:
#             f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])

# import argparse
# from tqdm import tqdm
# import text
# from utils import load_filepaths_and_text

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--out_extension", default="cleaned")
  parser.add_argument("--text_index", default=2, type=int)
  # parser.add_argument("--filelists", nargs="+", default=["filelists/ljs_audio_text_val_filelist.txt", "filelists/ljs_audio_text_test_filelist.txt"])
  parser.add_argument("--filelists", nargs="+", default=["path/to/filelist.txt"])
  # parser.add_argument("--filelists", nargs="+", default=["filelists/meld_audio_sid_text_emotion_intensity_test_filelist.txt"])
  # parser.add_argument("--filelists", nargs="+", default=["filelists/meld_audio_sid_text_emotion_intensity_dev_filelist.txt", "filelists/meld_audio_sid_text_emotion_intensity_test_filelist.txt","filelists/meld_audio_sid_text_emotion_intensity_train_filelist.txt"])
  parser.add_argument("--text_cleaners", nargs="+", default=["english_cleaners2"])


  args = parser.parse_args()

  t = 0
  for filelist in args.filelists:
    print("START:", filelist)
    new_filelist = filelist + "." + args.out_extension
    with open(new_filelist, "a", encoding="utf-8") as f:
      filepaths_and_text = load_filelist(filelist)
      for i in tqdm(range(len(filepaths_and_text))):
        if i < t:
          continue
        original_text = filepaths_and_text[i][args.text_index]
        cleaned_text = text._clean_text(original_text, args.text_cleaners)
        filepaths_and_text[i][args.text_index] = cleaned_text
        # f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])
        f.write("|".join(filepaths_and_text[i]) + "\n")
        # print("|".join(filepaths_and_text[i]) + "\n")
        # if i == t + 499:
        #   break
