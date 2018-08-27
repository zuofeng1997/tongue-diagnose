from TongueDiagnose.help.Classify import classifyTongue
from TongueDiagnose.help.SegmentationTongue import Segmentation
if __name__ == '__main__':
    data_dir = "/home/zf/testdata/"

    good_pics = classifyTongue(data_dir)
    # Segmentation(good_pics)


