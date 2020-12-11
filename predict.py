from model_utils import *
from data_utils import *
from sklearn.metrics import confusion_matrix, accuracy_score
import argparse
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def predict(model, test_directory, batch_size):
    
    dataTest = tf.keras.preprocessing.image_dataset_from_directory(
        test_directory,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(256, 256),
        shuffle=False,
        seed=1,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
    )

    dataTest = dataTest.map(lambda x, y: (x / 255., y))
    
    predicted_labels = model.predict(dataTest, batch_size=batch_size)
    predicted_labels = np.argmax(predicted_labels, axis=1)
    np_dataset = dataTest.as_numpy_iterator()
    groundTruth = []
    for item in np_dataset:
        groundTruth.extend(item[1].tolist())
    groundTruth = np.array(groundTruth)
    assert groundTruth.shape == predicted_labels.shape
    return groundTruth, predicted_labels


def load_model(model_name='res', pretrain=False, row=False):
    model = get_model(name=model_name, pretrain=True, target_size=256, n_class=9)
    if row:
        model_name += "_row"
    if pretrain:
        model_name += "_pretrain"
    model_folder = os.path.join('model', model_name)
    best_model_weights = os.path.join(model_folder,
                                      [item for item in os.listdir(model_folder) if ".index" in item][0].replace(
                                          ".index", ""))
    model.load_weights(best_model_weights)
    return model

def analyse(y_true, y_pred, model_name, pretrain, row):
    confu = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    if row:
        model_name += "_row"
    if pretrain:
        model_name += "_pretrain"
        
    df_cm = pd.DataFrame(confu, range(9), range(9))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d')  # font size
    plt.savefig(os.path.join("logs", model_name + "_%.4f" % acc + ".jpg"))

def statistically_test(result_csv_file, type="part"):
    df = pd.read_csv(result_csv_file)
    constant_p_threshold = 0.05
    if type == "part":
        values_row = df.loc[df["Part"]=="row"]['Acc']
        values_window = df.loc[df["Part"]=="window"]['Acc']
        stat_result = stats.ttest_rel(values_row, values_window)
        print(stat_result)
        if stat_result[1] < constant_p_threshold:
            print("P-value is smaller than %.2f, we can reject the null hypothesis of equal averages."%(constant_p_threshold))
    elif type == "init":
        values_random = df.loc[df["Init"] == "random"]['Acc']
        values_imagenet = df.loc[df["Init"] == "imagenet"]['Acc']
        stat_result = stats.ttest_rel(values_random, values_imagenet)
        print(stat_result)
        if stat_result[1] < constant_p_threshold:
            print("P-value is smaller than %.2f, we can reject the null hypothesis of equal averages." % (
                constant_p_threshold))
        else:
            print("P-value is larger than %.2f, we cannot reject the null hypothesis of equal averages." % (
                constant_p_threshold))
    elif type == "model":
        values_mobile = df.loc[df["Model"] == "mobile"]['Acc']
        values_vgg = df.loc[df["Model"] == "vgg"]['Acc']
        values_inception = df.loc[df["Model"] == "inception"]['Acc']
        values_res = df.loc[df["Model"] == "res"]['Acc']
        values_dense = df.loc[df["Model"] == "dense"]['Acc']
        stat_result = stats.friedmanchisquare(values_mobile, values_vgg, values_inception, values_res, values_dense)
        print(stat_result)



if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-test_dir', type=str,
    #                     help='Test directory',
    #                     dest='test_dir',
    #                     default="/home/junzheng/course/CSI5137/csi5137Project/malware-classification/test_imgs")
    # parser.add_argument('-model', type=str,
    #                     help=''' Models to be used, options are: 'res','vgg','googLeNet','dense','mobile', defaut is 'res' ''',
    #                     dest='model',
    #                     default='res')
    # parser.add_argument('-pretrain', type=bool,
    #                     help='Initialize the model with ImageNet pretrained weights',
    #                     dest='pretrain', const=True, default=False, nargs='?')
    # parser.add_argument('-row', type=bool,
    #                     help='use the row wise entropy comparison',
    #                     dest='row', const=True, default=False, nargs='?')
    # parser.add_argument('-batch_size', type=int,
    #                     help='batch size',
    #                     dest='batch_size',
    #                     default=32)
    #
    # args = parser.parse_args()
    #
    # model = load_model(model_name=args.model, pretrain=args.pretrain, row=args.row)
    # y_true, y_pred = predict(model, args.test_dir, args.batch_size)
    # analyse(y_true, y_pred, args.model, args.pretrain, args.row)
    statistically_test("logs/result.csv", type="init")
