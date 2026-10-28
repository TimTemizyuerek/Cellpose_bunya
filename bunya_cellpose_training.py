import os
import numpy as np
import torch
from cellpose import models, train, io

data_dir = os.environ["TMPDIR"] + "/training_images"

io.logger_setup()
output = io.load_train_test_data(data_dir, data_dir, image_filter="_img",
                                mask_filter="_masks", look_one_level_down=False)
images, labels, image_names, test_images, test_labels, image_names_test = output

if not isinstance(images, list):
    images = [images]
if not isinstance(labels, list):
    labels = [labels]
for i in range(len(labels)):
    if labels[i].ndim == 3 and labels[i].shape[2] == 1:
        labels[i] = labels[i][:, :, 0]
    labels[i] = labels[i].astype(np.int64)

model = models.CellposeModel(gpu=True)
model_path, train_losses, test_losses = train.train_seg(model.net,
                            train_data=images, train_labels=labels,
                            test_data=test_images, test_labels=test_labels,
                            weight_decay=0.1, learning_rate=1e-5,
                            n_epochs=100, model_name="standard_model_31_training_img")