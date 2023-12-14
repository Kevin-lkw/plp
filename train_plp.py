from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from seq_dataset import Raw_Data5k_Dataset, Segment_Hint5k_Dataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
resume_path = './models/plp_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

print('qwq')


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/plp_model.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
# dataset = Raw_Data5k_Dataset()
dataset = Segment_Hint5k_Dataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])


# Train!
trainer.fit(model, dataloader)
