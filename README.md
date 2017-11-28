# speech-systhesis-duration-model

## baseline

In the baseline dir, u can see that there are two pys:
+ overall.py
+ phone_level.py

### overall.py
compute all the wav duration of the source speaker and the target speaker, then we can get
the scale, and for every test wav, we can use scale to predict the target duration

### phone_level.py
Before u do this, u may need to use htk tools to do a state level or phone level align for the wav.
then u will get a mlf file for each speaker which contains every phone's frames.

In this file, u only to compute scale of each phone, then use phone frames to predict the target phone frames.

## NN Based Model
align.py : do a force align for mlf file, because some phone list of the different speaker (same wav) is not the same.

load_data.py: from mlf file to load train, dev, test data

train.py: based model is a small blstm model, use this can train extremely good in this task, only 1-2 frame between in the source speaker and the target speaker.
 
