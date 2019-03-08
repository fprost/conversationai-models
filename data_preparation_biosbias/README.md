# Data Preparation for biosbias dataset.


## Instructions

```
NOW=$(date +%Y%m%d%H%M%S)
JOB_NAME=data-preparation-$NOW


python run_preprocessing_biosbias.py \
  --job_name $JOB_NAME \
  --job_dir gs://conversationai-models/biosbias/dataflow_dir/$JOB_NAME \
  --input_data_path gs://conversationai-models/biosbias/input_data/BIOS_python2.pkl \
  --cloud
```

## Description


3 goals:
- Parse example (see below)
- Split into train/eval/test
- Write as TF records.

Input example:

{u'raw_title': u'assistant professor', u'bio': u'_ is also a Ronald D. Asmus Policy Entrepreneur Fellow with the German Marshall Fund and is a Visiting Fellow at the Centre for International Studies (CIS) at the University of Oxford. This commentary first appeared at Sada, an online journal published by the Carnegie Endowment for International Peace.', u'name': (u'Nora', u'Fisher', u'Onar'), u'title': u'professor', u'gender': u'F', u'URI': u'http://acturca.wordpress.com/2012/04/13/turkey-model-mideast/', u'raw': u'* Nora Fisher Onar is an assistant professor of international relations at Bahcesehir University in Istanbul. She is also a Ronald D. Asmus Policy Entrepreneur Fellow with the German Marshall Fund and is a Visiting Fellow at the Centre for International Studies (CIS) at the University of Oxford. This commentary first appeared at Sada, an online journal published by the Carnegie Endowment for International Peace.', u'start_pos': 109, u'path': u'crawl-data/CC-MAIN-2013-20/segments/1368696381249/wet/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.wet.gz'}


Output spec:
{
  'comment_text': tf.FixedLenFeature([], dtype=tf.string), # Select the second part of the snipped
  'gender': tf.FixedLenFeature([], dtype=tf.string), # gender are extracted ('M' or 'F')
  'title': tf.FixedLenFeature([], dtype=tf.int32), # Labels are parsed to integers
}