This repo contains work done by Daniel Kong, Ethan Deng, Jason Gu, Irving Zhao on the DSC 180B UCSD capstone project.\
Requirements for reproducibility are included in an environment file, and additionally include a Hugging Face account and Segments.ai account (for login tokens via api).

- Semantic Segmentation Models folder contains notebooks/code for the following:
  -  Binary Segmentation model testing (contours and edges) in seg_test.py
  -  trial_model, a trial version of a semantic segmentation model built on the base of Nvidia's mit-b0 lightweight transformer model, and trained on the hugging-face semantic-sidewalk dataset, to determine viability of approach and use as a template if promising.
  -  custom_segformer_training, containing the creation of public dataset repos, as well as the development, training, and deployment of our trained semantic segmentation model trained on our gathered dataset.
  -  custom_segformer_b3_training, containing the training of model on our gathered dataset using mit-b3 as baseline.
  -  upgraded_model_template, a notebook containing a template for our team to use when testing alternative baseline transformers to train.
- Classification Models folder contains notebooks/code for the following:
  -  The development, training, and deployment of binary inference models to determine broken-ness of the cables, service availability (oos, body damage, etc), screens, and plugs on ev chargers.
  -  The cleaning and generation of public dataset repos used to train each individual model.
- the notebook files should includes all requirements and steps for external frameworks to function.
- environment.yml file is included for the purposes of reproducibility and testing
- variations for publicly available repos for our dataset, including annotations and masks, for all datasets and models used can be found at: https://huggingface.co/dskong07
- The final version of our semantic segmentation model, utilizing mit-b3, can be found at: https://huggingface.co/irvingz/segformer-b3-finetuned-segments-chargers-full-v3.1
  - https://app.segments.ai/dskong07/chargers/samples (including all unlabeled/annotated images and filetypes)
  - https://app.segments.ai/dskong07/chargers-full (only jpeg, annotated and labeled images; used in the training of our most up-to-date model)

example of preliminary training output:\
![image](https://github.com/user-attachments/assets/a6ca6746-69df-44d6-8b82-a374d9bdc66a)


example of final trained model output:\
![image](https://github.com/user-attachments/assets/f2842610-d117-4ca0-a9fc-1bcbca10af60)
