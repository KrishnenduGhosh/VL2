# Video Lecture Augmentation using CQA

# Introduction:
This project presents a video lecture augmentation system leveraging community question-answer (CQA) forums. Question-answer (QA) pairs, relevant to the off-topic concepts, are retrieved to cater better insight to the learners. The proposed augmentation model comprises five modules: video lecture segmentation, concept extraction, off-topic identification, concept classification and retrieval of relevant QA pairs.
1. Video Lecture Segmentation: Video lectures are segregated by identifying the topical shifts using a word embedding-based technique.
2. Concept Extraction: The segmented video lectures are indexed using its concepts, extracted with an entity annotation service.
3. Retrieval of relevant QA pairs: For each off-topic concept which needs augmentation, appropriate QA pairs are fetched.
4. Categorization of retrieved QA pairs: For each off-topic concept which needs augmentation, appropriate QA pairs are fetched.

# Steps:
## 1. Data collection:
Transcript data for 2581 Lectures present in 68 Courses (39 and 19 related to Computer science and Mathematics domains, respectively) collected from https://nptel.ac.in/course.html. These data are collected and stored in 1_Data folder in PDF format. Further details of the data is provided in 'Data.md' file. You can download the data from https://drive.google.com/open?id=1KTWhbPk-N8_rz-p-wSIMo9nKKYWU7PU9
## 2. Preprocessing:
Transcripts (PDFs) are converted into TXT format and pre-processed by removing spurious data (course metadata). The code '1_preprocess.py' converts and preprocesses the data from folder '1_Data' and stores in '2_Text' folder.
## 3. Segmentation:
The transcript data are segmented into topical segments. The code '2_segment.py' segments transcripts from '2_Text' folder and stores in '3_Segment' folder.
## 4. Concept Extraction:
Topics are extracted for each video lecture segments. The code '3_tag.py' extracts the topics and stores in '4_Topic' folder in JSON format.
## 5. Retrieval of Relevant Video Segments:
Video lecture segments relevant to each of the off-topics are retrieved. The code '6_retrieval.py' retrieves the segments and stores in '6_Retrieved' folder in JSON format and as 'RT.txt' in '8_Retrieved/trec-eval/test' folder in TREC suggested text format. The '8_Result' folder is downloadable from https://drive.google.com/open?id=17-IxebyTtNsSXY98FfkTJWHK9goHhkOT which contains the folder 'trec-eval', providing the performance evaluation codes.
## 6. Categorization of Retrieved QA Pairs:
The retrieved video lecture segments are reranked using code '7_rerank.py' where the learned weights are used. The reranked segments are stored in '7_Reranked' folder in JSON format and as 'RR.txt' in '8_Retrieved/trec-eval/test' folder in TREC suggested text format.
## 7. Evaluation:
A. The retrieved and reranked segmnets are shown to the annotators and their relevance are tagged. The gold standard is present in 'GS.txt' file. The file can be downloaded from https://drive.google.com/open?id=1sKfmBveCkUtaL_5cJqKG0li_z-c0wns4 . The code '8_eval.py' evaluates the retrieval and re-ranking performance. The '8_Result' folder is downloadable from https://drive.google.com/open?id=17-IxebyTtNsSXY98FfkTJWHK9goHhkOT which contains the folder 'trec-eval', providing the performance evaluation codes.

# Run:
## Prepare the pre-requisites:
A. One needs a ist of supporting files to be present in the current directory. One can download these files (as recipients of 'lib' folder) from https://drive.google.com/open?id=11PJ0Y-3RavS2F0B8lj247M5pK19fK11I

## Execute:
Finally, run 'main.py' which offers a menu-based control to execute each of the above-mentioned modules.

# Contacts
In case of any queries, you can reach us at kghosh.cs@iitkgp.ac.in
