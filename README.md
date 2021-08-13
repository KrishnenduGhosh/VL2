# Augmentating Video Lectures using CQA

# Introduction:
This project presents a video lecture augmentation system leveraging community question-answer (CQA) forums. Question-answer (QA) pairs, relevant to the course-relevant concepts, are retrieved to cater better insight to the learners. The proposed augmentation model comprises four modules: Concept extraction, Course-relevant concept identification, Retrieving augmentations (QA pairs) and Categorizing augmentations.
1. Concept Extraction: The segmented video lectures are indexed using its concepts, extracted with an entity annotation service.
2. Course-relevant concept identification: The course-relevant concepts are identified basde on their connectivity with other concepts in the concept space.
3. Retrieval of relevant QA pairs: For each course-relevant concept which needs augmentation, appropriate QA pairs are fetched.
4. Categorization of retrieved QA pairs: Each augmentation is classified.

# Steps:
## 1. Data collection:
Transcript data for 2581 Lectures present in 68 Courses (39 and 19 related to Computer science and Mathematics domains, respectively) collected from https://nptel.ac.in/course.html. These data are collected and stored in 1_Data folder in PDF format. Further details of the data is provided in 'Data.md' file. You can download the data from https://drive.google.com/open?id=1KTWhbPk-N8_rz-p-wSIMo9nKKYWU7PU9
## 2. Preprocessing:
Transcripts (PDFs) are converted into TXT format and pre-processed by removing spurious data (course metadata). The code '1_preprocess.py' converts and preprocesses the data from folder '1_Data' and stores in '2_Text' folder.
## 3. Concept Extraction:
Topics are extracted for each video lecture segments. The code '3_tag.py' extracts the topics and stores in '4_Topic' folder in JSON format.
## 4. Retrieving augmentations:
Video lecture segments relevant to each of the off-topics are retrieved. The code '6_retrieval.py' retrieves the segments and stores in '6_Retrieved' folder in JSON format and as 'RT.txt' in '8_Retrieved/trec-eval/test' folder in TREC suggested text format. The '8_Result' folder is downloadable from https://drive.google.com/open?id=17-IxebyTtNsSXY98FfkTJWHK9goHhkOT which contains the folder 'trec-eval', providing the performance evaluation codes.
## 6. Categorizing augmentations:
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

# Cite
If this work is helpful for your research, please cite our paper 'Augmenting Video Lectures: Identifying Off-topic Concepts and Linking to Relevant Video Lecture Segments' available at https://link.springer.com/article/10.1007/s40593-021-00257-z.

    @article{ghosh2021augmenting,
        title = "Augmenting Video Lectures: Identifying Off-topic Concepts and Linking to Relevant Video Lecture Segments",
        journal = "International Journal of Artificial Intelligence in Education",
        year = "2021",
        doi = "https://doi.org/10.1007/s40593-021-00257-z",
        url = "https://link.springer.com/article/10.1007/s40593-021-00257-z",
        author = "Krishnendu Ghosh, Sharmila Reddy Nangi, Yashasvi Kanchugantla, Pavan Gopal Rayapati, Plaban Kumar Bhowmick and Pawan Goyal ",
        keywords = "Video lecture augmentation, Off-topic concept identification, MOOCs, Concept similarity, Community detection, Retrieval and re-ranking"
    }

The module on retrieving questions is discussed in details in our paper 'Using Re-Ranking to Boost Deep Learning Based Community Question Retrieval' available at https://link.springer.com/article/10.1007/s40593-021-00257-z.

    @inproceedings{ghosh2017using,
    author = {Ghosh, Krishnendu and Bhowmick, Plaban Kumar and Goyal, Pawan},
    title = {Using Re-Ranking to Boost Deep Learning Based Community Question Retrieval},
    year = {2017},
    isbn = {9781450349512},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3106426.3106442},
    doi = {10.1145/3106426.3106442},
    booktitle = {Proceedings of the International Conference on Web Intelligence},
    pages = {807–814},
    numpages = {8},
    keywords = {question retrieval, re-ranking, community question answering},
    location = {Leipzig, Germany},
    series = {WI '17}
    }

