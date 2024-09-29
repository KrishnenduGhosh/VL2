# Augmentations for Video Lectures using CQA

# Introduction:
This project presents a video lecture augmentation system leveraging community question-answer (CQA) forums. Question-answer (QA) pairs, relevant to the course-relevant concepts, are retrieved to cater better insight to the learners. The proposed augmentation model comprises four modules: Concept extraction, Course-relevant concept identification, Retrieving augmentations (QA pairs) and Categorizing augmentations.
1. Concept Extraction: The segmented video lectures are indexed using its concepts, extracted with an entity annotation service.
2. Course-relevant concept identification: The course-relevant concepts are identified basde on their connectivity with other concepts in the concept space.
3. Retrieval of relevant QA pairs: For each course-relevant concept which needs augmentation, appropriate QA pairs are fetched.
4. Categorization of retrieved QA pairs: Each augmentation is classified.

# Steps:
## 1. Data collection:
Transcript data for 2581 Lectures present in 68 Courses (39 and 19 related to Computer science and Mathematics domains, respectively) collected from https://nptel.ac.in/course.html. These data are collected and stored in 1_Data folder in PDF format. Further details of the data is provided in 'Data.md' file. You can download the data from [https://drive.google.com/open?id=1KTWhbPk-N8_rz-p-wSIMo9nKKYWU7PU9](https://drive.google.com/drive/folders/1LtTD1bECyaQrlgJ74z0lVg6x8RlUjWud?usp=sharing)
## 2. Preprocessing:
Transcripts (PDFs) are converted into TXT format and pre-processed by removing spurious data (course metadata). The code '1_preprocess.py' converts and preprocesses the data from folder '1_Data' and stores in '2_Text' folder.
## 3. Concept Extraction:
A. Topics are extracted for each video lecture segments. The code '3_tag.py' extracts the topics and stores in '4_Topic' folder in JSON format.

B. Wikipedia articles are fetched for the extracted topics. The code '4_topic_out.py' extracts the articles and stores in '5_Topic_pkl' folder in Pickle format.

C. Outlinks for the extracted Wikipedia articles are extracted to generate the concept-graph. The code '5_course_out.py' extracts the backlinks and stores in '6_Course_pkl' folder in Pickle format.

D. The concepts from '4_Topic' folder is shown to the annotators and the annotated concepts arestored in the '4_Annotated' folder in JSON format.
## 4. Course-relevant topic Identification:
The course-relevant topics are identified automatically. The code '6_course_predict.py' identifies the course-relevant topics, stores them in '7_course' folder and also evaluates the concerned modules.
## 5. Retrieving augmentations:
QA pairs relevant to each of the course-relevant topics are retrieved. The code '7_retrieval.py' retrieves the QA pairs and stores in '8_Retrieved' folder in JSON format and the associated evaluation as 'RT.txt' inside '8_Result/trec-eval/test' folder in TREC suggested text format. The retrieved QA pairs are shown to the annotators and their relevance are tagged. The gold standard is present as 'GS1.txt' file inside '8_Result/trec-eval/test' folder. This '8_Result' folder is downloadable from https://drive.google.com/open?id=17-IxebyTtNsSXY98FfkTJWHK9goHhkOT which contains the folder 'trec-eval', providing the performance evaluation codes.
## 6. Categorizing augmentations:
The retrieved QA pairs are categorized using code '8_categorize.py' and stored in '9_Categorization' folder. These categorized QA pairs are again shown to the annotators and the associated gold-standard file 'GS2.txt' is stored inside '8_Result/trec-eval/test' folder. '8_categorize.py' also does the evaluate the categorization performance of the cocnerned module. The '8_Result' folder is downloadable from [https://drive.google.com/open?id=17-IxebyTtNsSXY98FfkTJWHK9goHhkOT](https://drive.google.com/drive/folders/1GEU8VBjIEItmsz8N5GcqvaOwzgtXcLyO?usp=sharing) which contains the folder 'trec-eval', providing the performance evaluation codes.

# Run:
## Prepare the pre-requisites:
A. One needs a ist of supporting files to be present in the current directory. One can download these files (as recipients of 'lib' folder) from [https://drive.google.com/open?id=11PJ0Y-3RavS2F0B8lj247M5pK19fK11I](https://drive.google.com/drive/folders/144lSB61RGqfjuuSTMz3ir7spOUrLgRiY?usp=sharing)

B. Geckodriver is also required. Download this from [https://drive.google.com/open?id=1Mf92NT_MNV-z2ZXVkkuneIGw7hLoe8n1](https://drive.google.com/file/d/1Rdkq4OSDVSJG2aekY1_8OVBFs9hF2A3L/view?usp=sharing) and export it in PATH before running the codes.

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

The module on retrieving questions is discussed in details in our paper 'Using Re-Ranking to Boost Deep Learning Based Community Question Retrieval' available at https://dl.acm.org/doi/pdf/10.1145/3106426.3106442.

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

