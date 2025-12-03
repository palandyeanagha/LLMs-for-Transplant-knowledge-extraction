The code files directory for this part contains the following executions:     

(i) step 1: rpys analysis files for each transplant (suitable for parallel runs) that take bibtex WoS metadata as well as pubmed csv metadata as input to generate peak years required further.   
(ii) step 2: acts as input for either manifest/weighted union scripts, depending on whether we are running additional bibliographic methods or not.  
(iii) step 3: manifest.py script outputs frome each transplant act as input for the bulk downloader scripts and similarly outputs from weighted union scripts act as input for the bulk downloader scripts for each transplant respectively. Manifest scripts are used to organise and ensure structure, whereas weighted union cleans DOIs and uses them to score and rank papers according to the bibliometric methods used.    
(iv) step 4: the bulk downloader finally downloads pdfs, checks validity using paper size and is multi-sourced.   

side notes: The pubmed_pipeline scripts are used to automate metadata download via pubmed. The scripts ending with _trial are for new experimentations that I am currently working on to increase corpus size.    
: drive link for passed on dataset: https://drive.google.com/drive/folders/12ww0h9DOlKGRUVDHqzEPY9B9p0hAb3wM?usp=drive_link  
