from easse.sari import corpus_sari
from rouge import Rouge

rg1 = Rouge()
abs_path = 'score_files'
for i in range(20, 36):
    inp_file_name = 'score_files/txt/test_0' + str(i) + '.txt'
    # inp_file_name = 'txt/test_075.txt'
    csv_file_name = inp_file_name.replace(".txt", ".csv").replace("txt/", "csv/")
    score_file_name = csv_file_name.replace("test_", "score_").replace("csv/", "score/")
    ################################
    test_file = open(inp_file_name, 'r', encoding='utf-8')
    example_sentences = open(csv_file_name, 'w', encoding='utf-8')
    example_sentences.write("article,reference,decoded\n")

    pair = ""
    for row in test_file:
        if (row == "\n"):
            pair += "\n"
            example_sentences.write(pair)
            pair = ""
        else:
            row = row.replace("\n", "").replace(",", " ").replace("article: ", "").replace("ref: ", "").replace("dec: ",
                                                                                                                "").replace(
                '"', '')
            pair += row + ","

    test_file.close()
    example_sentences.close()

    ###############################
    file = open(csv_file_name, 'r', encoding='utf-8')
    file2 = open(score_file_name, 'w', encoding='utf-8')

    for i, row in enumerate(file):
        if (i == 0):
            file2.write("article,reference,decoded,rouge1,rouge2,rouge_L,sari\n")
            continue
        row = row.split("\n")[0]
        row = row.split(",")
        rough_score = rg1.get_scores(row[2], row[1])
        sari_score = corpus_sari(orig_sents=[row[0]], sys_sents=[row[2]], refs_sents=[[row[1]]])
        pair = row[0] + "," + row[1] + "," + row[2] + "," + str(rough_score[0]['rouge-1']['f']) + "," \
               + str(rough_score[0]['rouge-2']['f']) + "," + \
               str(rough_score[0]['rouge-l']['f']) + "," + str(sari_score) + "\n"
        file2.write(pair)
    print("score file with name", score_file_name, "written into disk")
    file2.close()
    file.close()
