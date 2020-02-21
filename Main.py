import pandas

import Preprocess
import MessageMe
import DopeSheet
import Shoot


loan_data = pandas.read_excel(r"C:\Users\twmar\OneDrive\Documents\Data\MLsamples\LendingClub\Cleaner_Lending_club_workbook.xlsx")

X = loan_data.drop("loan_status", axis=1)
y = loan_data["loan_status"]

try:
    for scaled_data, scale in Preprocess.scale(X):
        for pca, n in Preprocess.pri_comp_an(scaled_data):
            for model, classifier in Shoot.classify(pca, y):
                DopeSheet.append_dope_sheet(model, classifier, scale, n)
except BaseException as e:
    print(e)
finally:
    # MessageMe.send_message("Test Done")
    i = input("done")


