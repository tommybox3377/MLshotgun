from openpyxl import load_workbook, Workbook
import openpyxl
import datetime
import pickle
import pandas
import json
import os
import GlobalVars as gv

# user defined
pickle_jar_loc = gv.pickle_jar_loc

# init
catalog = f"{pickle_jar_loc}PickleCatalog.xlsx"
model_data = f"{pickle_jar_loc}ModelData.xlsx"
f = open(catalog, "a")
f.close()
f = open(model_data, "a")
f.close()

def pickle_grid(model, info):
    now = datetime.datetime.now()
    file_name = f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}.sav"
    pickle.dump(model, open(pickle_jar_loc + file_name, 'wb'))
    model_info = pandas.DataFrame(model.cv_results_).sort_values("rank_test_score")
    catalog_data = pandas.DataFrame(
        {
            "file_name": file_name,
            "model": [info[0]],
            "training_size": [info[1]],
            "best_score" : [model_info.iloc[0]["mean_test_score"]]
        }
    )
    if os.path.getsize(catalog) > 0:
        cat = pandas.read_excel(catalog, sheet_name="Catalog")
        cat = cat.append(catalog_data, ignore_index=True)
    else:
        cat = catalog_data
    with pandas.ExcelWriter(catalog, mode="w") as writer:
        cat.to_excel(writer, sheet_name="Catalog", index=False)
    mode = "w" if os.path.getsize(model_data) == 0 else "a"
    with pandas.ExcelWriter(model_data, mode=mode) as writer:
        model_info.to_excel(writer, sheet_name=file_name, index=False)


def unpickle_model(pickle_file_name):
    # return pickle.load(open(f"{pickle_jar_loc}\{pickle_file_name}", 'rb'))
    file = pickle_jar_loc + pickle_file_name
    print(file)
    return pickle.load(open(file, 'rb'))




