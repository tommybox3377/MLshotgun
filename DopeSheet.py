import pandas
import os

# init
catalog = f"Dope_Sheet.xlsx"
f = open(catalog, "a")
f.close()


def append_dope_sheet(model, classifier, scale, pca_n):
    model_info = pandas.DataFrame(model.cv_results_).sort_values("rank_test_score")
    catalog_data = pandas.DataFrame(
        {
            "model": classifier,
            "scale": scale,
            "pca_n": pca_n,
            "best_score": [model_info.iloc[0]["mean_test_score"]],
            "best_params": [model_info.iloc[0]["params"]],
            "worse_score": [model_info.iloc[-1]["mean_test_score"]],
            "full_model_info": model_info.to_json()
        }
    )
    if os.path.getsize(catalog) > 0:
        cat = pandas.read_excel(catalog, sheet_name="Catalog")
        cat = cat.append(catalog_data, ignore_index=True)
    else:
        cat = catalog_data
    with pandas.ExcelWriter(catalog, mode="w") as writer:
        cat.to_excel(writer, sheet_name="Catalog", index=False)


