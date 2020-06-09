#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This code is part of the paper :
# IJCAI 2020 paper "Metric Learning in Optimal Transport for Domain Adaptation"
# Written by Tanguy Kerdoncuff
# If there is any bug, don't hesitate to send me a mail to my personal email:
# tanguy.kerdoncuff@laposte.net

import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# ----------------- Comparaison between cheating cross validation and our method ------------


def best_test_cheat(name="OT", source="amazon", target="caltech10", feature="surf_SA",
                    reformat_pickle=False, pickle_file="pickle"):
    if reformat_pickle:
        list_iter = []
        pickle_in = open(pickle_file + "/" + feature + "/" + feature + name + source + target + ".pickle", "rb")
        error = 0
        while True:
            try:
                pickle_loaded = pickle.load(pickle_in)
                if type(pickle_loaded) is list:
                    list_iter = list_iter + pickle_loaded
                else:
                    list_iter.append(pickle_loaded)
            except:
                error += 1
                if error > 100:
                    break

        if len(list_iter) == 1:
            list_iter = list_iter[0]
        pickle_in.close()
        pickle_out = open(pickle_file + "/" + feature + "/" + feature + name + source + target + ".pickle", "wb")
        pickle.dump(list_iter, pickle_out)
        pickle_out.close()
    pickle_in = open(pickle_file + "/" + feature + "/" + feature + name + source + target + ".pickle", "rb")
    list_iter = pickle.load(pickle_in)
    pickle_in.close()

    best_target_result_cheat = 0
    for iter_i in list_iter:
        if np.mean(iter_i["target_result"]) > best_target_result_cheat:
            best_target_result_cheat = np.mean(iter_i["target_result"])

    best_source_result = 0
    for iter_i in list_iter:
        if np.mean(iter_i["result"]) > best_source_result:
            best_source_result = np.mean(iter_i["result"])
            best_target_result = np.mean(iter_i["target_result"])
    return best_target_result, best_target_result_cheat, len(list_iter)


def create_latex_cheat(names, feature, number_dataset=12, reformat_pickle=False, cheat=True,
                       pickle_file="pickle"):
    latex_tabular = {"dataset": []}
    nb_iter_cross = np.zeros((len(names), number_dataset))
    mean = np.zeros((len(names), number_dataset + 1))
    mean_cheat = np.zeros((len(names), number_dataset + 1))
    if number_dataset == 12:
        datasets_s = ["amazon", "caltech10", "dslr", "webcam"]
        datasets_t = ["amazon", "caltech10", "dslr", "webcam"]
    else:
        datasets_s = ["amazon", "dslr", "webcam"]
        datasets_t = ["amazon", "dslr", "webcam"]
    i = 0
    for name in names:
        dataset_j = 0
        for source in datasets_s:
            for target in datasets_t:
                if target != source:
                    mean[i, dataset_j], mean_cheat[i, dataset_j], nb_iter_cross[i, dataset_j] = \
                        best_test_cheat(name=name,
                                        source=source,
                                        target=target,
                                        feature=feature,
                                        reformat_pickle=reformat_pickle,
                                        pickle_file=pickle_file)
                    dataset_j += 1
                    if name == names[0]:
                        latex_tabular["dataset"].append(str(source[0].upper()) +
                                                        "$\rightarrow$" + str(target[0].upper()))
        i += 1
    mean_cheat[:, number_dataset] = np.mean(mean_cheat[:, :number_dataset], axis=1)
    mean[:, number_dataset] = np.mean(mean[:, :number_dataset], axis=1)

    for i, name in enumerate(names):
        if cheat:
            latex_tabular[name] = list(np.round(mean_cheat[i, :], 1))
        else:
            latex_tabular[name] = list(np.round(mean[i, :], 1))
        latex_tabular[name].append(np.round(np.mean(nb_iter_cross[i, :]), 1))
    if cheat:
        textbf = np.argmax(mean_cheat, axis=0)
    else:
        textbf = np.argmax(mean, axis=0)

    i = 0
    for name in names:
        for j in range(len(textbf)):
            if textbf[j] == i:
                latex_tabular[name][j] = "\textbf{" + str(latex_tabular[name][j]) + '}'
        i += 1

    latex_tabular["dataset"].append("AVG")
    latex_tabular["dataset"].append("nb")
    df = pd.DataFrame(latex_tabular)
    print(df.to_latex(index=False, escape=False))


def creat_all_cheat_tables(names="NA,SA,CORAL,TCA,OT,OTDA,OTDA_pca,MLOT_id,MLOT",
                           features="surf_SA,decaf6_SA,office31fc7_SA",
                           reformat_pickle=False):
    names = names.split(",")
    features = features.split(",")
    for feature in features:
        if "office" in feature:
            number_dataset = 6
        else:
            number_dataset = 12
        for cheat in [True, False]:
            print("\\newpage", "Feature :$", feature, "$ Cheat ", cheat, "\\newline")
            create_latex_cheat(names,
                               feature,
                               number_dataset=number_dataset,
                               reformat_pickle=reformat_pickle,
                               cheat=cheat,
                               pickle_file="pickle")


def average_result():
    """
    :return: Display the average result that are hard coded in the function. Quite ugly...
    """
    surf = np.array([31.4,
                     44.8,
                     45.8,
                     1.,
                     35.7,
                     44.5,
                     42.2,
                     48.2,
                     48.8,
                     0.9,
                     47.0,
                     49.7,
                     0.8])
    decaf6 = np.array([71.0,
                       79.4,
                       83.7,
                       0.5,
                       77.2,
                       83.4,
                       83.9,
                       83.2,
                       82.6,
                       0.5,
                       78.2,
                       84.7,
                       0.3])
    decaf7 = np.array([64.3,
                       64.7,
                       66.5,
                       0.2,
                       64.1,
                       64.1,
                       65.3,
                       65.3,
                       65.2,
                       0.1,
                       64.4,
                       66.2,
                       0.1])
    a = (surf * 12 + decaf6 * 12 + decaf7 * 6) / (12 + 12 + 6)
    a = list(a)
    i = 0
    print("AVG & ", end="")
    while i < len(a):
        #     print(i)
        if i in [2, 8, 11]:
            if i == 10:
                print(round(a[i], 1), "$\pm$", round(a[i + 1], 1), end="")
            else:
                print(round(a[i], 1), "$\pm$", round(a[i + 1], 1), "& ", end="")
            i += 2
        else:
            print(round(a[i], 1), "& ", end="")
            i += 1
    print("\\\\")

# ------------------- Complex Cross validation --------------------


def load_pickle_cross(name, rule, features, number_dataset=6, average=True):
    pickle_in = open("./pickle_latex/" + name + rule + features + ".pickle", "rb")
    specific_run = pickle.load(pickle_in)[:number_dataset,:]
    a = pickle.load(pickle_in)
    average_run = pickle.load(pickle_in)
    pickle_in.close()
    name_line = []
    keys_dataset = list(average_run.keys())
    name_aux = list(average_run[list(average_run.keys())[0]].keys())[0]
    if average:
        mean = []
        std = []
        for key in keys_dataset:
            mean.append(average_run[key][name_aux]["mean"])
            std.append(average_run[key][name_aux]["std"])
            average_run[key][name_aux]["mean"] = round(average_run[key][name_aux]["mean"], 1)
            average_run[key][name_aux]["std"] = round(average_run[key][name_aux]["std"], 1)
            if name not in ["MLOT_id", "OTDA", "OT", "NA", "Tused", "TCA", "CORAL", "JDA", "JDOT", "JDOTSVM", "LMNN",
                            "JDOTe", "JDOTSVMe"]:
                name_line.append(str(average_run[key][name_aux]["mean"])[:4] +
                                 " $\pm$ " + str(average_run[key][name_aux]["std"])
                                 [:min(4,len(str(average_run[key][name_aux]["std"])))])
            else:
                name_line.append(str(average_run[key][name_aux]["mean"])[:4])
        mean_mean = round(np.mean(mean), 1)
        std_mean = round(np.mean(std) ,1)
        if name not in ["MLOT_id", "OTDA", "OT", "NA", "Tused", "TCA", "CORAL", "JDA", "JDOT", "JDOTSVM", "LMNN",
                        "JDOTe", "JDOTSVMe"]:
            a = [str(mean_mean)[:4] + "$\pm$" + str(std_mean)[:min(4, len(str(std_mean)))]]
            return name_line + a, keys_dataset, mean
        else:
            return name_line + [str(mean_mean)[:4]], keys_dataset, mean
    else:
        specific_run = np.array(specific_run)
        for i in range(len(specific_run)):
            if name not in ["MLOT_id", "OTDA", "OT", "NA", "Tused", "TCA", "CORAL", "JDA", "JDOT", "JDOTSVM", "LMNN",
                            "JDOTe", "JDOTSVMe"]:
                name_line.append(str(round(specific_run[i, 0], 1))[:4] +
                                 "$\pm$" + str(round(specific_run[i, 1], 1))
                                 [:min(4,len(str(round(specific_run[i, 1], 1))))])
            else:
                name_line.append(str(round(specific_run[i, 0], 1))[:4])
        mean_mean = round(np.mean(specific_run[:, 0]), 1)
        std_mean = round(np.mean(specific_run[:, 1]), 1)
        if name not in ["MLOT_id", "OTDA", "OT", "NA", "Tused", "TCA", "CORAL", "JDA", "JDOT", "JDOTSVM", "LMNN",
                        "JDOTe", "JDOTSVMe"]:
            a = [str(mean_mean)[:4] + "$\pm$" + str(std_mean)[:min(4, len(str(std_mean)))]]
            return name_line + a, keys_dataset, specific_run[:, 0]
        else:
            return name_line + [str(mean_mean)[:4]], keys_dataset, specific_run[:, 0]


def create_latex(names, rule, features, average, number_dataset=12):
    latex_tabular = {"dataset": []}
    mean = np.zeros((len(names), number_dataset))
    i = 0
    for name in names:
        latex_tabular[name], latex_tabular["dataset"], mean[i, :] = load_pickle_cross(name, rule, features,
                                                                                      number_dataset=number_dataset,
                                                                                      average=average)
        i += 1
    textbf = np.argmax(mean, axis=0)
    textbf_mean = np.array([np.argmax(np.mean(mean, axis=1))])

    textbf = np.concatenate((textbf, textbf_mean))
    i = 0
    for name in names:
        for j in range(len(textbf)):
            if textbf[j] == i:
                if name in ["MLOT_id", "OTDA", "OT", "NA", "Tused", "TCA", "CORAL", "JDA", "JDOT", "JDOTSVM",
                            "LMNN", "JDOTe", "JDOTSVMe"]:
                    latex_tabular[name][j] = '\textbf{' + latex_tabular[name][j] + '}'
                else:
                    latex_tabular[name][j] = '\textbf{' + latex_tabular[name][j].split("$\pm$")[0] + '}' + \
                                             "$\pm$" + latex_tabular[name][j].split("$\pm$")[1]
        i += 1

    for i in range(len(latex_tabular["dataset"])):
        latex_tabular["dataset"][i] = latex_tabular["dataset"][i][0] + \
                                      "$\rightarrow$" + \
                                      latex_tabular["dataset"][i][-1]
    latex_tabular["dataset"].append("AVG")
    df = pd.DataFrame(latex_tabular)
    print(df.to_latex(index=False, escape=False))


def creat_latex_tables(name="NA,CORAL,SA,TCA,OT,OTDA,OTDA_pca,MLOT_id,MLOT",
                       rules="max",
                       features="surf_SA"):
    for feature in features.split(","):
        if "office" in feature:
            number_dataset = 6
        else:
            number_dataset = 12
        for rule in rules.split(","):
            for average in [False]:
                print("\\newpage", "Feature :", feature, "Rule :", rule, "Average :",
                      average, "\\newline")
                create_latex(name.split(","), rule, feature, average, number_dataset=number_dataset)


# --------------------- Display the images --------------------------

def load_pickle(path, features):
    pickle_in = open("./pickle_specific/" + path + features + ".pickle", "rb")
    param = []
    result = []
    while True:
        try:
            param.append(pickle.load(pickle_in))
            result.append(pickle.load(pickle_in))
        except:
            break
    pickle_in.close()
    return param, result


def plot_image(param, result, comparaison, color_map='Greys', save_image=False):
    unique_params_X = np.sort(np.unique(param[:, 0]))
    unique_params_Y = np.sort(np.unique(param[:, 1]))
    result_mean = np.zeros((len(unique_params_X), len(unique_params_Y)))
    for i in range(len(unique_params_X)):
        for j in range(len(unique_params_Y)):
            for k in range(len(result)):
                if param[k, 0] == unique_params_X[i]:
                    if param[k, 1] == unique_params_Y[j]:
                        if comparaison:
                            result_mean[i, j] = result[k, 1] - result[k, 0]
                        else:
                            result_mean[i, j] = result[k, 1]
    X, Y = unique_params_X, unique_params_Y
    Z = np.transpose(result_mean)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)

    if comparaison:
        im = ax.imshow(Z, cmap=color_map, vmin=0, vmax=6, extent=(-0.5, 6.5, -0.5, 4.5))
    else:
        im = ax.imshow(Z, cmap=color_map, vmin=43, vmax=55, extent=(-0.5, 6.5, -0.5, 4.5))

    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_xticks(np.arange(len(X)))
    if comparaison:
        ax.set_yticks(np.arange(len(Y))[::-1])
    else:
        ax.set_yticks([])
    ax.set_xticklabels(X, size=20)
    if comparaison:
        ax.set_yticklabels(Y, size=20)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel('Entropy regularization', size=20)
    if comparaison:
        ax.set_ylabel('Classes regularization', size=20)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(Y)):
        for j in range(len(X)):
            if comparaison:
                text = ax.text(j, i, "+" + str(round(Z[(len(Y) - 1) - i, j], 1)),  # from top to bottom
                               ha="center", va="center", color="k", size=20)
            else:
                text = ax.text(j, i, str(round(Z[(len(Y) - 1) - i, j], 1)),
                               ha="center", va="center", color="k", size=20)

    fig.tight_layout()
    form = "pdf"
    if save_image:
        if comparaison:
            plt.savefig("./PDF/MLOT_comparaison_heatmap." + form, format=form, bbox_inches='tight')
        else:
            plt.savefig("./PDF/MLOT_alone_heatmap." + form, format=form, bbox_inches='tight')
    plt.show()


def organise_param_result(param, result):
    param_list = np.zeros((len(param), 2))
    result_list = np.zeros((len(param), 12, 2))
    for i in range(len(param)):
        param_list[i, 0] = param[i]["reg_e"]
        param_list[i, 1] = param[i]["reg_cl"]
        k = 0
        for key_dataset in result[i]:
            j = 0
            # Different names can appear here.
            if "MLOT_SS_TT" in list(result[i][key_dataset].keys()) and "OT" in list(result[i][key_dataset].keys()):
                list_key_algo = ["OT", "MLOT_SS_TT"]
            else:
                list_key_algo = ["OTDA", "MLOT"]

            for key_algo in list_key_algo:
                result_list[i, k, j] = result[i][key_dataset][key_algo]["mean"]
                j += 1
            k += 1
    result_list = np.mean(result_list, axis=1)
    return param_list, result_list


def display_image(path="OTDAvsMLOT",
                  features="surf",
                  color_map="Greens",
                  comparaison=True,
                  save_image=False):
    param, result = load_pickle(path, features)
    param_list, result_list = organise_param_result(param, result)
    plot_image(param=param_list, result=result_list, comparaison=comparaison, color_map=color_map,
               save_image=save_image)
