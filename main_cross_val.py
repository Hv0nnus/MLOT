#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This code is part of the paper :
# IJCAI 2020 paper "Metric Learning in Optimal Transport for Domain Adaptation"
# Written by Tanguy Kerdoncuff
# If there is any bug, don't hesitate to send me a mail to my personal email:
# tanguy.kerdoncuff@laposte.net

# This code can be run only if the cross validation file are already saved.
# IMPORTANT : The -r parameter can solve some bug by deleting wrong run of the cross validation and reformating the
# pickle file.
# The parameter "rule" is used to chose how we find the best hyperparameter set.
# The easiest way is to use max which take the best set of hyperparameter.
# This is not the only solution, for example, "mean" is also implemented which take the best hyperparameter on average.
# "precentile" is also implemented.

import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import argparse
import sklearn


def plot_cross_val(name="SA", source="amazon", target="caltech10", param_to_order="d", feature="surf",
                   other_plots=[([], [], "b")], list_iter=None):
    if source == target:
        return
    if list_iter is None:
        pickle_in = open("pickle/" + feature + "/" + feature + name + source + target + ".pickle", "rb")
        #         pickle_in = open("pickle/" + "debug" + name + source + target + ".pickle", "rb")
        list_iter = pickle.load(pickle_in)
    fig1, axes = plt.subplots(figsize=(20, 10))
    x_axis = []
    for other_plot in other_plots:
        axes.plot(other_plot[0], other_plot[1], other_plot[2])
    for i, dict_i in enumerate(list_iter):
        param = dict_i[param_to_order]
        if False:
            axes.plot([param for _ in range(len(dict_i["result"]))],
                      dict_i["result"],
                      "+r",
                      alpha=0.05)
            axes.plot([param for _ in range(len(dict_i["target_result"]))],
                      dict_i["target_result"],
                      "+b",
                      alpha=0.05)
        axes.plot([param], np.mean(dict_i["result"]), "or")
        axes.plot([param], np.mean(dict_i["target_result"]), "ob")
        x_axis.append(param)
    if param_to_order in ["reg_e", "reg_cl", "reg_l", "margin"]:
        axes.set_xscale('log')
    axes.set_xticks(x_axis)
    axes.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes.set_xlabel(param_to_order)
    axes.set_title("Feature : " + feature + "  Dataset : "
                   + source.upper()[0] + '->' + target.upper()[0] + " Algo : " + name)
    plt.show()


def reduce_list_iter(list_iter, select_param_around):
    """Useless in classic case, never used in cross val"""
    def find_nearest(array, value):
        """find the value of the list closest to the value"""
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    nearest = {}
    for j in select_param_around:
        if j not in nearest:
            nearest[j] = []
        for i, dict_i in enumerate(list_iter):
            nearest[j].append(dict_i[j])
        nearest[j] = np.unique(nearest[j])
        nearest[j] = find_nearest(nearest[j], select_param_around[j])

    i = 0

    while i < len(list_iter):
        counter = 0
        for j in select_param_around:
            if nearest[j] == list_iter[i][j]:
                counter += 1
        if counter < len(select_param_around):
            del list_iter[i]
        else:
            i += 1


def best_test(name="SA", source="amazon", target="caltech10", rule="mean", feature="surf",
              param_to_choose="d", launch_code_associated=False, verbose=False, type_classifier="1NN",
              show_plot=False, select_param_around={}):
    pickle_in = open("pickle/" + feature + "/" + feature + name + source + target + ".pickle", "rb")
    list_iter = pickle.load(pickle_in)
    pickle_in.close()
    if len(select_param_around) > 0:
        reduce_list_iter(list_iter, select_param_around)

    param_dict = {}
    param_dict_target = {}
    for i, dict_i in enumerate(list_iter):
        param = dict_i[param_to_choose]  # it may crash here, use the -r to reformat the pickle file
        if param not in param_dict:
            param_dict[param] = []
            param_dict_target[param] = []
        param_dict[param].append(dict_i["result"])
        param_dict_target[param].append(dict_i["target_result"])
    best_param = []
    other_plots = [([], [])]
    for key in sorted(param_dict):
        # print(key)
        try:
            source_temp = np.mean(param_dict[key], axis=1)
            target_temp = np.mean(param_dict_target[key], axis=1)
        except:
            i = 0
            while i < len(param_dict[key]):
                if len(param_dict[key][i]) != 100 or len(param_dict_target[key][i]) != 10:
                    del param_dict[key][i]
                    del param_dict_target[key][i]
                else:
                    i += 1
            source_temp = np.mean(param_dict[key], axis=1)
            target_temp = np.mean(param_dict_target[key], axis=1)
        param_dict[key] = source_temp
        param_dict_target[key] = target_temp
        if verbose:
            print(key, end=" ")

        if rule == "mean":
            best_param.append((np.mean(param_dict[key]), np.mean(param_dict_target[key]), key))
        elif "percentile" in rule:
            percentile = int((int(rule.split("_")[1]) / 100) * len(param_dict[key]))
            max_percent = int(np.argpartition(param_dict[key], percentile)[percentile])
            best_param.append((param_dict[key][max_percent],
                               param_dict_target[key][max_percent],
                               key))
        elif rule == "max":
            arg_max = np.argmax(param_dict[key])
            best_param.append((param_dict[key][arg_max],
                               param_dict_target[key][arg_max],
                               key))

        if verbose:
            print("Mean source {:4.1f}".format(np.mean(param_dict[key])),
                  "Mean target {:4.1f}".format(np.mean(param_dict_target[key])))

    best_param = np.array(best_param)
    arg_max = np.argmax(best_param[:, 0])
    other_plots = [(best_param[:, 2], best_param[:, 0], 'r'), (best_param[:, 2], best_param[:, 1], 'b')]
    other_plots.append(([best_param[arg_max, 2], best_param[arg_max, 2]],
                        [best_param[arg_max, 1] - 10, best_param[arg_max, 1] + 10], 'g'))
    if show_plot:
        plot_cross_val(name=name, source=source, target=target, param_to_order=param_to_choose,
                       feature=feature, other_plots=other_plots, list_iter=list_iter)
    return best_param[arg_max, 2], best_param[arg_max, 1]


def best_test_together(name="SA", source="amazon", target="caltech10", rule="mean", feature="surf",
                       launch_code_associated=False, verbose=False, type_classifier="1NN",
                       show_plot=False,
                       list_param=["reg_e", "reg_cl", "margin", "reg_l", "max_iter", "d"],
                       select_param_around={}):
    pickle_in = open("pickle/" + feature + "/" + "test" + name + source + target + ".pickle", "rb")
    list_iter = pickle.load(pickle_in)
    if len(select_param_around) > 0:
        reduce_list_iter(list_iter, select_param_around)
    param_dict = []
    param_dict_target = []
    for i, dict_i in enumerate(list_iter):
        param_dict.append(dict_i["result"])
        param_dict_target.append(dict_i["target_result"])

    best_param = []
    other_plots = [([], [])]
    try:
        source_temp = np.mean(param_dict, axis=1)
        target_temp = np.mean(param_dict_target, axis=1)
    except:
        i = 0
        while i < len(param_dict):
            if len(param_dict[i]) != 100 or len(param_dict_target[i]) != 10:
                del param_dict[i]
                del param_dict_target[i]
            else:
                i += 1
        source_temp = np.mean(param_dict, axis=1)
        target_temp = np.mean(param_dict_target, axis=1)

    param_dict = source_temp
    param_dict_target = target_temp

    if "percentile" in rule:
        percentile1 = int((int(rule.split("_")[1]) / 100) * len(param_dict))
        percentile2 = int((int(rule.split("_")[2]) / 100) * len(param_dict))
        arg_percent = np.argpartition(param_dict,
                                      [i for i in range(percentile1, percentile2)])[percentile1:percentile2]
        best_param.append((param_dict[arg_percent], param_dict_target[arg_percent]))
    elif "max" in rule:
        arg_percent = param_dict.argsort()[-int(rule.split("_")[1]):][::-1]
        best_param.append((param_dict[arg_percent], param_dict_target[arg_percent]))
    dict_return = {}
    for key in list_param:
        dict_return[key] = []
        for run in range(len(arg_percent)):
            dict_return[key].append(list_iter[arg_percent[run]][key])
        dict_return[key] = np.mean(dict_return[key])

    return dict_return, best_param[0][1]


def all_param(name="SA", source="amazon", target="caltech10", rule="mean", feature="surf",
              launch_code_associated=False, verbose=False, type_classifier="1NN",
              list_param=['d', 'reg_e', 'reg_cl', 'reg_l', 'max_iter', 'margin'],
              best_param={'d': -1, 'reg_e': -1, 'reg_cl': -1, 'reg_l': -1, 'max_iter': -1, 'margin': -1},
              show_plot=False,
              how_to_select="per_features",
              select_param_around={}):        
    from main import main
    print(source, target, name, rule, feature)
    best_result = []
    if how_to_select == "per_features":
        for param in list_param:
            best_param[param], a = best_test(name=name,
                                             source=source,
                                             target=target,
                                             rule=rule,
                                             feature=feature,
                                             param_to_choose=param,
                                             launch_code_associated=launch_code_associated,
                                             show_plot=show_plot,
                                             verbose=verbose,
                                             type_classifier=type_classifier,
                                             select_param_around=select_param_around)
            best_result.append(a)
    elif how_to_select == "together":
        best_param_temps, a = best_test_together(name=name,
                                                 source=source,
                                                 target=target,
                                                 rule=rule,
                                                 feature=feature,
                                                 launch_code_associated=launch_code_associated,
                                                 show_plot=show_plot,
                                                 verbose=verbose,
                                                 type_classifier=type_classifier,
                                                 list_param=list_param,
                                                 select_param_around=select_param_around)
        best_result.append(a)
        for key in best_param_temps:
            best_param[key] = best_param_temps[key]

    print(best_param)
    # print("")

    if launch_code_associated:
        which_dataset = 0
        if "surf" in feature or "decaf6" in feature or "GoogleNet1024" in feature:
            if source == "amazon":
                if target == "dslr":
                    which_dataset += 1
                if target == "webcam":
                    which_dataset += 2
            elif source == "caltech10":
                which_dataset += 3
                if target == "dslr":
                    which_dataset += 1
                if target == "webcam":
                    which_dataset += 2
            elif source == "dslr":
                which_dataset += 6
                if target == "caltech10":
                    which_dataset += 1
                if target == "webcam":
                    which_dataset += 2
            elif source == "webcam":
                which_dataset += 9
                if target == "caltech10":
                    which_dataset += 1
                if target == "dslr":
                    which_dataset += 2
        elif "office31fc" in feature:
            if source == "amazon":
                if target == "webcam":
                    which_dataset += 1
            elif source == "dslr":
                which_dataset += 2
                if target == "webcam":
                    which_dataset += 1
            elif source == "webcam":
                which_dataset += 4
                if target == "dslr":
                    which_dataset += 1
        elif source == "USPS_vs_MNIST":
            which_dataset = 1

        if "surf" in feature:
            featuresToUse = "surf"
        elif "decaf6" in feature:
            featuresToUse = "decaf6"
        elif "office31fc6" in feature:
            featuresToUse = "office31fc6"
        elif "office31fc7" in feature:
            featuresToUse = "office31fc7"
        else:
            featuresToUse = feature
        return main(featuresToUse=featuresToUse,
                    numberIteration=1 + 9 * (name in ["SA", "OTDA_pca", "MLOT", "OTSAML"]),
                    adaptationAlgoUsed=[name],
                    type_classifier=type_classifier,
                    d=int(best_param["d"]),
                    reg_e=best_param["reg_e"],
                    reg_cl=best_param["reg_cl"],
                    reg_l=best_param["reg_l"],
                    reg_pca=best_param["reg_pca"],
                    lr=best_param["lr"],
                    max_iter=int(best_param["max_iter"]),
                    max_inner_iter_sink=10,
                    max_inner_iter_grad=1,
                    new_space=True,
                    margin=best_param["margin"],
                    ML_init="full_identity",
                    verbose=False,
                    save_pickle=False,
                    cross_val=False,
                    which_dataset=which_dataset,
                    test_exp=False,
                    pickle_name="testDecaf6",
                    specific_comparaison="None",
                    svd_solver=args.svd_solver,
                    cuda=args.cuda,
                    rule=args.rule_OTSAML,
                    detach=args.detach), best_result, best_param
    else:
        return best_result, best_param


def all_dataset(name="SA", rule="mean", feature="surf",
                launch_code_associated=False,
                launch_code_associated_average=False,
                verbose=False, type_classifier="1NN",
                datasets_s=["amazon", "caltech10", "dslr", "webcam"],
                datasets_t=["amazon", "caltech10", "dslr", "webcam"],
                show_plot=False,
                save_plot=None,
                list_param=None,
                param_default={'d': -1, 'reg_e': -1, 'reg_cl': -1, 'reg_l': -1, 'max_iter': -1, 'margin': -1},
                how_to_select="per_features",
                select_param_around={},
                save_pickle=None,
                reformat_pickle=0):
    if list_param == ['']:
        if name in ["CORAL", "NA", "Tused"]:
            list_param = ["d"]  # useless but avoid a bug
        if name in ["SA", "TCA"]:
            list_param = ["d"]
        elif name in ["OT"]:
            list_param = ["reg_e"]
        elif name in ["OTDA"]:
            list_param = ["reg_e", "reg_cl"]
        elif name in ["OTDA_pca", "JDA"]:
            list_param = ["reg_e", "reg_cl", "d"]
        elif name in ["MLOT_id"]:
            list_param = ["reg_e", "reg_cl", "margin", "reg_l", "max_iter"]
        elif name in ["MLOT"]:
            list_param = ["reg_e", "reg_cl", "margin", "reg_l", "max_iter", "d"]
        elif name in ["OTSAMLid"]:
            list_param = ["reg_e", "reg_cl", "margin", "reg_l", "lr", "reg_pca", "max_iter"]
        elif name in ["OTSAML"]:
            list_param = ["reg_e", "reg_cl", "margin", "reg_l", "lr", "d", "reg_pca", "max_iter"]
        elif name in ["JDOT"]:
            list_param = ["reg_l", "max_iter"]
        elif name in ["JDOTSVM"]:
            list_param = ["reg_l", "max_iter"]
        elif name in ["JDOTe"]:
            list_param = ["reg_l", "max_iter", "reg_e"]
        elif name in ["JDOTSVMe"]:
            list_param = ["reg_l", "max_iter", "reg_e"]
        elif name in ["LMNN"]:
            list_param = ["margin", "d", "max_iter"]

    i = 0
    for source in datasets_s:
        for target in datasets_t:
            if target != source:
                i = i + 1
    mean_std = np.zeros((i, 2))
    i = 0
    best_param_all = None
    best_result_all = [0, 0]
    for source in datasets_s:
        for target in datasets_t:
            if target != source:
                # print(target, source)
                if reformat_pickle:
                    list_iter = []
                    pickle_in = open("pickle/" + feature + "/" + feature + name + source + target + ".pickle", "rb")
                    while True:
                        try:
                            pickle_loaded = pickle.load(pickle_in)
                            if type(pickle_loaded) is list:
                                list_iter = list_iter + pickle_loaded
                            else:
                                list_iter.append(pickle_loaded)
                        except:
                            break
                    if len(list_iter) == 1:
                        list_iter = list_iter[0]
                    pickle_in.close()
                    pickle_out = open("pickle/" + feature + "/" + feature + name + source + target + ".pickle", "wb")
                    pickle.dump(list_iter, pickle_out)
                    pickle_out.close()
                if launch_code_associated:
                    a = all_param(name=name,
                                  source=source,
                                  target=target,
                                  rule=rule,
                                  feature=feature,
                                  launch_code_associated=launch_code_associated,
                                  verbose=verbose,
                                  list_param=list_param,
                                  show_plot=show_plot,
                                  best_param=param_default,
                                  type_classifier=type_classifier,
                                  how_to_select=how_to_select,
                                  select_param_around=select_param_around)
                    a, best_result, best_param = a[0][source.upper()[0] + '->' + target.upper()[0]][name], a[1], a[2]
                    mean_std[i, 0] = a["mean"]
                    mean_std[i, 1] = a["std"]
                    # print(np.mean(best_result))
                    best_result_all[0] = best_result_all[0] + np.mean(best_result)
                    best_result_all[1] = best_result_all[1] + 1
                else:
                    best_result, best_param = all_param(name=name,
                                                        source=source,
                                                        target=target,
                                                        rule=rule,
                                                        feature=feature,
                                                        launch_code_associated=launch_code_associated,
                                                        verbose=verbose,
                                                        list_param=list_param,
                                                        show_plot=show_plot,
                                                        best_param=param_default,
                                                        type_classifier=type_classifier,
                                                        how_to_select=how_to_select,
                                                        select_param_around=select_param_around)
                    mean_std[i, 0] = np.mean(best_result)
                if best_param_all is None:
                    best_param_all = dict(param_default)
                    for key in list_param:
                        best_param_all[key] = [best_param[key]]
                else:
                    for key in list_param:
                        best_param_all[key].append(best_param[key])
                i += 1
    best_param_all_mean = {}
    for key in best_param_all:
        best_param_all_mean[key] = np.median(best_param_all[key])

    a = None
    if launch_code_associated_average:
        from main import main
        if "surf" in feature:
            featuresToUse = "surf"
        elif "decaf6" in feature:
            featuresToUse = "decaf6"
        elif "office31fc6" in feature:
            featuresToUse = "office31fc6"
        elif "office31fc7" in feature:
            featuresToUse = "office31fc7"
        else:
            featuresToUse = feature
        a = main(featuresToUse=featuresToUse,
                 numberIteration=1 + 9 * (name in ["SA", "OTDA_pca", "MLOT", "OTSAML"]),
                 adaptationAlgoUsed=[name],
                 type_classifier=type_classifier,
                 d=int(best_param_all_mean["d"]),
                 reg_e=best_param_all_mean["reg_e"],
                 reg_cl=best_param_all_mean["reg_cl"],
                 reg_l=best_param_all_mean["reg_l"],
                 reg_pca=best_param_all_mean["reg_pca"],
                 lr=best_param_all_mean["lr"],
                 max_iter=int(best_param_all_mean["max_iter"]),
                 max_inner_iter_sink=10,
                 max_inner_iter_grad=1,
                 new_space=True,
                 margin=best_param_all_mean["margin"],
                 ML_init="full_identity",
                 verbose=False,
                 save_pickle=False,
                 cross_val=False,
                 which_dataset=-1,
                 test_exp=False,
                 pickle_name="testDecaf6",
                 svd_solver=args.svd_solver,
                 specific_comparaison="None",
                 cuda=args.cuda,
                 rule=args.rule_OTSAML,
                 detach=args.detach)
    print(save_pickle + name + rule + feature + ".pickle")
    if save_pickle is not None:
        if type_classifier == "1NN":
            pickle_out = open(save_pickle + name + rule + feature + ".pickle", "wb")
        else:
            pickle_out = open(save_pickle + name + rule + feature + type_classifier + ".pickle", "wb")
        pickle.dump(mean_std, pickle_out)
        pickle.dump(best_param_all, pickle_out)
        if a is not None:
            pickle.dump(a, pickle_out)
        pickle_out.close()


parser = argparse.ArgumentParser(description='main_cross_val')
parser.add_argument('--featuresToUse', type=str, default='surf',
                    choices=["CaffeNet4096", "surf", "decaf6", "office31fc6", "office31fc7",
                             "CaffeNet4096_SA", "surf_SA", "decaf6_SA", "office31fc6_SA", "office31fc7_SA"])
parser.add_argument('--adaptationAlgoUsed', type=str, default='MLOT',
                    help="Name of the method that should be used. Method should be separate by comma if more than \
                         one method is needed (SA,TCA,NA)")
parser.add_argument('--type_classifier', type=str, default='1NN',
                    help="Final classifier used, only NN and SVM is implemented. 4NN will search for the 4\
                     NearestNeighbors. SVM_10.1 will launch the SVM with a margin of 10.1 ")
parser.add_argument('--rule', type=str, default='max',
                    help="Important parameter that design how to choose the best set of hyperparameter. If max is \
                         choosen, the best set will be the one that have the best result on the source dataset.\
                         If mean is choosen, for each hyperparameter, the one that give best result on average is\
                         selected. If percentile_50 is choosen, the hyperparameter selected is the one that give\
                         the best result order by median. When mean or percentile is used, each hyperparameter is\
                         choosen idependently from the other")
parser.add_argument('--datasets_s', type=str, default="amazon,caltech10,dslr,webcam")
parser.add_argument('--datasets_t', type=str, default="amazon,caltech10,dslr,webcam")

parser.add_argument('--d', type=int, default=70,
                    help="Dimension for PCA")
parser.add_argument('--reg_e', type=float, default=0.1,
                    help="Entropy regularisation of the sinkhorn method")
parser.add_argument('--reg_cl', type=float, default=0.1,
                    help="Class regularisation from OTDA")
parser.add_argument('--reg_l', type=float, default=1,
                    help="Metric Learning regularisation term")
parser.add_argument('--lr', type=float, default=0.001,
                    help="Learning rate of the gradient descent of the pytorch method")
parser.add_argument('--reg_pca', type=float, default=1,
                    help="Regularisation to force to not go far from PCA init, used only in the pytorch version")
parser.add_argument('--max_iter', type=int, default=10,
                    help="number max of iterations of MLOT")
parser.add_argument('--margin', type=float, default=10,
                    help="Margin used in LMNN (Large Margin Nearest Neighbors)")
parser.add_argument('--list_param', default="", type=str,
                    help="List of parameter to cross validate, if reg_l,d,reg_e in choosen, only this 3 hyperparameter\
                         will be selected, the other will be default value define above.")
parser.add_argument('--how_to_select', type=str, default="per_features", choices=["together", "per_features"],
                    help="if per_features is selected, the choose of hyperparameter will be independant from the other \
                         hyperparameters. If together is choosen, the code will compare all the value together. \
                         This parameter make sense only if there is more than one result for each set of \
                         hyperparameter, if not, this is similar too per_features with max rule.")
parser.add_argument('--select_param_around', default={},
                    help="This parameter is not working totally,the idea was to agregate close value of hyperparameter")
parser.add_argument('-r', "--reformat_pickle", action="store_true",
                    help="Use this list if the code crash. It will delete cross validation run that bug and save a \
                         clean pickle")
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-l", "--launch_code_associated", action="store_true",
                    help="If used, each dataset is launch with the best set of hyperparameter. A pickle is saved.")
parser.add_argument("-la", "--launch_code_associated_average", action="store_true")
parser.add_argument("-s", "--show_plot", action="store_true", help="Show plot")
parser.add_argument("--save_plot", default=None)
parser.add_argument("--save_pickle", default=None, help="Location of the pickle to save.")
parser.add_argument('--svd_solver', type=str, default="auto", choices=["auto", "full", "arpack", "randomized"],
                    help="Solver of the numpy PCA")

parser.add_argument("-cu", "--cuda", action="store_true",
                        help="Put the Neural Network on GPU (only Pytorch version)")
parser.add_argument('--rule_OTSAML', type=str, choices=["median", "max", "mean"], default="median",
                    help="Rule of normalization for the cost matrix of the transport")
parser.add_argument("-d", "--detach", action="store_true",
                    help="Will detach the normalization term of the cost matrix, it will be considered has a \
                    constant for pytorch")
args = parser.parse_args()

all_dataset(name=args.adaptationAlgoUsed, rule=args.rule, feature=args.featuresToUse,
            launch_code_associated=args.launch_code_associated,
            launch_code_associated_average=args.launch_code_associated_average,
            verbose=args.verbose, type_classifier=args.type_classifier,
            datasets_s=args.datasets_s.split(","),
            datasets_t=args.datasets_t.split(","),
            show_plot=args.show_plot,
            save_plot=args.save_plot,
            list_param=args.list_param.split(","),
            param_default={'d': args.d, 'reg_e': args.reg_e, 'reg_cl': args.reg_cl, 'reg_l': args.reg_l,
                           'max_iter': args.max_iter, 'margin': args.margin, 'lr': args.lr, 'reg_pca': args.reg_pca},
            how_to_select=args.how_to_select,
            select_param_around=args.select_param_around,
            save_pickle=args.save_pickle,
            reformat_pickle=args.reformat_pickle)
