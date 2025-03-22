#! /home/omert/miniconda3/bin/python
#
import numpy as np
import pandas as pd
import time

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
#  sns.set()
sns.set_context("paper", rc={"grid.linewidth": 0.8})
sns.set_style("whitegrid", rc={"grid.linestyle": "--"})


import argparse

parser = argparse.ArgumentParser(description="Give something ...")
#  parser.add_argument("-mof_num", "--mof_num",
#                      type=int, required=True,
                    #  help="..")
parser.add_argument("-RESULT_DIR", type=str, required=True)
parser.add_argument("-delimiter", type=str, required=True)
parser.add_argument("-start", type=int, required=True)
parser.add_argument("-end", type=int, required=True)

args = parser.parse_args()
RESULT_DIR = args.RESULT_DIR

if args.delimiter == "space":
    df = pd.read_csv("%s/loss.csv" % args.RESULT_DIR, delimiter=' ')
else:
    df = pd.read_csv("%s/loss.csv" % args.RESULT_DIR, delimiter=' ')


def plot():
    plt.rcParams["figure.figsize"] = (12, 6) # set figure size
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #  ax.set_ylim(0, 0.1)
    ax.set_xlabel("#Epochs", size=12)
    ax.set_ylabel("Loss RMSE (eV)", size=12)
    ax.set_yscale('log')
    #  ax.locator_params(axis='y', nbins=10) # increase number of axis tick (x and y)
    column_names = df.columns.values
    n_epoch = df.shape[0]
    x = range(n_epoch)
    for column_name in column_names[args.start:args.end]:
        print(column_name)
        ax.plot(x, df[column_name], label=column_name)
        ax.legend(prop={'size': 10})

    #ax.plot(x, df["Train loss"], label="Train loss")
    #ax.plot(x, df["Validation loss"], label= "Validation loss")
    #ax.plot(x, df["MAE_cohesive_E_perAtom"], label= "Error")
    # same result with seaborn
    #palette = sns.color_palette("bright", 2)
    #sns.lineplot(ax=ax, x="Time", y="value",  hue='variable', markers=True,
    #             palette=palette, data=pd.melt(df, "Time"))
    #ax.xaxis.set_major_formatter(ticker.EngFormatter())
    plt.savefig("%s/singleTrain_model_loss_epochs" % RESULT_DIR, dpi=600)
    #  plt.show()
plot()

def multiTrainPlot():
    plt.rcParams["figure.figsize"] = (16, 10) # set figure siz
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.set_ylim(0, 2)
    ax.set_xlabel("Epochs", size=16)
    ax.set_ylabel("Loss RMSE (eV)", size=16)
    ax.locator_params(axis='y', nbins=10) # increase number of axis tick (x and y)
    ax.tick_params(axis='both', labelsize=16) # increase number of axis tick (x and y)
    for i, df in zip([4, 5, 6], [df1, df2, df3]):
        column_names = df.columns.values
        n_epoch = df.shape[0]
        x = range(n_epoch)
        #  for column_name in column_names[2:3]:
        ax.plot(x, df[column_names[args.start:args.end]], label="Model ${IR-MOF-%d}$" %i)
        ax.legend(prop={'size':16})
    #  plt.show()
    plt.savefig("%s/multiTrain_model_loss_epochs" % RESULT_DIR, dpi=300)

def plotTr():
    df = df1
    plt.rcParams["figure.figsize"] = (14, 10) # set figure siz

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim(0, 1)
    #ax.set_title("Eğitim ve Validasyon Kayıp (Loss) Grafiği")
    ax.set_xlabel("Devir Sayısı (Epoch)", size=14)
    ax.set_ylabel("Kayıp (Loss)", size=14)
    ax.locator_params(axis='y', nbins=10) # increase number of axis tick (x and y)
    ax.tick_params(axis='both', labelsize=12) # increase number of axis tick (x and y)
    column_names = df.columns.values
    n_epoch = df.shape[0]
    x = range(n_epoch)
    for column_name, label in zip(column_names[args.start:args.end], ["Eğitim Kaybı (Training Loss)", "Validasyon Kaybı (Validation Loss)"]):
        ax.plot(x, df[column_name], label=label)
        ax.legend(fontsize=12)

    #ax.plot(x, df["Train loss"], label="Train loss")
    #ax.plot(x, df["Validation loss"], label= "Validation loss")
    #ax.plot(x, df["MAE_cohesive_E_perAtom"], label= "Error")
    # same result with seaborn
    #palette = sns.color_palette("bright", 2)
    #sns.lineplot(ax=ax, x="Time", y="value",  hue='variable', markers=True,
    #             palette=palette, data=pd.melt(df, "Time"))
    #ax.xaxis.set_major_formatter(ticker.EngFormatter())
    #  plt.show()
    plt.savefig("%s/model_loss_epochs" % RESULT_DIR, dpi=300)

def dynamic_plot():
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ##ax.set_ylim(0, 1.2)
    while True:
        df = pd.read_csv("./mof5_model_hdnnp_forces_v4_v2/log.csv")#[["Time", "Train loss", "Validation loss"]]
        n_epoch = df.shape[0]
        x = range(n_epoch)
        ax.plot(x, df["Train loss"], label="Train loss")
        ax.plot(x, df["Validation loss"], label= "Validation loss")
        ax.plot(x, df["MAE_cohesive_E_perAtom"], label= "Error")
        # same result with seaborn
        #palette = sns.color_palette("bright", 2)
        #sns.lineplot(ax=ax, x="Time", y="value",  hue='variable', markers=True,
        #             palette=palette, data=pd.melt(df, "Time"))
        #ax.xaxis.set_major_formatter(ticker.EngFormatter())
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(60)

#  plotTr()
#  multiTrainPlot()
