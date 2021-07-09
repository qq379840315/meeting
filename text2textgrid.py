# -*- coding: utf-8 -*-
"""
Process the textgrid files
"""
import argparse
import codecs
from distutils.util import strtobool
from pathlib import Path
import textgrid
import pdb

class Segment(object):
    def __init__(self, uttid, spkr, stime, etime, text):
        self.uttid = uttid
        self.spkr = spkr
        self.stime = round(stime, 2)
        self.etime = round(etime, 2)
        self.text = text

    def change_stime(self, time):
        self.stime = time

    def change_etime(self, time):
        self.etime = time


def get_args():
    parser = argparse.ArgumentParser(description="process the textgrid files")
    parser.add_argument("--path", type=str, required=True, help="Data path")
    args = parser.parse_args()
    return args


def main(args):
    text = codecs.open(Path(args.path) / "text", "r", "utf-8")
    # get the path of textgrid file for each utterance
    spk2textgrid = {}
    xmin = 0
    xmax = 0
    for line in text:
        uttlist = line.split()
        utt_id = uttlist[0]
        if utt_id == "编号":
            continue
        utt_text = uttlist[1]
        utt_use = uttlist[2]
        utt_time_s, utt_time_e=uttlist[-1].strip('[').strip(']').split('][')
        if utt_use == "有效":
            utt_speaker = uttlist[3]
            if utt_speaker not in spk2textgrid:
                spk2textgrid[utt_speaker] = []
            xmax = max(xmax,float(utt_time_e))
            spk2textgrid[utt_speaker].append(
                Segment(
                    utt_id,
                    utt_speaker,
                    float(utt_time_s),
                    float(utt_time_e),
                    utt_text.strip(),
                )
            )
    text.close()
    #pdb.set_trace()
    #for segments in spk2textgrid.keys():
    #    spk2textgrid[segments] = sorted(spk2textgrid[segments], key=lambda x: x.stime)
    textgrid = codecs.open(Path(args.path) / "textgrid", "w", "utf-8")
    textgrid.write("File type = \"ooTextFile\"\n")
    textgrid.write("Object class = \"TextGrid\"\n\n")

    textgrid.write("xmin = %s\n" % (xmin))
    textgrid.write("xmax = %s\n" % (xmax))
    textgrid.write("tiers? <exists>\n")
    textgrid.write("size = %s\n" % (len(spk2textgrid)))
    textgrid.write("item []:\n")
    num_spk = 1
    for segments in spk2textgrid.keys():
        textgrid.write("\titem [%s]:\n" % (num_spk))
        num_spk = num_spk + 1
        textgrid.write("\t\tclass = \"IntervalTier\"\n")
        textgrid.write("\t\tname = \"%s\"\n" % spk2textgrid[segments][0].spkr)
        textgrid.write("\t\txmin = %s\n" % (xmin))
        textgrid.write("\t\txmax = %s\n" % (xmax))
        textgrid.write("\t\tintervals: size = %s\n" % (len(spk2textgrid[segments])))
        #pdb.set_trace()
        for i in range(len(spk2textgrid[segments])):
            #spk2textgrid[segments][i]
            #pdb.set_trace()
            textgrid.write("\t\tintervals [%s]\n" % (i+1))
            textgrid.write("\t\t\txmin = %s\n" % (spk2textgrid[segments][i].stime))
            textgrid.write("\t\t\txmax = %s\n" % (spk2textgrid[segments][i].etime))
            textgrid.write("\t\t\ttext = \"%s\"\n" % (spk2textgrid[segments][i].text))
            #textgrid.write("%s %s %s %s %s \n" % (spk2textgrid[segments][i].uttid, spk2textgrid[segments][i].spkr,spk2textgrid[segments][i].stime,
            #spk2textgrid[segments][i].etime,spk2textgrid[segments][i].text))
    textgrid.close()


if __name__ == "__main__":
    args = get_args()
    main(args)
