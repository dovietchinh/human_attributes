import os
import cv2
import pandas as pd
import shutil
import random
import xml.etree.ElementTree as ET
import yaml
with open("params.yaml") as f:
    params = yaml.safe_load(f)["prepared"]

tasks = params["tasks"]
ROOT = params["ROOT"]
seed = params["seed"]
labels_dir = params["label_dir"]
labels_mapping_index = params["labels_mapping_index"]

def xml2csv(file):
    data = {}
    data["path"] = []
    data["age"] = []
    data["extract_age"] = []
    data["ub_length"] = []
    data["lb_length"] = []
    data["visible"] = []

    root = ET.parse(file).getroot()
    for children in root:
        if not children.tag=="image":
            continue
        name = children.attrib["name"]
        height = int(float(children.attrib["width"]))
        width = int(float(children.attrib["height"]))
        path = os.path.join(file.replace(".xml",""), name)
        data["path"].append(path)
        for attribute in box.getchildren():
            value = attribute.text
            if value =="unknow":
                value = -1
            if attribute.attrib["name"] != "extract_age":
                value = labels_mapping_index[attribute.attrib["name"]].index(value)
            else:
                value = int(value)
            data[attribute.attrib["name"]].append(value)
    df = pd.DataFrame(data)
    return df
            
def main():
    for task in tasks:
        file = os.path.join("labels_dir",task+".xml")
        df = xml2csv(file)
        df.to_csv(os.path.join(csv_dirs,task+".csv"))


if __name__ =='__main__':
    main()














        





