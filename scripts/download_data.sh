#!/bin/bash
mkdir -p ./data
cd data

wget https://dms.sztaki.hu/sites/dms.sztaki.hu/files/download/2018/uo17_data.zip
echo "UO17 data set was downloaded."
unzip uo17_data.zip
echo "Compressed UO17 data set was unzipped."
wget https://dms.sztaki.hu/sites/dms.sztaki.hu/files/download/2018/rg17_data_0.zip
echo "RG17 data set was downloaded."
unzip rg17_data_0.zip
echo "Compressed RG17 data set was unzipped."
