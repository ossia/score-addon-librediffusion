#!/bin/bash
rm -rf release
mkdir -p release

cp -rf LTC *.{hpp,cpp,txt,json} LICENSE release/

mv release score-addon-librediffusion
7z a score-addon-librediffusion.zip score-addon-librediffusion
