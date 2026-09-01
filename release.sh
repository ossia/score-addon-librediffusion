#!/bin/bash
set -e
rm -rf release score-addon-librediffusion score-addon-librediffusion.zip
mkdir -p release

cp -rf LibreDiffusion cmake presets CMakeLists.txt addon.json LICENSE README.md release/
mkdir -p release/3rdparty/librediffusion
cp -rf 3rdparty/librediffusion/{src,tools,train-lora.py,pyproject.toml,uv.lock,LICENSE} release/3rdparty/librediffusion/

mv release score-addon-librediffusion
7z a score-addon-librediffusion.zip score-addon-librediffusion
