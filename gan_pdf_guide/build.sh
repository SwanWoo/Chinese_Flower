#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:/usr/local/bin:/Library/TeX/texbin:$PATH"

pandoc Chinese_Flower_GAN_Guide.md -o Chinese_Flower_GAN_Guide.pdf \
  --pdf-engine=xelatex \
  --template=templates/eisvogel.latex \
  -H header-eisvogel.tex \
  --resource-path=. \
  -V lang=zh-CN \
  -V mainfont="Times New Roman" \
  -V sansfont="Helvetica Neue" \
  -V monofont="Menlo" \
  -V CJKmainfont="Songti SC" \
  -V CJKsansfont="Heiti SC" \
  -V CJKmonofont="Heiti SC" \
  -V fontsize=11pt \
  -V geometry=margin=22mm \
  -V geometry=includehead=true \
  -V geometry=includefoot=true \
  -V linestretch=1.22 \
  -V titlepage=true \
  -V titlepage-color="F7F9FB" \
  -V titlepage-text-color="1F2933" \
  -V titlepage-rule-color="2F5D7C" \
  -V titlepage-rule-height=2 \
  -V toc-own-page=true \
  -V toc-title=目录 \
  -V colorlinks=true \
  -V linkcolor="NavyBlue" \
  -V urlcolor="NavyBlue" \
  -V book=false \
  -V code-block-font-size="\\small" \
  -V table-use-row-colors=false \
  -V caption-justification=centering \
  -V float-placement-figure=H \
  -V footnote-disable-urls=false \
  -V header-left="GAN 原理与 Chinese\_Flower 项目深度解析" \
  -V header-right="机器学习数学推导与源码精讲" \
  --toc --toc-depth=3 2>&1 | tee build.log

echo "Generated: $(pwd)/Chinese_Flower_GAN_Guide.pdf"
