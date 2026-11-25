#!/usr/bin/env bash
# Build docs/report/reporte-final.pdf: report + annex with every screenshot in docs/evidence/.
# The PDF can show the AWS account id (screenshots): it is gitignored, deliver it directly.
set -euo pipefail
cd "$(dirname "$0")/../docs/report"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
# Keep each heading / table label with what follows: break the page if < N lines remain.
awk '/^## 2\.2 / {print "\\clearpage"} /^#+ / {print "\\FloatBarrier\\needspace{10\\baselineskip}"} /^\*\*[^*]+\*\*$/ {print "\\needspace{12\\baselineskip}"} {print}' \
  reporte-final.md > "$tmp/report.md"
{
  echo
  echo "Capturas de pantalla en orden cronológico (archivo \`docs/evidence/\`)."
  echo
  for img in ../evidence/*.png; do
    [ -e "$img" ] || continue
    name=$(basename "$img" .png)
    caption="${name:0:10} · ${name:11}"
    caption="${caption//-/ }"
    echo "![${caption}](${img}){width=88%}"
    echo
  done
} >> "$tmp/report.md"

pandoc "$tmp/report.md" -o reporte-final.pdf \
  --pdf-engine=xelatex --resource-path=.:.. --toc \
  -V mainfont="DejaVu Sans" -V monofont="DejaVu Sans Mono" -V fontsize=10pt \
  -V header-includes="\\usepackage{needspace}\\usepackage{placeins}" -V geometry:margin=2cm -V colorlinks=true -V linkcolor=blue -V toc-title="Contenido"
echo "OK: docs/report/reporte-final.pdf"
