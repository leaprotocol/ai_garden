ls ../../lea2
tree -L 3 --filelimit 50 --prune -f -i ../../lea2 | while read filepath; do
  [ -f "$filepath" ] && echo "File: $filepath; Type: $(file --mime-type -b "$filepath")"
  mimetype=$(file --mime-type -b "$filepath")
  if echo "$mimetype" | grep -qE 'text|application/(javascript|json|xml|x-sh|x-python|x-markdown)'; then
    echo -n "<<<"
    head -c 1000 "$filepath" | tr '\n' ' '
    echo ">>>"
  fi
done > output