echo FileName, > nameFromData.csv
awk '{for(i=1;i<=NF;i++) if ($i=="comment") print $(i+1) ","}' $1 >> nameFromData.csv
